"""Read saved Adam endpoints on a frozen eligible manifest; optimizer steps = 0."""
import argparse
import copy
import time
import numpy as np
import torch
import runtime as rt
from data import inputs, batch_data
import effect_common as common
from measure_detail import measure, assert_legacy_equal

PREVIOUS = rt.ROOT / 'experiments/c_effective_update_v1'


def frozen_check():
    frozen = rt.read_json(rt.E / 'PROTOCOL_FREEZE.json')
    for name, expected in frozen['files'].items():
        assert rt.sha(rt.E / name) == expected, name
    for name, expected in frozen['reused_sources'].items():
        assert rt.sha(name) == expected, name
    return rt.sha(rt.E / 'PROTOCOL_FREEZE.json')


def original_entry_records():
    records = {}
    for path in sorted((PREVIOUS / 'pairs').glob('pair_*.json')):
        old = rt.read_json(path)
        for role in ('update', 'observer'):
            index = old[role + '_index']
            assert index not in records
            records[index] = (old, role)
    return records


def run(shard, shards):
    protocol = frozen_check()
    rt.setup()
    state = inputs()
    model, optimizer, checkpoint, _, _ = common.initialize()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    theta = torch.cat([p.detach().reshape(-1) for p in parameters]).clone()
    initial = rt.tensor_hash(model.state_dict())
    initial_opt = common.optimizer_hash(optimizer.state_dict())
    frozen_hash = rt.tensor_hash(rt.frozen_state(model))
    bn_hash = rt.tensor_hash(rt.bn_state(model))
    rng = common.get_rng()
    rng_digest = common.rng_hash()
    plan = rt.read_json(rt.E / 'PAIR_PLAN.json')
    old_entries = original_entry_records()
    endpoint_path = rt.E / 'replay/endpoints.npz'
    with np.load(endpoint_path, allow_pickle=False) as source:
        recovered_endpoints = {name: source[name].copy() for name in source.files}
    records = []
    began = time.perf_counter()

    for pair in plan['pairs'][shard::shards]:
        pair_id = pair['pair_id']
        destination = rt.E / 'pairs' / f'pair_{pair_id:02d}.json'
        if destination.exists():
            raise FileExistsError(destination)
        update_old, update_role = old_entries[pair['update_index']]
        assert update_role == 'update', 'Saved update endpoint must use exactly this update batch'
        old_id = update_old['pair_id']
        raw_path = PREVIOUS / 'pairs' / f'pair_{old_id:02d}.npz'
        old_path = raw_path.with_suffix('.json')
        assert pair['reuse_update_from_pair_id'] == old_id
        assert update_old['initial_model_sha256'] == initial
        assert update_old['initial_optimizer_sha256'] == initial_opt
        assert update_old['initial_rng_sha256'] == rng_digest
        assert update_old['parameter_names_in_vector_order'] == [n for n, _ in named]
        with np.load(raw_path, allow_pickle=False) as source:
            raw = {name: source[name].copy() for name in source.files}
        assert torch.equal(theta, torch.as_tensor(raw['initial_parameters'], device=theta.device))
        batches = {role: batch_data(pair[role + '_index'], state) for role in ('update', 'observer')}
        caches = {role: common.cache(pair[role + '_index'], batches[role]) for role in batches}
        entries = {role: plan['entries'][str(pair[role + '_index'])] for role in batches}
        cohorts = {role: rt.read_json(entries[role]['cohort_path']) for role in batches}
        for role in batches:
            assert rt.sha(entries[role]['cohort_path']) == entries[role]['cohort_sha256']
            for key in ('input_sha256', 'label_sha256', 'auxiliary_input_sha256'):
                assert batches[role][key] == entries[role][key], (pair_id, role, key)

        def observe(role, baseline=None):
            with torch.no_grad():
                z = common.forward(model, batches[role])
                risk = common.risk(z, caches[role])
                return measure(z, risk, cohorts[role], caches[role], entries[role]['windows'], baseline)

        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        common.set_rng(rng)
        baseline = {role: observe(role) for role in batches}
        for role in batches:
            reference_old, reference_role = old_entries[pair[role + '_index']]
            assert_legacy_equal(baseline[role], reference_old['baseline'][reference_role])
        result = {**pair, 'status': 'RUNNING', 'protocol_sha256': protocol,
            'initial_checkpoint_sha256': rt.sha(rt.INIT_CP), 'initial_model_sha256': initial,
            'initial_optimizer_sha256': initial_opt, 'initial_rng_sha256': rng_digest,
            'parameter_names_in_vector_order': [n for n, _ in named],
            'baseline': baseline, 'conditions': {}, 'initial_losses': update_old['initial_losses'],
            'gradients': update_old['gradients'], 'new_optimizer_steps': 0,
            'historical_parameter_updates_per_copy': 1, 'copies_accumulated': False,
            'saved_update_provenance': {'pair_id': old_id, 'json_path': str(old_path),
                'json_sha256': rt.sha(old_path), 'raw_path': str(raw_path), 'raw_sha256': rt.sha(raw_path)},
            'exact_endpoint_archive_sha256': rt.sha(endpoint_path),
            'cohort_sha256': {role: entries[role]['cohort_sha256'] for role in batches}}

        for condition in ('full_F', 'no_keep', 'no_track'):
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            # Rounded FP32 subtraction can lose low bits on near-zero parameters.
            # The saved archive resolves its finite inverse candidates with the
            # old complete-model hash, without recalculating an optimizer step.
            values = torch.as_tensor(recovered_endpoints[f'pair_{old_id:02d}_{condition}'], device=theta.device)
            assert torch.equal(values - theta, torch.as_tensor(raw['delta_' + condition], device=theta.device))
            offset = 0
            with torch.no_grad():
                for parameter in parameters:
                    parameter.copy_(values[offset:offset + parameter.numel()].reshape(parameter.shape))
                    offset += parameter.numel()
            assert offset == values.numel()
            expected = update_old['conditions'][condition]
            assert rt.tensor_hash(model.state_dict()) == expected['model_after_sha256'], (pair_id, condition)
            common.set_rng(rng)
            current = {role: observe(role, baseline[role]) for role in batches}
            assert_legacy_equal(current['update'], expected['roles']['update'])
            if pair['reuse_mode'] == 'original_pair':
                assert pair['observer_index'] == update_old['observer_index']
                assert_legacy_equal(current['observer'], expected['roles']['observer'])
            assert rt.tensor_hash(rt.frozen_state(model)) == frozen_hash
            assert rt.tensor_hash(rt.bn_state(model)) == bn_hash
            assert common.rng_hash() == rng_digest
            copied = copy.deepcopy(expected)
            copied['roles'] = current
            copied['reconstructed_model_hash_exact'] = True
            copied['new_optimizer_steps'] = 0
            copied['update_record_reused_from'] = str(old_path)
            result['conditions'][condition] = copied
        assert common.optimizer_hash(optimizer.state_dict()) == initial_opt
        assert all(p.grad is None for p in model.parameters())
        result['status'] = 'COMPLETE'
        result['replay_checks'] = {'old_baselines_exact': True, 'old_update_metrics_exact': True,
            'old_observer_metrics_exact': pair['reuse_mode'] == 'original_pair',
            'all_reconstructed_model_hashes_exact': True, 'optimizer_untouched': True,
            'no_backward': True, 'bn_frozen': True, 'rng_unchanged': True}
        rt.save_json(destination, result)
        records.append({'pair_id': pair_id, 'sha256': rt.sha(destination), 'new_optimizer_steps': 0})
        print({'pair_id': pair_id, 'source_relation': pair['source_relation'],
               'reused_update_pair': old_id, 'new_optimizer_steps': 0}, flush=True)
    frozen_check()
    rt.save_json(rt.E / 'pairs' / f'SHARD_{shard}.json', {'status': 'COMPLETE',
        'protocol_sha256': protocol, 'records': records, 'new_optimizer_steps': 0,
        'seconds': time.perf_counter() - began, 'initial_model_sha256': initial,
        'initial_optimizer_sha256': initial_opt, 'initial_rng_sha256': rng_digest})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=3)
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    run(args.shard, args.shards)
