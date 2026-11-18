"""Post-freeze, CPU-only read audit; never changes training state or protocol.

This script is deliberately outside the pre-training freeze. It independently
checks saved tensors and receipts and writes only evaluation audit artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import runtime as rt

CONDITIONS = ('ORIGINAL', 'REPLAY')
ADDITIONAL = (0, 128, 256, 512)


def optimizer_hash(state):
    tensors, scalars = {}, {}
    for key, values in state['state'].items():
        for name, value in values.items():
            if torch.is_tensor(value):
                tensors[f'{key}/{name}'] = value
            else:
                scalars[f'{key}/{name}'] = value
    encoded = json.dumps({'groups': state['param_groups'], 'scalars': scalars}, sort_keys=True)
    return hashlib.sha256((rt.tensor_hash(tensors) + encoded).encode()).hexdigest()


def frozen_identity():
    freeze = rt.read_json(rt.E / 'PROTOCOL_FREEZE.json')
    for name, digest in freeze['files'].items():
        assert rt.sha(rt.E / name) == digest, name
    for name, digest in freeze['reused_sources'].items():
        assert rt.sha(name) == digest, name
    assert rt.sha(rt.INIT_CP) == freeze['initial_checkpoint_sha256']
    assert 'code/postrun_audit.py' not in freeze['files']
    return freeze


def load_checkpoint(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    assert all(torch.isfinite(t).all() for t in value['model_state_dict'].values())
    assert all(torch.isfinite(t).all() for s in value['optimizer_state_dict']['state'].values()
               for t in s.values() if torch.is_tensor(t))
    return value


def initial():
    freeze = frozen_identity()
    reference = load_checkpoint(rt.INIT_CP)
    expected_model = rt.tensor_hash(reference['model_state_dict'])
    expected_optimizer = optimizer_hash(reference['optimizer_state_dict'])
    assert expected_optimizer == freeze['initial_optimizer_sha256']
    rows, sources = {}, {str(rt.INIT_CP): rt.sha(rt.INIT_CP)}
    for condition in CONDITIONS:
        folder = rt.E / 'training' / condition / 'run'
        run = rt.read_json(folder / 'RUN.json')
        path = folder / 'snapshot_2304.pth'
        cp = load_checkpoint(path)
        assert run['condition'] == cp['condition'] == condition
        assert run['seed'] == cp['seed'] == 46
        assert cp['step'] == run['start_step'] == 2304
        assert cp['additional_step'] == 0 and cp['additional_steps'] == run['additional_steps'] == 512
        assert run['initial_optimizer_restored']
        assert cp['initial_checkpoint_sha256'] == run['initial_checkpoint_sha256'] == rt.sha(rt.INIT_CP)
        assert cp['protocol_sha256'] == run['protocol_sha256'] == rt.sha(rt.E / 'PROTOCOL_FREEZE.json')
        assert cp['schedule_sha256'] == run['schedule_sha256'] == rt.sha(rt.E / (condition + '_SCHEDULE.csv'))
        assert cp['settings'] == run['settings'] == reference['settings']
        assert cp['trainable_parameter_names'] == run['trainable_parameter_names'] == reference['trainable_parameter_names']
        assert rt.tensor_hash(cp['model_state_dict']) == run['initial_state_sha256'] == expected_model
        assert optimizer_hash(cp['optimizer_state_dict']) == run['initial_optimizer_sha256'] == expected_optimizer
        assert len(cp['optimizer_state_dict']['state']) == 28
        assert all(float(s['step']) == 2304 for s in cp['optimizer_state_dict']['state'].values())
        snapshot = [r for r in run['snapshots'] if r['step'] == 2304]
        assert len(snapshot) == 1 and snapshot[0]['sha256'] == rt.sha(path)
        diagnostic = rt.read_json(folder / 'TRAIN_DIAGNOSTICS_000.json')
        assert diagnostic['additional_step'] == 0 and diagnostic['uses_development'] is False
        rows[condition] = dict(model_tensor_sha256=expected_model, optimizer_sha256=expected_optimizer,
            recorded_initial_rng_sha256=run['initial_rng_sha256'], checkpoint_sha256=rt.sha(path),
            initial_diagnostic_rows=diagnostic['rows'])
        sources[str(path)] = rt.sha(path)
        sources[str(folder / 'TRAIN_DIAGNOSTICS_000.json')] = rt.sha(folder / 'TRAIN_DIAGNOSTICS_000.json')
    assert rows['ORIGINAL']['recorded_initial_rng_sha256'] == rows['REPLAY']['recorded_initial_rng_sha256']
    initial_diagnostics_equal = rows['ORIGINAL']['initial_diagnostic_rows'] == rows['REPLAY']['initial_diagnostic_rows']
    for row in rows.values():
        del row['initial_diagnostic_rows']
    result = dict(status='PASS', phase='INITIAL', protocol_sha256=rt.sha(rt.E / 'PROTOCOL_FREEZE.json'),
        audit_kind='post_freeze_read_only_CPU_not_preregistered_training_code', audit_source_sha256=rt.sha(__file__),
        sources=sources, conditions=rows, initial_diagnostic_rows_exactly_equal=initial_diagnostics_equal,
        model_and_full_optimizer_exactly_equal_to_original_F2304=True,
        initial_rng_hash_equal=True,
        rng_evidence_limit='Full RNG states were not saved in checkpoints; equality uses training RUN initial RNG hashes.',
        added_optimizer_updates=0, model_inference_executed=False)
    destination = rt.E / 'evaluation/PAIRED_INITIAL_IDENTITY.json'
    if destination.exists():
        previous = rt.read_json(destination)
        assert previous['conditions'] == result['conditions']
        return previous
    rt.save_json(destination, result)
    return result


def final():
    start = initial()
    reference = load_checkpoint(rt.INIT_CP)
    names = set(reference['trainable_parameter_names'])
    before = reference['model_state_dict']
    rows, sources = {}, dict(start['sources'])
    for condition in CONDITIONS:
        folder = rt.E / 'training' / condition / 'run'
        run = rt.read_json(folder / 'RUN.json')
        assert run['status'] == 'complete' and run['final_step'] == 2816
        assert run['frozen_frontend_unchanged'] and run['bn_unchanged']
        assert [r['additional_step'] for r in run['snapshots']] == list(ADDITIONAL)
        assert [r['step'] for r in run['snapshots']] == [2304 + k for k in ADDITIONAL]
        logs = [json.loads(line) for line in (folder / 'TRAIN.jsonl').read_text().splitlines()]
        assert len(logs) == 512
        schedule = rt.read_json(rt.E / 'SCHEDULE.json')
        for i, row in enumerate(logs):
            assert row['condition'] == condition and row['step'] == 2305 + i and row['additional_step'] == i + 1
            assert row['original_plan_index'] == schedule['entries'][i]['original_index']
            assert torch.isfinite(torch.tensor([row['loss'], row['grad_norm']], dtype=torch.float64)).all()
        snapshots = []
        for additional, item in zip(ADDITIONAL, run['snapshots']):
            path = folder / item['path']
            assert rt.sha(path) == item['sha256']
            cp = load_checkpoint(path)
            assert cp['condition'] == condition and cp['step'] == 2304 + additional and cp['additional_step'] == additional
            assert len(cp['optimizer_state_dict']['state']) == 28
            assert all(float(s['step']) == 2304 + additional for s in cp['optimizer_state_dict']['state'].values())
            changed = [k for k, t in cp['model_state_dict'].items() if not torch.equal(t, before[k])]
            assert set(changed) <= names
            assert all(torch.equal(t, before[k]) for k, t in cp['model_state_dict'].items()
                       if k.endswith(('running_mean', 'running_var', 'num_batches_tracked')))
            diagnostic = rt.read_json(folder / f'TRAIN_DIAGNOSTICS_{additional:03d}.json')
            assert diagnostic['additional_step'] == additional and diagnostic['uses_development'] is False
            snapshots.append(dict(additional_step=additional, step=2304 + additional,
                changed_trainable_tensors=len(changed), checkpoint_sha256=rt.sha(path)))
            sources[str(path)] = rt.sha(path)
            sources[str(folder / f'TRAIN_DIAGNOSTICS_{additional:03d}.json')] = rt.sha(folder / f'TRAIN_DIAGNOSTICS_{additional:03d}.json')
        assert rt.tensor_hash(cp['model_state_dict']) == run['final_state_sha256']
        assert optimizer_hash(cp['optimizer_state_dict']) == run['final_optimizer_sha256']
        total_delta = sum((cp['model_state_dict'][k].double() - before[k].double()).square().sum().item() for k in names) ** .5
        assert total_delta > 0
        rows[condition] = dict(updates=512, final_Adam_step=2816, final_parameter_delta_L2=total_delta,
            frozen_frontend_exactly_unchanged=True, BN_buffers_exactly_unchanged=True, snapshots=snapshots)
        sources[str(folder / 'RUN.json')] = rt.sha(folder / 'RUN.json')
        sources[str(folder / 'TRAIN.jsonl')] = rt.sha(folder / 'TRAIN.jsonl')
    result = dict(status='PASS', phase='FINAL', protocol_sha256=rt.sha(rt.E / 'PROTOCOL_FREEZE.json'),
        audit_kind='post_freeze_read_only_CPU_not_preregistered_training_code', audit_source_sha256=rt.sha(__file__),
        sources=sources, initial_audit_sha256=rt.sha(rt.E / 'evaluation/PAIRED_INITIAL_IDENTITY.json'),
        conditions=rows, added_optimizer_updates=0, model_inference_executed=False,
        initial_model_optimizer_RNG_hashes_identical=True)
    rt.save_json(rt.E / 'evaluation/PAIRED_TRAINING_AUDIT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['initial', 'final'], required=True)
    args = parser.parse_args()
    result = initial() if args.phase == 'initial' else final()
    print(json.dumps({'status': result['status'], 'phase': result['phase']}), flush=True)
