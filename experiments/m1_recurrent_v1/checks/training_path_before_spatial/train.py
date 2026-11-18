"""Paired M1-only adaptation: fixed existing F exposure 256 x 8, final2048."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import time
import torch
import runtime as rt


def rng_state():
    import random
    import numpy as np
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import random
    import numpy as np
    random.setstate(state['python'])
    value = state['numpy']; np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def train(condition, resume=False, device='cuda'):
    out = rt.E / 'training' / condition / 'run'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'execution.lock').open('a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return train_locked(condition, out, resume, device)


def train_locked(condition, out, resume, device):
    freeze = rt.read_json(rt.E / 'PROTOCOL_FREEZE.json')
    assert freeze['source_sha256'] == rt.source_hashes()
    assert freeze['settings'] == rt.SETTINGS and freeze['model_settings'] == rt.MODEL_SETTINGS
    assert freeze['steps'] == rt.STEPS and freeze['exposure_plan'] == {'entries': 256, 'repetitions': 8}
    assert freeze['preflight_status'] == 'PASS'
    metadata_path = out / 'RUN.json'
    if metadata_path.exists() and rt.read_json(metadata_path)['status'] == 'complete':
        prior = rt.read_json(metadata_path)
        assert prior['final_step'] == rt.STEPS
        assert prior['source_sha256'] == rt.source_hashes()
        assert prior['snapshots'][-1]['sha256'] == rt.sha(out / 'snapshot_2048.pth')
        print({'condition': condition, 'status': 'ALREADY_COMPLETE'}, flush=True)
        return
    if metadata_path.exists() and not resume:
        raise FileExistsError('Existing incomplete run requires explicit --resume')
    model, initialization = rt.load_initialized(condition, device)
    assert freeze['initial_checkpoint_sha256'] == rt.sha(rt.INITIAL)
    assert freeze['initial_model_state_sha256'] == initialization['model_state_sha256']
    model.train()
    frozen = rt.frozen_f()
    state = frozen.data.inputs()
    cache_manifest = rt.frozen_cache_manifest()
    assert rt.SETTINGS == rt.read_json(rt.V2 / 'CALIBRATION.json')['fixed_settings']
    criterion = rt.load_module('_m1_official_loss', rt.MODEL_SOURCE).SoftLoUloss().to(device)
    parameters = {k: p for k, p in model.named_parameters() if p.requires_grad}
    assert parameters and all(not name.startswith(('original.', 'base.', 'backbone.')) for name in parameters)
    optimizer = torch.optim.Adam(list(parameters.values()), lr=1e-5, weight_decay=1e-4,
                                 betas=(.9, .999), eps=1e-8)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    original_hash = initialization['original_state_sha256']
    sources = rt.source_hashes(); start_step = 0
    metadata = {'status': 'running', 'condition': condition, 'mode': rt.MODES[condition],
        'seed': 46, 'steps': rt.STEPS, 'final_step': None, 'main_step': rt.STEPS,
        'initial_checkpoint_sha256': rt.sha(rt.INITIAL), 'release_checkpoint_sha256': rt.sha(rt.CP),
        'initial_model_state_sha256': initialization['model_state_sha256'],
        'original_state_sha256': original_hash, 'source_sha256': sources,
        'protocol_sha256': rt.protocol_hash(), 'cache_sha256': cache_manifest,
        'settings': rt.SETTINGS, 'model_settings': rt.MODEL_SETTINGS, 'runtime_settings': rt.RUNTIME_SETTINGS,
        'optimizer': {'name': 'fresh_Adam', 'lr': 1e-5,
            'weight_decay': 1e-4, 'betas': [.9, .999], 'eps': 1e-8},
        'exposure': 'existing F plan entries 0..255 in order repeated8; 4 detection+4 auxiliary windows; no new sampling',
        'checkpoint_policy': 'fixed2048; 128/512/1024 process only; latest128 for resume',
        'all_original_parameters_and_bn_frozen': True, 'trainable_parameter_names': list(parameters),
        'trainable_parameters': sum(p.numel() for p in parameters.values()),
        'parameters': sum(p.numel() for p in model.parameters()), 'precision': 'FP32_no_AMP_no_TF32',
        'torch_version': str(torch.__version__), 'snapshots': [], 'resume_events': []}
    log_path = out / 'TRAIN.jsonl'
    if resume:
        if not metadata_path.exists(): raise FileNotFoundError(metadata_path)
        previous = rt.read_json(metadata_path)
        for key in ('initial_checkpoint_sha256', 'release_checkpoint_sha256', 'source_sha256',
                    'protocol_sha256', 'cache_sha256', 'settings', 'initial_model_state_sha256'):
            assert previous[key] == metadata[key], ('Resume identity mismatch', key)
        checkpoint = torch.load(out / 'latest.pth', map_location='cpu', weights_only=True)
        assert checkpoint['condition'] == condition
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']; restore_rng(checkpoint['rng_state'])
        metadata = previous; metadata['status'] = 'running'; metadata.pop('error', None)
        metadata['snapshots'] = [x for x in metadata['snapshots'] if x['step'] <= start_step]
        metadata['resume_events'].append({'from_step': start_step, 'unix': time.time()})
        lines = log_path.read_text(encoding='utf-8').splitlines() if log_path.exists() else []
        retained = [line for line in lines if json.loads(line)['step'] <= start_step]
        assert len(retained) == start_step
        log_path.write_text(''.join(line + '\n' for line in retained), encoding='utf-8')
    elif log_path.exists() and log_path.stat().st_size:
        raise FileExistsError(log_path)
    rt.assert_frozen(model, original_hash)
    rt.save_json(out / 'INITIAL_STATE.json', {'model_state_sha256': initialization['model_state_sha256'],
        'initial_checkpoint_sha256': rt.sha(rt.INITIAL), 'original_state_sha256': original_hash,
        'condition': condition, 'seed': 46, 'trainable_parameter_names': list(parameters)})

    def snapshot(step, main=False):
        rt.assert_frozen(model, original_hash)
        checkpoint = {'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(), 'step': step, 'condition': condition,
            'mode': rt.MODES[condition], 'seed': 46, 'rng_state': rng_state(),
            'initial_checkpoint_sha256': metadata['initial_checkpoint_sha256'],
            'original_state_sha256': original_hash, 'source_sha256': sources,
            'protocol_sha256': metadata['protocol_sha256']}
        rt.atomic_torch_save(out / 'latest.pth', checkpoint)
        if main:
            path = out / f'snapshot_{step:03d}.pth'; rt.atomic_torch_save(path, checkpoint)
            metadata['snapshots'] = [r for r in metadata['snapshots'] if r['step'] != step]
            metadata['snapshots'].append({'step': step, 'path': path.name, 'sha256': rt.sha(path),
                'model_state_sha256': rt.tensor_hash(checkpoint['model_state_dict'])})
        metadata['last_saved_step'] = step; rt.save_json(metadata_path, metadata)

    if not resume: snapshot(0, True)
    if torch.device(device).type == 'cuda': torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    elapsed_before = float(metadata.get('elapsed_seconds', 0.))
    try:
        with log_path.open('a', buffering=1, encoding='utf-8') as stream:
            for index in range(start_step, rt.STEPS):
                plan_index = index % 256
                batch = frozen.data.batch_data(plan_index, state)
                optimizer.zero_grad(set_to_none=True)
                terms, stats, features, logits, risk, cache = frozen.common.compute(
                    model, batch, plan_index, criterion, rt.SETTINGS)
                loss = terms['base'] + rt.SETTINGS['lambda_readout'] * terms['frame']
                assert bool(torch.isfinite(loss))
                loss.backward()
                gradients = {k: p.grad for k, p in parameters.items()}
                assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients.values())
                row = {'step': index + 1, 'plan_index': plan_index, 'cycle': index // 256,
                    'condition': condition, 'loss': float(loss.detach()),
                    'loss_terms': {k: float(v.detach()) for k, v in terms.items() if k != 'gap'},
                    'grad_norm': float(torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients.values()]))),
                    **{key: batch[key] for key in ('input_sha256', 'label_sha256', 'auxiliary_input_sha256')}}
                inspect = index < 2 or (index + 1) % 16 == 0
                if inspect:
                    before = {k: p.detach().clone() for k, p in parameters.items()}
                    row['parameter_grad_norms'] = {k: float(g.norm()) for k, g in gradients.items()}
                    row['readout'] = frozen.common.readout_metrics(risk, cache)
                optimizer.step()
                if inspect:
                    row['parameter_update_norms'] = {k: float((p.detach() - before[k]).norm()) for k, p in parameters.items()}
                row['elapsed_seconds'] = elapsed_before + time.perf_counter() - began
                row['max_cuda_memory_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.device(device).type == 'cuda' else 0
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                metadata['elapsed_seconds'] = row['elapsed_seconds']
                if index < 2 or (index + 1) % 16 == 0:
                    print(json.dumps({key: row[key] for key in ('step', 'condition', 'loss', 'grad_norm', 'elapsed_seconds',
                                                             'max_cuda_memory_allocated_bytes')}), flush=True)
                if (index + 1) % 128 == 0:
                    stream.flush(); os.fsync(stream.fileno())
                    snapshot(index + 1, index + 1 in rt.SNAPSHOTS)
                del terms, stats, features, logits, risk, cache, loss, gradients
        rt.assert_frozen(model, original_hash)
        assert sources == rt.source_hashes()
        assert all(float(value['step']) == rt.STEPS for value in optimizer.state.values())
        rows = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()]
        assert len(rows) == rt.STEPS
        assert all(row['step'] == i + 1 and row['plan_index'] == i % 256 for i, row in enumerate(rows))
        exposure_hash = hashlib.sha256()
        for row in rows:
            for key in ('input_sha256', 'label_sha256', 'auxiliary_input_sha256'): exposure_hash.update(row[key].encode())
        changes = {k: float((p.detach().cpu() - initial[k]).double().norm()) for k, p in parameters.items()}
        assert any(value > 0 for value in changes.values())
        metadata.update(status='complete', final_step=rt.STEPS, original_unchanged=True,
            final_model_state_sha256=rt.tensor_hash(model.state_dict()),
            elapsed_seconds=elapsed_before + time.perf_counter() - began,
            max_cuda_memory_allocated_bytes=max(row['max_cuda_memory_allocated_bytes'] for row in rows),
            parameter_changes=changes, exposure_digest_sha256=exposure_hash.hexdigest())
        rt.save_json(metadata_path, metadata)
        other = next(c for c in rt.CONDITIONS if c != condition)
        other_path = rt.E / 'training' / other / 'run/RUN.json'
        if other_path.exists() and rt.read_json(other_path)['status'] == 'complete':
            paired = rt.read_json(other_path)
            for key in ('initial_model_state_sha256', 'initial_checkpoint_sha256', 'original_state_sha256',
                        'exposure_digest_sha256', 'settings', 'source_sha256', 'trainable_parameter_names'):
                assert paired[key] == metadata[key], ('Paired identity differs', key)
            rt.save_json(rt.E / 'checks/PAIRED_TRAINING.json', {'status': 'PASS', 'steps_each': rt.STEPS,
                'identical_initialization': True, 'identical_ordered_exposure': True,
                'original_unchanged_both': True, 'initial_model_state_sha256': metadata['initial_model_state_sha256'],
                'exposure_digest_sha256': metadata['exposure_digest_sha256'],
                'run_sha256': {condition: rt.sha(metadata_path), other: rt.sha(other_path)}})
    except BaseException as error:
        metadata.update(status='failed', error=repr(error), elapsed_seconds=elapsed_before + time.perf_counter() - began)
        rt.save_json(metadata_path, metadata)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=rt.CONDITIONS, required=True)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); train(args.condition, args.resume, args.device)
