"""128 matched exposures for independently trained V12 Current and History."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import time
import torch
import runtime as rt


def optimizer_steps(optimizer, parameters, old_names, gate_names, additional):
    old_steps = [float(optimizer.state[parameters[name]]['step']) for name in old_names]
    assert old_steps == [2048 + additional] * 53
    if additional == 0:
        assert all(parameters[name] not in optimizer.state for name in gate_names)
        gate_steps = []
    else:
        gate_steps = [float(optimizer.state[parameters[name]]['step']) for name in gate_names]
        assert gate_steps == [additional] * 6
    return {'old_steps': sorted(set(old_steps)), 'gate_steps': sorted(set(gate_steps)),
            'old_state_count': 53, 'gate_state_count': len(gate_steps)}


def train(condition, device='cuda', resume=False):
    assert condition in rt.CONDITIONS
    out = rt.E / 'training' / condition; out.mkdir(parents=True, exist_ok=True)
    with (out / 'execution.lock').open('a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return train_locked(condition, device, resume, out)


def train_locked(condition, device, resume, out):
    prior = rt.old(); sources = rt.source_hashes()
    freeze = rt.read_json(rt.E / 'PROTOCOL_FREEZE.json')
    assert freeze['source_sha256'] == sources
    protocol_hash = rt.sha(rt.E / 'PROTOCOL_FREEZE.json')
    run_path, log_path = out / 'RUN.json', out / 'TRAIN.jsonl'
    if run_path.exists():
        previous = rt.read_json(run_path)
        if previous['status'] == 'COMPLETE':
            assert previous['source_sha256'] == sources and previous['protocol_sha256'] == protocol_hash
            assert previous['checkpoint_sha256'] == rt.sha(out / 'snapshot_2176.pth')
            print({'status': 'ALREADY_COMPLETE', 'condition': condition}, flush=True)
            return
        if not resume: raise FileExistsError('Incomplete run requires explicit --resume: ' + str(run_path))
    elif resume:
        raise FileNotFoundError('No recorded run to resume')
    elif log_path.exists():
        raise FileExistsError('Preserve unreceipted training log: ' + str(log_path))

    model, parent = rt.load_initial(condition, device)
    checkpoint = torch.load(rt.CHECKPOINT, map_location='cpu', weights_only=True)
    optimizer, migration = rt.make_optimizer(model, parent, checkpoint)
    parameters, old_names, gate_names = rt.parameter_groups(model, parent)
    initial_state = torch.load(rt.INITIAL, map_location='cpu', weights_only=True)
    assert prior.tensor_hash(model.state_dict()) == initial_state['model_state_sha256']
    assert sum(parameter.numel() for parameter in model.parameters()) == 182280
    assert sum(parameter.numel() for parameter in parameters.values()) == 111367
    initial_parameters = {name: initial_state['model_state_dict'][name] for name in parameters}
    model.train()
    frozen = prior.frozen_f(); data = frozen.data.inputs()
    assert prior.frozen_cache_manifest() == parent['cache_sha256']
    criterion = prior.load_module('_m12_loss', prior.MODEL_SOURCE).SoftLoUloss().to(device)
    training_impl = rt.old_train()
    # All constructors, gate initialization and optimizer migration are complete.
    training_impl.restore_rng(checkpoint['rng_state'])
    run = {'status': 'RUNNING', 'condition': condition, 'mode': rt.MODES[condition], 'seed': 46,
        'initial_step': 2048, 'additional_steps': 128, 'completed_additional_steps': 0,
        'initial_checkpoint_sha256': rt.CHECKPOINT_SHA,
        'parent_initial_model_state_sha256': parent['final_model_state_sha256'],
        'initial_model_state_sha256': initial_state['model_state_sha256'],
        'gate_initial_state_sha256': initial_state['gate_state_sha256'],
        'v12_initial_checkpoint_sha256': rt.sha(rt.INITIAL),
        'original_state_sha256': parent['original_state_sha256'],
        'source_sha256': sources, 'frozen_v1_source_sha256': prior.source_hashes(),
        'protocol_sha256': protocol_hash, 'reference_state_update_sha256': rt.sha(rt.REFERENCE),
        'settings': prior.SETTINGS, 'runtime_settings': prior.RUNTIME_SETTINGS, 'cache_sha256': parent['cache_sha256'],
        'trainable_parameter_names': list(parameters), 'old_trainable_parameter_names': old_names,
        'gate_parameter_names': gate_names, 'trainable_parameters': 111367, 'parameters': 182280,
        'added_parameters': 2017, 'added_parameter_tensors': 6,
        'optimizer': 'Restore identical History2048 Adam state for old53 in recorded order; append fresh gate6 group; lr1e-5 wd1e-4',
        'optimizer_migration': migration,
        'rng': 'Restore identical old History2048 checkpoint RNG after model/gate/optimizer/data/loss construction',
        'exposure': 'Existing frozen F entries0..127 once, in order; 4 detection+4 auxiliary per entry',
        'revision': 'BoundedStateUpdate(pixel)', 'update_scope': 'pixel',
        'gate_initial_retention': 0.5, 'initial_output_equal_to_old_model_claimed': False,
        'first_frame': 'M0=X0; no gate; unchanged window-local state and checkpoint_recurrence=True',
        'all_original_parameters_and_BN_frozen': True,
        'gradient_logging': 'Loss gradients separate from optimizer updates; gate hidden gradients may be zero on first step with zero final gate layer',
        'training_diagnostics': 'Existing readout_metrics target/fixed-background logits/margin and gate forward statistics; no extra forward or supervision',
        'checkpoint_policy': 'additional0/32/64/128; fixed2176 main, earlier process diagnostics only',
        'snapshots': [], 'resume_events': []}
    additional_start = 0
    if resume:
        for key in ('condition', 'mode', 'source_sha256', 'protocol_sha256', 'initial_checkpoint_sha256',
                    'initial_model_state_sha256', 'gate_initial_state_sha256', 'v12_initial_checkpoint_sha256',
                    'settings', 'cache_sha256', 'trainable_parameter_names'):
            assert previous[key] == run[key], ('Resume identity mismatch', key)
        resumed = torch.load(out / 'latest.pth', map_location='cpu', weights_only=True)
        assert resumed['condition'] == condition and resumed['source_sha256'] == sources
        assert resumed['protocol_sha256'] == protocol_hash and resumed['step'] in rt.SNAPSHOTS
        model.load_state_dict(resumed['model_state_dict'], strict=True)
        optimizer.load_state_dict(resumed['optimizer_state_dict'])
        training_impl.restore_rng(resumed['rng_state'])
        additional_start = resumed['additional_step']
        run = previous; run.update(status='RUNNING', completed_additional_steps=additional_start)
        run.pop('error', None)
        run['snapshots'] = [item for item in run['snapshots'] if item['step'] <= resumed['step']]
        run['resume_events'].append({'step': resumed['step'], 'unix': time.time()})
        if log_path.exists():
            original_log = log_path.read_text(encoding='utf-8')
            kept = [line for line in original_log.splitlines() if json.loads(line)['additional_step'] <= additional_start]
            assert len(kept) == additional_start
            if len(original_log.splitlines()) != len(kept):
                preserved = out / f'TRAIN_before_resume_{time.time_ns()}.jsonl'
                preserved.write_text(original_log, encoding='utf-8')
                run['resume_events'][-1]['preserved_log'] = str(preserved)
            log_path.write_text(''.join(line + '\n' for line in kept), encoding='utf-8')
    optimizer_steps(optimizer, parameters, old_names, gate_names, additional_start)
    prior.assert_frozen(model, run['original_state_sha256'])
    rt.save(run_path, run)

    def parameter_changes():
        return {name: float((parameter.detach().cpu() - initial_parameters[name]).double().norm())
                for name, parameter in parameters.items()}

    def snapshot(additional):
        step = 2048 + additional
        assert step in rt.SNAPSHOTS and sources == rt.source_hashes()
        prior.assert_frozen(model, run['original_state_sha256'])
        steps = optimizer_steps(optimizer, parameters, old_names, gate_names, additional)
        saved = {'model_state_dict': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(), 'rng_state': training_impl.rng_state(),
            'step': step, 'additional_step': additional, 'condition': condition, 'mode': rt.MODES[condition],
            'source_sha256': sources, 'protocol_sha256': protocol_hash,
            'initial_checkpoint_sha256': rt.CHECKPOINT_SHA, 'v12_initial_checkpoint_sha256': run['v12_initial_checkpoint_sha256'],
            'optimizer_parameter_names': [old_names, gate_names]}
        path = out / f'snapshot_{step}.pth'
        if path.exists(): raise FileExistsError('Preserve existing named snapshot: ' + str(path))
        prior.atomic_torch_save(path, saved)
        prior.atomic_torch_save(out / 'latest.pth', saved)
        run['snapshots'].append({'step': step, 'additional_step': additional, 'path': path.name,
            'sha256': rt.sha(path), 'model_state_sha256': prior.tensor_hash(model.state_dict()),
            'parameter_changes': parameter_changes(), 'optimizer_steps': steps,
            'checkpoint_is_main': step == rt.FINAL_STEP})
        run['last_saved_step'] = step
        rt.save(run_path, run)

    if not resume: snapshot(0)
    if torch.device(device).type == 'cuda': torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter(); elapsed_before = float(run.get('elapsed_seconds', 0.0))
    cell = model.adapter.recurrent_cell
    try:
        with log_path.open('a', buffering=1, encoding='utf-8') as stream:
            for index in range(additional_start, rt.STEPS):
                batch = frozen.data.batch_data(index, data)
                optimizer.zero_grad(set_to_none=True)
                cell.begin_gate_statistics()
                terms, stats, features, logits, risk, cache = frozen.common.compute(model, batch, index, criterion, prior.SETTINGS)
                gate_statistics = cell.end_gate_statistics()  # BEFORE activation-checkpoint backward recomputation.
                readout = frozen.common.readout_metrics(risk, cache)
                loss = terms['base'] + prior.SETTINGS['lambda_readout'] * terms['frame']
                assert bool(torch.isfinite(loss))
                loss.backward()
                assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in parameters.values())
                inspect = index < 2 or (index + 1) in (32, 64, 128)
                before = {name: parameter.detach().clone() for name, parameter in parameters.items()} if inspect else None
                row = {'additional_step': index + 1, 'step': 2049 + index, 'plan_index': index,
                    'condition': condition, 'mode': rt.MODES[condition], 'loss': float(loss.detach()),
                    'terms': {key: float(value.detach()) for key, value in terms.items() if key != 'gap'},
                    'grad_norm': float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters.values()]))),
                    'old_grad_norm': float(torch.linalg.vector_norm(torch.stack([parameters[name].grad.norm() for name in old_names]))),
                    'gate_grad_norm': float(torch.linalg.vector_norm(torch.stack([parameters[name].grad.norm() for name in gate_names]))),
                    'gate': gate_statistics, 'readout': readout,
                    **{key: batch[key] for key in ('input_sha256', 'label_sha256', 'auxiliary_input_sha256')}}
                if inspect: row['parameter_grad_norms'] = {name: float(parameter.grad.norm()) for name, parameter in parameters.items()}
                optimizer.step()
                if inspect:
                    row['parameter_update_norms'] = {name: float((parameter.detach() - before[name]).norm()) for name, parameter in parameters.items()}
                row['elapsed_seconds'] = elapsed_before + time.perf_counter() - began
                row['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.device(device).type == 'cuda' else 0
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                run.update(completed_additional_steps=index + 1, elapsed_seconds=row['elapsed_seconds'])
                if inspect or (index + 1) % 16 == 0:
                    rt.save(run_path, run)
                    print({key: row[key] for key in ('condition', 'step', 'loss', 'grad_norm', 'gate_grad_norm', 'elapsed_seconds')}, flush=True)
                if 2049 + index in rt.SNAPSHOTS:
                    stream.flush(); os.fsync(stream.fileno()); snapshot(index + 1)
                del terms, stats, features, logits, risk, cache, loss, batch, before
        records = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()]
        assert len(records) == 128 and all(row['additional_step'] == i + 1 and row['plan_index'] == i for i, row in enumerate(records))
        digest = hashlib.sha256()
        for row in records:
            for key in ('input_sha256', 'label_sha256', 'auxiliary_input_sha256'): digest.update(row[key].encode())
        prior.assert_frozen(model, run['original_state_sha256'])
        assert sources == rt.source_hashes()
        final_steps = optimizer_steps(optimizer, parameters, old_names, gate_names, 128)
        run.update(status='COMPLETE', final_step=2176, checkpoint_sha256=rt.sha(out / 'snapshot_2176.pth'),
            final_model_state_sha256=prior.tensor_hash(model.state_dict()), exposure_digest_sha256=digest.hexdigest(),
            original_unchanged=True, elapsed_seconds=elapsed_before + time.perf_counter() - began,
            peak_cuda_allocated_bytes=max(row['peak_cuda_allocated_bytes'] for row in records),
            parameter_changes=parameter_changes(), optimizer_final_steps=final_steps)
        rt.save(run_path, run)
    except BaseException as error:
        if cell._gate_statistics is not None: cell.end_gate_statistics()
        run.update(status='FAILED', error=repr(error), elapsed_seconds=elapsed_before + time.perf_counter() - began)
        rt.save(run_path, run)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=rt.CONDITIONS, required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); train(args.condition, args.device, args.resume)
