"""512 matched exposures for independently trained Bank Current and History."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import time
import torch
import runtime as rt


def optimizer_steps(optimizer, parameters, additional):
    steps = [float(optimizer.state[p]['step']) for p in parameters.values()]
    assert steps == [2048 + additional] * 53
    return {'old_steps': sorted(set(steps)), 'old_state_count': 53, 'new_state_count': 0}


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
            assert previous['checkpoint_sha256'] == rt.sha(out / 'snapshot_2560.pth')
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
    parameters = {name:p for name,p in model.named_parameters() if p.requires_grad}
    assert prior.tensor_hash(model.state_dict()) == parent['final_model_state_sha256']
    assert sum(p.numel() for p in model.parameters()) == 180263
    assert sum(p.numel() for p in parameters.values()) == 109350
    initial_parameters = {name: p.detach().cpu().clone() for name,p in parameters.items()}
    model.train()
    frozen = prior.frozen_f(); data = frozen.data.inputs()
    assert prior.frozen_cache_manifest() == parent['cache_sha256']
    criterion = prior.load_module('_m13_loss', prior.MODEL_SOURCE).SoftLoUloss().to(device)
    training_impl = rt.old_train()
    # All constructors and strict optimizer restoration are complete.
    training_impl.restore_rng(checkpoint['rng_state'])
    run = {'status': 'RUNNING', 'condition': condition, 'mode': rt.MODES[condition], 'seed': 46,
        'initial_step': 2048, 'additional_steps': 512, 'completed_additional_steps': 0,
        'initial_checkpoint_sha256': rt.CHECKPOINT_SHA,
        'parent_initial_model_state_sha256': parent['final_model_state_sha256'],
        'initial_model_state_sha256': parent['final_model_state_sha256'],
        'original_state_sha256': parent['original_state_sha256'],
        'source_sha256': sources, 'frozen_v1_source_sha256': prior.source_hashes(),
        'protocol_sha256': protocol_hash, 'reference_evidence_bank_sha256': rt.sha(rt.REFERENCE/'evidence_bank.py'),
        'settings': prior.SETTINGS, 'runtime_settings': prior.RUNTIME_SETTINGS, 'cache_sha256': parent['cache_sha256'],
        'trainable_parameter_names': list(parameters), 'trainable_parameters': 109350, 'parameters': 180263,
        'added_parameters': 0, 'added_parameter_tensors': 0,
        'optimizer': 'Restore identical History2048 Adam state for all53 unchanged tensors; lr1e-5 wd1e-4',
        'optimizer_migration': migration,
        'rng': 'Restore identical old History2048 checkpoint RNG after model/optimizer/data/loss construction',
        'exposure': 'Existing frozen F entries0..255 twice in order; 4 detection+4 auxiliary per entry',
        'revision': 'source-feature evidence bank', 'capacity': 4, 'pairs_per_window': 150,
        'initial_output_equal_to_old_model_claimed': False,
        'first_frame': 'R0=X0; read-before-write FIFO4; window reset; checkpoint PDA/TSFM without detach',
        'all_original_parameters_and_BN_frozen': True,
        'gradient_logging': 'Loss gradients and optimizer updates for all53 unchanged tensors',
        'training_diagnostics': 'Existing readout_metrics target/fixed-background logits/margin; no extra forward or supervision',
        'checkpoint_policy': 'additional0/128/256/512; fixed2560 main, earlier process diagnostics only',
        'snapshots': [], 'resume_events': []}
    additional_start = 0
    if resume:
        for key in ('condition', 'mode', 'source_sha256', 'protocol_sha256', 'initial_checkpoint_sha256',
                    'initial_model_state_sha256',
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
    optimizer_steps(optimizer, parameters, additional_start)
    prior.assert_frozen(model, run['original_state_sha256'])
    rt.save(run_path, run)

    def parameter_changes():
        return {name: float((parameter.detach().cpu() - initial_parameters[name]).double().norm())
                for name, parameter in parameters.items()}

    def snapshot(additional):
        step = 2048 + additional
        assert step in rt.SNAPSHOTS and sources == rt.source_hashes()
        prior.assert_frozen(model, run['original_state_sha256'])
        steps = optimizer_steps(optimizer, parameters, additional)
        saved = {'model_state_dict': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(), 'rng_state': training_impl.rng_state(),
            'step': step, 'additional_step': additional, 'condition': condition, 'mode': rt.MODES[condition],
            'source_sha256': sources, 'protocol_sha256': protocol_hash,
            'initial_checkpoint_sha256': rt.CHECKPOINT_SHA,
            'optimizer_parameter_names': [list(parameters)]}
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
    try:
        with log_path.open('a', buffering=1, encoding='utf-8') as stream:
            for index in range(additional_start, rt.STEPS):
                plan_index = index % 256
                batch = frozen.data.batch_data(plan_index, data)
                optimizer.zero_grad(set_to_none=True)
                terms, stats, features, logits, risk, cache = frozen.common.compute(model, batch, plan_index, criterion, prior.SETTINGS)
                readout = frozen.common.readout_metrics(risk, cache)
                loss = terms['base'] + prior.SETTINGS['lambda_readout'] * terms['frame']
                assert bool(torch.isfinite(loss))
                loss.backward()
                assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in parameters.values())
                inspect = index < 2 or (index + 1) in (128, 256, 512)
                before = {name: parameter.detach().clone() for name, parameter in parameters.items()} if inspect else None
                row = {'additional_step': index + 1, 'step': 2049 + index, 'plan_index': plan_index,
                    'condition': condition, 'mode': rt.MODES[condition], 'loss': float(loss.detach()),
                    'terms': {key: float(value.detach()) for key, value in terms.items() if key != 'gap'},
                    'grad_norm': float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters.values()]))),
                    'readout': readout,
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
                    print({key: row[key] for key in ('condition', 'step', 'loss', 'grad_norm', 'elapsed_seconds')}, flush=True)
                if 2049 + index in rt.SNAPSHOTS:
                    stream.flush(); os.fsync(stream.fileno()); snapshot(index + 1)
                del terms, stats, features, logits, risk, cache, loss, batch, before
        records = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()]
        assert len(records) == 512 and all(row['additional_step'] == i + 1 and row['plan_index'] == i % 256 for i, row in enumerate(records))
        digest = hashlib.sha256()
        for row in records:
            for key in ('input_sha256', 'label_sha256', 'auxiliary_input_sha256'): digest.update(row[key].encode())
        prior.assert_frozen(model, run['original_state_sha256'])
        assert sources == rt.source_hashes()
        final_steps = optimizer_steps(optimizer, parameters, 512)
        run.update(status='COMPLETE', final_step=2560, checkpoint_sha256=rt.sha(out / 'snapshot_2560.pth'),
            final_model_state_sha256=prior.tensor_hash(model.state_dict()), exposure_digest_sha256=digest.hexdigest(),
            original_unchanged=True, elapsed_seconds=elapsed_before + time.perf_counter() - began,
            peak_cuda_allocated_bytes=max(row['peak_cuda_allocated_bytes'] for row in records),
            parameter_changes=parameter_changes(), optimizer_final_steps=final_steps)
        rt.save(run_path, run)
    except BaseException as error:
        run.update(status='FAILED', error=repr(error), elapsed_seconds=elapsed_before + time.perf_counter() - began)
        rt.save(run_path, run)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=rt.CONDITIONS, required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); train(args.condition, args.device, args.resume)
