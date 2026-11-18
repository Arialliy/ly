#!/usr/bin/env python3
"""Fixed N0/N1/N2 branch-only adaptation; original detector state is immutable."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

Q = Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
N = Path(__file__).resolve().parents[1]
PREVIOUS = Q / 'supervision_alignment_control_20261012'
sys.path.insert(0, str(Q / 'mutual_temporal_pilot_20261012/code'))
import train_mutual as historical
previous = historical.previous
sha, save_json, load_module = previous.sha, previous.save_json, previous.load_module
tensor_hash, data_hash, label_hash = previous.tensor_hash, previous.data_hash, previous.label_hash
EXPECTED, LABEL_CONFIG = previous.EXPECTED, historical.LABEL_CONFIG
build_targets, common_loss = historical.build_targets, historical.common_loss
LABEL_PATH, TOP1_PATH = historical.LABEL_PATH, historical.TOP1_PATH
EFFECTIVE_SUPERVISION = historical.EFFECTIVE_SUPERVISION
OLD_LOG = Q / 'mutual_temporal_pilot_20261012/training/E0/run/TRAIN.jsonl'
VARIANTS = {'N0': 'current', 'N1': 'difference', 'N2': 'relative'}
SNAPSHOTS = (0, 128, 256, 384, 512)
TELEMETRY_RULE = {
    'steps': 'step 1 and every 16th update, before optimizer.step',
    'target_queries': 'first at most 8 valid positive-label pixels per window, flattened T/H/W ascending',
    'background_queries': 'top at most 8 old-detector logits among frozen valid background pixels per window; ties flat index ascending',
    'selection_uses_new_logits': False,
    'model_receives_labels': False,
    'values': 'actual mode input I,D1,mu1,s1,E1,m1,D4,mu4,s4,E4,m4; old logit and signed delta',
    'limitation': 'sampled training telemetry; neighboring target pixels need not be distinct objects',
}


def deterministic_setup():
    import numpy as np
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(46); np.random.seed(46); torch.manual_seed(46); torch.cuda.manual_seed_all(46)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def wrap_model(original, variant):
    # Import by absolute file to avoid the historical sys.path's old module.py.
    module = load_module('local_reference_evidence_module', Path(__file__).with_name('module.py'))
    return module.FrozenEvidenceDetector(original, VARIANTS[variant])


def source_hashes(model_path):
    return {name: sha(path) for name, path in {
        'train_source_sha256': Path(__file__),
        'module_source_sha256': Path(__file__).with_name('module.py'),
        'model_source_sha256': model_path,
        'label_source_sha256': LABEL_PATH,
        'top1_loss_source_sha256': TOP1_PATH,
        'common_loss_source_sha256': Path(historical.__file__),
        'prior_train_source_sha256': PREVIOUS / 'code/train_supervision.py',
        'batch_source_sha256': Q / 'top1_background_control_20261011/code/train_pilot.py',
    }.items()}


def gradient_stats(model):
    import torch
    result = {}
    for name, parameter in model.evidence.named_parameters():
        gradient = parameter.grad
        result[name] = {'present': gradient is not None,
                        'norm': None if gradient is None else float(gradient.detach().norm()),
                        'nonzero': 0 if gradient is None else int(torch.count_nonzero(gradient)),
                        'parameter_norm': float(parameter.detach().norm())}
    return result


def sparse_telemetry(model, targets, valid, background, step):
    """Inspect same-forward tensors after selection; never alter model input."""
    import torch
    rows = []
    channels = ['I', 'D1', 'mu1', 's1', 'E1', 'm1', 'D4', 'mu4', 's4', 'E4', 'm4']
    values = model.evidence.last_input
    old = model.last_original_logits
    correction = model.last_delta
    if values.shape[:3] != (targets.shape[0], targets.shape[1], 11):
        raise AssertionError('Sparse telemetry expects detached [B,T,11,H,W] input')
    h, w = targets.shape[-2:]
    for wi in range(targets.shape[0]):
        positive = torch.nonzero(((targets[wi] > 0) & valid[wi]).flatten(), as_tuple=False).flatten()[:8]
        bg = torch.nonzero(background[wi].flatten(), as_tuple=False).flatten()
        if bg.numel():
            scores = old[wi].flatten()[bg]
            # Stable sort gives deterministic smaller flat-index precedence on ties.
            bg = bg[torch.argsort(scores, descending=True, stable=True)[:8]]
        for kind, indices in [('target', positive), ('hard_background', bg)]:
            for index in indices.detach().cpu().tolist():
                t, remainder = divmod(index, h*w); y, x = divmod(remainder, w)
                vector = values[wi, t, :, y, x].detach().cpu().tolist()
                rows.append({'step': step, 'window': wi, 'kind': kind, 'slot': t, 'y': y, 'x': x,
                             **dict(zip(channels, vector)), 'original_logit': float(old[wi,t,y,x]),
                             'delta_logit': float(correction[wi,t,y,x])})
    return rows


def train(args):
    import numpy as np
    import torch
    base, plan, names, sequences, model_file = previous.validate_inputs(args)
    old_rows = [json.loads(line) for line in OLD_LOG.read_text().splitlines()]
    if len(old_rows) != 512:
        raise AssertionError('Historical exposure comparison requires 512 rows')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Formal output must be new and empty')
    args.output.mkdir(parents=True, exist_ok=True)
    deterministic_setup()
    original, loaded = base.load_model(args.source, args.checkpoint)
    if loaded.resolve() != model_file.resolve() or sum(p.numel() for p in original.parameters()) != 70913:
        raise AssertionError('Frozen original architecture mismatch')
    original_state = {k: v.detach().cpu().clone() for k,v in original.state_dict().items()}
    original_hash = tensor_hash(original_state.items())
    model = wrap_model(original, args.variant)
    device = torch.device(args.device)
    model = model.to(device=device, dtype=torch.float32)
    model.train()
    if any(p.requires_grad for p in model.original.parameters()) or model.original.training:
        raise AssertionError('Every original parameter and its mode must be frozen')
    official = load_module('local_evidence_official_loss', model_file)
    criterion = official.SoftLoUloss().to(device)
    initial_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    original_bn = base.bn_snapshot(model.original)
    bn_hash = base.buffer_hash(original_bn)
    branch_hash = tensor_hash(model.evidence.state_dict().items())
    sources = source_hashes(model_file)
    optimizer = torch.optim.Adam(model.evidence.parameters(), lr=1e-4, weight_decay=1e-4)
    parameter_count = sum(p.numel() for p in model.parameters())
    new_parameters = sum(p.numel() for p in model.evidence.parameters())
    metadata = {
        'status': 'running', 'condition': args.variant, 'variant': args.variant, 'mode': VARIANTS[args.variant],
        'steps': 512, 'seed': 46, 'lr': 1e-4, 'weight_decay': 1e-4,
        'optimizer': 'fresh Adam on evidence parameters only', 'betas': [.9, .999], 'eps': 1e-8,
        'precision': 'FP32_no_AMP_no_TF32', 'bn_policy': 'original_entire_state_and_affine_frozen_eval; evidence_no_BN',
        'bn_initial_sha256': bn_hash, 'checkpoint_sha256': EXPECTED['checkpoint'],
        'initial_checkpoint_sha256': EXPECTED['checkpoint'], 'reference_checkpoint_sha256': EXPECTED['checkpoint'],
        'plan_sha256': EXPECTED['plan'], 'selection_sha256': EXPECTED['selection'],
        'old_log_sha256': sha(OLD_LOG), 'initial_model_state_sha256': tensor_hash(initial_state.items()),
        'original_initial_state_sha256': original_hash, 'branch_initial_sha256': branch_hash,
        'label_config': LABEL_CONFIG, 'effective_supervision_config': EFFECTIVE_SUPERVISION,
        'frozen_plan_config': base.CONFIG, 'model_config': model.config,
        'training_sequences': names, 'torch_version': str(torch.__version__), 'device': str(device),
        'parameter_count': parameter_count, 'new_parameters': new_parameters, 'trainable_parameters': new_parameters,
        'source_path': str(args.source.resolve()), 'checkpoint_path': str(args.checkpoint.resolve()),
        'criterion_source_path': str(model_file.resolve()), 'label_source_path': str(LABEL_PATH),
        'protocol': 'frozen_original_positive_masked_official_SoftIoU_empty_old_top1pct',
        'loss_reduction': 'unchanged official per positive window; empty old top1 negative; four-window equal mean',
        'checkpoint_selection': 'fixed final update512; snapshots0/128/256/384/512 never selected by validation',
        'snapshots': [], 'telemetry_rule': TELEMETRY_RULE, 'model_labels_access': False, **sources,
    }
    save_json(args.output / 'INITIAL_STATE.json', {
        'variant': args.variant, 'mode': VARIANTS[args.variant], 'original_checkpoint_sha256': EXPECTED['checkpoint'],
        'model_state_sha256': tensor_hash(initial_state.items()), 'branch_initial_sha256': branch_hash,
        'original_initial_state_sha256': original_hash,
        'state_tensor_sha256': {name: tensor_hash([(name, value)]) for name, value in initial_state.items()},
        'parameter_names': sorted(dict(model.named_parameters())),
        'trainable_parameter_names': [name for name,p in model.named_parameters() if p.requires_grad],
        'parameter_count': parameter_count, 'new_parameters': new_parameters,
    })

    def assert_original():
        state = model.original.state_dict()
        if set(state) != set(original_state) or any(not torch.equal(state[k].detach().cpu(), v) for k,v in original_state.items()):
            raise AssertionError('Released original parameter or buffer changed')
        return tensor_hash(state.items())

    def snapshot(step):
        unchanged = assert_original()
        checkpoint = {'model_state_dict': {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                      'optimizer_state_dict': optimizer.state_dict(), 'step': step, 'variant': args.variant,
                      'condition': args.variant, 'mode': VARIANTS[args.variant], 'model_config': model.config,
                      'label_config': LABEL_CONFIG, 'effective_supervision_config': EFFECTIVE_SUPERVISION,
                      'initial_checkpoint_sha256': EXPECTED['checkpoint'], 'plan_sha256': EXPECTED['plan'],
                      'selection_sha256': EXPECTED['selection'], 'original_state_sha256': unchanged,
                      'protocol': metadata['protocol'], **sources}
        path = args.output / f'snapshot_{step:03d}.pth'
        torch.save(checkpoint, path)
        metadata['snapshots'].append({'step': step, 'path': path.name, 'sha256': sha(path),
                                     'original_state_sha256': unchanged,
                                     'model_state_sha256': tensor_hash(checkpoint['model_state_dict'].items())})
        save_json(args.output / 'RUN.json', metadata)

    snapshot(0)
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counts = Counter(); input_chain, label_chain = hashlib.sha256(), hashlib.sha256()
    telemetry_seconds = 0.0
    try:
        with (args.output / 'TRAIN.jsonl').open('w', buffering=1) as log, (args.output / 'EVIDENCE_SAMPLES.jsonl').open('w', buffering=1) as samples:
            for index, entry in enumerate(plan['entries']):
                if entry['step'] != index: raise AssertionError('Frozen order changed')
                batch, positives, masks, specs = base.build_batch(entry, sequences)
                target_np, valid_np, label_stats = build_targets(entry, sequences, background_masks=masks)
                input_sha, labels_sha = data_hash(batch, positives, masks, specs), label_hash(target_np, valid_np)
                if input_sha != old_rows[index]['input_sha256'] or labels_sha != old_rows[index]['label_sha256']:
                    raise AssertionError(f'Exact exposure/labels differ at step {index+1}')
                input_chain.update(input_sha.encode()); label_chain.update(labels_sha.encode())
                x, targets = torch.from_numpy(batch).to(device), torch.from_numpy(target_np).to(device)
                valid, background = torch.from_numpy(valid_np).to(device), torch.from_numpy(masks).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)[1]
                if logits.shape != (4,40,128,128): raise AssertionError('Wrong logits dimensions')
                loss, parts = common_loss(logits, targets, valid, background, criterion)
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite loss')
                loss.backward()
                gradients = [p.grad for p in model.evidence.parameters() if p.grad is not None]
                if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise FloatingPointError('Missing or nonfinite branch gradients')
                if any(p.grad is not None or p.requires_grad for p in model.original.parameters()) or model.original.training:
                    raise AssertionError('Frozen detector gradient/mode changed')
                row = {'step': index+1, 'variant': args.variant, 'condition': args.variant,
                       'loss': float(loss.detach()), 'grad_norm': float(torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in gradients]))),
                       'parameters_with_gradient': len(gradients), 'original_parameters_with_gradient': 0,
                       'finite': True, 'input_sha256': input_sha, 'label_sha256': labels_sha,
                       'label_stats': label_stats, **parts}
                if index < 2 or (index+1) % 16 == 0:
                    row['module_gradients'] = gradient_stats(model)
                if index == 0 or (index+1) % 16 == 0:
                    if device.type == 'cuda': torch.cuda.synchronize(device)
                    stamp = time.perf_counter()
                    for point in sparse_telemetry(model, targets, valid, background, index+1):
                        samples.write(json.dumps(point, allow_nan=False)+'\n')
                    if device.type == 'cuda': torch.cuda.synchronize(device)
                    telemetry_seconds += time.perf_counter()-stamp
                optimizer.step()
                current_bn = base.bn_snapshot(model.original)
                if current_bn.keys() != original_bn.keys() or any(not torch.equal(current_bn[k],v) for k,v in original_bn.items()):
                    raise AssertionError('Frozen BN buffers changed')
                row['bn_running_sha256'] = base.buffer_hash(current_bn)
                for key in ['pure_negative_windows','valid_pixels','positive_pixels']: counts[key] += parts[key]
                row['elapsed_seconds'] = time.perf_counter()-started
                row['max_cuda_memory_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
                row['max_cuda_memory_reserved_bytes'] = torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0
                log.write(json.dumps(row, allow_nan=False)+'\n')
                if index % 16 == 0 or index == 511: print(json.dumps(row, allow_nan=False), flush=True)
                if index+1 in SNAPSHOTS: snapshot(index+1)
                del x, targets, valid, background, logits, loss, gradients
        if device.type=='cuda': torch.cuda.synchronize(device)
        if sources != source_hashes(model_file): raise AssertionError('Source changed during training')
        final_original = assert_original()
        final_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
        if set(initial_state) != set(final_state) or not all(bool(torch.isfinite(v).all()) for v in final_state.values()):
            raise AssertionError('Invalid final state')
        differences = {}
        for name,p in model.named_parameters():
            difference = final_state[name].double()-initial_state[name].double()
            differences[name] = {'numel': difference.numel(), 'trainable': p.requires_grad,
                                 'changed_elements': int(torch.count_nonzero(difference)),
                                 'max_absolute_difference': float(difference.abs().max()), 'l2_difference': float(difference.norm())}
        save_json(args.output / 'PARAMETER_DIFF.json', {'parameter_count': parameter_count, 'new_parameter_count': new_parameters,
                  'parameter_tensors': len(differences), 'changed_parameter_tensors': sum(d['changed_elements']>0 for d in differences.values()),
                  'per_parameter': differences})
        os.link(args.output / 'snapshot_512.pth', args.output / 'final_model.pth')
        metadata.update(status='complete', final_step=512, duration_seconds=time.perf_counter()-started,
                        counts=dict(counts), bn_final_sha256=base.buffer_hash(base.bn_snapshot(model.original)),
                        original_final_state_sha256=final_original, frozen_original_unchanged=True,
                        final_checkpoint_sha256=sha(args.output/'final_model.pth'), final_model_state_sha256=tensor_hash(final_state.items()),
                        cumulative_input_sha256=input_chain.hexdigest(), cumulative_label_sha256=label_chain.hexdigest(),
                        train_log_sha256=sha(args.output/'TRAIN.jsonl'), parameter_diff_sha256=sha(args.output/'PARAMETER_DIFF.json'),
                        telemetry_sha256=sha(args.output/'EVIDENCE_SAMPLES.jsonl'), telemetry_seconds=telemetry_seconds,
                        max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                        max_cuda_memory_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0)
        save_json(args.output/'RUN.json', metadata)
        print(json.dumps({'status':'complete','variant':args.variant,'duration_seconds':metadata['duration_seconds']}), flush=True)
    except BaseException as error:
        metadata.update(status='failed', error=repr(error), duration_seconds=time.perf_counter()-started)
        save_json(args.output/'RUN.json', metadata)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source','checkpoint','dataset-root','selection','plan','output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--steps', type=int, default=512); p.add_argument('--seed', type=int, default=46)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.steps!=512 or args.seed!=46 or args.lr!=1e-4: p.error('Frozen steps512 seed46 lr1e-4 required')
    train(args)


if __name__=='__main__': main()
