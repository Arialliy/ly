#!/usr/bin/env python3
"""GT-conditioned internal shared-support ablation of a frozen DeepPro checkpoint.

Dropping exclusive projected terms is a potentially out-of-distribution internal
intervention, not a naturally occurring context, deployable gain, or evidence that
a label is visible. Original model, windows, frames, and weights stay frozen.
"""
import argparse
import collections
import csv
import datetime
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

FROZEN_PATH = Path(__file__).resolve().with_name('run_factorial_frozen.py')
spec = importlib.util.spec_from_file_location('frozen_factorial_support_dependency', FROZEN_PATH)
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


def summarize_map(result, prefix, logits, peaks):
    if logits is None:
        # Keep a stable CSV schema without inventing zero evidence for empty Ssafe.
        result[prefix + '_box_max_probability'] = ''
        result[prefix + '_box_max_logit'] = ''
        for tag in peaks:
            result[prefix + '_' + tag + '_peak_logit'] = ''
            result[prefix + '_' + tag + '_peak_probability'] = ''
        return
    result[prefix + '_box_max_probability'] = float(logits.sigmoid().max())
    result[prefix + '_box_max_logit'] = float(logits.max())
    for tag, (py, px) in peaks.items():
        result[prefix + '_' + tag + '_peak_logit'] = float(logits[py, px])
        result[prefix + '_' + tag + '_peak_probability'] = float(logits[py, px].sigmoid())


def support_pair(model, frames, row, device, tolerance):
    import torch
    n, height, width = frames.shape
    x0, y0, x1, y1 = frozen.crop_indices(row, height, width)
    assert x1 > x0 and y1 > y0
    xa, ya, xb, yb = max(0, x0-16), max(0, y0-16), min(width, x1+16), min(height, y1+16)
    ys, xs = slice(y0-ya, y1-ya), slice(x0-xa, x1-xa)
    result = dict(row)
    result.update(crop_x0=x0, crop_y0=y0, crop_x1=x1, crop_y1=y1)

    def input_tensor(start, end):
        raw = np.asarray(frames[start:end, ya:yb, xa:xb], dtype=np.float32)
        normed = np.ascontiguousarray((raw-85.0799)/47.4845)
        assert np.isfinite(normed).all()
        return torch.from_numpy(normed)[None, None].to(device)

    originals, peaks = {}, {}
    for tag in ['a', 'b']:
        start, slot = row[tag+'_start'], row[tag+'_slot']
        assert start+slot == int(row['local_frame']) and 0 <= start <= n-40
        capture = {}

        def feature_hook(module, inputs, output):
            capture['feature'] = output[..., ys, xs].clone()

        def q_hook(module, inputs):
            capture['q_official'] = inputs[0][:, :, slot:slot+1, ys, xs].clone()

        h1 = model.layer1.register_forward_hook(feature_hook)
        h2 = model.TPro.norm1.register_forward_pre_hook(q_hook)
        try:
            output = model(input_tensor(start, start+40))
            logits = output[1][0, slot, ys, xs].clone()
            del output
        finally:
            h1.remove()
            h2.remove()
        probability = logits.sigmoid()
        peak = int(probability.flatten().argmax())
        peaks[tag] = divmod(peak, x1-x0)
        result[tag+'_peak_y_index'] = y0+peaks[tag][0]
        result[tag+'_peak_x_index'] = x0+peaks[tag][1]
        result[tag+'_probability_peak_tie_count'] = int((probability == probability.max()).sum())
        replay_error = abs(float(probability.max())-row[tag+'_saved_box_max'])
        result[tag+'_replay_abs_error'] = replay_error
        if replay_error > tolerance:
            raise AssertionError('Saved original response replay mismatch: %g' % replay_error)
        for threshold in [.1, .5, .9]:
            suffix = str(threshold).replace('.', '_')
            result[tag+'_replay_threshold_disagreement_'+suffix] = int(
                (float(probability.max()) >= threshold) != (row[tag+'_saved_box_max'] >= threshold))
        q = frozen.projection(model, capture['feature'], slot)
        result[tag+'_projection_vs_official_max_abs_error'] = float((q-capture['q_official']).abs().max())
        torch.testing.assert_close(q, capture['q_official'], atol=1e-4, rtol=1e-4)
        originals[tag] = dict(feature=capture['feature'], q=q, logits=logits)

    starts = [row[tag+'_start'] for tag in ['a', 'b']]
    shared = sorted(set(range(starts[0], starts[0]+40)) & set(range(starts[1], starts[1]+40)))
    assert shared
    safe_positions = [i for i,t in enumerate(shared) if t-8 >= shared[0] and t+8 <= shared[-1]]
    safe_shared = [shared[i] for i in safe_positions]
    union_start, union_end = min(starts), max(starts)+40
    global_start, global_end = max(0, union_start-8), min(n, union_end+8)
    # One real sequence frontend gives identical shared evidence to both rows.
    global_feature = model.layer1(model.conv_in(input_tensor(global_start, global_end)))
    global_shared = global_feature[:, :, [t-global_start for t in shared], ys, xs].clone()
    del global_feature
    result.update(shared_frames=len(shared), global_input_start=global_start, global_input_end=global_end,
                  global_halo_missing_before=max(0, 8-union_start),
                  global_halo_missing_after=max(0, union_end+8-n),
                  shared_absolute_first=shared[0], shared_absolute_last=shared[-1],
                  safe_shared_frame_count=len(safe_shared), safe_shared_eligible=int(bool(safe_shared)),
                  current_frame_in_safe_shared=int(int(row['local_frame']) in safe_shared),
                  safe_shared_absolute_first=safe_shared[0] if safe_shared else '',
                  safe_shared_absolute_last=safe_shared[-1] if safe_shared else '')
    lambda_values = [(0., '000'), (.25, '025'), (.5, '050'), (.75, '075'), (1., '100')]
    for tag in ['a', 'b']:
        start, slot = row[tag+'_start'], row[tag+'_slot']
        feature, q = originals[tag]['feature'], originals[tag]['q']
        indices = [t-start for t in shared]
        exclusive_indices = sorted(set(range(40))-set(indices))
        weights, bias = frozen.row_weights(model, slot)
        bias = bias[None, :, None, None, None]
        q_shared = frozen.weighted_sum(weights[:, indices], feature[:, :, indices]) + bias
        q_exclusive = frozen.weighted_sum(weights[:, exclusive_indices], feature[:, :, exclusive_indices])
        q_global_shared = frozen.weighted_sum(weights[:, indices], global_shared) + bias
        q_safe = (frozen.weighted_sum(weights[:, [t-start for t in safe_shared]],
                                     global_shared[:, :, safe_positions]) + bias) if safe_shared else None
        recon_error = float((q_shared+q_exclusive-q).abs().max())
        result[tag+'_shared_exclusive_reconstruction_max_abs_error'] = recon_error
        torch.testing.assert_close(q_shared+q_exclusive, q, atol=1e-4, rtol=1e-4)
        # Any shared frame >=8 away from both local boundaries must agree with
        # the globally processed real sequence, up to floating point error.
        inside_positions = [i for i, idx in enumerate(indices) if 8 <= idx < 32]
        inside_error = 0.
        if inside_positions:
            left = feature[:, :, [indices[i] for i in inside_positions]]
            right = global_shared[:, :, inside_positions]
            inside_error = float((left-right).abs().max())
            torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4)
        result[tag+'_global_shared_interior_max_abs_error'] = inside_error
        result[tag+'_global_shared_interior_frame_count'] = len(inside_positions)
        result[tag+'_global_vs_local_shared_feature_max_abs_change'] = float((feature[:, :, indices]-global_shared).abs().max())
        result[tag+'_exclusive_frame_count'] = len(exclusive_indices)
        result[tag+'_exclusive_projection_l2'] = float(q_exclusive.square().sum().sqrt())
        map_inputs = [q_shared+lam*q_exclusive for lam,_ in lambda_values]
        map_inputs += [q_global_shared.clone(), q.clone()]
        if q_safe is not None:
            map_inputs.append(q_safe)
        maps = frozen.phi(model, torch.cat(map_inputs, dim=0))
        original_logits = originals[tag]['logits']
        for endpoint_index in [4, 6]:
            torch.testing.assert_close(maps[endpoint_index], original_logits, atol=1e-4, rtol=1e-4)
            endpoint_error = float((maps[endpoint_index].sigmoid()-original_logits.sigmoid()).abs().max())
            result[tag+'_endpoint_%d_max_abs_probability_error' % endpoint_index] = endpoint_error
            result[tag+'_endpoint_%d_max_abs_logit_error' % endpoint_index] = float((maps[endpoint_index]-original_logits).abs().max())
            if endpoint_error > tolerance:
                raise AssertionError('Support probe full endpoint mismatch: %g' % endpoint_error)
        summarize_map(result, tag+'_full', original_logits, peaks)
        summarize_map(result, tag+'_shared', maps[0], peaks)
        summarize_map(result, tag+'_global_shared', maps[5], peaks)
        summarize_map(result, tag+'_safe_shared', maps[7] if safe_shared else None, peaks)
        for index, (_, suffix) in enumerate(lambda_values):
            summarize_map(result, tag+'_lambda_'+suffix, maps[index], peaks)
        full_p, shared_p = float(original_logits.sigmoid().max()), float(maps[0].sigmoid().max())
        result[tag+'_shared_to_full_delta_box_max_probability'] = full_p-shared_p
        result[tag+'_shared_to_full_high_low_0_5'] = int(shared_p >= .5 and full_p < .5)
        result[tag+'_shared_to_full_low_high_0_5'] = int(shared_p < .5 and full_p >= .5)
        result[tag+'_shared_to_full_strong_high_low'] = int(shared_p >= .9 and full_p < .1)
        result[tag+'_shared_to_full_strong_low_high'] = int(shared_p < .1 and full_p >= .9)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--progress-every', type=int, default=10)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards or args.limit < 0 or args.progress_every < 1:
        parser.error('Invalid shard, limit, or progress arguments')
    root = args.root.resolve()
    out = args.output_dir or root/'mechanism_followup/support_probe'/('shard_%02d_of_%02d' % (args.shard_index,args.num_shards))
    out.mkdir(parents=True, exist_ok=True)
    if (out/'results.csv').exists():
        raise FileExistsError('Refuse to overwrite existing results: '+str(out))
    started = time.perf_counter()
    pairs, input_paths = frozen.select_pairs(root)
    assigned = [r for r in pairs if r['pair_index'] % args.num_shards == args.shard_index]
    selected = assigned[:args.limit] if args.limit else assigned
    assert selected
    with (out/'selected_pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    launches, raw_paths = {}, {}
    for sequence in sorted({r['sequence'] for r in selected}):
        path = root/'runs'/sequence/'LAUNCH_TILED.json'
        launch = json.loads(path.read_text())['cmd']
        launches[sequence] = launch
        raw_paths[sequence] = Path(frozen.launch_value(launch,'--input-npy'))
        input_paths.append(path)
    first = next(iter(launches.values()))
    source = Path(frozen.launch_value(first,'--source'))
    checkpoint = Path(frozen.launch_value(first,'--checkpoint'))
    for launch in launches.values():
        assert frozen.launch_value(launch,'--source') == str(source)
        assert frozen.launch_value(launch,'--checkpoint') == str(checkpoint)
        assert frozen.launch_value(launch,'--normalization') == 'RGB-T'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    random.seed(46)
    np.random.seed(46)
    torch.manual_seed(46)
    torch.cuda.manual_seed_all(46)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    model, model_path, epoch, radius = frozen.model_setup(source, checkpoint, device)
    arrays = {seq:np.load(path,mmap_mode='r',allow_pickle=False) for seq,path in raw_paths.items()}
    metadata = dict(status='RUNNING', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    purpose=__doc__, cohort_total=len(pairs), shard_total=len(assigned), selected_total=len(selected),
                    shard_index=args.shard_index, num_shards=args.num_shards, limit=args.limit,
                    wrapper_sha256=frozen.digest(__file__), frozen_dependency_sha256=frozen.digest(FROZEN_PATH),
                    input_sha256={str(p):frozen.digest(p) for p in input_paths},
                    raw_inputs_sha256={str(p):frozen.digest(p) for p in raw_paths.values()},
                    checkpoint_sha256=frozen.digest(checkpoint), model_source_sha256=frozen.digest(model_path),
                    layer_source_sha256={str(p):frozen.digest(p) for p in [source/'networks/layers/basic.py',source/'networks/layers/TPro.py']},
                    checkpoint_epoch=epoch, receptive_radius_tyx=radius, seed=46, tf32=False, deterministic_algorithms=True,
                    device=str(device), torch=torch.__version__, numpy=np.__version__, python=sys.version,
                    command=sys.argv, lambda_values=[0,.25,.5,.75,1],
                    projection='q_shared=sum_shared(w*F)/sqrt40+bias; q_exclusive=sum_exclusive(w*F)/sqrt40; q_lambda=q_shared+lambda*q_exclusive',
                    global_shared='Same shared absolute-frame frontend features computed once on real union window plus up to 8 real frames either side; each original output row retains its own coefficients and bias.',
                    dependency_safe_shared='Ssafe={u in intersection: u-8>=shared_first and u+8<=shared_last}; use unified global frontend features at these times and each original row weight+bias. Every feature temporal dependency is inside shared raw frames. Empty Ssafe yields eligible=0 and blank responses, not zeros.',
                    replay_probability_tolerance=1e-4, logit_endpoint_tolerance={'atol':1e-4,'rtol':1e-4},
                    fixed_peak_rule='Original probability argmax, first row-major tie; tie counts recorded.',
                    limitations=['Exclusive-term deletion is a potentially out-of-distribution internal activation ablation, not a natural input intervention.',
                                 'Ordinary shared features can depend on nonshared raw frames within radius 8; only safe_shared features restrict all temporal dependencies to common raw content.',
                                 'safe_shared is still an out-of-distribution projection subset; eligibility and whether the current frame lies in Ssafe must accompany subgroup results.',
                                 'Strong internal response does not establish physical target visibility or deployable detection gain.',
                                 'GT boxes and prior extreme responses select diagnostic windows; no labels are model input.',
                                 'No full-frame false-alarm or point-matching evaluation; strong response alone is not a new detection.',
                                 'Four development clips and target frames within tracks are not independent random samples.'])
    frozen.json_write(out/'metadata.json',metadata)
    maxima, disagreements = collections.defaultdict(float), collections.Counter()
    completed = 0
    try:
        with torch.inference_mode(), (out/'results.csv').open('w',newline='') as stream:
            writer = None
            for row in selected:
                result = support_pair(model,arrays[row['sequence']],row,device,1e-4)
                if writer is None:
                    writer = csv.DictWriter(stream,fieldnames=list(result))
                    writer.writeheader()
                writer.writerow(result)
                stream.flush()
                completed += 1
                for name,value in result.items():
                    if 'error' in name and isinstance(value,(float,int)):
                        maxima[name] = max(maxima[name],float(value))
                    if 'replay_threshold_disagreement' in name:
                        disagreements[name] += int(value)
                if completed % args.progress_every == 0 or completed == len(selected):
                    progress = dict(completed=completed,total=len(selected),elapsed_seconds=time.perf_counter()-started,
                                    pair_index=row['pair_index'],maximum_errors=dict(maxima),
                                    replay_threshold_disagreements=dict(disagreements))
                    frozen.json_write(out/'progress.json',progress)
                    print(json.dumps(progress),flush=True)
    except BaseException as error:
        metadata.update(status='FAILED',completed=completed,error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started)
        frozen.json_write(out/'metadata.json',metadata)
        raise
    metadata.update(status='COMPLETE',completed=completed,elapsed_seconds=time.perf_counter()-started,
                    maximum_errors=dict(maxima),replay_threshold_disagreements=dict(disagreements),
                    result_sha256=frozen.digest(out/'results.csv'),selected_pairs_sha256=frozen.digest(out/'selected_pairs.csv'))
    frozen.json_write(out/'metadata.json',metadata)
    print(json.dumps({'status':'COMPLETE','completed':completed,'elapsed_seconds':metadata['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
