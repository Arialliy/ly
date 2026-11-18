#!/usr/bin/env python3
"""Frozen-checkpoint mechanism diagnosis, never a deployable method evaluation.

GT boxes and previously observed minimum/maximum-response windows define each
diagnostic pair. All target-frame images stay unchanged. Three additive terms
exactly decompose the pre-BN TPro projection change; eight internal combinations
are read through the unchanged eval backend. A separate real temporal halo
intervention changes frontend boundary context only. See PROTOCOL.json.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def json_write(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def key(row):
    return (row['sequence'], int(row['local_frame']), int(row['track_id']),
            int(row['annotation_id']))


def select_pairs(root):
    choices_path = root / 'result_stratification/context/same_gt_frame_choices.csv'
    choices = [r for r in read_csv(choices_path)
               if float(r['threshold']) == .5 and int(r['strict']) == 1
               and int(r['multi']) == 1 and float(r['max_side']) <= 16]
    choices.sort(key=key)
    if len(choices) != 1417:
        raise ValueError('Frozen cohort should contain 1417 target frames, got %d' % len(choices))
    if len({key(r) for r in choices}) != len(choices):
        raise ValueError('Duplicate target frame in cohort')
    by_key = collections.defaultdict(list)
    context_paths = []
    for sequence in sorted({r['sequence'] for r in choices}):
        path = root / 'window_offset_experiment/analysis' / sequence / 'per_context_all_tracks.csv'
        context_paths.append(path)
        for row in read_csv(path):
            if float(row['threshold']) == .5:
                by_key[key(row)].append(row)
    result = []
    for pair_index, choice in enumerate(choices):
        windows = sorted(by_key[key(choice)], key=lambda r: (float(r['box_max']), int(r['window_start'])))
        if len({int(r['window_start']) for r in windows}) != len(windows):
            raise ValueError('Repeated context start for %r' % (key(choice),))
        if len(windows) != int(choice['context_count']) or len(windows) < 2:
            raise ValueError('Context count mismatch for %r' % (key(choice),))
        low, high = windows[0], windows[-1]
        assert low['window_start'] != high['window_start']
        for field in ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'source_frame']:
            assert low[field] == high[field]
        assert abs(float(low['box_max']) - float(choice['context_boxmax_min'])) < 1e-12
        assert abs(float(high['box_max']) - float(choice['context_boxmax_max'])) < 1e-12
        record = dict(pair_index=pair_index, **choice)
        for tag, window in [('a', low), ('b', high)]:
            record[tag + '_start'] = int(window['window_start'])
            record[tag + '_slot'] = int(window['slot'])
            record[tag + '_saved_box_max'] = float(window['box_max'])
            record[tag + '_window_source'] = window['window_source']
        for field in ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']:
            record[field] = float(low[field])
        result.append(record)
    return result, [choices_path] + context_paths


def crop_indices(row, height, width):
    # Exact analysis/analyze.py rule: original half-open box contains pixel centers.
    x, y, w, h = [row[k] for k in ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']]
    return (max(0, min(width, math.ceil(x - .5))),
            max(0, min(height, math.ceil(y - .5))),
            max(0, min(width, math.ceil(x + w - .5))),
            max(0, min(height, math.ceil(y + h - .5))))


def launch_value(command, name):
    return command[command.index(name) + 1]


def model_setup(source, checkpoint_path, device):
    import torch
    sys.path.insert(0, str(source))
    model_path = checkpoint_path.parent.parent / 'DeepPro-Plus.py'
    if not model_path.is_file():
        raise FileNotFoundError('Checkpoint-adjacent frozen model required: ' + str(model_path))
    spec = importlib.util.spec_from_file_location('official_frozen_factorial_model', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    numpy_core = np._core if hasattr(np, '_core') else np.core
    numeric_globals = [(numpy_core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
                       np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(numeric_globals):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = module.detector(1, 40, 40)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.to(device).eval()
    # Verify the longest spatial and temporal paths rather than assuming halos.
    path = [model.conv_in[0]]
    for block in model.layer1:
        path.extend([block.conv1, block.conv2, block.conv3])
    radius = [0, 0, 0]
    for layer in path:
        assert tuple(layer.stride) == (1, 1, 1)
        for axis in range(3):
            assert layer.kernel_size[axis] % 2 == 1
            radius[axis] += (layer.kernel_size[axis] - 1) * layer.dilation[axis] // 2
    assert radius == [8, 9, 9], radius
    for module_ in [model.TPro.conv[0], model.conv_out1[0], model.conv_out2]:
        assert module_.kernel_size == (1, 1, 1)
    assert model.TPro.seqlen == 40 and model.TPro.num_head == 8 and model.TPro.hidden_dim == 4
    return model, model_path, checkpoint.get('epoch'), radius


def phi(model, q):
    # Eval BN uses frozen running statistics, so batching combinations is exact.
    value = model.TPro.relu(model.TPro.norm1(q))
    value = model.TPro.conv(value)
    return model.conv_out2(model.conv_out1(value))[:, 0, 0]


def projection(model, feature, slot):
    """One official Linear output row; scale features, but never scale bias."""
    import torch
    chunks = []
    for head_index, head in enumerate(model.TPro.QK_heads):
        f = feature[:, head_index * 4:(head_index + 1) * 4].permute(0, 3, 4, 1, 2)
        q = torch.nn.functional.linear(f / math.sqrt(40), head.weight[slot:slot+1], head.bias[slot:slot+1])
        chunks.append(q[..., 0].permute(0, 3, 1, 2))
    return torch.cat(chunks, dim=1).unsqueeze(2)


def row_weights(model, slot):
    import torch
    weights = torch.stack([head.weight[slot] for head in model.TPro.QK_heads])
    bias = torch.stack([head.bias[slot] for head in model.TPro.QK_heads])
    return weights.repeat_interleave(4, 0), bias.repeat_interleave(4, 0)


def weighted_sum(weights, features):
    # [C,T] x [B,C,T,H,W] -> [B,C,1,H,W]
    return (features * weights[None, :, :, None, None]).sum(2, keepdim=True) / math.sqrt(40)


def decompose(model, fa, fb, start_a, start_b, slot_a, slot_b):
    import torch
    wa, ba = row_weights(model, slot_a)
    wb, bb = row_weights(model, slot_b)
    shared = sorted(set(range(start_a, start_a + 40)) & set(range(start_b, start_b + 40)))
    ai, bi = [t - start_a for t in shared], [t - start_b for t in shared]
    only_a = sorted(set(range(40)) - set(ai))
    only_b = sorted(set(range(40)) - set(bi))
    fsa, fsb = fa[:, :, ai], fb[:, :, bi]
    d_slot = weighted_sum(wb[:, bi] - wa[:, ai], (fsa + fsb) / 2)
    d_slot += (bb - ba)[None, :, None, None, None]
    d_feature = weighted_sum((wa[:, ai] + wb[:, bi]) / 2, fsb - fsa)
    d_membership = weighted_sum(wb[:, only_b], fb[:, :, only_b]) - weighted_sum(wa[:, only_a], fa[:, :, only_a])
    q_a, q_b = projection(model, fa, slot_a), projection(model, fb, slot_b)
    residual = q_a + d_slot + d_feature + d_membership - q_b
    scale = max(float(q_a.abs().max()), float(q_b.abs().max()), 1.)
    error = float(residual.abs().max())
    if error > 1e-4 + 1e-5 * scale:
        raise AssertionError('Projection decomposition fails: error=%g scale=%g' % (error, scale))
    interior_pairs = [(ia, ib) for ia, ib in zip(ai, bi) if 8 <= ia < 32 and 8 <= ib < 32]
    interior_error, interior_scale = 0., 0.
    if interior_pairs:
        ia, ib = zip(*interior_pairs)
        left, right = fa[:, :, list(ia)], fb[:, :, list(ib)]
        interior_error = float((left - right).abs().max())
        interior_scale = max(float(left.abs().max()), float(right.abs().max()), 1.)
        torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4)
    stats = dict(shared_frames=len(shared), exclusive_a_frames=len(only_a), exclusive_b_frames=len(only_b),
                 shared_interior_frames=len(interior_pairs), shared_interior_feature_max_abs_error=interior_error,
                 shared_interior_feature_scale=interior_scale, projection_reconstruction_max_abs_error=error,
                 projection_scale=scale, shared_feature_max_abs_difference=float((fsa-fsb).abs().max()))
    for name, delta in [('slot', d_slot), ('feature', d_feature), ('membership', d_membership)]:
        stats['delta_' + name + '_l2'] = float(delta.square().sum().sqrt())
        stats['delta_' + name + '_max_abs'] = float(delta.abs().max())
    # Sensitivity to allocating the product interaction: the primary symmetric
    # decomposition splits it equally. A/B anchors assign it to feature/slot.
    interaction_half = weighted_sum(wb[:, bi] - wa[:, ai], (fsb - fsa) / 2)
    stats['slot_feature_interaction_half_l2'] = float(interaction_half.square().sum().sqrt())
    return q_a, q_b, [d_slot, d_feature, d_membership], stats, interaction_half


def run_pair(model, frames, row, device, mean, std, replay_tolerance):
    import torch
    n, height, width = frames.shape
    x0, y0, x1, y1 = crop_indices(row, height, width)
    assert x1 > x0 and y1 > y0
    xa, ya, xb, yb = max(0, x0-16), max(0, y0-16), min(width, x1+16), min(height, y1+16)
    ys, xs = slice(y0-ya, y1-ya), slice(x0-xa, x1-xa)
    result = dict(row)
    result.update(crop_x0=x0, crop_y0=y0, crop_x1=x1, crop_y1=y1,
                  input_crop_x0=xa, input_crop_y0=ya, input_crop_x1=xb, input_crop_y1=yb)

    def input_tensor(start, end):
        raw = np.asarray(frames[start:end, ya:yb, xa:xb], dtype=np.float32)
        normed = np.ascontiguousarray((raw - mean) / std)
        if not np.isfinite(normed).all():
            raise ValueError('Nonfinite input')
        return torch.from_numpy(normed)[None, None].to(device)

    originals, extended = {}, {}
    for tag in ['a', 'b']:
        start, slot = int(row[tag + '_start']), int(row[tag + '_slot'])
        assert int(row['local_frame']) == start + slot and 0 <= start <= n-40
        capture = {}

        def feature_hook(module, inputs, output):
            capture['feature'] = output[..., ys, xs].clone()

        def projection_hook(module, inputs):
            capture['q'] = inputs[0][:, :, slot:slot+1, ys, xs].clone()

        h1 = model.layer1.register_forward_hook(feature_hook)
        h2 = model.TPro.norm1.register_forward_pre_hook(projection_hook)
        try:
            model_output = model(input_tensor(start, start+40))
            logits = model_output[1][0, slot, ys, xs].clone()
            del model_output
        finally:
            h1.remove()
            h2.remove()
        probability = logits.sigmoid()
        saved = float(row[tag + '_saved_box_max'])
        observed = float(probability.max())
        error = abs(saved - observed)
        result[tag + '_replay_box_max'] = observed
        result[tag + '_replay_box_max_logit'] = float(logits.max())
        result[tag + '_replay_abs_error'] = error
        for threshold in [.1, .5, .9]:
            suffix = str(threshold).replace('.', '_')
            result[tag + '_replay_threshold_disagreement_' + suffix] = int((saved >= threshold) != (observed >= threshold))
        if error > replay_tolerance:
            raise AssertionError('Replay mismatch for %r %s: saved=%g observed=%g error=%g' % (key(row), tag, saved, observed, error))
        peak = int(probability.flatten().argmax())
        py, px = divmod(peak, x1-x0)
        result[tag + '_probability_peak_tie_count'] = int((probability == probability.max()).sum())
        result[tag + '_peak_x_index'] = x0+px
        result[tag + '_peak_y_index'] = y0+py
        originals[tag] = dict(feature=capture['feature'], q=capture['q'], logits=logits, peak=(py, px))
        # Restore only real frontend context. No repeated/zero synthetic frames
        # are added; the original per-layer padding persists at source ends.
        ext_start, ext_end = max(0, start-8), min(n, start+48)
        extended_feature = model.layer1(model.conv_in(input_tensor(ext_start, ext_end)))
        extended_feature = extended_feature[:, :, start-ext_start:start-ext_start+40, ys, xs].clone()
        inside_error = float((extended_feature[:, :, 8:32] - capture['feature'][:, :, 8:32]).abs().max())
        torch.testing.assert_close(extended_feature[:, :, 8:32], capture['feature'][:, :, 8:32], atol=1e-4, rtol=1e-4)
        extended_logits = phi(model, projection(model, extended_feature, slot))[0]
        extended[tag] = extended_logits
        result[tag + '_halo_real_before'] = start-ext_start
        result[tag + '_halo_real_after'] = ext_end-(start+40)
        result[tag + '_halo_missing_before'] = 8-(start-ext_start)
        result[tag + '_halo_missing_after'] = 8-(ext_end-(start+40))
        result[tag + '_halo_interior_feature_max_abs_error'] = inside_error
        result[tag + '_halo_box_max'] = float(extended_logits.sigmoid().max())
        result[tag + '_halo_box_max_logit'] = float(extended_logits.max())
        result[tag + '_halo_delta_box_max'] = result[tag + '_halo_box_max'] - observed
        result[tag + '_halo_feature_max_abs_change'] = float((extended_feature-capture['feature']).abs().max())
        result[tag + '_halo_changed_feature_time_count'] = int(((extended_feature-capture['feature']).abs().amax(dim=(0,1,3,4)) > 1e-4).sum())
        del extended_feature

    qa, qb, deltas, stats, interaction_half = decompose(
        model, originals['a']['feature'], originals['b']['feature'],
        row['a_start'], row['b_start'], row['a_slot'], row['b_slot'])
    result.update(stats)
    for tag, q in [('a', qa), ('b', qb)]:
        result[tag + '_projection_vs_official_max_abs_error'] = float((q-originals[tag]['q']).abs().max())
        torch.testing.assert_close(q, originals[tag]['q'], atol=1e-4, rtol=1e-4)
    # Bit ordering in names is slot, shared frontend feature, membership.
    combos, names = [], []
    for mask in range(8):
        q = qa.clone()
        bits = [(mask >> i) & 1 for i in range(3)]
        for bit, delta in zip(bits, deltas):
            if bit:
                q += delta
        combos.append(q)
        names.append(''.join(map(str, bits)))
    outputs = phi(model, torch.cat(combos, 0))
    for tag, i in [('a', 0), ('b', 7)]:
        error = float((outputs[i].sigmoid()-originals[tag]['logits'].sigmoid()).abs().max())
        result[tag + '_factorial_endpoint_max_abs_probability_error'] = error
        result[tag + '_factorial_endpoint_max_abs_logit_error'] = float((outputs[i]-originals[tag]['logits']).abs().max())
        torch.testing.assert_close(outputs[i], originals[tag]['logits'], atol=1e-4, rtol=1e-4)
        if error > replay_tolerance:
            raise AssertionError('Factorial endpoint mismatch: ' + str(error))
    for name, output in zip(names, outputs):
        prefix = 'combo_' + name
        result[prefix + '_box_max_logit'] = float(output.max())
        result[prefix + '_box_max_probability'] = float(output.sigmoid().max())
        for tag in ['a', 'b']:
            py, px = originals[tag]['peak']
            result[prefix + '_' + tag + '_peak_logit'] = float(output[py, px])
            result[prefix + '_' + tag + '_peak_probability'] = float(output[py, px].sigmoid())
    for anchor, sign in [('a', -1), ('b', 1)]:
        anchored_deltas = [deltas[0] + sign * interaction_half,
                           deltas[1] - sign * interaction_half, deltas[2]]
        anchored_combos = []
        for mask in range(8):
            q = qa.clone()
            for i, delta in enumerate(anchored_deltas):
                if (mask >> i) & 1:
                    q += delta
            anchored_combos.append(q)
        anchor_outputs = phi(model, torch.cat(anchored_combos, 0))
        endpoint_error = float((anchor_outputs[7].sigmoid() - originals['b']['logits'].sigmoid()).abs().max())
        result['anchor_' + anchor + '_endpoint_max_abs_probability_error'] = endpoint_error
        result['anchor_' + anchor + '_endpoint_max_abs_logit_error'] = float((anchor_outputs[7] - originals['b']['logits']).abs().max())
        torch.testing.assert_close(anchor_outputs[0], originals['a']['logits'], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(anchor_outputs[7], originals['b']['logits'], atol=1e-4, rtol=1e-4)
        if endpoint_error > replay_tolerance:
            raise AssertionError('Anchored factorial endpoint mismatch: ' + str(endpoint_error))
        for name, output in zip(names, anchor_outputs):
            prefix = 'anchor_' + anchor + '_combo_' + name
            result[prefix + '_box_max_logit'] = float(output.max())
            result[prefix + '_box_max_probability'] = float(output.sigmoid().max())
            for tag in ['a', 'b']:
                py, px = originals[tag]['peak']
                result[prefix + '_' + tag + '_peak_logit'] = float(output[py, px])
                result[prefix + '_' + tag + '_peak_probability'] = float(output[py, px].sigmoid())
    for halo_tag in ['a', 'b']:
        for peak_tag in ['a', 'b']:
            py, px = originals[peak_tag]['peak']
            result[halo_tag + '_halo_' + peak_tag + '_peak_logit'] = float(extended[halo_tag][py, px])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0, help='Pilot prefix of assigned shard; zero means all.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--progress-every', type=int, default=10)
    parser.add_argument('--replay-tolerance', type=float, default=1e-4)
    parser.add_argument('--selection-only', action='store_true', help='CPU CSV selection audit; no torch import or model execution.')
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error('Require 0 <= shard-index < num-shards')
    if args.limit < 0 or args.progress_every < 1:
        parser.error('limit must be nonnegative and progress-every positive')
    started = time.perf_counter()
    root = args.root.resolve()
    out = args.output_dir or root / 'mechanism_followup' / ('shard_%02d_of_%02d' % (args.shard_index, args.num_shards))
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.csv').exists():
        raise FileExistsError('Use a fresh output directory; results.csv already exists: ' + str(out))
    pairs, input_paths = select_pairs(root)
    assigned = [row for row in pairs if row['pair_index'] % args.num_shards == args.shard_index]
    selected = assigned[:args.limit] if args.limit else assigned
    if not selected:
        raise ValueError('No selected pairs')
    with (out / 'selected_pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    launches, raw_paths = {}, {}
    for sequence in sorted({row['sequence'] for row in selected}):
        launch_path = root / 'runs' / sequence / 'LAUNCH_TILED.json'
        launch = json.loads(launch_path.read_text())['cmd']
        launches[sequence] = launch
        raw_paths[sequence] = Path(launch_value(launch, '--input-npy'))
        input_paths.append(launch_path)
    first = next(iter(launches.values()))
    source = Path(launch_value(first, '--source'))
    checkpoint_path = Path(launch_value(first, '--checkpoint'))
    for launch in launches.values():
        assert launch_value(launch, '--source') == str(source)
        assert launch_value(launch, '--checkpoint') == str(checkpoint_path)
        assert launch_value(launch, '--normalization') == 'RGB-T'
    metadata = dict(status='SELECTION_READY', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    purpose='GT-conditioned extreme-window mechanism diagnosis; no claimed detection improvement',
                    cohort_total=len(pairs), shard_total=len(assigned), selected_total=len(selected),
                    shard_index=args.shard_index, num_shards=args.num_shards, limit=args.limit,
                    selection='Sort existing contexts by (box_max, start), choose first and last; tie endpoints remain distinct.',
                    factor_bit_order=['shared_slot_weights_and_bias', 'shared_frontend_features', 'context_membership'],
                    decomposition_anchors={'primary':'symmetric split of slot-feature interaction',
                                           'a':'d_slot=dw*FA+db, d_feature=wB*dF',
                                           'b':'d_slot=dw*FB+db, d_feature=wA*dF'},
                    bbox_rule='ceil(x-.5), ceil(y-.5), ceil(x+w-.5), ceil(y+h-.5), clipped; pixel centers in half-open GT box',
                    fixed_peak_rule='Endpoint probability maximum; ties choose first row-major pixel, tie counts reported. Saturation may differ from logit argmax.',
                    spatial_halo=16, temporal_halo=8, seed=46, tf32=False, deterministic_algorithms=True,
                    replay_probability_tolerance=args.replay_tolerance, source=str(source), checkpoint=str(checkpoint_path),
                    raw_inputs={k:str(v) for k,v in raw_paths.items()}, wrapper_sha256=digest(__file__),
                    input_sha256={str(p):digest(p) for p in input_paths},
                    python=sys.version, numpy=np.__version__, command=sys.argv,
                    limitations=['Target frames share tracks/clips and are not independent samples.',
                                 'Extreme windows selected from prior outputs; intervention differences are conditioned diagnostics.',
                                 'Halo only uses real frames available inside each source clip; missing context is reported.',
                                 'Shapley/internal combinations do not isolate unique external causal mechanisms.',
                                 'No full-frame false-alarm assessment or official Pd/Fa evaluation.'])
    json_write(out / 'metadata.json', metadata)
    print(json.dumps({'selection_ready': len(selected), 'cohort':len(pairs), 'output_dir':str(out)}), flush=True)
    if args.selection_only:
        return
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
    model, model_path, epoch, radius = model_setup(source, checkpoint_path, device)
    metadata.update(status='RUNNING', torch=torch.__version__, cuda=torch.version.cuda, device=str(device),
                    device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                    checkpoint_epoch=epoch, model_file=str(model_path), receptive_radius_tyx=radius,
                    checkpoint_sha256=digest(checkpoint_path), model_source_sha256=digest(model_path),
                    layer_source_sha256={str(p):digest(p) for p in [source/'networks/layers/basic.py', source/'networks/layers/TPro.py']},
                    raw_input_sha256={str(p):digest(p) for p in raw_paths.values()})
    json_write(out / 'metadata.json', metadata)
    arrays = {seq:np.load(path, mmap_mode='r', allow_pickle=False) for seq,path in raw_paths.items()}
    maxima = collections.defaultdict(float)
    disagreements = collections.Counter()
    completed = 0
    try:
        with torch.inference_mode(), (out / 'results.csv').open('w', newline='') as stream:
            writer = None
            for row in selected:
                result = run_pair(model, arrays[row['sequence']], row, device, 85.0799, 47.4845, args.replay_tolerance)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(result))
                    writer.writeheader()
                writer.writerow(result)
                stream.flush()
                completed += 1
                for name, value in result.items():
                    if 'error' in name and isinstance(value, (int, float)):
                        maxima[name] = max(maxima[name], float(value))
                    if 'replay_threshold_disagreement' in name:
                        disagreements[name] += int(value)
                if completed % args.progress_every == 0 or completed == len(selected):
                    progress = dict(completed=completed, total=len(selected), elapsed_seconds=time.perf_counter()-started,
                                    sequence=row['sequence'], pair_index=row['pair_index'], maximum_errors=dict(maxima),
                                    replay_threshold_disagreements=dict(disagreements))
                    json_write(out / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
    except BaseException as error:
        metadata.update(status='FAILED', completed=completed, error_type=type(error).__name__, error=str(error),
                        elapsed_seconds=time.perf_counter()-started)
        json_write(out / 'metadata.json', metadata)
        raise
    metadata.update(status='COMPLETE', completed=completed, elapsed_seconds=time.perf_counter()-started,
                    maximum_errors=dict(maxima), replay_threshold_disagreements=dict(disagreements),
                    result_sha256=digest(out/'results.csv'), selected_pairs_sha256=digest(out/'selected_pairs.csv'))
    json_write(out / 'metadata.json', metadata)
    print(json.dumps({'status':'COMPLETE', 'completed':completed, 'elapsed_seconds':metadata['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
