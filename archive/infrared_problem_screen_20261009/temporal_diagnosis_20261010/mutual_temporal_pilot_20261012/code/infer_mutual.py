#!/usr/bin/env python3
"""Frozen E0/E1/E2 final inference and E2 mass/current no-training diagnostics."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import train_mutual as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'reference-checkpoint', 'checkpoint', 'dataset-root', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--variant', choices=runner.VARIANTS, required=True)
    parser.add_argument('--diagnostic', choices=['none', 'mass', 'current'], default='none')
    parser.add_argument('--step', type=int, choices=[128, 256, 384, 512], default=512)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--verify-tiling', action='store_true')
    parser.add_argument('--provenance', action='store_true', help='Read-only fixed E2 final query annotation telemetry')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.diagnostic != 'none' and args.variant != 'E2':
        parser.error('Only final E2 weights receive no-training diagnostic interventions')
    if args.diagnostic != 'none' and args.step != 512:
        parser.error('Mechanism diagnostics use only the prespecified final checkpoint')
    if args.provenance and (args.variant != 'E2' or args.diagnostic != 'none' or args.step != 512):
        parser.error('Provenance only applies to unmodified final E2 inference')
    condition = args.variant if args.diagnostic == 'none' else 'E2_' + args.diagnostic
    if args.step != 512:
        condition += f'_step{args.step}'
    if args.output.parent.name != condition:
        raise ValueError('Output parent must be the condition name')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Inference requires a new empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'windows').mkdir()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(46); np.random.seed(46); torch.manual_seed(46); torch.cuda.manual_seed_all(46)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    base = runner.load_module('mutual_inference_frozen_helpers', runner.Q / 'top1_background_control_20261011/code/train_pilot.py')
    inference = runner.load_module('mutual_inference_frozen_tiling', runner.Q / 'infer_with_windows.py')
    model, model_path = base.load_model(args.source, args.reference_checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    run = json.loads((args.checkpoint.parent / 'RUN.json').read_text())
    snapshot = next(x for x in run['snapshots'] if x['step'] == args.step)
    expected_sources = {'train_source_sha256': runner.sha(runner.__file__),
                        'label_source_sha256': runner.sha(runner.LABEL_PATH),
                        'module_source_sha256': runner.sha(Path(__file__).with_name('module.py')),
                        'prior_train_source_sha256': runner.sha(runner.PREVIOUS / 'code/train_supervision.py'),
                        'top1_loss_source_sha256': runner.sha(runner.TOP1_PATH),
                        'model_source_sha256': runner.sha(model_path)}
    if (checkpoint['variant'] != args.variant or checkpoint['step'] != args.step or run['status'] != 'complete'
            or run['variant'] != args.variant or checkpoint['radius'] != 2 or run['radius'] != 2
            or snapshot['sha256'] != runner.sha(args.checkpoint)
            or checkpoint['initial_checkpoint_sha256'] != runner.sha(args.reference_checkpoint)
            or checkpoint['plan_sha256'] != runner.EXPECTED['plan']
            or checkpoint['label_config'] != runner.LABEL_CONFIG
            or checkpoint['effective_supervision_config'] != runner.EFFECTIVE_SUPERVISION
            or any(run[name] != digest or checkpoint[name] != digest for name, digest in expected_sources.items())):
        raise AssertionError('Frozen completed-training provenance mismatch')
    aggregation = runner.attach_aggregation(model, args.variant, radius=2)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if aggregation is not None and args.diagnostic != 'none':
        aggregation.set_mode(args.diagnostic)
    # Statistics are logged during training; disable their reductions/synchronization
    # for all inference conditions so runtime comparisons reflect the same policy.
    if aggregation is not None:
        aggregation.collect_stats = False
    device = torch.device(args.device)
    model = model.to(device).eval()
    raw_path = args.dataset_root / args.sequence / 'raw_frames.npy'
    raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
    if raw.dtype != np.uint8 or raw.shape != (160, 512, 640):
        raise ValueError('Unexpected frozen development image dimensions')
    starts = [0, 24, 36, 60, 72, 96, 108, 120]
    official = [0, 36, 72, 108, 120]
    symmetric = [0, 24, 60, 96, 120]
    fused = np.zeros(raw.shape, dtype=np.float32)
    coverage = np.zeros(160, dtype=np.int32)
    halo, core = 20, 128
    recorder = None
    provenance_context = {'rows': [], 'queries': [], 'start': None, 'seconds': 0.0}
    if args.provenance:
        provenance_module = runner.load_module('mutual_provenance_recorder', Path(__file__).resolve().parents[1] / 'evaluation_code/provenance.py')
        recorder = provenance_module.Recorder(Path(__file__).resolve().parents[1], args.sequence, radius=2)

        def capture_sparse(module, inputs):
            if not provenance_context['queries']:
                return
            if device.type == 'cuda': torch.cuda.synchronize(device)
            stamp = time.perf_counter()
            result = aggregation.sparse_query_weights(inputs[0], provenance_context['queries'])
            recorder.add(provenance_context['rows'], provenance_context['start'], result)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            provenance_context['seconds'] += time.perf_counter() - stamp
        aggregation.register_forward_pre_hook(capture_sparse)

    def tiled_with_telemetry(image, start):
        if recorder is None:
            return inference.tiled_logits(model, image, core, halo)
        height, width = image.shape[-2:]
        output = torch.empty((image.shape[0], 40, height, width), dtype=image.dtype, device='cpu')
        for y in range(0, height, core):
            y1 = min(y + core, height)
            for x in range(0, width, core):
                x1 = min(x + core, width)
                ya, yb = max(0, y-halo), min(height, y1+halo)
                xa, xb = max(0, x-halo), min(width, x1+halo)
                rows, queries = recorder.select(start, y, y1, x, x1, ya, xa)
                provenance_context.update(rows=rows, queries=queries, start=start)
                result = model(image[..., ya:yb, xa:xb].contiguous())[1]
                output[..., y:y1, x:x1] = result[..., y-ya:y1-ya, x-xa:x1-xa].cpu()
                del result
        provenance_context.update(rows=[], queries=[], start=None)
        return output
    metadata = {
        'sequence': args.sequence, 'condition': condition, 'variant': condition,
        'training_variant': args.variant, 'diagnostic': args.diagnostic,
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': runner.sha(args.checkpoint),
        'reference_checkpoint_sha256': runner.sha(args.reference_checkpoint),
        'model_source': str(model_path), **expected_sources,
        'raw_sha256': runner.sha(raw_path), 'script_sha256': runner.sha(__file__),
        'normalization': [85.0799, 47.4845], 'seed': 46, 'precision': 'FP32_no_AMP_no_TF32',
        'torch': str(torch.__version__), 'cuda': str(torch.version.cuda),
        'official_starts': official, 'diagnostic_starts': starts,
        'fusion': 'pixelwise probability maximum over official starts only',
        'tile_core': core, 'tile_halo': halo, 'receptive_radius_tyx': [None, 13, 13],
        'conservative_halo_derivation': 'old halo16 plus mutual search 2r=4; all conditions use20',
        'model_labels_access': False, 'read_only_query_labels_access': args.provenance,
        'tiling_checks': [], 'window_timing': [],
        'label_config': runner.LABEL_CONFIG, 'parameters': sum(p.numel() for p in model.parameters()),
        'radius': 2, 'descriptor_dim': 8, 'module_mode': 'off' if aggregation is None else aggregation.mode,
        'main_checkpoint_step': 512, 'checkpoint_step': args.step,
        'peak_memory_scope': 'includes sparse provenance telemetry' if args.provenance else 'model inference and output copy',
        'checkpoint_is_main': args.step == 512,
        'inference_module_statistics': 'disabled equally for cost measurement',
    }
    runner.save_json(args.output / 'RUNNING.json', metadata)
    begin = time.perf_counter()
    try:
        with torch.inference_mode():
            if args.verify_tiling:
                for y, x in [(0, 0), (176, 224), (352, 448)]:
                    probe = torch.from_numpy(np.ascontiguousarray((raw[:40, y:y+160, x:x+192].astype(np.float32) - 85.0799) / 47.4845))[None, None].to(device)
                    direct = model(probe)[1].cpu()
                    tiled = inference.tiled_logits(model, probe, 48, halo)
                    torch.testing.assert_close(tiled, direct, atol=1e-4, rtol=1e-4)
                    disagreement = int(((direct.sigmoid() >= .5) != (tiled.sigmoid() >= .5)).sum())
                    metadata['tiling_checks'].append({'yx': [y, x],
                        'max_logit_error': float((direct - tiled).abs().max()),
                        'max_probability_error': float((direct.sigmoid() - tiled.sigmoid()).abs().max()),
                        'threshold05_disagreements': disagreement, 'atol': 1e-4, 'rtol': 1e-4})
                    del probe, direct, tiled
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for start in starts:
                image = torch.from_numpy(np.ascontiguousarray((raw[start:start+40].astype(np.float32) - 85.0799) / 47.4845))[None, None].to(device)
                if device.type == 'cuda': torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                old_provenance_seconds = provenance_context['seconds']
                logits = tiled_with_telemetry(image, start)
                if device.type == 'cuda': torch.cuda.synchronize(device)
                total_seconds = time.perf_counter() - t0
                telemetry_seconds = provenance_context['seconds'] - old_provenance_seconds
                metadata['window_timing'].append({'start': start,
                    'model_plus_cpu_copy_seconds': total_seconds - telemetry_seconds,
                    'total_including_provenance_seconds': total_seconds,
                    'provenance_seconds': telemetry_seconds})
                probabilities = logits.sigmoid()[0].numpy()
                if not np.isfinite(probabilities).all():
                    raise FloatingPointError('Nonfinite window probability')
                np.savez_compressed(args.output / 'windows' / f'window_{start:03d}.npz', probabilities=probabilities, start=start, end=start+40)
                if start in official:
                    fused[start:start+40] = np.maximum(fused[start:start+40], probabilities)
                    coverage[start:start+40] += 1
                print(json.dumps({'window_complete': start, 'condition': condition, 'elapsed_seconds': time.perf_counter() - begin}), flush=True)
                del image, logits, probabilities
        assert np.all(coverage > 0)
        if recorder is not None:
            recorder.finish(args.output)
        np.savez_compressed(args.output / 'official_fused.npz', probabilities=fused, coverage=coverage, starts=np.array(official))
        metadata.update(status='COMPLETE', elapsed_seconds=time.perf_counter() - begin,
            new_model_inference_elapsed_seconds=time.perf_counter() - begin,
            O_model_seconds=sum(r['model_plus_cpu_copy_seconds'] for r in metadata['window_timing'] if r['start'] in official),
            S_model_seconds=sum(r['model_plus_cpu_copy_seconds'] for r in metadata['window_timing'] if r['start'] in symmetric),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
            fused_sha256=runner.sha(args.output / 'official_fused.npz'),
            window_sha256={p.name: runner.sha(p) for p in sorted((args.output / 'windows').glob('*.npz'))})
        runner.save_json(args.output / 'COMPLETE.json', metadata)
        print(json.dumps({'status': 'COMPLETE', 'sequence': args.sequence, 'condition': condition, 'seconds': metadata['elapsed_seconds']}), flush=True)
    except BaseException as error:
        metadata.update(status='FAILED', error=repr(error), elapsed_seconds=time.perf_counter() - begin)
        runner.save_json(args.output / 'FAILED.json', metadata)
        raise


if __name__ == '__main__':
    main()
