#!/usr/bin/env python3
"""Original Plus supervision control: frozen eight-window union, five per arrangement."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from gaussian_labels import LABEL_CONFIG


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--reference-checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--variant', choices=['gaussian_softiou_approx'], required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--sequence', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify-tiling', action='store_true')
    a = p.parse_args()
    if (a.output/'COMPLETE.json').exists():
        raise FileExistsError('Completed output exists; verify it instead of restarting')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'windows').mkdir(exist_ok=True)
    random.seed(46); np.random.seed(46); torch.manual_seed(46); torch.cuda.manual_seed_all(46)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    diag = Path(__file__).resolve().parents[2]
    helpers = module_at('bc_original_training_helpers', diag/'top1_background_control_20261011/code/train_pilot.py')
    inference = module_at('pilot_original_tiling', diag/'infer_with_windows.py')
    model, model_path = helpers.load_model(a.source, a.reference_checkpoint)
    cp = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    variant = cp['variant']
    run=json.loads((a.checkpoint.parent/'RUN.json').read_text())
    if (variant!=a.variant or a.output.parent.name!=variant or cp.get('step')!=512
            or run.get('status')!='complete' or run.get('variant')!=variant
            or run['final_checkpoint_sha256']!=digest(a.checkpoint)
            or cp['initial_checkpoint_sha256']!=digest(a.reference_checkpoint)
            or cp['plan_sha256']!=run['plan_sha256']
            or run['label_source_sha256']!=digest(Path(__file__).with_name('gaussian_labels.py'))
            or run['train_source_sha256']!=digest(Path(__file__).with_name('train_supervision.py'))):
        raise ValueError('Training completion, model source, checkpoint provenance or variant mismatch')
    if cp['label_config'] != LABEL_CONFIG:
        raise ValueError('Frozen supervision configuration mismatch')
    model.load_state_dict(cp['model_state_dict'], strict=True)
    model = model.cuda()
    epoch = cp.get('epoch', cp.get('step'))
    radius = [None,9,9]
    model.eval()
    raw_path = a.dataset_root/a.sequence/'raw_frames.npy'
    raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
    if raw.dtype != np.uint8 or raw.shape != (160,512,640):
        raise ValueError('Unexpected pilot raw image shape/dtype')
    starts = [0,24,36,60,72,96,108,120]
    official = [0,36,72,108,120]
    fused = np.zeros(raw.shape, dtype=np.float32)
    coverage = np.zeros(160, dtype=np.int32)
    metadata = {'sequence':a.sequence,'checkpoint':str(a.checkpoint),
                'checkpoint_sha256':digest(a.checkpoint),'reference_checkpoint_sha256':digest(a.reference_checkpoint),
                'model_source':str(model_path),'model_source_sha256':digest(model_path),
                'raw_sha256':digest(raw_path),'script_sha256':digest(__file__),
                'normalization':[85.0799,47.4845],'seed':46,'precision':'FP32',
                'torch':str(torch.__version__),'cuda':str(torch.version.cuda),
                'official_starts':official,'diagnostic_starts':starts,
                'fusion':'pixelwise probability maximum over official starts only',
                'tile_core':128,'tile_halo':16,'receptive_radius_tyx':radius,
                'model_labels_access':False,'tiling_checks':[],
                'condition':variant,'variant':variant,'label_config':LABEL_CONFIG,
                'label_source_sha256':digest(Path(__file__).with_name('gaussian_labels.py')),
                'train_source_sha256':digest(Path(__file__).with_name('train_supervision.py')),
                'parameters':sum(p.numel() for p in model.parameters()),'window_timing':[]}
    (a.output/'RUNNING.json').write_text(json.dumps(metadata,indent=2)+'\n')
    begin = time.perf_counter()
    with torch.inference_mode():
        if a.verify_tiling:
            for y,x in [(0,0),(176,224),(352,448)]:
                v = torch.from_numpy(np.ascontiguousarray((raw[:40,y:y+160,x:x+192].astype(np.float32)-85.0799)/47.4845))[None,None].cuda()
                direct = model(v)[1].cpu()
                tiled = inference.tiled_logits(model,v,48,16)
                torch.testing.assert_close(tiled,direct,atol=1e-4,rtol=1e-4)
                metadata['tiling_checks'].append({'yx':[y,x],
                    'max_logit_error':float((direct-tiled).abs().max()),
                    'max_probability_error':float((direct.sigmoid()-tiled.sigmoid()).abs().max()),
                    'threshold05_disagreements':int(((direct.sigmoid()>=.5)!=(tiled.sigmoid()>=.5)).sum())})
                del v,direct,tiled
        torch.cuda.reset_peak_memory_stats()
        for start in starts:
            x = torch.from_numpy(np.ascontiguousarray((raw[start:start+40].astype(np.float32)-85.0799)/47.4845))[None,None].cuda()
            torch.cuda.synchronize(); t0=time.perf_counter()
            logits = inference.tiled_logits(model,x,128,16)
            torch.cuda.synchronize(); model_seconds=time.perf_counter()-t0
            metadata['window_timing'].append({'start':start,'model_plus_cpu_copy_seconds':model_seconds})
            prob = logits.sigmoid()[0].numpy()
            if not np.isfinite(prob).all():
                raise ValueError('Nonfinite inference')
            np.savez_compressed(a.output/'windows'/f'window_{start:03d}.npz',probabilities=prob,start=start,end=start+40)
            if start in official:
                fused[start:start+40] = np.maximum(fused[start:start+40],prob)
                coverage[start:start+40] += 1
            print(json.dumps({'window_complete':start,'official':start in official,'elapsed_seconds':time.perf_counter()-begin}),flush=True)
            del x,logits,prob
    metadata['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    metadata['O_model_seconds']=sum(r['model_plus_cpu_copy_seconds'] for r in metadata['window_timing'] if r['start'] in official)
    metadata['S_model_seconds']=sum(r['model_plus_cpu_copy_seconds'] for r in metadata['window_timing'] if r['start'] in [0,24,60,96,120])
    assert np.all(coverage>0)
    np.savez_compressed(a.output/'official_fused.npz',probabilities=fused,coverage=coverage,starts=np.array(official))
    metadata["new_model_inference_elapsed_seconds"]=time.perf_counter()-begin
    metadata.update(elapsed_seconds=time.perf_counter()-begin,status='COMPLETE',
                    fused_sha256=digest(a.output/'official_fused.npz'),
                    window_sha256={f.name:digest(f) for f in sorted((a.output/'windows').glob('*.npz'))})
    (a.output/'COMPLETE.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'status':'COMPLETE','sequence':a.sequence,'seconds':metadata['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
