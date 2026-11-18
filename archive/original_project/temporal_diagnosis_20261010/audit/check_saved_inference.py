import json
import hashlib
from pathlib import Path
import numpy as np

q = Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
wrapper_sha = hashlib.sha256((q/'infer_with_windows.py').read_bytes()).hexdigest()
results = []
for run in sorted((q/'runs').iterdir()):
    path = run/'predictions.npz'
    if not path.is_file():
        continue
    with np.load(path, allow_pickle=False) as z:
        p = z['probabilities']
        windows = z['windows']
        coverage = z['coverage']
        meta = json.loads(str(z['metadata_json']))
    assert p.shape == (160, 512, 640), p.shape
    assert p.dtype == np.float32
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    assert windows.tolist() == [[0,40],[36,76],[72,112],[108,148],[120,160]]
    assert meta['wrapper_sha256'] == wrapper_sha
    assert meta['seed'] == 46
    assert meta['normalization'] == 'RGB-T' and meta['mean'] == 85.0799 and meta['std'] == 47.4845
    assert meta['spatial_tile_core'] == 128 and meta['spatial_halo'] == 16
    assert meta['checkpoint_sha256'] == '4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc'
    assert len(meta['tiling_check']) == 5 and all(c['status']=='passed' for c in meta['tiling_check'])
    rebuilt = np.zeros_like(p)
    counted = np.zeros(160, dtype=np.int32)
    files = sorted((run/'windows').glob('window_*.npz'))
    assert len(files) == 5
    for file, (start,end) in zip(files, windows):
        with np.load(file, allow_pickle=False) as z:
            pp = z['probabilities']
            assert int(z['start']) == start and int(z['end']) == end
            assert pp.shape == (40,512,640) and np.isfinite(pp).all()
            np.maximum(rebuilt[start:end], pp, out=rebuilt[start:end])
            counted[start:end] += 1
    assert np.array_equal(rebuilt,p)
    assert np.array_equal(counted, coverage)
    results.append({'sequence':run.name, 'status':'PASS',
                    'shape':list(p.shape), 'dtype':str(p.dtype),
                    'min':float(p.min()), 'max':float(p.max()),
                    'windows':windows.tolist(), 'coverage_counts':{str(x):int((coverage==x).sum()) for x in np.unique(coverage)},
                    'max_fusion_exact':True,
                    'wrapper_sha256':wrapper_sha,
                    'checkpoint_sha256':meta['checkpoint_sha256'],
                    'checkpoint_epoch':meta['checkpoint_epoch'],
                    'tiling_checks':meta['tiling_check']})
    print(json.dumps({'sequence':run.name,'status':'PASS'}),flush=True)
assert len(results)==4
(q/'audit'/'SAVED_INFERENCE_CHECK.json').write_text(json.dumps({'status':'PASS','results':results},indent=2)+'\n')
