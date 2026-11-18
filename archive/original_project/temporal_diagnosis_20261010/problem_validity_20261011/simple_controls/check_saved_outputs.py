#!/usr/bin/env python3
"""Post-save consistency check; does not select operating points."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
checks=[]
launches=json.loads((ROOT/'LAUNCHES.json').read_text())
script_sha=hashlib.sha256((ROOT/'score_controls.py').read_bytes()).hexdigest()
assert all(x['script_sha256']==script_sha and not x['gpu_requested'] for x in launches)
states=json.loads((ROOT/'RUNTIME_STATUS.json').read_text())['jobs']
assert all(not x['state']['Running'] and x['state']['Status']=='exited' and x['state']['ExitCode']==0
           and not x['state']['OOMKilled'] and not x['devices'] for x in states)
for control in ['official5_probmean','symmetric5_probmax']:
    for sequence in ['DJI_0026_2','DJI_0067_2']:
        p=ROOT/control/sequence
        metadata=json.loads((p/'fused_COMPLETE.json').read_text())
        assert metadata['scoring_script_sha256']==script_sha
        rows=list(csv.DictReader((p/'fused_curves.csv').open()))
        with np.load(p/'fused_matches.npz',allow_pickle=False) as z:
            assert np.array_equal(z['thresholds'],metadata['thresholds'])
            assert len(rows)==23==len(z['thresholds'])
            assert len(metadata['objects'])==z['matched'].shape[1]==z['offsets'][-1]
            assert z['frame_counts'].shape==(23,160,6)
            assert np.array_equal(z['frame_counts'][:,:,0],np.broadcast_to(np.diff(z['offsets']),(23,160)))
            assert np.all(z['frame_counts'][:,:,2]-z['frame_counts'][:,:,1]==z['frame_counts'][:,:,3])
            assert np.all(z['frame_counts'][:,:,5]<=z['frame_counts'][:,:,4])
            assert len({(x['frame'],x['annotation_id']) for x in metadata['objects']})==len(metadata['objects'])
            for i,r in enumerate(rows):
                counts=z['frame_counts'][i].sum(0)
                assert float(r['threshold'])==z['thresholds'][i]
                assert int(r['matched'])==int(z['matched'][i].sum())==int(counts[1])
                assert int(r['strict_small_matched'])==int(z['matched'][i,z['strict_small']].sum())
                assert int(r['strict_small_gt'])==int(z['strict_small'].sum())
                for column,k in [('gt',0),('components',2),('unmatched_components',3),('positive_pixels',4),('positive_pixels_outside_boxes',5)]:
                    assert int(r[column])==int(counts[k])
                for t in range(160):
                    a,b=z['offsets'][t:t+2]
                    assert int(z['matched'][i,a:b].sum())==int(z['frame_counts'][i,t,1])
        for filename,key in [('fused_matches.npz','matches_sha256'),('fused_curves.csv','curves_sha256')]:
            assert hashlib.sha256((p/filename).read_bytes()).hexdigest()==metadata[key]
        checks.append(dict(control=control,sequence=sequence,status='PASS',curve_rows=23))
result=dict(status='PASS',checks=checks,runtime_exited_zero_cpu_only=True,
            checks_description='Output hashes, current script against launch hash, CSV/NPZ counts, per-frame matching conservation, threshold and object alignment; no threshold selected.')
(ROOT/'OUTPUT_INTEGRITY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
