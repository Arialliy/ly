from pathlib import Path
from collections import defaultdict
import difflib
import hashlib
import json
import math
import numpy as np
from independent_component_helpers import flood_components, assignment

q=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
seq='DJI_0047_3'; track=4621500
base=q/'runs'/seq; off=q/'window_offset_experiment/runs'/seq
with np.load(base/'predictions.npz',allow_pickle=False) as z:
    original=z['probabilities']; om=json.loads(str(z['metadata_json']))
with np.load(off/'predictions_partial.npz',allow_pickle=False) as z:
    nm=json.loads(str(z['metadata_json']))
fields=['model_type','model_file','checkpoint_sha256','model_source_sha256','input','input_shape','normalization','mean','std','torch_version','seed','deterministic_algorithms','tf32','spatial_tile_core','spatial_halo','checkpoint_epoch']
assert all(om[k]==nm[k] for k in fields)
raw=Path(om['input']).parent
ann=json.loads((raw/'annotations.json').read_text())
manifest=json.loads((raw/'FRAME_MANIFEST.json').read_text())
byimage=defaultdict(list)
for a in ann['annotations']: byimage[a['image_id']].append(a)
allgt=[]; chosen=[]
for f in manifest['frames']:
    gg=[]
    for a in byimage[f['source_image_id']]:
        x,y,w,h=a['bbox']
        if not a.get('ignore',0) and w>0 and h>0 and min(640,x+w)>max(0,x) and min(512,y+h)>max(0,y): gg.append(a)
    gg.sort(key=lambda a:(a['track_id'],a['id']));allgt.append(gg)
    chosen.append(next(a for a in gg if a['track_id']==track))
assert [f['raw_frame'] for f in manifest['frames']]==list(range(160))

def boxmax(p,a):
    x,y,w,h=a['bbox']; H,W=p.shape
    left=max(0,math.ceil(x-.5));right=min(W,math.ceil(x+w-.5));top=max(0,math.ceil(y-.5));bottom=min(H,math.ceil(y+h-.5))
    return float(p[top:bottom,left:right].max())

oldmax=np.array([boxmax(p,a) for p,a in zip(original,chosen)])
assert np.all(oldmax[18:160]<.5)
windows=[]
for new,folder in [(False,base/'windows'),(True,off/'windows')]:
    for p in sorted(folder.glob('window_*.npz')):
        with np.load(p,allow_pickle=False) as z:
            start,end=int(z['start']),int(z['end']); probs=z['probabilities']
        assert probs.shape==(40,512,640) and end-start==40
        windows.append({'start':start,'end':end,'new':new,'probs':probs,'path':str(p)})
windows.sort(key=lambda w:w['start'])
assert [w['start'] for w in windows]==list(range(0,121,12))
contextmax=np.full((11,160),np.nan)
for wi,ww in enumerate(windows):
    for frame in range(ww['start'],ww['end']): contextmax[wi,frame]=boxmax(ww['probs'][frame-ww['start']],chosen[frame])
count=np.isfinite(contextmax).sum(axis=0)
hasnew=np.any(np.isfinite(contextmax[[i for i,w in enumerate(windows) if w['new']]]),axis=0)
anymax=np.nanmax(contextmax,axis=0)
eligible=np.arange(18,160)
recovered=[int(i) for i in eligible if anymax[i]>=.5]
assert int((count[eligible]>=2).sum())==130
assert int(hasnew[eligible].sum())==118
assert [int(i) for i in eligible if count[i]==1]==list(range(148,160))
match_details=[]
for frame in recovered:
    for wi,ww in enumerate(windows):
        if not ww['start']<=frame<ww['end']:continue
        p=ww['probs'][frame-ww['start']];gts=allgt[frame]
        comps=flood_components(p); matches,edges=assignment(gts,comps)
        gi=next(i for i,a in enumerate(gts) if a['track_id']==track)
        matched=gi in matches
        c=comps[matches[gi]] if matched else None
        match_details.append({'source_frame':frame,'start':ww['start'],'slot':frame-ww['start'],'new_context':ww['new'],
                              'boxmax':float(contextmax[wi,frame]),'matched':matched,
                              'peak_xy':([c['x'],c['y']] if c else None),
                              'peak_probability':(c['score'] if c else None),
                              'eligible_component_count':len(edges[gi]),
                              'chosen_component_eligible_gt_count':(sum(matches[gi] in e for e in edges) if matched else None)})
for frame in recovered:
    assert any(r['matched'] and r['new_context'] for r in match_details if r['source_frame']==frame)
reported=json.loads((q/'window_offset_experiment/analysis'/seq/'SUMMARY.json').read_text())
rep=next(s for s in reported['stats'] if s['threshold']==.5)
assert recovered==rep['recovered_source_frames']
assert len(recovered)==rep['previously_observed_source_18_to_159']['oracle_recovery_requires_new_context']
diff=''.join(difflib.unified_diff((q/'infer_with_windows.py').read_text().splitlines(True),(q/'window_offset_experiment/infer_offsets.py').read_text().splitlines(True),fromfile='infer_with_windows.py',tofile='infer_offsets.py'))
result={'status':'PASS','sequence':seq,'selected_track_id':track,'threshold':.5,
        'old_interval':[18,159],'interval_frames':142,'all_original_boxmax_below_threshold':True,
        'frames_with_multiple_real_contexts':int((count[eligible]>=2).sum()),
        'frames_with_new_context':int(hasnew[eligible].sum()),
        'single_context_frames':[int(i) for i in eligible if count[i]==1],
        'boxmax_crossing_frames':recovered,'boxmax_crossing_count':len(recovered),
        'point_matching_verified_in_new_context_for_every_crossing_frame':True,
        'remaining_low_response_all_142_frames':int((anymax[eligible]<.5).sum()),
        'remaining_low_response_multiple_context_frames':int(((anymax[eligible]<.5)&(count[eligible]>=2)).sum()),
        'remaining_low_response_new_context_frames':int(((anymax[eligible]<.5)&hasnew[eligible]).sum()),
        'matching_details':match_details,
        'source_and_runtime_metadata_equal_fields':fields,
        'inference_adapter_diff':diff,
        'source_review':'Diff only adds specified window starts, guards bounds, permits uncovered diagnostic archive frames and labels purpose/coverage. Model execution, weights, normalization and temporal length are unchanged.',
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'independent_helpers_sha256':hashlib.sha256((q/'audit/independent_component_helpers.py').read_bytes()).hexdigest(),
        'limits':['This is a GT-guided any-context diagnostic envelope, not a deployable method result.',
                  'Changing window start jointly changes context content, temporal position and boundary padding.',
                  'Frames148-159 did not receive multiple contexts and cannot demonstrate persistence under rewindowing.',
                  'This spotcheck covers only selected track4621500 and the requested0.5 claim.']}
(q/'audit/OFFSET_SPOTCHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['matching_details','inference_adapter_diff']},indent=2))
