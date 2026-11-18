"""Report scoped diagnostics; bbox response is never called detection recall."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

p = Path(__file__).resolve().parent
result = {'seed':46,'new_method_tested':False,'training_run':False,'probes':{}}
ann = json.loads((p/'public_subset_vt/annotations.json').read_text())
with np.load(p/'vt_small_seed46.npz',allow_pickle=False) as z:
    scores=z['probabilities']; meta=json.loads(str(z['metadata_json']))
assert meta['seed']==46 and np.isfinite(scores).all()
image_to_index={im['id']:i for i,im in enumerate(ann['images'])}
tracks={}; rows=[]
for a in ann['annotations']:
    i=image_to_index[a['image_id']]
    x,y,w,h=a['bbox']
    roi=scores[i,max(0,int(np.floor(y))):min(scores.shape[1],int(np.ceil(y+h))),
               max(0,int(np.floor(x))):min(scores.shape[2],int(np.ceil(x+w)))]
    row={'frame':int(ann['images'][i]['frame_id']),'track_id':int(a['track_id']),
         'category_id':int(a['category_id']),'ignore':int(a.get('ignore',0)),
         'width':w,'height':h,'center_x':x+w/2,'center_y':y+h/2,
         'box_max_probability':float(roi.max()),'box_mean_probability':float(roi.mean())}
    rows.append(row);tracks.setdefault(a['track_id'],[]).append(row)
track_summary=[]
for tid,rr in sorted(tracks.items()):
    rr.sort(key=lambda r:r['frame'])
    centers=np.array([[r['center_x'],r['center_y']] for r in rr])
    steps=np.linalg.norm(np.diff(centers,axis=0),axis=1)
    mx=np.array([r['box_max_probability'] for r in rr])
    track_summary.append({'track_id':int(tid),'frames':len(rr),
                          'width_min_max':[min(r['width'] for r in rr),max(r['width'] for r in rr)],
                          'height_min_max':[min(r['height'] for r in rr),max(r['height'] for r in rr)],
                          'center_step_median':float(np.median(steps)),
                          'center_step_p90':float(np.quantile(steps,.9)),
                          'box_max_probability_min_median_max':[float(mx.min()),float(np.median(mx)),float(mx.max())],
                          'frames_with_any_probability_above_0_5_inside_box':int((mx>.5).sum())})
with (p/'vt_small_seed46_frame_responses.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result['probes']['real_vt']={'shape':list(scores.shape),'frames':len(scores),'annotations':len(rows),
                           'selected_track_id':3300000,'tracks':track_summary,
                           'scope':'real extra development subset; not official benchmark or independent-scene evaluation',
                           'metric':'bbox probability proxy; not Pd, recall, or AP',
                           'tiling_check':meta['tiling_check']}
Image.fromarray(np.uint8(np.clip(scores[0],0,1)*255)).save(p/'vt_small_seed46_probability_frame0.png')
del scores
with np.load(p/'nudt_native_seed46.npz',allow_pickle=False) as z:
    scores=z['probabilities'];meta=json.loads(str(z['metadata_json']))
assert meta['seed']==46 and np.isfinite(scores).all()
data=p/'public_subset_nudt/NUDT-MIRSDT/Sequence85'
masks=np.stack([np.asarray(Image.open(f)) for f in sorted((data/'masks').glob('*.png'))])>0
assert masks.shape==scores.shape
binary=scores>.5
intersection=int(np.logical_and(binary,masks).sum());union=int(np.logical_or(binary,masks).sum())
native={'shape':list(scores.shape),'threshold':.5,'mask_intersection':intersection,'mask_union':union,
        'mask_IoU':intersection/union if union else None,
        'scope':'40 frames of one original synthesized test sequence only; not full paper benchmark',
        'tiling_check':meta['tiling_check']}
try:
    sys.path.insert(0,str(p.parent/'official_deeppro'))
    from ShootingRules import ShootingRules
    rule=ShootingRules(); counts=np.zeros(3,dtype=np.int64)
    for score, path in zip(scores,sorted((data/'masks_centroid').glob('*.png'))):
        centroid=np.asarray(Image.open(path),dtype=np.float32)/255
        counts+=np.asarray(rule(score[None],centroid[None],.5),dtype=np.int64)
    native['original_ShootingRules']={'false_positive_pixels':int(counts[0]),'matched_targets':int(counts[1]),'targets':int(counts[2]),'Pd':float(counts[1]/counts[2]) if counts[2] else None,'Fa':float(counts[0]/scores.size)}
except ImportError as error:
    native['original_ShootingRules']={'status':'not_executed','reason':str(error)}
result['probes']['native_nudt']=native
f=p/'first_val_seed46.npz'
if f.exists():
    with np.load(f,allow_pickle=False) as z:
        new=z['probabilities'];new_meta=json.loads(str(z['metadata_json']))
    with np.load(p/'first_val_inference.npz',allow_pickle=False) as z:
        old=z['probabilities']
    assert new_meta['seed']==46
    result['probes']['anti_uav_seed46_rerun']={'shape':list(new.shape),'seed':46,
        'max_absolute_probability_difference_from_seed0_engineering_run':float(np.max(np.abs(new-old))),
        'role':'seed correction on fixed pretrained eval model, not training seed comparison'}
(p/'SEED46_PROBE_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
