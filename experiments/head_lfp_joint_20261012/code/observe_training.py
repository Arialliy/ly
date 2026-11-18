"""Read fixed training crops at saved nodes; never train or select development data."""
import argparse
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import train_joint as r

E=r.E
SOURCE=r.Q.parent/'official_deeppro'
REFERENCE=SOURCE/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['release','H','HL'],required=True)
    p.add_argument('--step',type=int,choices=[0,128,512,1024,2048],required=True);a=p.parse_args()
    if (a.condition=='release')!=(a.step==0):p.error('Release uses step0; candidates use saved nonzero nodes')
    dest=E/'observations'/a.condition/f'step_{a.step:04d}'
    if dest.exists() and any(dest.iterdir()):raise FileExistsError(dest)
    dest.mkdir(parents=True,exist_ok=True)
    r.deterministic_setup()
    inputs=SimpleNamespace(source=SOURCE,checkpoint=REFERENCE,dataset_root=r.Q/'stage1_goal/data',
        selection=r.Q/'stage1_goal/DATA_SELECTION.json',plan=r.Q/'stage1_goal/pilot/TRAIN_PLAN.json')
    base,plan,names,sequences,model_file=r.previous.validate_inputs(inputs)
    specification=json.loads((E/'OBSERVATION_PLAN.json').read_text())
    assert specification['plan_sha256']==r.sha(inputs.plan)
    original,_=base.load_model(SOURCE,REFERENCE)
    checkpoint=REFERENCE
    if a.condition=='release':model=original
    else:
        checkpoint=E/'training'/a.condition/'run'/f'snapshot_{a.step:03d}.pth'
        cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert cp['step']==a.step and cp['condition']==a.condition
        assert all(cp[k]==v for k,v in r.source_hashes(model_file).items())
        model=r.wrap_model(original,a.condition);model.load_state_dict(cp['model_state_dict'],strict=True)
    model=model.cuda().eval();before=r.tensor_hash(model.state_dict().items())
    match=r.load_module('observation_fixed_match',r.Q/'analysis/analyze.py')
    release_dir=E/'observations/release/step_0000'
    if a.condition=='release':baseline={};frozen_bg={}
    else:
        baseline={row['key']:row for row in csv.DictReader((release_dir/'OBJECTS.csv').open())}
        frozen_bg={row['key']:row for row in csv.DictReader((release_dir/'FRAMES.csv').open())}
    objects=[];frames=[];groups=[]
    with torch.no_grad():
        for item in specification['pairs']:
            entry=plan['entries'][item['entry_index']];j=item['pair_index'];pair=entry['pairs'][j]
            assert pair==item['pair'] and pair['sequence'] in names
            seq=sequences[pair['sequence']];batch,_,masks,_=base.build_batch(entry,sequences)
            logits=model(torch.from_numpy(batch[2*j:2*j+2].copy()).cuda())[1].cpu()
            probs=logits.sigmoid().numpy();zs=logits.numpy();cy,cx=pair['crop_yx']
            for wi,start in enumerate(pair['starts']):
                for slot in range(40):
                    absolute=start+slot;prefix=f'{seq.name}:{item["entry_index"]}:{j}:{wi}:{slot}'
                    local=[]
                    for ann in seq.annotations[absolute]:
                        ann=copy.deepcopy(ann);x,y,w,h=ann['bbox'];ann['bbox']=[x-cx,y-cy,w,h];local.append(ann)
                    gt,_=match.prepare_gt(local,128,128)
                    rows,stats,components=match.match_frame(probs[wi,slot],gt,.5)
                    bg=masks[2*j+wi,slot];negative=np.flatnonzero(bg)
                    if negative.size:
                        current_index=int(negative[np.argmax(probs[wi,slot].ravel()[negative])])
                        fixed_index=current_index if a.condition=='release' else int(frozen_bg[prefix]['fixed_bg_index'])
                        assert bg.ravel()[fixed_index]
                        bg_current=float(probs[wi,slot].ravel()[current_index]);bg_fixed=float(probs[wi,slot].ravel()[fixed_index])
                        bg_logit=float(zs[wi,slot].ravel()[fixed_index])
                    else:fixed_index=-1;bg_current=bg_fixed=bg_logit=None
                    frame={'key':prefix,'sequence':seq.name,'frame':absolute,'window_start':start,'crop_y':cy,'crop_x':cx,
                        'fixed_bg_index':fixed_index,'fixed_bg_probability':bg_fixed,'fixed_bg_logit':bg_logit,
                        'current_background_max':bg_current,'background_pixels':int(bg.sum()),
                        'unmatched_component_area':sum(c['area'] for c in components if c['unmatched_component']),**stats}
                    frames.append(frame)
                    for row,ann in zip(rows,gt):
                        x,y,w,h=ann['bbox'];core=x>=9 and y>=9 and x+w<=119 and y+h<=119
                        strict=core and max(w,h)<=16 and not ann.get('iscrowd',0) and not ann['label_overlap_track_ids']
                        x0,y0,x1,y1=ann['crop'];key=f'{prefix}:{ann["id"]}'
                        release_match=int(row['matched']) if a.condition=='release' else int(baseline[key]['matched'])
                        release_max=float(row['box_max']) if a.condition=='release' else float(baseline[key]['box_max'])
                        objects.append({'key':key,'sequence':seq.name,'frame':absolute,'window_start':start,
                            'crop_y':cy,'crop_x':cx,'core_strict_small':int(strict),'release_matched':release_match,
                            'release_box_max':release_max,'gain_vs_release':int(not release_match and row['matched']),
                            'loss_vs_release':int(release_match and not row['matched']),
                            'box_max_logit':float(zs[wi,slot,y0:y1,x0:x1].max()),
                            'fixed_background_probability':bg_fixed,
                            'margin_vs_fixed_background_logit':None if bg_logit is None else float(zs[wi,slot,y0:y1,x0:x1].max())-bg_logit,
                            **row})
            del logits,probs,zs
    assert before==r.tensor_hash(model.state_dict().items())
    write_csv(dest/'OBJECTS.csv',objects);write_csv(dest/'FRAMES.csv',frames)
    for seq in ['ALL']+names:
        all_rows=[row for row in objects if seq=='ALL' or row['sequence']==seq]
        target=[row for row in all_rows if row['core_strict_small']]
        ff=[row for row in frames if seq=='ALL' or row['sequence']==seq]
        low=[row for row in target if row['release_box_max']<.5]
        groups.append({'condition':a.condition,'step':a.step,'sequence':seq,
            'core_strict_small_window_target_frames':len(target),'matched':sum(row['matched'] for row in target),
            'gain_vs_release':sum(row['gain_vs_release'] for row in target),'loss_vs_release':sum(row['loss_vs_release'] for row in target),
            'release_low_window_objects':len(low),'release_low_recovered':sum(row['matched'] for row in low),
            'unique_core_strict_target_frames':len({(row['sequence'],row['frame'],row['track_id']) for row in target}),
            'tracks':len({(row['sequence'],row['track_id']) for row in target}),
            'recovered_tracks':len({(row['sequence'],row['track_id']) for row in target if row['gain_vs_release']}),
            'unmatched_components':sum(row['unmatched_components'] for row in ff),
            'outside_pixels':sum(row['positive_pixels_outside_boxes'] for row in ff),
            'unmatched_component_area':sum(row['unmatched_component_area'] for row in ff)})
    write_csv(dest/'SUMMARY.csv',groups)
    r.save_json(dest/'COMPLETE.json',{'status':'COMPLETE','condition':a.condition,'step':a.step,
        'checkpoint_sha256':r.sha(checkpoint),'observation_plan_sha256':r.sha(E/'OBSERVATION_PLAN.json'),
        'model_state_unchanged':True,'training_updates':0,'scope':specification['scope'],
        'frozen_sampling':True,'windows':16,'sequence_count':8,'object_instance_count':len(objects),
        'csv_sha256':{p.name:r.sha(p) for p in dest.glob('*.csv')}})
    print(json.dumps(groups[0]),flush=True)

if __name__=='__main__':main()
