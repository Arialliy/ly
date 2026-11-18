#!/usr/bin/env python3
"""Freeze A target/hard-background regions using only old release logits."""
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT.parent
LOW = Q / 'low_response_information_20261011'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def main():
    output=ROOT/'A_REGIONS.csv'
    if output.exists():raise FileExistsError('Never replace a frozen region selection')
    old=module('old_background_geometry',LOW/'analysis/build_low_response_objects.py')
    helper=module('old_point_matching_geometry',Q/'analysis/analyze.py')
    with (LOW/'LOW_RESPONSE_OBJECTS.csv').open() as stream:targets=list(csv.DictReader(stream))
    if len(targets)!=10400:raise AssertionError('Frozen target-arrangement universe changed')
    rows=[];sources={str(LOW/'LOW_RESPONSE_OBJECTS.csv'):sha(LOW/'LOW_RESPONSE_OBJECTS.csv'),
        str(LOW/'analysis/build_low_response_objects.py'):sha(LOW/'analysis/build_low_response_objects.py'),
        str(Q/'analysis/analyze.py'):sha(Q/'analysis/analyze.py')}
    counts=Counter()
    for sequence in ['DJI_0026_2','DJI_0067_2']:
        annpath=Q/'stage1_goal/data'/sequence/'annotations.json'
        if not annpath.exists():annpath=Q/'stage1_goal/data_diagnostics'/sequence/'annotations.json'
        annotation=json.loads(annpath.read_text());sources[str(annpath)]=sha(annpath)
        frame_for_image={image['id']:i for i,image in enumerate(annotation['images'])}
        annotations=defaultdict(list)
        for ann in annotation['annotations']:annotations[frame_for_image[ann['image_id']]].append(ann)
        excluded=old.exclusion_masks(annotations,helper)
        for arrangement in ['O','S']:
            logpath=LOW/'logits'/sequence/f'fused_{arrangement}_logits.npy';sources[str(logpath)]=sha(logpath)
            logits=np.load(logpath,mmap_mode='r')
            selected=[r for r in targets if r['sequence']==sequence and r['arrangement']==arrangement]
            byframe=defaultdict(list)
            for row in selected:byframe[int(row['frame'])].append(row)
            for frame,objects in sorted(byframe.items()):
                logit=np.asarray(logits[frame]);mask=excluded[frame];globalcache={}
                for target in objects:
                    x0,y0,bw,bh=(int(target[k]) for k in ['crop_x0','crop_y0','crop_w','crop_h'])
                    identity={k:target[k] for k in ['sequence','track_id','frame','source_frame','annotation_id','arrangement','low_all_05','extreme_all_001','both_fixed_success','extra_only']}
                    candidates={}
                    candidates['target']=(x0,y0,x0+bw,y0+bh,1)
                    _,blocks=old.local_tiles(logit,mask,bw,bh,x0,y0,old.expanded_neighborhood(x0,y0,bw,bh))
                    if blocks:
                        chosen=min(blocks,key=lambda b:(-b[0],b[4],b[3]))
                        candidates['hard_local']=(chosen[3],chosen[4],chosen[3]+bw,chosen[4]+bh,len(blocks))
                    else:candidates['hard_local']=None
                    if (bw,bh) not in globalcache:
                        _,scores,valid=old.global_tiles(logit,mask,bw,bh)
                        if valid.any():
                            yy,xx=np.unravel_index(int(np.argmax(np.where(valid,scores,-np.inf))),scores.shape)
                            globalcache[bw,bh]=(int(xx*bw),int(yy*bh),int((xx+1)*bw),int((yy+1)*bh),int(valid.sum()))
                        else:globalcache[bw,bh]=None
                    candidates['hard_global']=globalcache[bw,bh]
                    for kind,rect in candidates.items():
                        row={**identity,'region_id':f"{sequence}:{target['annotation_id']}:{arrangement}:{kind}",
                            'region_type':kind,'target_width':bw,'target_height':bh,'valid':int(rect is not None),
                            'x0':'','y0':'','x1':'','y1':'','valid_background_blocks':'',
                            'release_max_logit':'','release_peak_x_index':'','release_peak_y_index':'','background_insufficient':''}
                        if rect is not None:
                            xx,yy,xx1,yy1,n=rect
                            score,px,py=old.peak(logit[yy:yy1,xx:xx1],xx,yy)
                            row.update(x0=xx,y0=yy,x1=xx1,y1=yy1,valid_background_blocks=n,
                                release_max_logit=score,release_peak_x_index=int(px-.5),release_peak_y_index=int(py-.5),
                                background_insufficient=0 if kind=='target' else int(n<20))
                            if kind!='target' and mask[yy:yy1,xx:xx1].any():raise AssertionError('A selected background overlaps excluded GT')
                        counts[f'{kind}:valid{row["valid"]}']+=1;rows.append(row)
    if len(rows)!=31200 or len({r['region_id'] for r in rows})!=31200:raise AssertionError('Expected three region records per frozen target-arrangement')
    with output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    receipt={'status':'COMPLETE','rows':len(rows),'target_arrangement_rows':10400,'target_frames':5200,
        'counts':dict(counts),'regions_sha256':sha(output),'source_sha256':sources,'script_sha256':sha(__file__),
        'selection':'Original released logits only; same-size old local/global grids; all annotations plus2px excluded; maximum logit, ties y then x; invalid backgrounds retained as missing',
        'model_response_for_new_onoff_seen':False}
    (ROOT/'A_REGIONS_COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'COMPLETE','rows':len(rows),'counts':dict(counts),'regions_sha256':receipt['regions_sha256']}))


if __name__=='__main__':main()
