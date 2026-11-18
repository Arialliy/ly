#!/usr/bin/env python3
"""Independent raw annotation/cache audit and discrete peak-inheritance check."""
from pathlib import Path
import collections,csv,hashlib,itertools,json,time
import numpy as np
OUT=Path(__file__).resolve().parent;Q=OUT.parent.parent
SEQS=['DJI_0026_2','DJI_0067_2'];STARTS=list(range(0,121,12))
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def independent_components(a,h=2,w=3):
    foreground={i for i,x in enumerate(a) if x>=1};answer=[]
    while foreground:
        seed=min(foreground);foreground.remove(seed);seen={seed};queue=[seed]
        while queue:
            v=queue.pop();y,x=divmod(v,w)
            for dy,dx in itertools.product((-1,0,1),repeat=2):
                ny,nx=y+dy,x+dx
                n=ny*w+nx
                if 0<=ny<h and 0<=nx<w and n in foreground:foreground.remove(n);seen.add(n);queue.append(n)
        answer.append(min(seen,key=lambda i:(-a[i],i)))
    return set(answer)
def main():
    tic=time.time();matrix=read(OUT/'MATCH_MATRIX.csv');response=read(OUT/'BOX_RESPONSE_MATRIX.csv')
    bykey={(r['sequence'],int(r['frame']),int(r['annotation_id'])):i for i,r in enumerate(matrix)}
    assert len(bykey)==5200
    hashes={};npz_rows=0;cells=0;strict_count=0;all_valid_count=0
    for seq in SEQS:
        p=Q/'stage1_goal/data'/seq/'annotations.json';hashes[str(p)]=digest(p);data=json.loads(p.read_text())
        ims=data['images'];lookup={im['id']:i for i,im in enumerate(ims)};anns=collections.defaultdict(list)
        for a in data['annotations']:anns[lookup[a['image_id']]].append(a)
        valid=[];smallkeys=set()
        for t,im in enumerate(ims):
            aa=[]
            for a in sorted(anns[t],key=lambda a:(a['track_id'],a['id'])):
                x,y,w,h=a['bbox']
                if a.get('ignore',0) or w<=0 or h<=0 or min(im['width'],x+w)<=max(0,x) or min(im['height'],y+h)<=max(0,y):continue
                aa.append(a)
            valid.append(aa);all_valid_count+=len(aa)
            for a in aa:
                x,y,w,h=a['bbox']
                overlaps=any(b['id']!=a['id'] and min(x+w,b['bbox'][0]+b['bbox'][2])>max(x,b['bbox'][0]) and min(y+h,b['bbox'][1]+b['bbox'][3])>max(y,b['bbox'][1]) for b in aa)
                strict=max(w,h)<=16 and x>=0 and y>=0 and x+w<=im['width'] and y+h<=im['height'] and not overlaps
                k=(seq,t,a['id'])
                assert strict==(k in bykey)
                if strict:
                    strict_count+=1;smallkeys.add(k);r=matrix[bykey[k]]
                    assert int(r['track_id'])==a['track_id']
                    assert [float(r[z]) for z in ['bbox_x','bbox_y','bbox_w','bbox_h']]==a['bbox']
        for s in STARTS:
            p=Q/'problem_validity_20261011/window_scores'/seq/f'window_{s:03d}_matches.npz';hashes[str(p)]=digest(p)
            a=np.load(p);assert a['thresholds'].tolist()==[.5]
            kk=[(seq,t,b['id']) for t in range(s,s+40) for b in valid[t]]
            assert len(kk)==a['matched'].shape[1];npz_rows+=len(kk)
            assert a['strict_small'].tolist()==[k in smallkeys for k in kk]
            for n,k in enumerate(kk):
                if k not in smallkeys:continue
                i=bykey[k];assert int(matrix[i][f'w{s:03d}'])==int(a['matched'][0,n])
                assert float(response[i][f'w{s:03d}'])==float(a['box_max'][n]);cells+=1
    assert strict_count==5200 and all_valid_count==16297 and cells==14343
    # Exhaustively check two ternary-valued 2x3 maps, including ties and component merges.
    maps=list(itertools.product(range(3),repeat=6));peaks=[independent_components(a) for a in maps]
    checked=0;multiple_component_cases=0
    for i,a in enumerate(maps):
        for j,b in enumerate(maps):
            f=tuple(max(x,y) for x,y in zip(a,b));fp=independent_components(f)
            assert fp<=peaks[i]|peaks[j]
            checked+=1;multiple_component_cases+=int(len(peaks[i])>1 or len(peaks[j])>1)
    result={'status':'PASS','strict_small_annotation_count':strict_count,'all_valid_annotation_count':all_valid_count,'strict_small_nonoverlap_with_all_valid':True,'source_window_score_npz_count':22,'source_window_gt_rows':npz_rows,'applicable_matrix_cells_directly_checked':cells,'synthetic_peak_inheritance':{'two_map_2x3_ternary_cases':checked,'cases_with_multiple_source_components':multiple_component_cases,'violations':0,'independent_BFS_and_lexicographic_peak_implementation':True,'scope':'computational sanity check; proof in PEAK_UPPER_BOUND.md'},'source_hashes':hashes,'elapsed_seconds':time.time()-tic}
    (OUT/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))
if __name__=='__main__':main()
