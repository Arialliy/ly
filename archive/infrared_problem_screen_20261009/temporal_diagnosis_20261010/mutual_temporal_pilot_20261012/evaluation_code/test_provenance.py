"""CPU checks for geometry selection and conservative coarse source classification."""
import json
from pathlib import Path
from provenance import Recorder,STARTS,sha,save
base=Path(__file__).resolve().parents[1]
checks=[]
def check(name,value):
    assert value,name
    checks.append(name)
r=Recorder.__new__(Recorder)
r.images={0:{'frame_id':10},1:{'frame_id':11}}
q={'frame':0,'x':10,'y':10,'track_id':1}
def a(t,box,**kwargs):return dict(track_id=t,bbox=box,ignore=0,iscrowd=0,**kwargs)
r.annotations={0:[a(1,[8,8,5,5])],1:[a(1,[9,8,5,5]),a(2,[20,20,5,5])]}
check('same track',r.classify(q,1,11,10)=='same_track_region')
check('other track',r.classify(q,1,21,21)=='other_target_region')
check('outside all',r.classify(q,1,30,30)=='outside_all_annotated_boxes')
check('outside image uncertain',r.classify(q,1,-1,10)=='uncertain')
r.annotations[1].append(a(3,[10,10,3,3]))
check('overlap uncertain',r.classify(q,1,11,11)=='uncertain')
r.annotations[1]=[a(2,[20,20,5,5])]
check('missing same-track uncertain even background',r.classify(q,1,30,30)=='uncertain')
r.annotations[1]=[a(1,[9,8,5,5])]
r.images[1]['frame_id']=13
check('source gap uncertain',r.classify(q,1,11,10)=='uncertain')
r.images[1]['frame_id']=11
r.annotations[0].append(a(3,[9,9,3,3]))
check('query ambiguity uncertain',r.classify(q,1,11,10)=='uncertain')
counts={}
for seq in ['DJI_0026_2','DJI_0067_2']:
    rec=Recorder(base,seq,2);seen=[]
    for start in STARTS:
        for y in range(0,512,128):
            for x in range(0,640,128):
                y1=min(y+128,512);x1=min(x+128,640);py0=max(0,y-20);px0=max(0,x-20)
                rows,qs=rec.select(start,y,y1,x,x1,py0,px0)
                for row,query in zip(rows,qs):
                    seen.append((start,row['frame'],row['annotation_id']))
                    check(seq+' query location '+str(seen[-1]),query['slot']==row['frame']-start
                          and query['y']+py0==row['y'] and query['x']+px0==row['x'])
    expected={(s,q['frame'],q['annotation_id']) for q in rec.queries for s in STARTS if s<=q['frame']<s+40}
    check(seq+' exactly once per window query',len(seen)==len(set(seen)) and set(seen)==expected)
    counts[seq]={'target_frames':len(rec.queries),'query_window_exposures':len(seen)}
check('4988 frozen query frames',sum(r['target_frames'] for r in counts.values())==4988)
result=dict(status='PASS',checks_passed=len(checks),checks=checks,counts=counts,script_sha256=sha(__file__),provenance_source_sha256=sha(base/'evaluation_code/provenance.py'))
save(base/'evaluation_preflight/PROVENANCE_CHECK.json',result)
print(json.dumps({k:result[k] for k in ['status','checks_passed','counts']}))
