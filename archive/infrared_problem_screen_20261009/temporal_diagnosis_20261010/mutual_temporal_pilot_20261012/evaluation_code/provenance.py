"""Frozen geometry-only query sampling and descriptive coarse-box source attribution.
Labels are used after sparse module probabilities have been computed, never as model input.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

SEQS=['DJI_0026_2','DJI_0067_2']
STARTS=[0,24,36,60,72,96,108,120]
CATEGORIES=['same_track_region','other_target_region','outside_all_annotated_boxes','uncertain']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def freeze_queries(base):
    q=base.parent;cohort=q/'low_response_information_20261011/COHORTS.csv'
    with cohort.open() as f:raw=list(csv.DictReader(f))
    queries=[]
    for row in raw:
        if not (int(row['low_all_05']) or int(row['both_fixed_success'])):continue
        x=math.floor(float(row['bbox_x'])+float(row['bbox_w'])/2)
        y=math.floor(float(row['bbox_y'])+float(row['bbox_h'])/2)
        assert 0<=x<640 and 0<=y<512
        queries.append(dict(sequence=row['sequence'],frame=int(row['frame']),source_frame=int(row['source_frame']),
                            track_id=int(row['track_id']),annotation_id=int(row['annotation_id']),x=x,y=y,
                            low_all_05=int(row['low_all_05']),extreme_all_001=int(row['extreme_all_001']),
                            both_fixed_success=int(row['both_fixed_success'])))
    assert len(queries)==4988
    source={str(cohort):sha(cohort)}
    for seq in SEQS:
        path=q/'stage1_goal/data'/seq/'annotations.json';source[str(path)]=sha(path)
    result=dict(status='FROZEN',queries=queries,query_count=len(queries),radius=2,starts=STARTS,
                selection='Union of frozen 4342 low and 646 both-fixed-success target frames, no new-output selection.',
                query_pixel='floor(original float bbox center), interpreted at its pixel center (x+0.5,y+0.5) for source-box classification.',
                classification='All retained annotations used. Query overlap/ignore, missing same-track reference, source-frame gap, candidate overlap/ignored/out-of-image => uncertain. Otherwise unique same track, unique other target, or outside all annotated boxes.',
                categories=CATEGORIES,scope='Coarse-box probability mass, not pixel correspondence ground truth, attention causality, or mechanism proof.',
                source_sha256=source,script_sha256=sha(__file__))
    path=base/'evaluation_preflight/PROVENANCE_QUERIES.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['queries']==result['queries'] and old['source_sha256']==source
    else:save(path,result)
    return path


def contains(ann,x,y):
    bx,by,w,h=map(float,ann['bbox'])
    return w>0 and h>0 and bx<=x+.5<bx+w and by<=y+.5<by+h


class Recorder:
    def __init__(self,base,sequence,radius=2):
        self.base=Path(base);self.sequence=sequence;self.radius=radius
        self.freeze_path=self.base/'evaluation_preflight/PROVENANCE_QUERIES.json'
        frozen=json.loads(self.freeze_path.read_text())
        assert frozen['radius']==radius and frozen['starts']==STARTS
        for path,digest in frozen['source_sha256'].items():assert sha(path)==digest
        self.queries=[r for r in frozen['queries'] if r['sequence']==sequence]
        self.ann_path=self.base.parent/'stage1_goal/data'/sequence/'annotations.json'
        data=json.loads(self.ann_path.read_text())
        # Annotation image order is chronological; preserve original source frame IDs.
        images=sorted(data['images'],key=lambda r:int(r['frame_id']))
        self.images={i:r for i,r in enumerate(images)}
        image_index={r['id']:i for i,r in enumerate(images)}
        self.annotations={i:[] for i in range(len(images))}
        for a in data['annotations']:self.annotations[image_index[a['image_id']]].append(a)
        self.rows=[];self.seen=set()

    def select(self,start,y0,y1,x0,x1,py0,px0):
        rows=[r for r in self.queries if start<=r['frame']<start+40 and y0<=r['y']<y1 and x0<=r['x']<x1]
        qs=[dict(batch=0,slot=r['frame']-start,y=r['y']-py0,x=r['x']-px0) for r in rows]
        return rows,qs

    def classify(self,q,ref_frame,x,y):
        if not (0<=x<640 and 0<=y<512):return 'uncertain'
        current=[a for a in self.annotations[q['frame']] if contains(a,q['x'],q['y'])]
        if len(current)!=1 or current[0].get('ignore',0) or current[0].get('iscrowd',0):return 'uncertain'
        if int(current[0]['track_id'])!=q['track_id']:return 'uncertain'
        refs=self.annotations[ref_frame]
        same=[a for a in refs if int(a['track_id'])==q['track_id'] and not a.get('ignore',0) and not a.get('iscrowd',0)]
        if len(same)!=1:return 'uncertain'
        # source frame continuity, not merely adjacent cropped frame index.
        if int(self.images[ref_frame]['frame_id'])-int(self.images[q['frame']]['frame_id'])!=ref_frame-q['frame']:return 'uncertain'
        hits=[a for a in refs if contains(a,x,y)]
        if len(hits)>1 or any(a.get('ignore',0) or a.get('iscrowd',0) for a in hits):return 'uncertain'
        if not hits:return 'outside_all_annotated_boxes'
        return 'same_track_region' if int(hits[0]['track_id'])==q['track_id'] else 'other_target_region'

    def add(self,queries,start,results):
        assert len(queries)==len(results)
        for qi,(q,result) in enumerate(zip(queries,results)):
            assert result.get('query_index',qi)==qi
            key=(start,q['frame'],q['annotation_id'])
            assert key not in self.seen,key
            self.seen.add(key)
            for ref in result['references']:
                ref_frame=start+int(ref['ref_slot']);assert abs(ref_frame-q['frame'])==1
                assert start<=ref_frame<start+40
                row=dict(**q,window_start=start,reference_frame=ref_frame,
                         reference_offset=ref_frame-q['frame'],null_probability=float(ref['null_probability']))
                for prefix in ['forward','mutual']:
                    for label in CATEGORIES:row[prefix+'_'+label]=0.
                valid_count=0
                for offset,valid,p,w in zip(ref['offsets'],ref['valid'],ref['forward'],ref['mutual']):
                    assert math.isfinite(p) and math.isfinite(w) and p>=0 and w>=0
                    if not valid:
                        assert p==0 and w==0
                        continue
                    valid_count+=1;dy,dx=offset
                    label=self.classify(q,ref_frame,q['x']+dx,q['y']+dy)
                    row['forward_'+label]+=float(p);row['mutual_'+label]+=float(w)
                row['valid_candidates']=valid_count
                row['forward_nonnull_mass']=sum(row['forward_'+c] for c in CATEGORIES)
                row['mutual_nonnull_mass']=sum(row['mutual_'+c] for c in CATEGORIES)
                assert abs(row['forward_nonnull_mass']+row['null_probability']-1)<2e-5
                assert row['mutual_nonnull_mass']<=row['forward_nonnull_mass']+1e-6
                self.rows.append(row)

    def finish(self,output):
        output=Path(output);expected={(s,q['frame'],q['annotation_id']) for q in self.queries for s in STARTS if s<=q['frame']<s+40}
        assert self.seen==expected,(len(self.seen),len(expected),list(expected-self.seen)[:3])
        self.rows.sort(key=lambda r:(r['window_start'],r['frame'],r['annotation_id'],r['reference_frame']))
        path=output/'PROVENANCE_ROWS.csv'
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
        result=dict(status='COMPLETE',sequence=self.sequence,query_target_frames=len(self.queries),
                    query_window_exposures=len(self.seen),query_reference_rows=len(self.rows),
                    rows_path=str(path),rows_sha256=sha(path),query_freeze_path=str(self.freeze_path),query_freeze_sha256=sha(self.freeze_path),
                    annotations_sha256=sha(self.ann_path),script_sha256=sha(__file__),
                    scope='Descriptive coarse-box source mass for E2 geometry-fixed queries; no identity supervision, precise correspondence or causal credit.')
        save(output/'PROVENANCE_COMPLETE.json',result)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    print(freeze_queries(args.base))
