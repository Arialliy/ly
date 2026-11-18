import csv, hashlib, json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone

Q=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
D=Q/'low_response_information_20261011/visual'
SRC=Q/'failure_attribution_20261011/stages/ERROR_STAGE_TABLE.csv'
SEED='low-response-visual-20261011-v1'
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
rows=list(csv.DictReader(SRC.open()))
def group(r):
    if float(r['context_box_max']) < .001: return 'extreme_low'
    if float(r['context_box_max']) < .5: return 'low_not_extreme'
    if r['cohort']=='both_success': return 'both_success'
    if r['system_case']=='extra_window_only_opportunity': return 'extra_only'
    if r['system_case']=='no_match_but_box_response': return 'high_unmatched'
    return 'other'
pool=defaultdict(lambda:defaultdict(list)); tracks=defaultdict(list)
for r in rows:
    r['visual_group']=group(r)
    pool[(r['sequence'],r['visual_group'])][r['track_id']].append(r)
    tracks[(r['sequence'],r['track_id'])].append(r)
selected=[]; frame_rows=[]; bins=[]
for (seq,g),ts in sorted(pool.items()):
    if g=='other': continue
    order=sorted(ts,key=lambda t:digest(f'{SEED}|track|{seq}|{g}|{t}'))
    bins.append(dict(sequence=seq,group=g,eligible_tracks=len(ts),eligible_frames=sum(map(len,ts.values())),selected_tracks=order[:2]))
    for rank,t in enumerate(order[:2],1):
        anchor=min(ts[t],key=lambda r:digest(f'{SEED}|anchor|{seq}|{g}|{t}|{r["source_frame"]}|{r["annotation_id"]}'))
        rr=sorted(tracks[(seq,t)],key=lambda r:int(r['source_frame']))
        ai=rr.index(anchor); lo=ai; hi=ai
        while lo>0 and int(rr[lo]['source_frame'])==int(rr[lo-1]['source_frame'])+1 and int(rr[lo]['frame'])==int(rr[lo-1]['frame'])+1: lo-=1
        while hi+1<len(rr) and int(rr[hi+1]['source_frame'])==int(rr[hi]['source_frame'])+1 and int(rr[hi+1]['frame'])==int(rr[hi]['frame'])+1: hi+=1
        start=max(lo,min(ai-2,hi-4)); stop=min(hi+1,start+5)
        ss=rr[start:stop]; case=f'{seq}_{g}_{rank}_track{t}'
        selected.append(dict(case_id=case,sequence=seq,group=g,rank=rank,track_id=t,anchor_frame=anchor['frame'],anchor_source_frame=anchor['source_frame'],anchor_annotation_id=anchor['annotation_id'],category=anchor['category'],n_selected_frames=len(ss),frames=';'.join(r['frame'] for r in ss),source_frames=';'.join(r['source_frame'] for r in ss),anchor_hash=digest(f'{SEED}|anchor|{seq}|{g}|{t}|{anchor["source_frame"]}|{anchor["annotation_id"]}'),track_hash=digest(f'{SEED}|track|{seq}|{g}|{t}')))
        for r in ss: frame_rows.append(dict(case_id=case,selected_group=g,**r))
def write(name,rr):
    with (D/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
write('SELECTED_CASES.csv',selected);write('SELECTED_FRAMES.csv',frame_rows)
manifest=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,source=str(SRC),source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),rule='Five mutually exclusive frame strata from previously frozen outputs. Select up to 2 tracks per sequence/group by ascending SHA256(seed|track|sequence|group|track_id). Anchor independently chosen within stratum by ascending SHA256(seed|anchor|sequence|group|track_id|source_frame|annotation_id). Show up to 5 consecutive strict-small frames around anchor, bounded by any local/source frame gap. Neighbor frames may belong to another stratum; their actual stratum is retained. Track identities may recur in different strata; retain and disclose.',candidate_bins=bins,n_cases=len(selected),n_unique_tracks=len(set((r['sequence'],r['track_id']) for r in selected)),n_case_frames=len(frame_rows),n_unique_object_frames=len(set((r['sequence'],r['annotation_id']) for r in frame_rows)),selected_cases_sha256=hashlib.sha256((D/'SELECTED_CASES.csv').read_bytes()).hexdigest(),selected_frames_sha256=hashlib.sha256((D/'SELECTED_FRAMES.csv').read_bytes()).hexdigest(),new_logit_or_background_statistics_read=False)
(D/'SELECTION_FREEZE.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
