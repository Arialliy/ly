"""Check reference arrays against the old immutable scoring caches, without rerun."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;L=O.parent;Q=L.parent.parent;P=Q/'stage1_goal'
report=json.loads((O/'CONTROL_AUDIT.json').read_text());assert report['status']=='PASS'
with (L/'METRICS.csv').open() as f:metrics=list(csv.DictReader(f))
refs={};sources={}
for seq in ['DJI_0026_2','DJI_0067_2']:
    current=np.load(L/f'{seq}_scores.npz')
    for pi,plan in enumerate(['O','S']):
        path=P/'evaluation/official'/seq/'official_fused_matches.npz' if plan=='O' else Q/'problem_validity_20261011/simple_controls/symmetric5_probmax'/seq/'fused_matches.npz'
        old=np.load(path);ti=list(old['thresholds']).index(.5)
        assert np.array_equal(old['matched'][ti],current['reference_matched'][pi])
        assert np.array_equal(old['frame_counts'][ti],current['reference_frame_counts'][pi])
        refs[seq,plan]=old['frame_counts'][ti].sum(0)
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
for row in metrics:
    seqs=['DJI_0026_2','DJI_0067_2'] if row['group']=='pooled' else [row['group']]
    totals=sum((refs[seq,row['arrangement']] for seq in seqs),np.zeros(6,np.int64))
    assert int(row['reference_U_same_arrangement'])==int(totals[3])
    assert int(row['reference_A_same_arrangement'])==int(totals[5])
plan=json.loads((P/'pilot/TRAIN_PLAN.json').read_text())
assert not(set(plan['train_sequences'])&{'DJI_0026_2','DJI_0067_2'})
report['checks']['all_reference_match_and_frame_arrays_equal_old_original_caches']=True
report['checks']['all_reference_U_and_A_columns_equal_original_caches']=True
report['checks']['evaluation_clips_absent_from_training_sequence_list']=True
report['original_reference_sha256']=sources
report['source_sha256']['audit/verify_reference_sources.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(O/'CONTROL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','reference_arrays':4,'evaluation_training_overlap':0}))
