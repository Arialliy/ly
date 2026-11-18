"""Audit the completed controlled run and its evaluation artifacts."""
import json
import math
import time
import torch
from runtime import E,SEQS,CP,sha,load_model,frozen_state,bn_state,tensor_hash,save_json


def main():
    freeze=json.loads((E/'PROTOCOL_FREEZE.json').read_text());protocol_hash=sha(E/'PROTOCOL_FREEZE.json')
    assert all(sha(E/name)==expected for name,expected in freeze['files'].items())
    reference,_=load_model(device='cpu');refhash=tensor_hash(reference.state_dict())
    frozen_hash=tensor_hash(frozen_state(reference));bn_hash=tensor_hash(bn_state(reference))
    rows_by_condition={};runs={};predictions={};all_pairs={}
    for condition in ('A','B','C'):
        run_dir=E/'training'/condition/'run';run=json.loads((run_dir/'RUN.json').read_text())
        assert run['status']=='complete' and run['final_step']==2048 and run['seed']==46
        assert run['protocol_sha256']==protocol_hash and run['initial_checkpoint_sha256']==sha(CP)
        assert run['initial_state_sha256']==refhash
        assert run['frozen_frontend_unchanged'] and run['bn_unchanged']
        assert run['frozen_state_sha256']==frozen_hash and run['bn_initial_sha256']==bn_hash
        assert run['trainable_parameters']==14593 and run['new_parameters']==0
        assert run['settings']==freeze['settings']
        assert [s['step'] for s in run['snapshots']]==[0,128,512,1024,2048]
        for snapshot in run['snapshots']:
            assert sha(run_dir/snapshot['path'])==snapshot['sha256']
        initial,_=load_model(run_dir/'snapshot_000.pth',device='cpu')
        assert tensor_hash(initial.state_dict())==refhash
        final,_=load_model(run_dir/'snapshot_2048.pth',device='cpu')
        assert tensor_hash(frozen_state(final))==frozen_hash and tensor_hash(bn_state(final))==bn_hash
        assert tensor_hash(final.state_dict())==run['final_state_sha256']
        rows=[json.loads(line) for line in (run_dir/'TRAIN.jsonl').read_text().splitlines()]
        assert len(rows)==2048
        for index,row in enumerate(rows):
            assert row['step']==index+1 and row['plan_index']==index%512 and row['condition']==condition
            assert math.isfinite(row['loss']) and math.isfinite(row['grad_norm'])
            assert all(math.isfinite(value) for value in row['loss_terms'].values())
            assert len(row['loss_statistics']['det']['loss_per_window'])==4
        rows_by_condition[condition]=rows
        if condition!='A':
            all_pairs[condition]=sum(row['loss_statistics'][condition]['pairs_used'] for row in rows)
            assert all_pairs[condition]==30476*4
        runs[condition]=dict(status=run['status'],steps=len(rows),
            elapsed_seconds=run['elapsed_seconds'],peak_memory_bytes=run['max_cuda_memory_allocated_bytes'],
            final_checkpoint_sha256=sha(run_dir/'snapshot_2048.pth'))
        predictions[condition]={}
        for sequence in SEQS:
            folder=E/'predictions'/condition/sequence
            record=json.loads((folder/'COMPLETE.json').read_text())
            assert record['status']=='COMPLETE' and record['all_model_state_unchanged_during_inference']
            assert record['bn_unchanged'] and record['frozen_frontend_unchanged']
            assert len(record['window_sha256'])==8
            assert all(sha(folder/'windows'/name)==value for name,value in record['window_sha256'].items())
            assert sha(folder/'official_fused.npz')==record['fused_sha256']
            predictions[condition][sequence]=sha(folder/'COMPLETE.json')
    for i in range(2048):
        for key in ('input_sha256','label_sha256','auxiliary_input_sha256'):
            assert len({rows_by_condition[c][i][key] for c in ('A','B','C')})==1
        assert rows_by_condition['B'][i]['loss_statistics']['B']['pairs_used']==rows_by_condition['C'][i]['loss_statistics']['C']['pairs_used']
    evaluation=json.loads((E/'evaluation/SUMMARY_COMPLETE.json').read_text())
    assert evaluation['conditions']==['release','A','B','C']
    assert evaluation['strict_small_target_frames']==5200
    assert all(sha(E/'evaluation'/name)==value for name,value in evaluation['output_sha256'].items())
    save_json(E/'VALIDATION.json',dict(status='PASS',created_unix=time.time(),protocol_sha256=protocol_hash,
        all_three_initial_models_identical_to_release=True,all_2048_detection_and_auxiliary_exposures_paired=True,
        fixed_frontend_and_bn_unchanged=True,original_forward_only_at_inference=True,
        formal_training_pairs=all_pairs,runs=runs,prediction_receipts=predictions,
        evaluation_summary_sha256=sha(E/'evaluation/SUMMARY_COMPLETE.json'),
        scope='Execution integrity; does not imply that the research hypothesis passed.'))
    print('Final execution integrity: PASS',flush=True)

if __name__=='__main__':main()
