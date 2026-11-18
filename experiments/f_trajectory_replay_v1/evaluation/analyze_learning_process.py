"""Post-hoc descriptive training process only; never reads development outputs.

Requires both registered 512-step runs complete. This script writes a separate
analysis artifact and does not modify frozen code, plans, or training records.
"""
from pathlib import Path
import hashlib
import json
import math
import statistics

E=Path(__file__).resolve().parents[1]
STEPS=(0,128,256,512)
CONDITIONS=('ORIGINAL','REPLAY')
METRICS=('target_logit_mean','background_logit_mean','logit_margin_mean',
         'probability_margin_mean','target_logit_above_zero_fraction','background_logit_above_zero_fraction')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(v)for v in value.values())
    if isinstance(value,list):return all(map(finite,value))
    return True
def aggregate(values):
    values=list(values)
    return {'n':len(values),'mean':statistics.mean(values),'median':statistics.median(values),
        'min':min(values),'max':max(values),'positive_count':sum(x>0 for x in values),
        'negative_count':sum(x<0 for x in values),'zero_count':sum(x==0 for x in values)}
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    plan=read(E/'TRAIN_DIAGNOSTIC_PLAN.json');indices=plan['original_entry_indices']
    assert len(indices)==18 and len(set(indices))==18
    expected_order=read(E/'SCHEDULE.json')['continuation_order']
    sources=[Path(__file__),E/'TRAIN_DIAGNOSTIC_PLAN.json',E/'PROTOCOL_FREEZE.json',E/'SCHEDULE.json']
    raw={};runs={};conditions={}
    for condition in CONDITIONS:
        root=E/'training'/condition/'run';run=read(root/'RUN.json');runs[condition]=run
        assert run['status']=='complete'and run['final_step']==2816 and run['additional_steps']==512
        assert run['frozen_frontend_unchanged'] and run['bn_unchanged']
        logs=[json.loads(x)for x in (root/'TRAIN.jsonl').read_text().splitlines()]
        assert len(logs)==512 and [x['additional_step']for x in logs]==list(range(1,513))
        assert [x['step']for x in logs]==list(range(2305,2817))
        assert [x['original_plan_index']for x in logs]==expected_order
        assert all(x['condition']==condition for x in logs)and finite(logs)
        assert all(x['grad_norm']>0 for x in logs)
        sources += [root/'RUN.json',root/'TRAIN.jsonl']
        raw[condition]={};snapshots={}
        for step in STEPS:
            path=root/f'TRAIN_DIAGNOSTICS_{step:03d}.json';d=read(path);sources.append(path)
            assert d['condition']==condition and d['additional_step']==step and d['uses_development']is False
            assert d['plan_sha256']==sha(E/'TRAIN_DIAGNOSTIC_PLAN.json')
            assert [x['original_entry_index']for x in d['rows']]==indices and finite(d)
            raw[condition][step]=d['rows']
            assert all(x['readout']['observations_with_multiplicity']>0 for x in d['rows'])
            weights=[x['readout']['observations_with_multiplicity']for x in d['rows']]
            if step:assert weights==[x['readout']['observations_with_multiplicity']for x in raw[condition][0]]
            snapshots[str(step)]={
                'batch_count':len(indices),'segment_observations_with_multiplicity':sum(weights),
                'loss_mean_equal_batch':statistics.mean(x['loss']for x in d['rows']),
                'loss_terms_mean_equal_batch':{k:statistics.mean(x['loss_terms'][k]for x in d['rows'])for k in ('det','keep','C','frame')},
                'readout_equal_batch':{k:aggregate(x['readout'][k]for x in d['rows'])for k in METRICS},
                'readout_segment_exposure_weighted_mean':{k:sum(x['readout'][k]*w for x,w in zip(d['rows'],weights))/sum(weights)for k in METRICS}}
        for step in STEPS[1:]:
            snapshots[str(step)]['change_from_initial_equal_batch']={
                'loss':aggregate(x['loss']-b['loss']for x,b in zip(raw[condition][step],raw[condition][0])),
                'loss_terms':{k:aggregate(x['loss_terms'][k]-b['loss_terms'][k]for x,b in zip(raw[condition][step],raw[condition][0]))for k in ('det','keep','C','frame')},
                'readout':{k:aggregate(x['readout'][k]-b['readout'][k]for x,b in zip(raw[condition][step],raw[condition][0]))for k in METRICS}}
        windows=[]
        for low,high in ((1,128),(129,256),(257,512)):
            chunk=logs[low-1:high]
            windows.append({'additional_steps':[low,high],'count':len(chunk),
                'loss':aggregate(x['loss']for x in chunk),'grad_norm':aggregate(x['grad_norm']for x in chunk),
                'note':'Different scheduled input windows across steps and conditions; these averages are descriptive, not a matched-window causal comparison.'})
        conditions[condition]={'training_integrity':{'logged_steps':len(logs),'all_logged_numbers_finite':True,
            'all_steps_nonzero_gradient':True,'gradient_norm':aggregate(x['grad_norm']for x in logs),
            'frozen_frontend_unchanged':run['frozen_frontend_unchanged'],'bn_unchanged':run['bn_unchanged'],
            'changed_trainable_tensor_count':sum(v>0 for v in run['parameter_changes'].values()),
            'final_parameter_delta_L2':math.sqrt(sum(v*v for v in run['parameter_changes'].values())),
            'actual_adam_step_range':[2304,2816]},'fixed_diagnostic_checkpoints':snapshots,'unmatched_training_log_intervals':windows}
    for k in ('initial_checkpoint_sha256','initial_optimizer_sha256','initial_state_sha256','initial_rng_sha256','settings','optimizer','trainable_parameter_names'):
        assert runs['ORIGINAL'][k]==runs['REPLAY'][k],k
    assert raw['ORIGINAL'][0]==raw['REPLAY'][0]
    paired={}
    for step in STEPS:
        a=raw['ORIGINAL'][step];b=raw['REPLAY'][step]
        paired[str(step)]={'loss_delta_replay_minus_original':aggregate(y['loss']-x['loss']for x,y in zip(a,b)),
            'loss_terms_delta_replay_minus_original':{k:aggregate(y['loss_terms'][k]-x['loss_terms'][k]for x,y in zip(a,b))for k in ('det','keep','C','frame')},
            'readout_delta_replay_minus_original':{k:aggregate(y['readout'][k]-x['readout'][k]for x,y in zip(a,b))for k in METRICS},
            'per_batch':[{'original_entry_index':x['original_entry_index'],'loss_delta':y['loss']-x['loss'],
                'readout_delta':{k:y['readout'][k]-x['readout'][k]for k in METRICS}}for x,y in zip(a,b)]}
    result={'status':'PASS','analysis_scope':'training logs and registered fixed eighteen training batches only',
        'uses_development':False,'uses_confirmation':False,'new_optimizer_updates':0,'changes_experiment':False,
        'interpretation':['Fixed diagnostic readout uses all valid FRAME segments in the eighteen batches, not only the thirty-nine identified weak tracks.',
            'Objects recur in overlapping windows/segments; segment exposures are not independent object counts.',
            'Target/background logits are original bounding-region readouts; neither logits nor nonzero gradients establish component-level recovery.',
            'Checkpoint 512 remains the registered endpoint. No checkpoint or training choice is selected from these descriptive summaries.',
            'Raw gradient/loss log finiteness comes from recorded scalars plus trainer completion assertions; no new backward or optimizer step is performed.'],
        'initial_state_model_optimizer_rng_equal':True,'initial_fixed_readout_exactly_equal':True,
        'diagnostic_indices':indices,'conditions':conditions,'fixed_batch_paired_differences':paired,
        'source_sha256':{str(p):sha(p)for p in sources}}
    save(E/'evaluation/TRAIN_LEARNING_PROCESS.json',result)
    print(json.dumps({'status':'PASS','integrity':{k:v['training_integrity']for k,v in conditions.items()},
        'endpoint_paired_mean_delta':{k:v['mean']for k,v in paired['512']['readout_delta_replay_minus_original'].items()},
        'endpoint_loss_delta_mean':paired['512']['loss_delta_replay_minus_original']['mean']},ensure_ascii=False))


if __name__=='__main__':main()
