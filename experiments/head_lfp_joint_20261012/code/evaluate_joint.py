#!/usr/bin/env python3
"""Score existing H / HL predictions through the frozen diagnostic matcher.

Only roots/condition labels are redirected. The pixel threshold, GT preparation,
component extraction, point matching, fusion and output arrays are unchanged.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
Q = Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
OLD = Q/'top1_background_control_20261011'
PREVIOUS = Q/'tlfp_pilot_20261012'
PLANS = {'O':[0,36,72,108,120], 'S':[0,24,60,96,120]}
SEQUENCES = ['DJI_0026_2', 'DJI_0067_2']
STEPS = [512,1024,2048]
CONDITIONS = ['release','H','HL']
FIELDS = ['gt_count','matched_count','component_count','unmatched_components','positive_pixels','positive_pixels_outside_boxes']


def digest(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(path, rows, fields=None):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    assert rows or fields, path
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluation_root(step):
    return BASE/'evaluation'/f'step_{step:04d}'


def record_root(condition, step):
    return OLD/'evaluation' if condition=='release' else evaluation_root(step)


def inference_identity(condition, sequence, step):
    root=BASE/'predictions'/condition/f'step_{step:04d}'/sequence
    path=root/'COMPLETE.json'
    inference=json.loads(path.read_text())
    checkpoint=BASE/'training'/condition/'run'/f'snapshot_{step:03d}.pth'
    run_path=checkpoint.with_name('RUN.json')
    run=json.loads(run_path.read_text())
    assert inference['status']=='COMPLETE' and run['status']=='complete'
    assert inference['condition']==condition and inference['sequence']==sequence
    assert inference['checkpoint_step']==step and inference['main_checkpoint_step']==2048
    assert inference['checkpoint_is_main']==(step==2048)
    assert inference['checkpoint']==str(checkpoint)
    snapshots={int(row['step']):row for row in run['snapshots']}
    assert inference['checkpoint_sha256']==digest(checkpoint)==snapshots[step]['sha256']
    assert inference['training_run_sha256']==digest(run_path)
    assert inference['official_starts']==PLANS['O'] and inference['symmetric_starts']==PLANS['S']
    assert inference['diagnostic_starts']==sorted(set(PLANS['O']+PLANS['S']))
    assert inference['normalization']==[85.0799,47.4845] and inference['seed']==46
    assert inference['precision']=='FP32_no_AMP_no_TF32'
    assert inference['trainable_parameters']==(273 if condition=='H' else 373)
    assert inference['frozen_nonhead_unchanged'] and inference['all_model_state_unchanged_during_inference']
    assert not inference['all_original_frozen'] and not inference['model_labels_access']
    assert inference['initial_inference_model_state_sha256']==inference['final_inference_model_state_sha256']
    assert inference['inference_source_sha256']==digest(inference['inference_source_path'])
    assert inference['annotations_sha256']==digest(Q/'stage1_goal/data'/sequence/'annotations.json')
    for name, value in run.items():
        if name.endswith('_source_sha256') or name in ['spatial_reference_sha256','adapter_reference_sha256']:
            assert inference[name]==value, name
    for check in inference['tiling_checks']:
        assert check['threshold05_disagreements']==0
    return {
        'training_identity_checked':True, 'step':step, 'main_step':2048, 'is_main':step==2048,
        'identity_source_sha256':{str(path):digest(path),str(checkpoint):digest(checkpoint),str(run_path):digest(run_path)},
    }


def score(condition, sequence, step):
    identity=inference_identity(condition,sequence,step)
    frozen_path=OLD/'evaluation_code/evaluate.py'
    reference=json.loads((OLD/'evaluation/lambda0_top1'/sequence/'COMPLETE.json').read_text())
    assert reference['script_sha256']==digest(frozen_path)
    sources=json.loads((PREVIOUS/'evaluation_code/MATCHER_SOURCES.json').read_text())['source_sha256']
    for path, expected in sources.items():
        assert digest(path)==expected, ('Frozen matcher source changed',path)
    frozen=load_module(frozen_path,'joint_frozen_matcher')
    frozen.E=evaluation_root(step)
    frozen.ROOTS[condition]=BASE/'predictions'/condition/f'step_{step:04d}'
    frozen.score(condition,sequence)
    output=frozen.E/condition/sequence/'COMPLETE.json'
    receipt=json.loads(output.read_text())
    receipt.update(
        wrapper_path=str(Path(__file__).resolve()),wrapper_sha256=digest(__file__),
        reused_scorer_path=str(frozen_path),reused_scorer_sha256=digest(frozen_path),
        matcher_source_sha256=sources,
        score_artifact_sha256={name:digest(output.with_name(name)) for name in ['scores.npz','component_areas.npz','FRAME_COUNTS.csv']},
        **identity,
    )
    write_json(output,receipt)


def read_record(condition, sequence, step):
    folder=record_root(condition,step)/condition/sequence
    receipt=json.loads((folder/'COMPLETE.json').read_text())
    assert receipt['status']=='COMPLETE' and receipt['condition']==condition and receipt['sequence']==sequence
    assert receipt['threshold']==.5
    if condition!='release':
        assert receipt['training_identity_checked'] and receipt['step']==step
        assert receipt['wrapper_sha256']==digest(__file__)
        assert receipt['reused_scorer_sha256']==digest(receipt['reused_scorer_path'])
        for name,expected in receipt['score_artifact_sha256'].items():
            assert digest(folder/name)==expected
        for key in ['identity_source_sha256','matcher_source_sha256']:
            for path, expected in receipt[key].items():
                assert digest(path)==expected, (key,path)
    with np.load(folder/'scores.npz',allow_pickle=False) as archive:
        scores={key:archive[key] for key in archive.files}
    with np.load(folder/'component_areas.npz',allow_pickle=False) as archive:
        areas={key:archive[key] for key in archive.files}
    assert scores['plans'].tolist()==list(PLANS) and float(scores['threshold'])==.5
    assert scores['matched'].shape==(2,len(receipt['objects']))
    assert scores['frame_counts'].shape==(2,160,len(FIELDS))
    assert np.all(scores['coverage']>0) and np.all(scores['coverage'].sum(axis=1)==200)
    hashes={str(folder/name):digest(folder/name) for name in ['COMPLETE.json','scores.npz','component_areas.npz','FRAME_COUNTS.csv']}
    return scores,areas,receipt,hashes


def collector(step, conditions=CONDITIONS):
    """Reuse the frozen 5200-object join and cohort definitions, not old conclusions."""
    module=load_module(PREVIOUS/'evaluation_code/evaluate.py','joint_frozen_table_join')
    module.CONDITIONS=conditions
    module.NEW=[c for c in conditions if c!='release']
    module.COMPARISONS=list(conditions)
    module.EVAL=evaluation_root(step)
    module.record_root=lambda condition:record_root(condition,step)
    module.read_record=lambda condition,sequence:read_record(condition,sequence,step)
    return module,module.collect(conditions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition',choices=['H','HL'],required=True)
    parser.add_argument('--step',choices=STEPS,type=int,required=True)
    parser.add_argument('--sequence',choices=SEQUENCES)
    args=parser.parse_args()
    for sequence in [args.sequence] if args.sequence else SEQUENCES:
        score(args.condition,sequence,args.step)


if __name__=='__main__':
    main()
