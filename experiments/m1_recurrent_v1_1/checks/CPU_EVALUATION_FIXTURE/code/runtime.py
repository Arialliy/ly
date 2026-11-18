"""One-factor M1 state-update revision, paired with identical short continuation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parents[1]
V1 = ROOT / 'experiments/m1_recurrent_v1'
DIAG = ROOT / 'experiments/m1_history_effect_v1'
CHECKPOINT = V1/'training/History/run/snapshot_2048.pth'
CHECKPOINT_SHA = 'eda063496757f368d7442a7fac99a64f7b33ef0414ef99d006716d77a6dcb725'
START_STEP, STEPS, FINAL_STEP = 2048, 128, 2176
CONDITIONS = ['Continuation', 'Revision']
SEQS = ['DJI_0026_2','DJI_0067_2']
PLANS = {'O':[0,36,72,108,120],'S':[0,24,60,96,120]}
STARTS = sorted(set(sum(PLANS.values(),[])))
Q = ROOT/'archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010'
DATA = Q/'stage1_goal/data'

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module

_old=None
def old():
    global _old
    if _old is None: _old=load_module('_m11_old_runtime',V1/'code/runtime.py')
    return _old

def old_train():
    prior=old()
    with prior.import_aliases({'runtime':prior}): return load_module('_m11_old_train',V1/'code/train.py')

def load_initial(condition,device='cuda'):
    assert condition in CONDITIONS and sha(CHECKPOINT)==CHECKPOINT_SHA
    prior=old()
    with prior.import_aliases({'runtime':prior}): infer=load_module('_m11_old_infer',V1/'code/infer.py')
    model,run,_,_=infer.load_trained('History',2048,device)
    before=prior.tensor_hash(model.state_dict())
    if condition=='Revision':
        from revision import install_revision
        install_revision(model)
    assert prior.tensor_hash(model.state_dict())==before
    return model,run

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)
save_json=save
def source_hashes():return {str(p):sha(p) for p in sorted((E/'code').glob('*.py'))}

def load_final(condition,device='cuda'):
    import torch
    model,_=load_initial(condition,device)
    path=E/'training'/condition/f'snapshot_{FINAL_STEP}.pth'
    run=read_json(path.with_name('RUN.json'))
    assert run['status']=='COMPLETE' and run['final_step']==FINAL_STEP
    assert run['source_sha256']==source_hashes()
    assert run['checkpoint_sha256']==sha(path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.eval()
    assert old().tensor_hash(model.state_dict())==run['final_model_state_sha256']
    old().assert_frozen(model,run['original_state_sha256'])
    return model,run,path
