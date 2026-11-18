"""Frozen inputs/environment and new experiment-only helpers."""
from pathlib import Path
import os, sys, random, hashlib, json
N=Path(__file__).resolve().parents[1]
Q=N.parent
R=Q.parent
sys.path.insert(0,str(Q/'supervision_alignment_control_20261012/code'))
import train_supervision as previous
sha,save_json,load_module=previous.sha,previous.save_json,previous.load_module
tensor_hash,data_hash=previous.tensor_hash,previous.data_hash
EXPECTED=previous.EXPECTED
SOURCE=R/'official_deeppro'
CHECKPOINT=SOURCE/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
VARIANTS={'R0':False,'R1':True,'R2':True}
OLD_LOG=Q/'mutual_temporal_pilot_20261012/training/E0/run/TRAIN.jsonl'

def setup():
    import numpy as np
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(46);np.random.seed(46);torch.manual_seed(46);torch.cuda.manual_seed_all(46)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

def default_args():
    from types import SimpleNamespace
    return SimpleNamespace(source=SOURCE,checkpoint=CHECKPOINT,dataset_root=Q/'stage1_goal/data',
        selection=Q/'stage1_goal/DATA_SELECTION.json',plan=Q/'stage1_goal/pilot/TRAIN_PLAN.json')

def sources():
    names=['common.py','model.py','targets.py','losses.py','train.py']
    return {str(N/'code'/name):sha(N/'code'/name) for name in names}

def create_model(base):
    # Old helper prepends its own module dirs; explicit import prevents collision.
    original,_=base.load_model(SOURCE,CHECKPOINT)
    module=load_module('ord_model',N/'code/model.py')
    return module.ObjectResponseDecoder(original)

def branches(model):
    return [(name,p) for name,p in model.named_parameters() if p.requires_grad]

def assert_original(model,initial):
    import torch
    state=model.original.state_dict()
    if model.original.training or any(p.requires_grad or p.grad is not None for p in model.original.parameters()):
        raise AssertionError('Original model not frozen/eval')
    if set(state)!=set(initial) or any(not torch.equal(state[k].detach().cpu(),v) for k,v in initial.items()):
        raise AssertionError('Original state changed')
    return tensor_hash(state.items())
