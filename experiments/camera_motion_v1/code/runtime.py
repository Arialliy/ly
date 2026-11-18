"""Read-only historical inputs and the task's isolated runtime."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

E = Path(__file__).resolve().parents[1]
B = E.parents[1]
R = B / 'archive/infrared_problem_screen_20261009'
Q = R / 'temporal_diagnosis_20261010'
O = [0, 36, 72, 108, 120]
S = [0, 24, 60, 96, 120]
STARTS = sorted(set(O + S))
SEQS = ['DJI_0026_2', 'DJI_0067_2']
C0 = Q / 'stage1_goal/predictions/official'
SOURCE = R / 'official_deeppro'
CP = SOURCE / 'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8388608), b''): h.update(chunk)
    return h.hexdigest()

def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(p)

def setup():
    import numpy as np
    import torch
    import random
    random.seed(46); np.random.seed(46); torch.manual_seed(46)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def load_model():
    import torch
    sys.path.insert(0, str(SOURCE))
    p = CP.parent.parent / 'DeepPro-Plus.py'
    spec = importlib.util.spec_from_file_location('camera_frozen_official', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    # Bundled public release checkpoint, checked against its retained duplicate.
    assert sha(CP) == sha(B/'results/references/deeppro_plus_release.pth')
    ckpt = torch.load(CP, map_location='cpu', weights_only=False)
    model = m.detector(1, 40, 40)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.requires_grad_(False)
    assert sum(x.numel() for x in model.parameters()) == 70913
    return model.eval().cuda(), p

def tiled_logits(model, x, core=128, halo=16):
    import torch
    h, w = x.shape[-2:]
    out = torch.empty((1, 40, h, w), dtype=torch.float32)
    for y in range(0, h, core):
        y1=min(h,y+core); ya=max(0,y-halo)//2*2; yb=min(h,((y1+halo+1)//2)*2)
        for z in range(0,w,core):
            z1=min(w,z+core); za=max(0,z-halo)//2*2; zb=min(w,((z1+halo+1)//2)*2)
            pred=model(x[...,ya:yb,za:zb].contiguous())[1]
            out[...,y:y1,z:z1]=pred[...,y-ya:y1-ya,z-za:z1-za].cpu()
    return out
