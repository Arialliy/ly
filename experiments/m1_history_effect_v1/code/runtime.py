"""Read-only bridge to the frozen M1-v1 checkpoint and runtime."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parents[1]
V1 = ROOT / 'experiments/m1_recurrent_v1'
Q = ROOT / 'archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010'
DATA = Q / 'stage1_goal/data'
PLANS = {'O': [0, 36, 72, 108, 120], 'S': [0, 24, 60, 96, 120]}
STARTS = sorted(set(sum(PLANS.values(), [])))
SEQS = ['DJI_0026_2', 'DJI_0067_2']
CHECKPOINT = V1 / 'training/History/run/snapshot_2048.pth'
CHECKPOINT_SHA = 'eda063496757f368d7442a7fac99a64f7b33ef0414ef99d006716d77a6dcb725'

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

_old = None
def old():
    global _old
    if _old is None:
        _old = load_module('_history_effect_frozen_runtime', V1 / 'code/runtime.py')
    return _old

def v1_infer():
    prior = old()
    with prior.import_aliases({'runtime': prior}):
        return load_module('_history_effect_frozen_infer', V1 / 'code/infer.py')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

save_json = save

def source_hashes():
    return {str(p): sha(p) for p in sorted((E / 'code').glob('*.py'))}
