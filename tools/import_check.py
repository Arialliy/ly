"""Check the collected source tree without data, checkpoints, GPU, or optimization."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009')
Q = ROOT / 'temporal_diagnosis_20261010'
N = Q / 'tlfp_pilot_20261012'
sys.path[:0] = [str(N/'code'), str(N/'evaluation_code'), str(ROOT/'official_deeppro')]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

import train_tlfp
import infer_tlfp
import temporal_lfp
import evaluate
import verify_identity
import mechanism_tables
import trajectory_tables

paths = [
    Q/'top1_background_control_20261011/code/train_pilot.py',
    Q/'top1_background_control_20261011/code/losses.py',
    Q/'top1_background_control_20261011/evaluation_code/evaluate.py',
    Q/'stage1_goal/pilot/score_pilot.py',
    Q/'analysis/analyze.py',
]
for i, path in enumerate(paths):
    load('collected_dependency_' + str(i), path)
official = load('collected_official_checkpoint_model', ROOT/'official_deeppro/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/DeepPro-Plus.py')
base = official.detector(num_classes=1, seqlen=40, out_len=40)
assert sum(p.numel() for p in base.parameters()) == 70913
spatial = temporal_lfp.FramewiseFeatureAdapter(temporal_lfp.SpatialLFPReference(channels=32), channels=32, frame_chunk=8)
counts = {}
for mode, expected in [('spatial', 100), ('temporal', 296)]:
    branch = temporal_lfp.TemporalLFPAdapter.from_spatial(spatial, mode=mode, frame_chunk=8)
    counts[mode] = sum(p.numel() for p in branch.parameters())
    assert counts[mode] == expected
record = dict(status='PASS', source_root=str(ROOT), original_parameters=70913,
              added_parameters=counts, training_updates=0, gpu_used=False,
              original_host_code_mounted=False,
              versions={p: importlib.metadata.version(p) for p in ['torch','numpy','scipy','matplotlib','Pillow','pytest']})
Path('/validation/IMPORT_CHECK.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
