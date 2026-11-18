"""CPU-only restoration compatibility check, no update or checkpoint writes."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

BASE = Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009')
SRC = BASE / 'official_deeppro'
CKPT = SRC / 'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('restore_check_model', CKPT.parent.parent / 'DeepPro-Plus.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with torch.serialization.safe_globals([(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype, type(np.dtype(np.float64))]):
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
model = module.detector(1, 40, 40)
model.load_state_dict(ckpt['model_state_dict'], strict=True)
optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=1e-4)
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
steps = sorted({int(s['step']) for s in optimizer.state.values()})
result = dict(strict_model_restore=True, optimizer_restore=True, parameter_tensors=sum(1 for _ in model.parameters()),
              optimizer_state_entries=len(optimizer.state), optimizer_step_values=steps,
              epoch=int(ckpt['epoch']), restored_lr=optimizer.param_groups[0]['lr'], optimizer_steps_executed=0)
for parameter, state in optimizer.state.items():
    for key in ('exp_avg', 'exp_avg_sq'):
        assert state[key].shape == parameter.shape
        assert torch.isfinite(state[key]).all()
result['moment_shapes_and_finiteness_pass'] = True
(BASE / 'temporal_diagnosis_20261010/stage1_goal/runtime/OPTIMIZER_RESTORE.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
