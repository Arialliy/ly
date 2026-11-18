"""Isolated synthetic-input forward/backward audit; never updates/saves weights."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch

BASE = Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009')
SOURCE = BASE / 'official_deeppro'
OUT = BASE / 'temporal_diagnosis_20261010/stage1_goal/runtime'
CHECKPOINT = SOURCE / 'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
MODEL_SOURCE = CHECKPOINT.parent.parent / 'DeepPro-Plus.py'
sys.path.insert(0, str(SOURCE))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

random.seed(46)
np.random.seed(46)
torch.manual_seed(46)
torch.cuda.manual_seed_all(46)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
spec = importlib.util.spec_from_file_location('archived_deeppro_model', MODEL_SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
numeric_globals = [(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype, type(np.dtype(np.float64))]
original_hash = sha(CHECKPOINT)
with torch.serialization.safe_globals(numeric_globals):
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
metadata = {k: v for k, v in checkpoint.items() if k not in ('model_state_dict', 'optimizer_state_dict')}
optimizer_groups = [{k: v for k, v in group.items() if k != 'params'} for group in checkpoint['optimizer_state_dict']['param_groups']]
results = []
for batch, patch in [(1, 64), (4, 64), (1, 96), (4, 96), (4, 128)]:
    model = module.detector(1, 40, 40).cuda()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.train()
    criterion = module.SoftLoUloss().cuda()
    x = torch.randn(batch, 1, 40, patch, patch, device='cuda', dtype=torch.float32)
    y = torch.zeros(batch, 40, patch, patch, device='cuda', dtype=torch.float32)
    y[:, :, patch // 2:patch // 2 + 2, patch // 2:patch // 2 + 2] = 1
    times = []
    peak_allocated = []
    peak_reserved = []
    losses = []
    for iteration in range(4):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        features, logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert torch.isfinite(loss)
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if iteration:
            times.append(elapsed)
            peak_allocated.append(torch.cuda.max_memory_allocated())
            peak_reserved.append(torch.cuda.max_memory_reserved())
            losses.append(float(loss.detach().cpu()))
        del loss, features, logits
    record = dict(batch=batch, frames=40, patch=patch, bn_mode='train', precision='float32',
                  timed_repetitions=3, seconds=times, median_seconds=statistics.median(times),
                  max_allocated_bytes=max(peak_allocated), max_reserved_bytes=max(peak_reserved),
                  synthetic_losses=losses, finite_gradients=True)
    results.append(record)
    print(json.dumps(record), flush=True)
    del model, criterion, x, y
    gc.collect()
    torch.cuda.empty_cache()
assert sha(CHECKPOINT) == original_hash
report = dict(scope='synthetic input forward+backward only, no optimizer step, no weight file writes, not detection performance',
              seed=46, torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
              device=torch.cuda.get_device_name(), tf32=False, cudnn_deterministic=True,
              source_sha256=sha(MODEL_SOURCE), checkpoint_sha256=original_hash, checkpoint_unchanged=True,
              checkpoint_metadata=metadata, checkpoint_optimizer_groups=optimizer_groups, results=results)
(OUT / 'BENCHMARK.json').write_text(json.dumps(report, indent=2) + '\n')
print('RUNTIME_BENCHMARK_OK', flush=True)
