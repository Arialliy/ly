"""Original DeepPro-Plus loader and deterministic runtime for track contrast."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parents[1]
V1 = ROOT / 'experiments/track_contrast_v1'
V2 = ROOT / 'experiments/track_contrast_v2'
INIT_CP = V1 / 'training/C/run/snapshot_2048.pth'
REFERENCE = ROOT / 'deeppro_c_v2_reference'
R = ROOT / 'archive/infrared_problem_screen_20261009'
Q = R / 'temporal_diagnosis_20261010'
SOURCE = R / 'official_deeppro'
CP = SOURCE / 'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
MODEL_SOURCE = CP.parent.parent / 'DeepPro-Plus.py'
DATA = Q / 'stage1_goal/data'
C0 = Q / 'stage1_goal/predictions/official'
O = [0, 36, 72, 108, 120]
S = [0, 24, 60, 96, 120]
PLANS = {'O': O, 'S': S}
STARTS = sorted(set(O + S))
SEQS = ['DJI_0026_2', 'DJI_0067_2']
TRAIN_PREFIXES = ('TPro.', 'conv_out1.', 'conv_out2.')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


save = save_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_hash(items):
    """Order-independent hash, preserving key, dtype, shape and exact bytes."""
    import torch
    if hasattr(items, 'items'):
        items = items.items()
    result = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        result.update(name.encode('utf-8'))
        result.update(str(value.dtype).encode('ascii'))
        result.update(str(tuple(value.shape)).encode('ascii'))
        # Flatten before viewing uint8: BatchNorm counters are scalar tensors.
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def setup(seed=46, threads=2):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_model(checkpoint=None, device='cuda'):
    """Return (unwrapped original detector, original source path).

    The verified public release contains trusted historical metadata; new task
    checkpoints consist of tensors and primitive values and use safe loading.
    The default model is eval/frozen. Training explicitly calls freeze_for_training.
    """
    import torch
    assert sha(CP) == sha(ROOT / 'results/references/deeppro_plus_release.pth')
    source = str(SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
    official = load_module('track_contrast_original_deeppro', MODEL_SOURCE)
    model = official.detector(1, 40, 40)
    path = CP if checkpoint is None else Path(checkpoint)
    checkpoint_data = torch.load(path, map_location='cpu', weights_only=(path.resolve() != CP.resolve()))
    assert 'model_state_dict' in checkpoint_data
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 70913
    model.requires_grad_(False)
    return model.eval().to(device), MODEL_SOURCE


def trainable_names(model):
    return sorted(name for name, _ in model.named_parameters() if name.startswith(TRAIN_PREFIXES))


def freeze_for_training(model):
    """Enable original TPro/head parameters incl BN affine; freeze all BN stats.

    Returns (sorted_parameter_names, trainable_parameter_count). The original
    forward is unchanged. Callers must retain model.eval() during optimization.
    """
    names = trainable_names(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
    model.eval()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert names and count > 273
    return names, count


def frozen_state(model):
    names = set(trainable_names(model))
    return {k: v for k, v in model.state_dict().items() if k not in names}


def bn_state(model):
    return {k: v for k, v in model.state_dict().items()
            if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


def tiled_logits(model, x, core=128, halo=16):
    """Original forward, absolute-even spatial tile origins, CPU logits output."""
    import torch
    height, width = x.shape[-2:]
    output = torch.empty((x.shape[0], 40, height, width), dtype=torch.float32, device='cpu')
    for y in range(0, height, core):
        y1 = min(height, y + core)
        ya = max(0, y - halo) // 2 * 2
        yb = min(height, ((y1 + halo + 1) // 2) * 2)
        for z in range(0, width, core):
            z1 = min(width, z + core)
            za = max(0, z - halo) // 2 * 2
            zb = min(width, ((z1 + halo + 1) // 2) * 2)
            prediction = model(x[..., ya:yb, za:zb].contiguous())[1]
            output[..., y:y1, z:z1] = prediction[..., y-ya:y1-ya, z-za:z1-za].cpu()
    return output
