"""M1 runtime. Importing this module never starts a workload."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parents[1]
V1 = ROOT / 'experiments/track_contrast_v1'
V2 = ROOT / 'experiments/track_contrast_v2'
R = ROOT / 'archive/infrared_problem_screen_20261009'
Q = R / 'temporal_diagnosis_20261010'
SOURCE = R / 'official_deeppro'
CP = SOURCE / 'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
MODEL_SOURCE = CP.parent.parent / 'DeepPro-Plus.py'
DATA = Q / 'stage1_goal/data'
O = [0, 36, 72, 108, 120]
S = [0, 24, 60, 96, 120]
PLANS = {'O': O, 'S': S}
STARTS = sorted(set(O + S))
SEQS = ['DJI_0026_2', 'DJI_0067_2']
CONDITIONS = ['Current', 'History']
MODES = {'Current': 'current_only', 'History': 'history'}
STEPS = 2048
SNAPSHOTS = (0, 128, 512, 1024, 2048)
SETTINGS = {'temperature': 0.47569116950035095, 'lambda_keep': 5.75443767610971,
            'lambda_temporal': 0.10885460558865433, 'lambda_readout': 0.0023821861727945912}
INITIAL = E / 'init/initial_model.pth'
MODEL_SETTINGS = {'state_channels': 16, 'checkpoint_recurrence': True}
RUNTIME_SETTINGS = {'deterministic_algorithms': 'warn_only',
    'nondeterministic_operators': ['CUDA adaptive_avg_pool2d backward', 'DCNv2 atomicAdd'],
    'cudnn_deterministic': True, 'cudnn_benchmark': False, 'TF32': False,
    'bitwise_training_reproducibility_claimed': False}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


save = save_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def import_aliases(mapping):
    """Historical scripts use bare imports; restore caller bindings afterwards."""
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in mapping}
    sys.modules.update(mapping)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is absent: sys.modules.pop(name, None)
            else: sys.modules[name] = value


_frozen = None


def frozen_f():
    global _frozen
    if _frozen is None:
        from types import SimpleNamespace
        for path in (ROOT, V1 / 'train_modules', V2 / 'train_modules'):
            if str(path) not in sys.path: sys.path.insert(0, str(path))
        rt = load_module('_m1_frozen_v2_runtime', V2 / 'code/runtime.py')
        with import_aliases({'runtime': rt}):
            data = load_module('_m1_frozen_v2_data', V2 / 'code/data.py')
            with import_aliases({'data': data}):
                common = load_module('_m1_frozen_v2_common', V2 / 'code/common.py')
        _frozen = SimpleNamespace(runtime=rt, data=data, common=common)
    return _frozen


def setup():
    frozen_f().runtime.setup(seed=46)
    import torch
    # Keep the official adaptive pooling and atomic DCNv2 operator semantics.
    # PyTorch cannot supply deterministic CUDA backward for these operations.
    torch.use_deterministic_algorithms(True, warn_only=True)


def tensor_hash(items):
    return frozen_f().runtime.tensor_hash(items)


def source_hashes():
    paths = list((E / 'code').glob('*.py'))
    paths += list((ROOT / 'models/adapters').glob('*.py'))
    paths += [V2 / 'code/common.py', V2 / 'code/data.py', V2 / 'code/runtime.py',
              V2 / 'train_modules/temporal_risk.py', V1 / 'code/train.py',
              V1 / 'train_modules/region_contrast_loss.py', V1 / 'train_modules/reliable_keep_loss.py',
              ROOT / 'deeppro_c_v2_reference/temporal_risk.py', MODEL_SOURCE]
    paths += list((E / 'upstream').glob('*.py'))
    compat = E / 'upstream/compat'
    extensions = {'.py', '.cpp', '.cc', '.c', '.cu', '.cuh', '.h', '.hpp', '.so'}
    paths += [p for p in compat.rglob('*') if p.is_file() and p.suffix in extensions
              and not {'build', '__pycache__'}.intersection(p.relative_to(compat).parts)]
    paths += [E / 'upstream/PROVENANCE.json', E / 'upstream/COMPATIBILITY.patch']
    return {str(p): sha(p) for p in sorted(set(paths)) if p.is_file()}


def protocol_hash():
    path = E / 'PROTOCOL_FREEZE.json'
    return sha(path) if path.exists() else None


def make_model(condition, device='cuda'):
    if condition not in MODES: raise ValueError(condition)
    import torch
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    from models.adapters.window_recurrent_adapter import WindowRecurrentAdapter, FrozenDeepProWithAdapter
    base, _ = frozen_f().runtime.load_model(device='cpu')
    adapter = WindowRecurrentAdapter(**MODEL_SETTINGS)
    model = FrozenDeepProWithAdapter(base, adapter, mode=MODES[condition])
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    base.requires_grad_(False)
    adapter.requires_grad_(True)
    return model


def original(model):
    for name in ('original', 'base', 'backbone'):
        value = getattr(model, name, None)
        if value is not None: return value
    raise AttributeError('Wrapper must expose original/base/backbone detector')


def assert_frozen(model, expected):
    import torch
    base = original(model)
    assert not base.training
    assert all(not p.requires_grad and p.grad is None for p in base.parameters())
    assert all(not module.training for module in base.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
    assert tensor_hash(base.state_dict()) == expected, 'Original model/BN changed'


def atomic_torch_save(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, temporary); temporary.replace(path)


def ensure_initialization():
    """One lock-protected seed46 state shared by both independent conditions."""
    import fcntl
    import torch
    INITIAL.parent.mkdir(parents=True, exist_ok=True)
    with (INITIAL.parent / 'initialization.lock').open('a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        setup()
        model = make_model('History', device='cpu')
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        identity = tensor_hash(state)
        if INITIAL.exists():
            checkpoint = torch.load(INITIAL, map_location='cpu', weights_only=True)
            assert checkpoint['model_state_sha256'] == identity
            assert checkpoint['release_checkpoint_sha256'] == sha(CP)
        else:
            atomic_torch_save(INITIAL, {'model_state_dict': state, 'model_state_sha256': identity,
                'release_checkpoint_sha256': sha(CP), 'seed': 46,
                'original_state_sha256': tensor_hash(original(model).state_dict())})
        save_json(INITIAL.with_suffix('.json'), {'status': 'COMPLETE', 'seed': 46,
            'checkpoint_sha256': sha(INITIAL), 'model_state_sha256': identity,
            'release_checkpoint_sha256': sha(CP), 'source_sha256': source_hashes()})
    return INITIAL


def load_initialized(condition, device='cuda'):
    import torch
    ensure_initialization()
    setup(); model = make_model(condition, device)
    checkpoint = torch.load(INITIAL, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    assert tensor_hash(model.state_dict()) == checkpoint['model_state_sha256']
    assert_frozen(model, checkpoint['original_state_sha256'])
    return model, checkpoint


def frozen_cache_manifest():
    result = {}
    for root in (V1, V2):
        freeze = read_json(root / 'PROTOCOL_FREEZE.json')
        for index in range(256):
            path = root / 'cache' / f'entry_{index:03d}.npz'
            value = sha(path)
            assert value == freeze['cache_sha256'][str(index)], path
            result[str(path)] = value
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--initialize', action='store_true')
    args = parser.parse_args()
    if args.initialize: print(str(ensure_initialization()), flush=True)
