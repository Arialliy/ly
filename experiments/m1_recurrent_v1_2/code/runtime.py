"""Strict paired V12 Current/History initialization and optimizer migration."""
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
REFERENCE = E / 'reference/M1_v12_state_update_reference/state_update.py'
CHECKPOINT = V1 / 'training/History/run/snapshot_2048.pth'
CHECKPOINT_SHA = 'eda063496757f368d7442a7fac99a64f7b33ef0414ef99d006716d77a6dcb725'
INITIAL = E / 'init/initial_model.pth'
START_STEP, STEPS, FINAL_STEP = 2048, 128, 2176
SNAPSHOTS = (2048, 2080, 2112, 2176)
CONDITIONS = ['V12Current', 'V12History']
MODES = {'V12Current': 'current_only', 'V12History': 'history'}
SEQS = ['DJI_0026_2', 'DJI_0067_2']
PLANS = {'O': [0, 36, 72, 108, 120], 'S': [0, 24, 60, 96, 120]}
STARTS = sorted(set(sum(PLANS.values(), [])))
Q = ROOT / 'archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010'
DATA = Q / 'stage1_goal/data'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_old = None
def old():
    global _old
    if _old is None: _old = load_module('_m12_old_runtime', V1 / 'code/runtime.py')
    return _old


def old_train():
    prior = old()
    with prior.import_aliases({'runtime': prior}):
        return load_module('_m12_old_train', V1 / 'code/train.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


save_json = save


def source_hashes():
    paths = list((E / 'code').glob('*.py')) + list((E / 'reference').rglob('*.py'))
    return {str(path): sha(path) for path in sorted(paths)}


def _construct_initial(condition, device):
    assert condition in CONDITIONS and sha(CHECKPOINT) == CHECKPOINT_SHA
    prior = old()
    with prior.import_aliases({'runtime': prior}):
        inference = load_module('_m12_old_infer', V1 / 'code/infer.py')
    model, parent, _, _ = inference.load_trained('History', 2048, device)
    assert prior.tensor_hash(model.state_dict()) == parent['final_model_state_sha256']
    revision = load_module('_m12_revision', E / 'code/revision.py')
    installed = revision.install_revision(model, gate_seed=46)
    model.mode = MODES[condition]
    prior.assert_frozen(model, parent['original_state_sha256'])
    return model, parent, installed


def load_initial(condition, device='cuda'):
    """Use one persisted full initialization; only mode differs between branches."""
    import fcntl
    import torch
    model, parent, installed = _construct_initial(condition, device)
    expected = old().tensor_hash(model.state_dict())
    INITIAL.parent.mkdir(parents=True, exist_ok=True)
    with (INITIAL.parent / 'initialization.lock').open('a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not INITIAL.exists():
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            gate = {name: value for name, value in state.items() if '.state_update.' in name}
            checkpoint = {'model_state_dict': state, 'model_state_sha256': expected,
                'gate_state_sha256': old().tensor_hash(gate), 'initial_checkpoint_sha256': CHECKPOINT_SHA,
                'reference_sha256': sha(REFERENCE), 'seed': 46, 'original_state_sha256': parent['original_state_sha256'],
                'installation': installed}
            old().atomic_torch_save(INITIAL, checkpoint)
            save(INITIAL.with_suffix('.json'), {'status': 'COMPLETE', 'checkpoint_sha256': sha(INITIAL),
                **{key: value for key, value in checkpoint.items() if key != 'model_state_dict'}})
        checkpoint = torch.load(INITIAL, map_location='cpu', weights_only=True)
        assert checkpoint['initial_checkpoint_sha256'] == CHECKPOINT_SHA and checkpoint['reference_sha256'] == sha(REFERENCE)
        assert checkpoint['model_state_sha256'] == expected
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    assert old().tensor_hash(model.state_dict()) == expected
    old().assert_frozen(model, parent['original_state_sha256'])
    return model, parent


load_initialized = load_initial


def parameter_groups(model, parent):
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    old_names = list(parent['trainable_parameter_names'])
    gate_names = [name for name in parameters if name not in old_names]
    assert len(old_names) == 53 and len(set(old_names)) == 53 and all(name in parameters for name in old_names)
    assert len(gate_names) == 6 and all(name.startswith('adapter.recurrent_cell.state_update.') for name in gate_names)
    assert sum(parameters[name].numel() for name in old_names) == 109350
    assert sum(parameters[name].numel() for name in gate_names) == 2017
    assert list(name for name in parameters if name in old_names) == old_names
    return parameters, old_names, gate_names


def make_optimizer(model, parent, checkpoint=None):
    """Restore old53 by their recorded order, then append a fresh gate group."""
    import torch
    if checkpoint is None: checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == 2048 and checkpoint['condition'] == 'History'
    parameters, old_names, gate_names = parameter_groups(model, parent)
    optimizer = torch.optim.Adam([parameters[name] for name in old_names], lr=1e-5,
        weight_decay=1e-4, betas=(.9, .999), eps=1e-8)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    original_group = checkpoint['optimizer_state_dict']['param_groups']
    assert len(original_group) == 1 and len(original_group[0]['params']) == 53
    for name, identifier in zip(old_names, original_group[0]['params']):
        before = checkpoint['optimizer_state_dict']['state'][identifier]
        restored = optimizer.state[parameters[name]]
        assert float(restored['step']) == 2048
        for key, value in before.items():
            if torch.is_tensor(value): assert torch.equal(value.cpu(), restored[key].cpu()), (name, key)
            else: assert value == restored[key]
    optimizer.add_param_group({'params': [parameters[name] for name in gate_names], 'lr': 1e-5,
        'weight_decay': 1e-4, 'betas': (.9, .999), 'eps': 1e-8})
    assert all(parameters[name] not in optimizer.state for name in gate_names)
    receipt = {'old_parameter_names': old_names, 'gate_parameter_names': gate_names,
        'old_state_count': 53, 'old_initial_step': 2048, 'old_adam_tensors_exact': True,
        'gate_state_count': 0, 'gate_initial_step': 0, 'gate_parameters': 2017,
        'param_group_order': ['old53', 'new_gate6'], 'gate_state_policy': 'Fresh Adam state created on first gradient step',
        'initial_checkpoint_sha256': CHECKPOINT_SHA}
    return optimizer, receipt


def load_snapshot(condition, step, device='cuda'):
    import torch
    assert condition in CONDITIONS and step in SNAPSHOTS
    model, _ = load_initial(condition, device)
    path = E / 'training' / condition / f'snapshot_{step}.pth'
    run = read_json(path.with_name('RUN.json'))
    assert run['source_sha256'] == source_hashes()
    assert run['protocol_sha256'] == sha(E / 'PROTOCOL_FREEZE.json')
    snapshot = next(item for item in run['snapshots'] if item['step'] == step)
    assert snapshot['sha256'] == sha(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == step and checkpoint['condition'] == condition
    model.load_state_dict(checkpoint['model_state_dict'], strict=True); model.eval()
    assert old().tensor_hash(model.state_dict()) == snapshot['model_state_sha256']
    old().assert_frozen(model, run['original_state_sha256'])
    return model, run, path


def load_final(condition, device='cuda'):
    model, run, path = load_snapshot(condition, FINAL_STEP, device)
    assert run['status'] == 'COMPLETE' and run['final_step'] == FINAL_STEP
    assert run['checkpoint_sha256'] == sha(path)
    assert old().tensor_hash(model.state_dict()) == run['final_model_state_sha256']
    return model, run, path
