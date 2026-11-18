"""Official DeepPro-Plus baseline plus a freshly initialized source-relations M1.

This module defines model construction only: no training plan, legacy M1 state,
optimizer restoration, data access, evaluation protocol, or automatic execution.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / 'sources/official_deeppro'
MODEL_SOURCE = OFFICIAL / 'networks/models/DeepPro-Plus.py'
MODEL_SOURCE_SHA = 'fa46a034a30d8bbb0f677271d6368f033fcb389d682478d331551d8bae087668'
CHECKPOINT = ROOT / 'results/references/deeppro_plus_release.pth'
CHECKPOINT_SHA = '4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc'
SEED = 46
CONFIGS = {}
for _suffix, _readout in (('', 'simple'), ('M2', 'selective')):
    for _name, _mode, _reduction in (
            ('CurrentRelations', 'current_only', 'relations'),
            ('HistoryEarlyMerge', 'history', 'early_merge'),
            ('HistoryRelations', 'history', 'relations')):
        CONFIGS[_name + _suffix] = dict(mode=_mode, reduction_mode=_reduction, readout=_readout)
CONDITIONS = list(CONFIGS)
MODES = {name: config['mode'] for name, config in CONFIGS.items()}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(state):
    """Hash named CPU tensor contents, including baseline BN buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _official_module():
    if sha(MODEL_SOURCE) != MODEL_SOURCE_SHA:
        raise RuntimeError('Official DeepPro-Plus model source SHA256 differs')
    # The official file uses absolute networks.* imports. Reject collisions;
    # never silently use another project's implementation of that package.
    for name, module in tuple(sys.modules.items()):
        if name == 'networks' or name.startswith('networks.'):
            origin = getattr(module, '__file__', None)
            if origin and not Path(origin).resolve().is_relative_to(OFFICIAL.resolve()):
                raise RuntimeError(f'Conflicting official networks import: {name}: {origin}')
            for directory in getattr(module, '__path__', ()):
                if not Path(directory).resolve().is_relative_to(OFFICIAL.resolve()):
                    raise RuntimeError(f'Conflicting networks package path: {directory}')
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    name = '_fresh_source_relations_official_deeppro_plus'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, MODEL_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _release_state_dict():
    """Read the exact release artifact with a restricted weights-only unpickler."""
    import numpy as np
    import torch
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError('Published baseline checkpoint SHA256 differs')
    # This fixed release includes a NumPy scalar metric in addition to tensors.
    # NumPy2 moved its scalar function; explicitly map the serialized old name.
    # Keep weights_only=True: do not fall back to arbitrary pickle execution.
    core_name = 'numpy._core.multiarray' if hasattr(np, '_core') else 'numpy.core.multiarray'
    scalar = importlib.import_module(core_name).scalar
    allowed = [(scalar, 'numpy.core.multiarray.scalar'), np.dtype,
               type(np.dtype('float32')), type(np.dtype('float64'))]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError('Baseline checkpoint must contain a state dictionary')
    key = next((name for name in ('model_state_dict', 'state_dict') if name in checkpoint), None)
    state = checkpoint[key] if key else checkpoint
    if not isinstance(state, Mapping) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError('Baseline model state contains non-tensor values')
    prefixed = [name.startswith('module.') for name in state]
    if any(prefixed) and not all(prefixed):
        raise RuntimeError('Mixed module-prefixed baseline state keys')
    strip_module = all(prefixed)
    state = {name[7:] if strip_module else name: value for name, value in state.items()}
    return state, {'checkpoint_state_key': key or 'direct_state_dict',
                   'stripped_module_prefix': strip_module, 'weights_only': True,
                   'allowed_numpy_metadata': ['numpy.core.multiarray.scalar', 'numpy.dtype',
                                              'float32 dtype', 'float64 dtype'],
                   'optimizer_state_restored': False}


def load_baseline(device='cpu'):
    """Load the official 40-frame release with strict key/shape matching."""
    import torch
    state, loading = _release_state_dict()
    module = _official_module()
    # Construction RNG is restored; the release weights replace every tensor.
    with torch.random.fork_rng(devices=[]):
        base = module.detector(num_classes=1, seqlen=40, out_len=40)
    base.load_state_dict(state, strict=True)
    base = base.to(device=device, dtype=torch.float32)
    base.eval()
    metadata = {'baseline_source': str(MODEL_SOURCE), 'baseline_source_sha256': MODEL_SOURCE_SHA,
                'baseline_checkpoint': str(CHECKPOINT), 'baseline_checkpoint_sha256': CHECKPOINT_SHA,
                'baseline_state_sha256': tensor_hash(base.state_dict()),
                'baseline_parameters': sum(p.numel() for p in base.parameters()),
                'baseline_settings': {'num_classes': 1, 'seqlen': 40, 'out_len': 40}, **loading}
    return base, metadata


def load_initial(condition='HistoryRelations', device='cpu', *, seed=SEED, freeze_baseline=True):
    """Build every M1/M2 tensor fresh; only baseline tensors come from a checkpoint."""
    import torch
    if condition not in CONFIGS:
        raise ValueError(f'Unknown condition: {condition}')
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    base, metadata = load_baseline('cpu')
    # Adapter/PDA imports are new code and third-party source only. No legacy
    # adapter module, learned M1 weight, optimizer, or training cache is read.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.m1_source_relations_v1.code.adapter import (
        SourceRelationsAdapter, DeepProWithSourceRelations,
    )
    config = CONFIGS[condition]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        adapter = SourceRelationsAdapter(reduction_mode=config['reduction_mode'], readout=config['readout'])
        model = DeepProWithSourceRelations(base, adapter, mode=config['mode'], freeze_baseline=freeze_baseline)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    new_parameters = dict(adapter.named_parameters())
    metadata.update(condition=condition, condition_config=dict(config), seed=seed,
                    baseline_frozen=freeze_baseline, new_M1_initialization='fresh seed; no M1 checkpoint',
                    new_parameter_sha256=tensor_hash(new_parameters),
                    new_state_sha256=tensor_hash(adapter.state_dict()),
                    new_parameters=sum(p.numel() for p in new_parameters.values()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    parameters=sum(p.numel() for p in model.parameters()),
                    new_parameter_shapes={name: list(p.shape) for name, p in new_parameters.items()},
                    model_state_sha256=tensor_hash(model.state_dict()),
                    training_started=False, native_GPU_execution_verified=False)
    if tensor_hash(base.state_dict()) != metadata['baseline_state_sha256']:
        raise RuntimeError('Fresh adapter construction changed baseline parameters or buffers')
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true', help='Construct on CPU and report hashes/parameters; no forward or training')
    parser.add_argument('--baseline-only', action='store_true', help='Inspect only the official release, without building PDA/M1')
    parser.add_argument('--condition', choices=CONDITIONS, default='HistoryRelations')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    if args.baseline_only:
        _, metadata = load_baseline('cpu')
        metadata.update(training_started=False, forward_executed=False)
    elif args.inspect:
        _, metadata = load_initial(args.condition, 'cpu', seed=args.seed)
        metadata['forward_executed'] = False
    else:
        metadata = {'conditions': CONFIGS, 'baseline_checkpoint': str(CHECKPOINT),
                    'new_M1_initialization': 'fresh seed', 'training_started': False,
                    'usage': 'Use --inspect for CPU construction or --baseline-only for the release.'}
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
