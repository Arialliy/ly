#!/usr/bin/env python3
"""Execute pinned official test.main, preserving loader, model and metric code.

Runtime adaptations: explicit source-declared seed46; safe NumPy checkpoint
metadata; output log relocation; optional disjoint sequence sharding. Raw metric
arrays are aggregated in the complete original order by a separate command.
The forward hook only records probabilities and never changes model outputs.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

D = Path(__file__).resolve().parent.parent
S = D / 'official_deeppro'
EXPERIMENT = 'NUDT-MIRSDT__2024-12-28_16-21__SoftLoUloss_DeepPro-Plus_DataL40'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source():
    snapshot = json.loads((D/'official_reproduction/SOURCE_SNAPSHOT.json').read_text())
    differences = []
    for name, expected in snapshot['tracked_sha256'].items():
        if not (S/name).is_file() or sha(S/name) != expected:
            differences.append(name)
    actual = {str(p.relative_to(S)) for p in S.rglob('*')
              if p.is_file() and '.git' not in p.relative_to(S).parts}
    differences.extend(sorted(actual-set(snapshot['tracked_sha256'])))
    if differences:
        raise RuntimeError(f'Pinned source snapshot changed: {differences}')
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    assert 0 <= args.shard_index < args.shards
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        raise RuntimeError('Completed output exists; choose a new output directory.')
    source_snapshot = verify_source()
    random.seed(46)
    np.random.seed(46)
    torch.manual_seed(46)
    torch.cuda.manual_seed_all(46)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(S))
    spec = importlib.util.spec_from_file_location('pinned_official_test', S / 'test.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    full = official.TestIRSeqDataLoader('NUDT-MIRSDT', str(D / 'official_reproduction/data/NUDT-MIRSDT'), 40, 4)
    full_names = full.seq_names.copy()
    indices = list(range(args.shard_index, len(full_names), args.shards))
    if args.max_sequences:
        indices = indices[:args.max_sequences]
    names = [full_names[i] for i in indices]
    plans = []
    for i in indices:
        for sample in full[i].samplelist:
            plans.append((full_names[i], sample[0][2], sample[-1][2]))
    base_loader = official.TestIRSeqDataLoader

    class ShardLoader(base_loader):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            assert self.seq_names == full_names
            self.seq_names = names.copy()

    official.TestIRSeqDataLoader = ShardLoader
    source_experiment = S / 'log/sem_seg' / EXPERIMENT
    target_experiment = out / 'log/sem_seg' / EXPERIMENT
    target_experiment.mkdir(parents=True, exist_ok=True)
    for name in ['logs', 'checkpoints', 'DeepPro-Plus.py']:
        dst = target_experiment / name
        if not dst.exists():
            dst.symlink_to(source_experiment / name, target_is_directory=(source_experiment / name).is_dir())
        if dst.resolve() != (source_experiment / name).resolve():
            raise RuntimeError(f'Unexpected existing experiment entry: {dst}')
    sys.path.insert(0, str(target_experiment))
    model_module = importlib.import_module('DeepPro-Plus')
    original_detector = model_module.detector
    probability_dir = out / 'probabilities'
    probability_dir.mkdir(exist_ok=True)
    capture = {'calls': 0, 'sequence': None, 'array': None, 'coverage': None, 'files': []}

    def save_capture():
        if capture['array'] is None:
            return
        assert np.all(capture['coverage'] > 0)
        p = probability_dir / (capture['sequence'] + '.npz')
        np.savez_compressed(p, probabilities=capture['array'], coverage=capture['coverage'])
        capture['files'].append({'sequence': capture['sequence'], 'path': str(p), 'sha256': sha(p), 'shape': list(capture['array'].shape)})

    def record_forward(module, inputs, outputs):
        k = capture['calls']
        if k >= len(plans):
            return  # official post-evaluation FLOP profiling is not a prediction
        name, start, end = plans[k]
        prob = torch.sigmoid(outputs[1]).detach().cpu().numpy()[0]
        if name != capture['sequence']:
            save_capture()
            frames = max(p[2] for p in plans if p[0] == name) + 1
            capture['sequence'] = name
            capture['array'] = np.zeros((frames, *prob.shape[-2:]), np.float32)
            capture['coverage'] = np.zeros(frames, np.int32)
        np.maximum(capture['array'][start:end+1], prob, out=capture['array'][start:end+1])
        capture['coverage'][start:end+1] += 1
        capture['calls'] += 1

    # Preserve the class name: official __init__ uses super(detector, self).
    # Replacing the module's class with a factory would invalidate that call.
    original_init = original_detector.__init__

    def recording_init(model, *pos, **kw):
        original_init(model, *pos, **kw)
        model.register_forward_hook(record_forward)

    original_detector.__init__ = recording_init
    callback_status = {'done': False}

    def record_counts(FalseNumAll, TrueNumAll, TgtNumAll, pixelsNumber,
                      intersection, union, thresholds, dataset, log_string):
        assert dataset.seq_names == names
        assert capture['calls'] == len(plans)
        save_capture()
        np.savez(out / 'official_counts.npz', false_num=FalseNumAll,
                 true_num=TrueNumAll, target_num=TgtNumAll, pixels=pixelsNumber,
                 intersection=intersection, union=union, thresholds=thresholds,
                 sequence_indices=np.array(indices), sequence_names=np.array(names))
        callback_status['done'] = True
        log_string('RAW_OFFICIAL_COUNTS_SAVED; complete-order ROC computed after shard merge')

    official.writeNUDTMIRSDT_ROC = record_counts
    old_argv = sys.argv
    sys.argv = ['test.py']
    try:
        test_args = official.parse_args()
    finally:
        sys.argv = old_argv
    test_args.gpu = args.gpu
    test_args.datapath = str(D / 'official_reproduction/data/NUDT-MIRSDT')
    test_args.logpath = str(out / 'log') + '/'
    test_args.log_dir = EXPERIMENT
    test_args.threshold_eval = 0.5
    test_args.batch_size = 1
    test_args.seqlen = 40
    # The official checkpoint has only one audited NumPy float64 metadata scalar.
    numpy_core = np._core if hasattr(np, '_core') else np.core
    numeric_globals = [(numpy_core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
                       np.dtype, type(np.dtype(np.float64))]
    started = time.time()
    print(json.dumps({'event': 'START', 'seed': 46, 'gpu': args.gpu, 'sequences': names}), flush=True)
    with torch.serialization.safe_globals(numeric_globals):
        official.main(test_args)
    assert callback_status['done']
    verify_source()
    record = {
        'status': 'complete', 'seed': 46, 'official_checkpoint_training_seed': 'unknown',
        'source_commit': source_snapshot['git_commit'],
        'source_dirty': '',
        'source_status_method': 'host git-clean snapshot; all tracked SHA256 and file set verified before and after runtime',
        'source_snapshot_sha256': sha(D/'official_reproduction/SOURCE_SNAPSHOT.json'),
        'checkpoint_sha256': sha(target_experiment / 'checkpoints/best_model.pth'),
        'loaded_checkpoint_path': str((target_experiment / 'checkpoints/best_model.pth').resolve()),
        'loaded_model_path': str(Path(model_module.__file__).resolve()),
        'script_sha256': sha(__file__), 'source_file_sha256': {p: sha(S/p) for p in ['test.py', 'ShootingRules.py', 'write_results.py', 'data_utils/TestDataLoader.py']},
        'full_sequence_order': full_names, 'sequence_indices': indices, 'sequences': names,
        'windows': plans, 'probabilities': capture['files'], 'gpu': args.gpu,
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'numpy': np.__version__,
        'elapsed_seconds_including_io_metrics_profile': time.time()-started,
        'execution': 'native full spatial frames, float32, original 40-frame windows and overlap maximum',
        'arithmetic_policy': {'cudnn_benchmark': False, 'cudnn_deterministic': True, 'matmul_tf32': False, 'cudnn_tf32': False, 'deterministic_algorithms': True},
        'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated(), 'max_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
        'training_performed': False,
    }
    assert not record['source_dirty'], record['source_dirty']
    (out / 'COMPLETE.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'event': 'COMPLETE', 'out': str(out), 'sequences': names}), flush=True)


if __name__ == '__main__':
    main()
