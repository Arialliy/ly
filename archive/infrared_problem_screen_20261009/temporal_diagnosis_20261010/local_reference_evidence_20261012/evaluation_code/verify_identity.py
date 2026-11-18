"""CPU-only checkpoint, frozen-backbone and exposure validation before scoring."""
import hashlib
import json
from pathlib import Path

MODES = {'N0': 'current', 'N1': 'difference', 'N2': 'relative'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_hash(items):
    h = hashlib.sha256()
    for name, value in sorted(items):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_identity(base, condition, sequence, predictions_root):
    import numpy as np
    import torch
    assert condition in MODES
    q = base.parent
    runpath = base/'training'/condition/'run/RUN.json'
    infpath = predictions_root/sequence/'COMPLETE.json'
    run = json.loads(runpath.read_text())
    inf = json.loads(infpath.read_text())
    freeze_path = base/'PROTOCOL_FREEZE.json'
    freeze = json.loads(freeze_path.read_text())
    assert sha(base/'PROTOCOL.md') == freeze['protocol_sha256']
    for path, expected in freeze['code_sha256'].items():
        assert sha(base/path) == expected, ('Frozen code changed', path)
    assert run['status'] == 'complete' and run['condition'] == run['variant'] == condition
    assert run['mode'] == MODES[condition] and run['steps'] == run['final_step'] == 512
    assert run['seed'] == 46 and run['lr'] == 1e-4 and run['weight_decay'] == 1e-4
    assert run['bn_initial_sha256'] == run['bn_final_sha256']
    assert run['original_initial_state_sha256'] == run['original_final_state_sha256']
    assert run['frozen_original_unchanged'] is True and run['model_labels_access'] is False
    assert run['parameter_count'] == 72442 and run['new_parameters'] == run['trainable_parameters'] == 1529
    assert run['protocol'] == 'frozen_original_positive_masked_official_SoftIoU_empty_old_top1pct'
    assert inf['status'] == 'COMPLETE' and inf['sequence'] == sequence
    assert inf['condition'] == inf['variant'] == inf['training_variant'] == condition
    assert inf['mode'] == MODES[condition] and inf['diagnostic'] == 'none'
    assert inf['checkpoint_step'] == inf['main_checkpoint_step'] == 512 and inf['checkpoint_is_main'] is True
    assert inf['model_labels_access'] is False and inf['read_only_query_labels_access'] is True
    assert inf['frozen_original_unchanged'] is True
    assert inf['seed'] == 46 and inf['normalization'] == [85.0799, 47.4845]
    assert inf['tile_core'] == 128 and inf['tile_halo'] == 16
    assert inf['official_starts'] == [0,36,72,108,120]
    assert inf['symmetric_starts'] == [0,24,60,96,120]
    assert inf['diagnostic_starts'] == [0,24,36,60,72,96,108,120]
    assert inf['parameters'] == run['parameter_count'] and inf['trainable_parameters'] == run['new_parameters']
    for key in ['original_initial_state_sha256', 'original_final_state_sha256', 'bn_initial_sha256', 'bn_final_sha256']:
        assert inf[key] == run[key], key
    assert all(row['status'] == 'passed' and row['threshold05_disagreements'] == 0
               for row in inf['tiling_checks'])

    checkpoint_path = Path(inf['checkpoint'])
    assert checkpoint_path.parent.resolve() == runpath.parent.resolve()
    assert checkpoint_path.name in ['snapshot_512.pth', 'final_model.pth']
    snapshot = next(row for row in run['snapshots'] if row['step'] == 512)
    cp_sha = sha(checkpoint_path)
    assert snapshot['sha256'] == run['final_checkpoint_sha256'] == inf['checkpoint_sha256'] == cp_sha
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert cp['condition'] == cp['variant'] == condition and cp['step'] == 512
    assert cp['mode'] == MODES[condition]
    assert cp['initial_checkpoint_sha256'] == run['checkpoint_sha256'] == inf['reference_checkpoint_sha256']
    assert cp['plan_sha256'] == run['plan_sha256'] == inf['plan_sha256']
    assert cp['selection_sha256'] == run['selection_sha256'] == inf['selection_sha256']
    assert cp['label_config'] == run['label_config'] == inf['label_config']
    assert cp['effective_supervision_config'] == run['effective_supervision_config']
    assert cp['model_config'] == run['model_config'] == inf['model_config']
    assert cp['original_state_sha256'] == run['original_initial_state_sha256']
    config = cp['model_config']
    assert config['original_frozen'] is True and config['original_bn'] == 'eval'
    evidence = config['evidence']
    assert evidence['mode'] == MODES[condition] and evidence['delays'] == [1,4]
    assert evidence['inner_radius'] == 5 and evidence['outer_radius'] == 7 and evidence['annulus_count'] == 104
    assert evidence['s_min'] == 1.0 / 47.4845 and evidence['clipping'] is None

    # Compare tensors themselves; agreement between two metadata strings is insufficient.
    reference_checkpoint = Path(run['checkpoint_path'])
    assert sha(reference_checkpoint) == run['checkpoint_sha256']
    numpy_core = np._core if hasattr(np, '_core') else np.core
    safe = [(numpy_core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(safe):
        release = torch.load(reference_checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
    original = {key[len('original.'):]: value for key, value in cp['model_state_dict'].items() if key.startswith('original.')}
    assert original.keys() == release.keys()
    assert all(torch.equal(original[key], release[key]) for key in release)
    assert tensor_hash(original.items()) == run['original_initial_state_sha256']
    assert tensor_hash(cp['model_state_dict'].items()) == run['final_model_state_sha256']
    assert all(key.startswith(('original.', 'evidence.')) for key in cp['model_state_dict'])

    previous_path = q/'mutual_temporal_pilot_20261012/training/E0/run/RUN.json'
    previous = json.loads(previous_path.read_text())
    for key in ['checkpoint_sha256', 'plan_sha256', 'selection_sha256', 'model_source_sha256',
                'cumulative_input_sha256', 'cumulative_label_sha256', 'training_sequences',
                'label_config', 'effective_supervision_config']:
        assert run[key] == previous[key], ('Common training exposure or recipe changed', key)
    logs_path = runpath.parent/'TRAIN.jsonl'
    assert sha(logs_path) == run['train_log_sha256']
    logs = [json.loads(line) for line in logs_path.read_text().splitlines()]
    assert len(logs) == 512 and [row['step'] for row in logs] == list(range(1,513))
    assert all(row['original_parameters_with_gradient'] == 0 and row['finite'] is True for row in logs)

    source_paths = {
        'train_source_sha256': base/'code/train_evidence.py',
        'module_source_sha256': base/'code/module.py',
        'model_source_sha256': Path(inf['model_source']),
        'label_source_sha256': q/'supervision_alignment_control_20261012/code/gaussian_labels.py',
        'top1_loss_source_sha256': q/'top1_background_control_20261011/code/losses.py',
        'common_loss_source_sha256': q/'mutual_temporal_pilot_20261012/code/train_mutual.py',
        'prior_train_source_sha256': q/'supervision_alignment_control_20261012/code/train_supervision.py',
        'batch_source_sha256': q/'top1_background_control_20261011/code/train_pilot.py',
    }
    sources = {str(p): sha(p) for p in [runpath, infpath, checkpoint_path, reference_checkpoint,
               freeze_path, base/'PROTOCOL.md', previous_path, logs_path]}
    for name, path in source_paths.items():
        actual = sha(path)
        assert cp[name] == run[name] == inf[name] == actual, name
        sources[str(path)] = actual
    script = base/'code/infer_evidence.py'
    assert sha(script) == inf['script_sha256']
    sources[str(script)] = sha(script)
    annotation_path = q/'stage1_goal/data'/sequence/'annotations.json'
    assert sha(annotation_path) == inf['annotations_sha256']
    sources[str(annotation_path)] = sha(annotation_path)
    return dict(identity_source_sha256=sources, checkpoint_sha256=cp_sha, training_variant=condition,
                checkpoint_step=512, main_endpoint=True, diagnostic='none',
                checkpoint_cpu_weights_only_metadata_checked=True,
                frozen_original_tensors_equal_release=True, common_training_exposure_checked=True)
