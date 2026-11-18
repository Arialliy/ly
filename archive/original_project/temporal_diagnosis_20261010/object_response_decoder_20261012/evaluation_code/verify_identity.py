"""CPU-only identity audit for the NEW 2048-step direct-response experiment.

The historical recipe is used only to verify input exposure. New labels/losses
are bound to this protocol and compared among R0/R1/R2, never to old weak masks.
"""
import hashlib
import json
from pathlib import Path

MODES = {'R0': (False, 'frame'), 'R1': (True, 'frame'), 'R2': (True, 'track')}
SOURCE_PLAN_SHA = '6f86d929ad1275bbb120456dd23a25c7ea9571517f60b4c64a9d5ce6413c975c'
SELECTION_SHA = '1f5e617764c0fe27a63440cc5b26d0e27a05db7dbc8c997b0bc5fd74a4b39c92'
RELEASE_SHA = '4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc'


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
    base = Path(base)
    q = base.parent
    runpath = base/'training'/condition/'run/RUN.json'
    infpath = Path(predictions_root)/sequence/'COMPLETE.json'
    run = json.loads(runpath.read_text())
    inf = json.loads(infpath.read_text())
    freeze_path = base/'PROTOCOL_FREEZE.json'
    freeze = json.loads(freeze_path.read_text())
    assert sha(base/'PROTOCOL.md') == freeze['protocol_sha256']
    for path, expected in freeze['code_sha256'].items():
        assert sha(base/path) == expected, ('Frozen code changed', path)
    temporal, assignment = MODES[condition]
    assert run['status'] == 'complete' and run['condition'] == run['variant'] == condition
    assert run['temporal_enabled'] is temporal and run['assignment'] == assignment
    assert run['steps'] == run['final_step'] == 2048 and run['seed'] == 46
    assert run['lr'] == 1e-4 and run['weight_decay'] == 1e-4 and run['optimizer'] == 'Adam'
    assert run['parameter_count'] == 102107 and run['new_parameters'] == 31194
    assert run['precision'] == 'FP32_no_AMP_no_TF32'
    assert run['microbatches'] == run['repeat_source_plan'] == 4
    assert run['loss_weights'] == [1, 1, 1, 1, 1]
    assert run['snapshots_selection'] == 'fixed2048only'
    assert [row['step'] for row in run['snapshots']] == [0, 512, 1024, 2048]
    assert run['bn_policy'] == 'entire_original_frozen_eval'
    assert run['original_initial_state_sha256'] == run['original_final_state_sha256']
    assert run['frozen_original_unchanged'] is True and run['model_labels_access'] is False
    assert run['source_plan_sha256'] == SOURCE_PLAN_SHA
    assert run['selection_sha256'] == SELECTION_SHA
    assert run['checkpoint_sha256'] == RELEASE_SHA
    assert inf['status'] == 'COMPLETE' and inf['sequence'] == sequence and inf['condition'] == condition
    assert inf['checkpoint_step'] == inf['main_checkpoint_step'] == 2048 and inf['checkpoint_is_main'] is True
    assert inf['model_labels_access'] is False and inf['global_queries'] is True
    assert inf['seed'] == 46 and inf['normalization'] == [85.0799, 47.4845]
    assert inf['tile_core'] == 128 and inf['tile_halo'] == 16
    assert inf['official_starts'] == [0, 36, 72, 108, 120]
    assert inf['symmetric_starts'] == [0, 24, 60, 96, 120]
    assert inf['diagnostic_starts'] == [0, 24, 36, 60, 72, 96, 108, 120]
    assert inf['parameters'] == run['parameter_count'] and inf['new_parameters'] == run['new_parameters']
    for key in ['original_initial_state_sha256', 'original_final_state_sha256']:
        assert inf[key] == run[key], key
    if 'tiling_checks' in inf:
        assert inf['tiling_checks'] and all(row['status'] == 'passed' and row['threshold05_disagreements'] == 0
                                            for row in inf['tiling_checks'])

    checkpoint_path = Path(inf['checkpoint'])
    assert checkpoint_path.parent.resolve() == runpath.parent.resolve()
    assert checkpoint_path.name in ['snapshot_2048.pth', 'final_model.pth']
    snapshot = next(row for row in run['snapshots'] if row['step'] == 2048)
    cp_sha = sha(checkpoint_path)
    assert snapshot['sha256'] == run['final_checkpoint_sha256'] == inf['checkpoint_sha256'] == cp_sha
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert cp['condition'] == condition and cp['step'] == 2048
    assert cp['initial_checkpoint_sha256'] == run['checkpoint_sha256'] == inf['reference_checkpoint_sha256']
    assert cp['plan_sha256'] == run['plan_sha256'] == sha(base/'TRAIN_PLAN.json')
    if 'plan_sha256' in inf:
        assert inf['plan_sha256'] == run['plan_sha256']
    if 'selection_sha256' in inf:
        assert inf['selection_sha256'] == run['selection_sha256']
    assert cp['model_config'] == run['model_config'] == inf['model_config']
    assert cp['original_state_sha256'] == run['original_initial_state_sha256']
    config = cp['model_config']
    assert config['original_frozen'] is True and config['original_bn'] == 'eval'
    assert config['d_model'] == 32 and config['num_queries'] == 100 and config['num_heads'] == 4
    assert config['decoder_layers'] == 2 and config['ffn_hidden'] == 64 and config['dropout'] == 0
    assert config['token_pool'] == 'average' and config['token_stride'] == 4
    assert config['anchor_grid'] == [10, 10]

    # Check actual tensors, including all BN buffers, against released weights.
    reference_checkpoint = Path(run['checkpoint_path'])
    assert sha(reference_checkpoint) == RELEASE_SHA
    numpy_core = np._core if hasattr(np, '_core') else np.core
    safe = [(numpy_core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(safe):
        release = torch.load(reference_checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
    state = cp['model_state_dict']
    original = {key[len('original.'):]: value for key, value in state.items() if key.startswith('original.')}
    assert original.keys() == release.keys()
    assert all(torch.equal(original[key], release[key]) for key in release)
    assert tensor_hash(original.items()) == run['original_initial_state_sha256']
    assert tensor_hash(state.items()) == run['final_model_state_sha256']
    prefixes = ('original.', 'raw_encoder.', 'project.', 'layers.', 'presence_head.', 'mask_head.')
    assert all(key in ('query_embed', 'anchors_yx') or key.startswith(prefixes) for key in state)

    initial_path = runpath.parent/'snapshot_0000.pth'
    initial_receipt = json.loads((runpath.parent/'INITIAL_STATE.json').read_text())
    assert sha(initial_path) == run['snapshots'][0]['sha256']
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    assert initial['step'] == 0 and initial['condition'] == condition
    initial_state = initial['model_state_dict']
    assert initial_state.keys() == state.keys()
    assert tensor_hash(initial_state.items()) == run['initial_model_state_sha256'] == initial_receipt['model_state_sha256']
    assert initial_receipt['original_state_sha256'] == run['original_initial_state_sha256']
    assert all(tensor_hash([(key, value)]) == initial_receipt['tensor_sha256'][key] for key, value in initial_state.items())
    trainable = initial_receipt['trainable_names']
    assert len(trainable) == len(set(trainable)) and all(not key.startswith('original.') for key in trainable)
    assert sum(state[key].numel() for key in trainable) == 31194
    for key in set(state) - set(trainable):
        assert torch.equal(state[key], initial_state[key]), ('Nontrainable tensor changed', key)
    if condition == 'R0':
        for key in trainable:
            if '.temporal_' in key:
                assert torch.equal(state[key], initial_state[key]), ('Disabled temporal parameter changed', key)
    groups = cp['optimizer_state_dict']['param_groups']
    assert len(groups) == 1 and len(groups[0]['params']) == len(trainable)
    assert groups[0]['lr'] == 1e-4 and groups[0]['weight_decay'] == 1e-4
    assert list(groups[0]['betas']) == [0.9, 0.999] and groups[0]['eps'] == 1e-8

    # Every input exposure equals the corresponding historical entry; new labels
    # repeat identically over four cycles and agree between completed new runs.
    old_logs_path = q/'mutual_temporal_pilot_20261012/training/E0/run/TRAIN.jsonl'
    old_logs = [json.loads(line) for line in old_logs_path.read_text().splitlines()]
    assert len(old_logs) == 512
    logs_path = runpath.parent/'TRAIN.jsonl'
    assert sha(logs_path) == run['train_log_sha256']
    logs = [json.loads(line) for line in logs_path.read_text().splitlines()]
    assert len(logs) == 2048 and [row['step'] for row in logs] == list(range(1, 2049))
    input_chain, label_chain = hashlib.sha256(), hashlib.sha256()
    for i, row in enumerate(logs):
        assert row['source_entry'] == i % 512 and row['cycle'] == i // 512
        assert row['condition'] == condition and row['finite'] is True and row['original_parameters_with_gradient'] == 0
        assert row['input_sha256'] == old_logs[i % 512]['input_sha256']
        assert row['label_sha256'] == logs[i % 512]['label_sha256']
        input_chain.update(row['input_sha256'].encode())
        label_chain.update(row['label_sha256'].encode())
    assert input_chain.hexdigest() == run['cumulative_input_sha256']
    assert label_chain.hexdigest() == run['cumulative_label_sha256']
    completed_peers = []
    for peer in MODES:
        peer_path = base/'training'/peer/'run/RUN.json'
        if not peer_path.exists():
            continue
        peer_run = json.loads(peer_path.read_text())
        if peer_run['status'] != 'complete':
            continue
        assert peer_run['initial_model_state_sha256'] == run['initial_model_state_sha256']
        peer_logs = [json.loads(line) for line in peer_path.with_name('TRAIN.jsonl').read_text().splitlines()]
        assert [row['label_sha256'] for row in peer_logs] == [row['label_sha256'] for row in logs]
        completed_peers.append(peer)

    sources = {str(p): sha(p) for p in [runpath, infpath, checkpoint_path, reference_checkpoint,
               initial_path, runpath.parent/'INITIAL_STATE.json', freeze_path, base/'PROTOCOL.md',
               base/'TRAIN_PLAN.json', q/'stage1_goal/pilot/TRAIN_PLAN.json',
               q/'stage1_goal/DATA_SELECTION.json', old_logs_path, logs_path]}
    assert sha(q/'stage1_goal/pilot/TRAIN_PLAN.json') == SOURCE_PLAN_SHA
    assert sha(q/'stage1_goal/DATA_SELECTION.json') == SELECTION_SHA
    assert cp['source_sha256'] == run['source_sha256'] == inf['source_sha256']
    for path, expected in run['source_sha256'].items():
        assert sha(path) == expected, ('Experiment source changed', path)
        sources[str(path)] = expected
    script = base/'code/infer.py'
    assert sha(script) == inf['script_sha256']
    sources[str(script)] = sha(script)
    annotation_path = q/'stage1_goal/data'/sequence/'annotations.json'
    raw_path = annotation_path.with_name('raw_frames.npy')
    assert sha(annotation_path) == inf['annotations_sha256']
    assert sha(raw_path) == inf['raw_sha256']
    sources[str(annotation_path)] = sha(annotation_path)
    sources[str(raw_path)] = sha(raw_path)
    return dict(identity_source_sha256=sources, checkpoint_sha256=cp_sha, training_variant=condition,
                checkpoint_step=2048, main_endpoint=True, diagnostic='none',
                checkpoint_cpu_weights_only_metadata_checked=True,
                frozen_original_tensors_equal_release=True, common_training_exposure_checked=True,
                global_queries_checked=True, new_label_equality_completed_peers=completed_peers,
                new_supervision_not_required_to_match_historical_labels=True)
