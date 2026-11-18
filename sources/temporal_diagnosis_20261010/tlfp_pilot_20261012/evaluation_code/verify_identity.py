"""Bind unchanged matching to the fixed S/T endpoint and unconditional T diagnostics."""
import hashlib
import json
from pathlib import Path

MODES = {'S': ('S', 'spatial'), 'T': ('T', 'temporal'),
         'T_current': ('T', 'current_only'), 'T_bypass': ('T', 'bypass')}
RELEASE_SHA = '4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_hash(items):
    h = hashlib.sha256()
    for name, value in sorted(items):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def read_run(base, variant):
    path = base/'training'/variant/'run/RUN.json'
    run = json.loads(path.read_text())
    assert run['status'] == 'complete' and run['condition'] == variant
    assert run['steps'] == run['final_step'] == 512 and run['seed'] == 46
    assert run['lr'] == run['weight_decay'] == 1e-4
    assert run['original_initial_state_sha256'] == run['original_final_state_sha256']
    assert run['frozen_original_unchanged'] is True
    assert run['new_parameters'] == (100 if variant == 'S' else 296)
    assert run['parameter_count'] == 70913 + run['new_parameters']
    assert run['checkpoint_sha256'] == RELEASE_SHA
    assert [row['step'] for row in run['snapshots']] == [0, 64, 128, 256, 512]
    return path, run


def verify_identity(base, condition, sequence, predictions_root):
    import numpy as np
    import torch
    base = Path(base); q = base.parent
    variant, diagnostic = MODES[condition]
    runpath, run = read_run(base, variant)
    infpath = Path(predictions_root)/sequence/'COMPLETE.json'
    inf = json.loads(infpath.read_text())
    freeze_path = base/'PROTOCOL_FREEZE.json'
    freeze = json.loads(freeze_path.read_text())
    assert sha(base/'PROTOCOL.md') == freeze['protocol_sha256']
    for path, expected in freeze['code_sha256'].items():
        actual = Path(path) if Path(path).is_absolute() else base/path
        assert sha(actual) == expected, ('Frozen code changed', path)
    assert inf['status'] == 'COMPLETE' and inf['sequence'] == sequence
    assert inf['condition'] == condition and inf['training_variant'] == variant
    assert inf['diagnostic_mode'] == diagnostic
    assert inf['checkpoint_step'] == 512
    assert inf['normalization'] == [85.0799, 47.4845]
    assert inf['official_starts'] == [0,36,72,108,120]
    assert inf['symmetric_starts'] == [0,24,60,96,120]
    assert inf['diagnostic_starts'] == [0,24,36,60,72,96,108,120]
    assert inf['model_labels_access'] is False
    cp_path = Path(inf['checkpoint'])
    assert cp_path.resolve().parent == runpath.parent.resolve()
    assert cp_path.name in ['snapshot_512.pth', 'snapshot_0512.pth', 'final_model.pth']
    snapshot = next(row for row in run['snapshots'] if row['step'] == 512)
    cp_sha = sha(cp_path)
    assert snapshot['sha256'] == run['final_checkpoint_sha256'] == inf['checkpoint_sha256'] == cp_sha
    cp = torch.load(cp_path, map_location='cpu', weights_only=True)
    assert cp['step'] == 512 and cp['condition'] == variant
    assert cp['initial_checkpoint_sha256'] == RELEASE_SHA
    assert cp['plan_sha256'] == run['plan_sha256']
    assert cp['original_state_sha256'] == run['original_initial_state_sha256']
    reference = Path(run['checkpoint_path'])
    assert sha(reference) == RELEASE_SHA
    numpy_core = np._core if hasattr(np, '_core') else np.core
    safe = [(numpy_core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(safe):
        release = torch.load(reference, map_location='cpu', weights_only=True)['model_state_dict']
    state = cp['model_state_dict']
    original = {key[len('original.'):]: value for key, value in state.items() if key.startswith('original.')}
    assert original.keys() == release.keys()
    assert all(torch.equal(original[key], release[key]) for key in release), 'Frozen original changed'
    assert tensor_hash(original.items()) == run['original_initial_state_sha256']
    assert tensor_hash(state.items()) == run['final_model_state_sha256']
    logs_path = runpath.with_name('TRAIN.jsonl')
    assert sha(logs_path) == run['train_log_sha256']
    logs = [json.loads(line) for line in logs_path.read_text().splitlines()]
    assert len(logs) == 512 and [row['step'] for row in logs] == list(range(1,513))
    assert all(row['finite'] is True and row['original_parameters_with_gradient'] == 0 for row in logs)
    chains = {}
    for name in ['input', 'label']:
        h = hashlib.sha256()
        for row in logs: h.update(row[name+'_sha256'].encode())
        chains[name] = h.hexdigest()
        assert chains[name] == run['cumulative_'+name+'_sha256']
    previous_path = q/'mutual_temporal_pilot_20261012/training/E0/run/RUN.json'
    previous = json.loads(previous_path.read_text())
    for key in ['checkpoint_sha256', 'plan_sha256', 'selection_sha256',
                'cumulative_input_sha256', 'cumulative_label_sha256', 'training_sequences',
                'label_config', 'effective_supervision_config']:
        assert run[key] == previous[key], ('Prior fixed exposure or supervision changed', key)
    # A diagnostic is a change of guide access in the same T checkpoint, never retraining.
    peers = []
    if variant == 'T':
        for name in ['T', 'T_current', 'T_bypass']:
            peer_path = base/'predictions'/name/sequence/'COMPLETE.json'
            assert peer_path.exists(), ('Missing mandatory same-checkpoint diagnostic', name, sequence)
            peer = json.loads(peer_path.read_text())
            assert peer['checkpoint_sha256'] == cp_sha and Path(peer['checkpoint']).resolve() == cp_path.resolve()
            assert peer['training_variant'] == 'T' and peer['checkpoint_step'] == 512
            assert peer['diagnostic_mode'] == MODES[name][1]
            peers.append(peer_path)
    annotation = q/'stage1_goal/data'/sequence/'annotations.json'
    raw = annotation.with_name('raw_frames.npy')
    assert sha(annotation) == inf['annotations_sha256'] and sha(raw) == inf['raw_sha256']
    sources = {str(p): sha(p) for p in [runpath, infpath, cp_path, reference, logs_path,
               freeze_path, base/'PROTOCOL.md', previous_path, annotation, raw, *peers]}
    for key in ['source_sha256']:
        if key in run:
            for path, expected in run[key].items():
                assert sha(path) == expected, ('Recorded implementation changed', path)
                sources[path] = expected
    return dict(identity_source_sha256=sources, checkpoint_sha256=cp_sha,
                training_variant=variant, checkpoint_step=512, main_endpoint=True,
                diagnostic=diagnostic, checkpoint_cpu_weights_only_metadata_checked=True,
                frozen_original_tensors_equal_release=True, common_training_exposure_checked=True,
                same_T_checkpoint_for_all_diagnostics=variant == 'T')


def verify_training_comparability(base):
    base = Path(base)
    paths, runs = {}, {}
    for variant in ['S', 'T']:
        paths[variant], runs[variant] = read_run(base, variant)
    shared = ['seed', 'steps', 'final_step', 'lr', 'weight_decay', 'optimizer', 'precision',
              'original_initial_state_sha256', 'original_final_state_sha256', 'plan_sha256',
              'selection_sha256', 'cumulative_input_sha256', 'cumulative_label_sha256',
              'training_sequences', 'label_config', 'effective_supervision_config']
    for key in shared:
        assert runs['S'][key] == runs['T'][key], ('Unequal controlled training', key)
    streams = {}
    for variant in ['S', 'T']:
        logs = [json.loads(line) for line in paths[variant].with_name('TRAIN.jsonl').read_text().splitlines()]
        streams[variant] = [(row['input_sha256'], row['label_sha256']) for row in logs]
    assert len(streams['S']) == 512 and streams['S'] == streams['T']
    return dict(status='PASS', conditions=['S','T'], shared_fields=shared,
                shared_values={key:runs['S'][key] for key in shared}, seed=46, steps=512,
                source_sha256={str(path):sha(path) for path in paths.values()},
                labels_and_inputs_equal_every_step=True,
                parameter_counts={variant:runs[variant]['new_parameters'] for variant in runs},
                inference_diagnostics=['T_current','T_bypass'],
                scope='S and T share release, original frozen state, exposure, labels and update budget; adapters have 100/296 parameters. Common initialization mapping checked separately, not asserted from unequal state shapes.')
