"""Read completed paired checkpoints once on CPU; never instantiate a model."""
from pathlib import Path
import hashlib
import json

import torch

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]
OUT = E / 'analysis/TRAINING_ARTIFACT_CHECK.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    freeze = read(E / 'PROTOCOL_FREEZE.json')
    for path, expected in freeze['source_sha256'].items():
        assert sha(path) == expected, path
    initial_path = ROOT / 'experiments/m1_recurrent_v1/training/History/run/snapshot_2048.pth'
    assert sha(initial_path) == freeze['checkpoint_sha256']
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    assert initial['step'] == 2048
    original_optimizer = initial['optimizer_state_dict']
    results, run_values, exposure_values = {}, {}, {}
    for condition in ('Continuation', 'Revision'):
        folder = E / 'training' / condition
        run = read(folder / 'RUN.json')
        assert run['status'] == 'COMPLETE'
        assert run['initial_step'] == 2048 and run['additional_steps'] == 128
        assert run['completed_additional_steps'] == 128 and run['final_step'] == 2176
        assert run['initial_checkpoint_sha256'] == freeze['checkpoint_sha256']
        assert run['source_sha256'] == freeze['source_sha256']
        assert run['protocol_sha256'] == sha(E / 'PROTOCOL_FREEZE.json')
        for path, expected in run['frozen_v1_source_sha256'].items():
            assert sha(path) == expected, path
        rows = [json.loads(line) for line in (folder / 'TRAIN.jsonl').read_text().splitlines()]
        assert len(rows) == 128
        for i, row in enumerate(rows):
            assert (row['step'], row['additional_step'], row['plan_index']) == (2049+i, 1+i, i)
        exposure = [(row['input_sha256'], row['label_sha256'], row['auxiliary_input_sha256']) for row in rows]
        digest = hashlib.sha256(''.join(''.join(row) for row in exposure).encode()).hexdigest()
        assert digest == run['exposure_digest_sha256']
        checkpoint_path = folder / 'snapshot_2176.pth'
        assert sha(checkpoint_path) == run['checkpoint_sha256']
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert checkpoint['step'] == 2176 and checkpoint['condition'] == condition
        assert checkpoint['source_sha256'] == freeze['source_sha256']
        state, initial_state = checkpoint['model_state_dict'], initial['model_state_dict']
        assert list(state) == list(initial_state)
        assert all(state[k].shape == initial_state[k].shape and state[k].dtype == initial_state[k].dtype for k in state)
        assert all(torch.isfinite(v).all() for v in state.values())
        frozen_keys = [key for key in state if key.startswith('base.')]
        assert frozen_keys and all(torch.equal(state[k], initial_state[k]) for k in frozen_keys)
        optimizer = checkpoint['optimizer_state_dict']
        assert optimizer['param_groups'] == original_optimizer['param_groups']
        assert list(optimizer['state']) == list(original_optimizer['state'])
        names = run['trainable_parameter_names']
        ids = [pid for group in optimizer['param_groups'] for pid in group['params']]
        assert len(ids) == len(names) == len(optimizer['state']) == 53
        steps = []
        for name, pid in zip(names, ids):
            values = optimizer['state'][pid]
            steps.append(float(values['step']))
            assert steps[-1] == 2176
            assert values['exp_avg'].shape == values['exp_avg_sq'].shape == state[name].shape
            assert bool(torch.isfinite(values['exp_avg']).all()) and bool(torch.isfinite(values['exp_avg_sq']).all())
        changed = [name for name in names if not torch.equal(state[name], initial_state[name])]
        assert changed
        run_values[condition], exposure_values[condition] = run, exposure
        results[condition] = {
            'run_sha256': sha(folder / 'RUN.json'), 'checkpoint_sha256': sha(checkpoint_path),
            'logged_steps': len(rows), 'first_step': rows[0]['step'], 'last_step': rows[-1]['step'],
            'optimizer_state_count': len(steps), 'optimizer_steps_unique': sorted(set(steps)),
            'optimizer_param_groups_unchanged': True,
            'optimizer_moment_shapes_match_ordered_parameter_names': True,
            'frozen_base_tensor_count': len(frozen_keys), 'all_frozen_base_tensors_exact_initial': True,
            'all_model_and_moment_tensors_finite': True, 'changed_trainable_tensor_count': len(changed),
            'exposure_digest_sha256': digest,
        }
        del checkpoint, state, optimizer
    shared = ('initial_checkpoint_sha256', 'initial_model_state_sha256', 'original_state_sha256',
              'source_sha256', 'frozen_v1_source_sha256', 'settings', 'cache_sha256',
              'trainable_parameter_names', 'trainable_parameters', 'parameters', 'exposure_digest_sha256')
    assert all(run_values['Continuation'][key] == run_values['Revision'][key] for key in shared)
    assert exposure_values['Continuation'] == exposure_values['Revision']
    result = {
        'status': 'PASS', 'scope': 'One read-only CPU checkpoint verification; no model/forward/optimizer step',
        'source_sha256': sha(__file__), 'protocol_sha256': sha(E / 'PROTOCOL_FREEZE.json'),
        'identical_initial_checkpoint': True, 'identical_128_ordered_exposures': True,
        'identical_RUN_fields': list(shared), 'conditions': results,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
