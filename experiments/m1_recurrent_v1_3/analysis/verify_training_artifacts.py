"""Read completed checkpoint tensors to verify the unchanged single-group Adam contract."""
from pathlib import Path
import hashlib
import json
import torch

E = Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def main():
    out = E/'analysis/TRAINING_ARTIFACT_CHECK.json'
    assert not out.exists()
    original_path = E.parent/'m1_recurrent_v1/training/History/run/snapshot_2048.pth'
    old = torch.load(original_path, map_location='cpu', weights_only=True)
    original_state = old['model_state_dict']
    initial = old
    freeze = read(E/'PROTOCOL_FREEZE.json')
    assert all(sha(path) == expected for path, expected in freeze['source_sha256'].items())
    identities, exposure = {}, {}
    for arm in ('BankCurrent', 'BankHistory'):
        folder = E/'training'/arm
        run = read(folder/'RUN.json')
        assert run['status'] == 'COMPLETE' and run['final_step'] == 2560
        log = [json.loads(line) for line in (folder/'TRAIN.jsonl').read_text().splitlines()]
        assert len(log) == 512
        assert all(row['plan_index'] == i % 256 and row['step'] == 2049+i for i,row in enumerate(log))
        exposure[arm] = [(row['plan_index'], row['input_sha256'], row['label_sha256'],
                          row['auxiliary_input_sha256']) for row in log]
        steps = []
        for extra in (0, 128, 256, 512):
            path = folder/f'snapshot_{2048+extra}.pth'
            metadata = next(item for item in run['snapshots'] if item['additional_step'] == extra)
            assert sha(path) == metadata['sha256']
            cp = torch.load(path, map_location='cpu', weights_only=True)
            assert cp['step'] == 2048+extra and cp['condition'] == arm
            assert cp['optimizer_parameter_names'] == [run['trainable_parameter_names']]
            assert all(torch.isfinite(value).all() for value in cp['model_state_dict'].values())
            assert all(torch.equal(value, cp['model_state_dict'][key]) for key, value in original_state.items()
                       if key.startswith('base.'))
            if extra == 0:
                assert all(torch.equal(value, cp['model_state_dict'][key])
                           for key, value in initial['model_state_dict'].items())
            opt = cp['optimizer_state_dict']
            assert [len(group['params']) for group in opt['param_groups']] == [53]
            observed = []
            for group_number, group in enumerate(opt['param_groups']):
                expected_step = 2048+extra
                assert group['lr'] == 1e-5 and group['weight_decay'] == 1e-4
                assert tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8
                group_steps = []
                for identifier in group['params']:
                    if expected_step == 0:
                        assert identifier not in opt['state']
                        continue
                    state = opt['state'][identifier]
                    assert float(state['step']) == expected_step
                    assert all(torch.isfinite(value).all() for value in state.values() if torch.is_tensor(value))
                    group_steps.append(float(state['step']))
                    if extra == 0 and group_number == 0:
                        assert all(torch.equal(value, old['optimizer_state_dict']['state'][identifier][key])
                                   for key, value in state.items() if torch.is_tensor(value))
                observed.append(sorted(set(group_steps)))
            steps.append({'additional_step': extra, 'old_steps': observed[0], 'new_steps': [],
                          'checkpoint_sha256': sha(path)})
        identities[arm] = {'snapshots': steps, 'original_base_tensors_exact': True,
            'initial_weights_and_old_moments_exact': True, 'all_parameters_and_moments_finite': True,
            'run_sha256': sha(folder/'RUN.json'), 'training_log_sha256': sha(folder/'TRAIN.jsonl')}
    assert exposure['BankCurrent'] == exposure['BankHistory']
    result = {'status': 'PASS', 'steps_each': 512, 'new_parameters': 0,
        'arms': identities, 'ordered_exposure_exact': True,
        'initial_checkpoint_sha256': sha(original_path), 'shared_initialization_sha256': sha(original_path),
        'protocol_sha256': sha(E/'PROTOCOL_FREEZE.json'), 'verification_source_sha256': sha(__file__)}
    out.write_text(json.dumps(result, indent=2)+'\n')
    print({'status': result['status'], 'steps_each': 512, 'old_final_step': 2560, 'new_parameter_tensors': 0}, flush=True)


if __name__ == '__main__': main()
