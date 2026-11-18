#!/usr/bin/env python3
"""CPU integrity audit of fixed C2048 -> global2304 paired continuations."""
import argparse
import hashlib
import json
import math
import time

import numpy as np
import torch

import runtime as rt
import evaluate as evaluation

CONDITIONS = ['CONT', 'FRAME', 'GAP']
EXPOSURE_KEYS = ['input_sha256', 'label_sha256', 'auxiliary_input_sha256']


def require(value, message):
    if not value:
        raise AssertionError(message)


def optimizer_hash(state):
    tensors, scalars = {}, {}
    for key, values in state['state'].items():
        for name, value in values.items():
            if torch.is_tensor(value):
                tensors[f'{key}/{name}'] = value
            else:
                scalars[f'{key}/{name}'] = value
    encoded = json.dumps({'groups': state['param_groups'], 'scalars': scalars}, sort_keys=True)
    return hashlib.sha256((rt.tensor_hash(tensors) + encoded).encode()).hexdigest()


def exact(left, right, label):
    if torch.is_tensor(left):
        require(torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.cpu(), right.cpu()), label)
    elif isinstance(left, dict):
        require(isinstance(right, dict) and left.keys() == right.keys(), label + ' keys')
        for key in left:
            exact(left[key], right[key], label + '/' + str(key))
    elif isinstance(left, (tuple, list)):
        require(type(left) is type(right) and len(left) == len(right), label + ' sequence')
        for index, (a, b) in enumerate(zip(left, right)):
            exact(a, b, label + '/' + str(index))
    else:
        require(type(left) is type(right) and left == right, label)


def adam_steps(state, expected_step):
    require(len(state['state']) == 28, 'Adam must contain exactly 28 parameter states')
    parameters = [p for group in state['param_groups'] for p in group['params']]
    require(len(parameters) == len(set(parameters)) == 28 and set(parameters) == set(state['state']),
            'Adam state/group parameter coverage differs')
    for index, values in state['state'].items():
        require(float(values['step']) == expected_step, f'Adam step differs: {index}')
        require('exp_avg' in values and 'exp_avg_sq' in values, 'Adam moments absent')
        require(values['exp_avg'].shape == values['exp_avg_sq'].shape, 'Adam moment shape differs')
        require(bool(torch.isfinite(values['exp_avg']).all()) and bool(torch.isfinite(values['exp_avg_sq']).all()),
                'Nonfinite Adam moments')
    for group in state['param_groups']:
        require(group['lr'] == 1e-5 and group['weight_decay'] == 1e-4
                and tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8,
                'Adam hyperparameters differ')


def baseline_check():
    result, source = [], {}
    for condition in ['release', 'C_init']:
        pooled = np.zeros((2, 2), dtype=np.int64)
        for sequence in rt.SEQS:
            scores, areas, receipt, hashes = evaluation.read_record(condition, sequence)
            source.update(hashes)
            reference = (evaluation.OLD / 'evaluation/release' / sequence if condition == 'release'
                         else rt.V1 / 'evaluation/scored/C' / sequence)
            for name, current in [('scores.npz', scores), ('component_areas.npz', areas)]:
                with np.load(reference / name, allow_pickle=False) as original:
                    require(set(current) == set(original.files), 'Baseline array key set differs')
                    for key in original.files:
                        equal = (np.array_equal(current[key], original[key], equal_nan=True)
                                 if current[key].dtype.kind not in ('U', 'S') else np.array_equal(current[key], original[key]))
                        require(equal, f'Baseline array differs: {condition}/{sequence}/{key}')
                source[str(reference / name)] = rt.sha(reference / name)
            small = scores['strict_small'].astype(bool)
            require(int(small.sum()) > 0, 'Empty strict population')
            pooled[:, 0] += scores['matched'][:, small].sum(axis=1).astype(np.int64)
            pooled[:, 1] += scores['frame_counts'][:, :, 3].sum(axis=1)
            require(receipt['threshold'] == .5, 'Baseline threshold differs')
        expected = [[700, 4015], [720, 3967]] if condition == 'release' else [[707, 4053], [724, 4002]]
        require(pooled.tolist() == expected, 'Baseline pooled M/U differs')
        result.append(dict(condition=condition, O=dict(M=int(pooled[0, 0]), U=int(pooled[0, 1])),
                           S=dict(M=int(pooled[1, 0]), U=int(pooled[1, 1])), all_score_and_component_arrays_exact=True))
    return dict(status='PASS', checks=result, source_sha256=source,
                scope='Read-only CPU exact comparison of existing release/C2048 evaluations; no model execution.')


def validate():
    freeze_path = rt.E / 'PROTOCOL_FREEZE.json'
    freeze = rt.read_json(freeze_path)
    protocol_hash = rt.sha(freeze_path)
    require(freeze['conditions'] == CONDITIONS and freeze['start_step'] == 2048
            and freeze['additional_steps'] == 256 and freeze['main_step'] == 2304, 'Frozen run endpoints differ')
    for name, expected in freeze['files'].items():
        require(rt.sha(rt.E / name) == expected, 'Frozen source differs: ' + name)
    for path, expected in freeze['training_sources'].items():
        require(rt.sha(path) == expected, 'Frozen training source differs: ' + path)
    require(rt.sha(rt.INIT_CP) == freeze['initial_checkpoint_sha256'], 'Initial C checkpoint changed')
    require(rt.sha(rt.CP) == freeze['release_checkpoint_sha256'], 'Release checkpoint changed')
    require(rt.sha(rt.V1 / 'PROTOCOL_FREEZE.json') == freeze['initial_v1_protocol_sha256'], 'v1 protocol changed')
    for path, expected in rt.read_json(rt.V1 / 'PROTOCOL_FREEZE.json')['training_sources'].items():
        require(rt.sha(path) == expected, 'v1 training implementation changed')
    expected_exposures = []
    for index in range(256):
        for root, key in [(rt.E, 'cache_sha256'), (rt.V1, 'v1_cache_sha256')]:
            require(rt.sha(root / 'cache' / f'entry_{index:03d}.npz') == freeze[key][str(index)], 'Cache changed')
        fresh = rt.read_json(rt.E / 'cache' / f'entry_{index:03d}.json')
        for path, expected in fresh['sources'].items():
            require(rt.sha(path) == expected, 'Risk-cache source changed: ' + path)
        with np.load(rt.V1 / 'cache' / f'entry_{index:03d}.npz', allow_pickle=False) as cache:
            previous = json.loads(str(cache['metadata_json']))
        require(all(fresh[k] == previous[k] for k in EXPOSURE_KEYS), 'Risk cache changes original C exposure')
        expected_exposures.append(tuple(previous[k] for k in EXPOSURE_KEYS))
    initial_cp = torch.load(rt.INIT_CP, map_location='cpu', weights_only=True)
    initial_model = initial_cp['model_state_dict']
    initial_optimizer = initial_cp['optimizer_state_dict']
    adam_steps(initial_optimizer, 2048)
    initial_model_hash = rt.tensor_hash(initial_model)
    initial_opt_hash = optimizer_hash(initial_optimizer)
    train_names = initial_cp['trainable_parameter_names']
    require(len(train_names) == 28 and all(n.startswith(rt.TRAIN_PREFIXES) for n in train_names), 'Trainable names differ')
    require(sum(initial_model[n].numel() for n in train_names) == 14593, 'Trainable parameter count differs')
    frozen = {k: v for k, v in initial_model.items() if k not in train_names}
    bn = {k: v for k, v in initial_model.items() if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    release, _ = rt.load_model(device='cpu')
    exact(frozen, rt.frozen_state(release), 'C_init/release frozen frontend and buffers')
    exact(bn, rt.bn_state(release), 'C_init/release BN')
    del release
    calibration = rt.read_json(rt.E / 'CALIBRATION.json')
    require(calibration['uses_development'] is False and calibration['no_parameter_updates'] is True,
            'Calibration access/update policy differs')
    require(calibration['initial_optimizer_sha256'] == initial_opt_hash, 'Calibration did not use initial optimizer identity')
    require(rt.sha(rt.E / 'CALIBRATION.json') == freeze['calibration_sha256'], 'Calibration changed')
    results, evidence = {}, {str(freeze_path): protocol_hash, str(rt.INIT_CP): rt.sha(rt.INIT_CP)}
    for condition in CONDITIONS:
        folder = rt.E / 'training' / condition / 'run'
        run = rt.read_json(folder / 'RUN.json')
        require(run['status'] == 'complete' and run['condition'] == condition and run['seed'] == 46,
                'Run incomplete or wrong identity')
        require(run['start_step'] == 2048 and run['final_step'] == 2304 and run['additional_steps'] == run['steps'] == 256,
                'Training update count differs')
        require(run['initial_optimizer_restored'] and run['initial_optimizer_sha256'] == initial_opt_hash
                and run['initial_state_sha256'] == initial_model_hash, 'Restoration receipt differs')
        require(run['initial_checkpoint_sha256'] == rt.sha(rt.INIT_CP) and run['release_checkpoint_sha256'] == rt.sha(rt.CP)
                and run['protocol_sha256'] == protocol_hash and run['model_source_sha256'] == rt.sha(rt.MODEL_SOURCE),
                'Run source identity differs')
        require(run['training_sources'] == freeze['training_sources'] and run['settings'] == calibration['fixed_settings'],
                'Run settings differ')
        require(run['frozen_frontend_unchanged'] and run['bn_unchanged'] and run['new_parameters'] == 0
                and run['trainable_parameters'] == 14593 and run['total_parameters'] == 70913
                and run['trainable_parameter_names'] == train_names, 'Run architecture/freeze receipt differs')
        require(run['frozen_state_sha256'] == rt.tensor_hash(frozen) and run['bn_initial_sha256'] == rt.tensor_hash(bn),
                'Run frozen-state hashes differ')
        require([r['step'] for r in run['snapshots']] == [2048, 2112, 2176, 2304], 'Snapshot schedule differs')
        for record in run['snapshots']:
            path = folder / record['path']
            require(rt.sha(path) == record['sha256'], 'Snapshot hash differs')
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            require(checkpoint['step'] == record['step'] and checkpoint['condition'] == condition
                    and checkpoint['protocol_sha256'] == protocol_hash, 'Snapshot identity differs')
            model = checkpoint['model_state_dict']
            require(model.keys() == initial_model.keys(), 'Model keys changed')
            exact(frozen, {k: model[k] for k in frozen}, 'Frozen frontend/buffers')
            exact(bn, {k: model[k] for k in bn}, 'BN state')
            adam_steps(checkpoint['optimizer_state_dict'], record['step'])
            exact(initial_optimizer['param_groups'], checkpoint['optimizer_state_dict']['param_groups'], 'Adam parameter groups')
            require(all(bool(torch.isfinite(v).all()) for v in model.values()), 'Nonfinite model state')
            if record['step'] == 2048:
                exact(initial_model, model, 'Restored model state')
                exact(initial_optimizer, checkpoint['optimizer_state_dict'], 'Restored complete Adam state')
            if record['step'] == 2304:
                require(rt.tensor_hash(model) == run['final_state_sha256'], 'Final model hash differs')
                require(optimizer_hash(checkpoint['optimizer_state_dict']) == run['final_optimizer_sha256'], 'Final optimizer hash differs')
        logs = [json.loads(line) for line in (folder / 'TRAIN.jsonl').read_text().splitlines()]
        require(len(logs) == 256, 'Training log length differs')
        for index, row in enumerate(logs):
            require(row['step'] == 2049 + index and row['additional_step'] == index + 1
                    and row['plan_index'] == index and row['condition'] == condition, 'Training schedule differs')
            require(tuple(row[k] for k in EXPOSURE_KEYS) == expected_exposures[index], 'Paired data exposure differs')
            require(math.isfinite(row['loss']) and math.isfinite(row['grad_norm'])
                    and all(math.isfinite(v) for v in row['loss_terms'].values()), 'Nonfinite training loss or gradient')
            if condition == 'CONT':
                require(row['loss'] == row['loss_terms']['base'], 'CONT adds an auxiliary risk loss')
        diagnostics = rt.read_json(folder / 'TRAIN_DIAGNOSTICS.json')
        require(diagnostics['uses_development'] is False and diagnostics['indices'] == calibration['calibration_entry_indices'],
                'Final training diagnostic inputs differ')
        pred_root = rt.E / 'predictions' / condition
        require({p.name for p in pred_root.iterdir() if p.is_dir()} == set(rt.SEQS), 'Unexpected prediction source directory')
        predictions = {}
        for sequence in rt.SEQS:
            receipt_path = pred_root / sequence / 'COMPLETE.json'
            prediction = rt.read_json(receipt_path)
            evaluation.inference_identity(condition, sequence)
            require(prediction['initial_model_state_sha256'] == prediction['final_model_state_sha256'] == run['final_state_sha256'],
                    'Inference state differs from final checkpoint')
            require(prediction['all_model_state_unchanged_during_inference'] and prediction['original_forward_unchanged']
                    and prediction['auxiliary_modules_at_inference'] is False and prediction['model_labels_access'] is False,
                    'Inference architecture/access policy differs')
            require(prediction['raw_sha256'] == rt.sha(rt.DATA / sequence / 'raw_frames.npy'), 'Inference raw frames changed')
            require(set(prediction['window_sha256']) == {f'window_{s:03d}.npz' for s in rt.STARTS}, 'Inference window set differs')
            for name, expected in prediction['window_sha256'].items():
                require(rt.sha(pred_root / sequence / 'windows' / name) == expected, 'Prediction window hash differs')
            require(rt.sha(pred_root / sequence / 'official_fused.npz') == prediction['fused_sha256'], 'Fused prediction hash differs')
            evaluation.read_record(condition, sequence)
            predictions[sequence] = rt.sha(receipt_path)
        results[condition] = dict(initial_model_exact=True, initial_complete_adam_exact=True,
            optimizer_parameter_states=28, optimizer_initial_step=2048, optimizer_final_step=2304,
            paired_training_updates=len(logs), frozen_all_saved_states=True, prediction_receipt_sha256=predictions,
            final_checkpoint_sha256=rt.sha(folder / 'snapshot_2304.pth'))
        evidence.update({str(folder / name): rt.sha(folder / name) for name in ['RUN.json', 'TRAIN.jsonl', 'TRAIN_DIAGNOSTICS.json']})
    summary_path = rt.E / 'evaluation/SUMMARY_COMPLETE.json'
    summary = rt.read_json(summary_path)
    require(summary['status'] == 'COMPLETE' and summary['conditions'] == ['release', 'C_init'] + CONDITIONS,
            'Five-condition summary incomplete')
    require(summary['fixed_main_step'] == 2304 and summary['strict_small_target_frames'] == 5200
            and summary['strict_small_tracks'] == 130 and summary['threshold'] == .5, 'Evaluation protocol differs')
    for path, expected in summary['source_sha256'].items():
        require(rt.sha(path) == expected, 'Summary source changed')
    for name, expected in summary['output_sha256'].items():
        require(rt.sha(rt.E / 'evaluation' / name) == expected, 'Evaluation table changed')
    baselines = baseline_check()
    rt.save_json(rt.E / 'VALIDATION.json', dict(status='PASS', created_unix=time.time(),
        protocol_sha256=protocol_hash, validator_sha256=rt.sha(__file__), runs=results,
        baseline_checks=baselines['checks'], initial_model_sha256=initial_model_hash,
        initial_optimizer_sha256=initial_opt_hash, evidence_sha256=evidence,
        evaluation_summary_sha256=rt.sha(summary_path),
        checks=['Complete Adam moments and parameter groups restored identically in all branches',
                '28 Adam states at 2048/2112/2176/2304; same 256 original input/label/auxiliary hashes',
                'Original model keys, frozen frontend and all BN buffers unchanged at all saved states',
                'Original inference only, exactly the two development sources, unchanged model state and prediction hashes',
                'Release/C_init baselines and final fixed five-condition evaluation verified'],
        scope='Execution and artifact integrity only; not evidence of detection improvement or validation of the research hypothesis.'))
    print('C-v2 final execution integrity: PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baselines-only', action='store_true')
    args = parser.parse_args()
    if args.baselines_only:
        result = baseline_check()
        result.update(validator_sha256=rt.sha(__file__), created_unix=time.time())
        rt.save_json(rt.E / 'evaluation/BASELINE_VALIDATION.json', result)
        print(json.dumps(dict(status=result['status'], checks=result['checks'])), flush=True)
    else:
        validate()
