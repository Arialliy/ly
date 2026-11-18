#!/usr/bin/env python3
"""Validate new D/N executions and cached-reference imports; no old experiment replay."""
import argparse
import hashlib
import json
import math
import time

import numpy as np
import torch

import runtime as rt
import evaluate as evaluation

CONDITIONS = ['D', 'N']
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


def references_check():
    """Only new alias/join wiring is checked; old matching/model execution is not repeated."""
    conditions = list(evaluation.REFERENCES)
    _, (records, areas, metadata, sources, frames, objects) = evaluation.collect(conditions)
    previous_path = rt.V2 / 'evaluation/SUMMARY.csv'
    previous = evaluation.read_csv(previous_path)
    prior_receipt = rt.read_json(rt.V2 / 'evaluation/SUMMARY_COMPLETE.json')
    require(rt.sha(previous_path) == prior_receipt['output_sha256']['SUMMARY.csv'], 'Old summary changed')
    sources[str(previous_path)] = rt.sha(previous_path)
    result = []
    for condition in conditions:
        for pi, plan in enumerate(rt.PLANS):
            selected = [r for r in objects if r['arrangement'] == plan]
            old = [r for r in previous if r['condition'] == evaluation.REFERENCES[condition]
                   and r['arrangement'] == plan and r['group'] == 'pooled']
            require(len(old) == 1, 'Missing reference summary row')
            old = old[0]
            counts = dict(M=sum(r[condition + '_matched'] for r in selected),
                low_all_05_matched=sum(r['low_all_05'] and r[condition + '_matched'] for r in selected),
                extreme_all_001_matched=sum(r['extreme_all_001'] and r[condition + '_matched'] for r in selected),
                **evaluation.background(records, areas, condition, pi, plan, rt.SEQS))
            for key, value in counts.items():
                require(math.isclose(float(old[key]), value, rel_tol=1e-12, abs_tol=1e-12),
                        f'Reference alias aggregate changed: {condition}/{plan}/{key}')
            require(len(selected) == 5200 and len({(r['sequence'], r['track_id']) for r in selected}) == 130,
                    'Reference cohort join differs')
            require(sum(r['low_all_05'] for r in selected) == 4342
                    and sum(r['extreme_all_001'] for r in selected) == 4170
                    and sum(r['both_fixed_success'] for r in selected) == 646, 'Reference cohorts differ')
            result.append(dict(condition=condition, original_condition=evaluation.REFERENCES[condition],
                               arrangement=plan, M=counts['M'], U=counts['unmatched_components']))
    require(len(frames) == 4 * 2 * 2 * 160 and {r['condition'] for r in frames} == set(conditions),
            'Aliased frame-count join differs')
    return dict(status='PASS', checks=result, source_sha256=sources,
        model_executions=0, matcher_executions=0,
        scope='Cached arrays/receipts and new reference alias/cohort/summary joins only; old audits reused.')


def validate():
    freeze_path = rt.E / 'PROTOCOL_FREEZE.json'
    freeze = rt.read_json(freeze_path)
    protocol_hash = rt.sha(freeze_path)
    require(freeze['conditions'] == CONDITIONS and freeze['start_step'] == 2048
            and freeze['additional_steps'] == 256 and freeze['main_step'] == 2304, 'Frozen endpoints differ')
    for name, expected in freeze['files'].items():
        require(rt.sha(rt.E / name) == expected, 'Frozen source differs: ' + name)
    for path, expected in freeze['training_sources'].items():
        require(rt.sha(path) == expected, 'Frozen training source differs: ' + path)
    require(rt.sha(rt.INIT_CP) == freeze['initial_checkpoint_sha256'], 'Initial checkpoint changed')
    require(rt.sha(rt.CP) == freeze['release_checkpoint_sha256'], 'Release checkpoint changed')
    require(rt.sha(rt.V2 / 'PROTOCOL_FREEZE.json') == freeze['initial_v2_protocol_sha256'], 'v2 protocol changed')
    previous_freeze = rt.read_json(rt.V2 / 'PROTOCOL_FREEZE.json')
    require(freeze['v1_cache_sha256'] == previous_freeze['v1_cache_sha256']
            and freeze['v2_cache_sha256'] == previous_freeze['cache_sha256'], 'Reused cache identity differs')
    previous_validation = rt.read_json(rt.V2 / 'VALIDATION.json')
    require(previous_validation['status'] == 'PASS', 'Prior execution validation absent')
    reuse_path = rt.E / 'resources/F_REUSE_AUDIT.json'
    reuse = rt.read_json(reuse_path)
    require(reuse['status'] == 'PASS' and reuse['all256exposure_hashes_match'], 'F reuse has not passed')
    for path, expected in reuse['reference_evidence_sha256'].items():
        require(rt.sha(path) == expected, 'F reference receipt changed: ' + path)
    initial_cp = torch.load(rt.INIT_CP, map_location='cpu', weights_only=True)
    initial_model, initial_optimizer = initial_cp['model_state_dict'], initial_cp['optimizer_state_dict']
    initial_model_hash, initial_opt_hash = rt.tensor_hash(initial_model), optimizer_hash(initial_optimizer)
    require(initial_model_hash == previous_validation['initial_model_sha256'] == reuse['initial_model_sha256']
            and initial_opt_hash == previous_validation['initial_optimizer_sha256'] == reuse['initial_optimizer_sha256'],
            'Reused initial state identity differs')
    adam_steps(initial_optimizer, 2048)
    train_names = initial_cp['trainable_parameter_names']
    require(len(train_names) == 28 and all(n.startswith(rt.TRAIN_PREFIXES) for n in train_names)
            and sum(initial_model[n].numel() for n in train_names) == 14593, 'Trainable parameters differ')
    frozen = {k: v for k, v in initial_model.items() if k not in train_names}
    bn = {k: v for k, v in initial_model.items() if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    calibration_path = rt.V2 / 'CALIBRATION.json'
    calibration = rt.read_json(calibration_path)
    require(rt.sha(calibration_path) == freeze['calibration_sha256'], 'Fixed v2 calibration changed')
    require(str(calibration_path) == freeze['calibration_path'], 'Unexpected new calibration path')
    old_logs_path = rt.V2 / 'training/FRAME/run/TRAIN.jsonl'
    old_logs = [json.loads(line) for line in old_logs_path.read_text().splitlines()]
    require(len(old_logs) == 256, 'Archived F exposure log differs')
    results, evidence = {}, {str(p): rt.sha(p) for p in [freeze_path, reuse_path, rt.INIT_CP,
        rt.V2 / 'VALIDATION.json', calibration_path, old_logs_path]}
    for condition in CONDITIONS:
        folder = rt.E / 'training' / condition / 'run'
        run = rt.read_json(folder / 'RUN.json')
        require(run['status'] == 'complete' and run['condition'] == condition and run['seed'] == 46, 'Run identity differs')
        require(run['start_step'] == 2048 and run['final_step'] == 2304
                and run['additional_steps'] == run['steps'] == 256, 'Update count differs')
        require(run['initial_optimizer_restored'] and run['initial_optimizer_sha256'] == initial_opt_hash
                and run['initial_state_sha256'] == initial_model_hash, 'Restoration receipt differs')
        require(run['initial_checkpoint_sha256'] == rt.sha(rt.INIT_CP)
                and run['release_checkpoint_sha256'] == rt.sha(rt.CP)
                and run['protocol_sha256'] == protocol_hash and run['model_source_sha256'] == rt.sha(rt.MODEL_SOURCE),
                'Run source identity differs')
        require(run['training_sources'] == freeze['training_sources']
                and run['settings'] == calibration['fixed_settings']
                and run['calibration_sha256'] == rt.sha(calibration_path), 'Run fixed settings differ')
        require(run['readout']['mode'] == {'D': 'confidence', 'N': 'matched_confidence'}[condition]
                and run['readout']['gamma'] == 2 and run['readout']['group'] == 'existing_segment8'
                and run['readout']['F_reference'] == 'track_contrast_v2/FRAME', 'Readout identity differs')
        require(run['frozen_frontend_unchanged'] and run['bn_unchanged'] and run['new_parameters'] == 0
                and run['trainable_parameters'] == 14593 and run['total_parameters'] == 70913
                and run['trainable_parameter_names'] == train_names, 'Architecture/freeze receipt differs')
        require(run['frozen_state_sha256'] == rt.tensor_hash(frozen)
                and run['bn_initial_sha256'] == rt.tensor_hash(bn), 'Frozen-state hashes differ')
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
            exact(initial_optimizer['param_groups'], checkpoint['optimizer_state_dict']['param_groups'], 'Adam groups')
            require(all(bool(torch.isfinite(v).all()) for v in model.values()), 'Nonfinite model state')
            if record['step'] == 2048:
                exact(initial_model, model, 'Restored model')
                exact(initial_optimizer, checkpoint['optimizer_state_dict'], 'Restored complete Adam')
            if record['step'] == 2304:
                require(rt.tensor_hash(model) == run['final_state_sha256']
                        and optimizer_hash(checkpoint['optimizer_state_dict']) == run['final_optimizer_sha256'],
                        'Final state hash differs')
        logs = [json.loads(line) for line in (folder / 'TRAIN.jsonl').read_text().splitlines()]
        require(len(logs) == 256, 'Training log length differs')
        for index, row in enumerate(logs):
            require(row['step'] == 2049 + index and row['additional_step'] == index + 1
                    and row['plan_index'] == index and row['condition'] == condition, 'Training schedule differs')
            require(all(row[k] == old_logs[index][k] for k in EXPOSURE_KEYS), 'Exposure differs from archived F')
            require(math.isfinite(row['loss']) and math.isfinite(row['grad_norm'])
                    and all(math.isfinite(v) for v in row['loss_terms'].values()), 'Nonfinite loss/gradient')
        pred_root = rt.E / 'predictions' / condition
        require({p.name for p in pred_root.iterdir() if p.is_dir()} == set(rt.SEQS), 'Unexpected prediction source')
        predictions = {}
        for sequence in rt.SEQS:
            receipt_path = pred_root / sequence / 'COMPLETE.json'
            prediction = rt.read_json(receipt_path)
            evaluation.inference_identity(condition, sequence)
            require(prediction['initial_model_state_sha256'] == prediction['final_model_state_sha256']
                    == run['final_state_sha256'], 'Inference changed model')
            require(prediction['raw_sha256'] == rt.sha(rt.DATA / sequence / 'raw_frames.npy'), 'Raw frames changed')
            require(set(prediction['window_sha256']) == {f'window_{s:03d}.npz' for s in rt.STARTS}, 'Window set differs')
            for name, expected in prediction['window_sha256'].items():
                require(rt.sha(pred_root / sequence / 'windows' / name) == expected, 'New prediction window changed')
            require(rt.sha(pred_root / sequence / 'official_fused.npz') == prediction['fused_sha256'], 'Fused prediction changed')
            evaluation.read_record(condition, sequence)
            predictions[sequence] = rt.sha(receipt_path)
        results[condition] = dict(initial_model_exact=True, initial_complete_adam_exact=True,
            optimizer_parameter_states=28, optimizer_initial_step=2048, optimizer_final_step=2304,
            paired_training_updates=256, all_exposures_equal_archived_F=True, frozen_all_saved_states=True,
            prediction_receipt_sha256=predictions, final_checkpoint_sha256=rt.sha(folder / 'snapshot_2304.pth'))
        evidence.update({str(folder / name): rt.sha(folder / name) for name in ['RUN.json', 'TRAIN.jsonl']})
    summary_path = rt.E / 'evaluation/SUMMARY_COMPLETE.json'
    summary = rt.read_json(summary_path)
    require(summary['status'] == 'COMPLETE' and summary['conditions'] == evaluation.CONDITIONS
            and summary['reused_reference_conditions'] == evaluation.REFERENCES, 'Six-condition summary incomplete')
    require(summary['fixed_main_step'] == 2304 and summary['strict_small_target_frames'] == 5200
            and summary['strict_small_tracks'] == 130 and summary['threshold'] == .5, 'Evaluation protocol differs')
    for path, expected in summary['source_sha256'].items():
        require(rt.sha(path) == expected, 'Summary source changed')
    for name, expected in summary['output_sha256'].items():
        require(rt.sha(rt.E / 'evaluation' / name) == expected, 'Evaluation table changed')
    reference_result = references_check()
    rt.save_json(rt.E / 'VALIDATION.json', dict(status='PASS', created_unix=time.time(),
        protocol_sha256=protocol_hash, validator_sha256=rt.sha(__file__), runs=results,
        reference_import_checks=reference_result['checks'], initial_model_sha256=initial_model_hash,
        initial_optimizer_sha256=initial_opt_hash, evidence_sha256=evidence,
        F_reuse_audit_sha256=rt.sha(reuse_path), evaluation_summary_sha256=rt.sha(summary_path),
        checks=['D/N original C2048 model and complete Adam restored exactly;28 Adam states finish at2304',
                'All256 D/N input/label/auxiliary exposure hashes equal archived F',
                'New saved models preserve frozen frontend/BN and original architecture',
                'New inference only two development sources;new score/summary identity checked',
                'Existing release/C_init/CONT/F scores imported without model or matcher replay'],
        scope='Execution/import integrity only;not detection improvement or validity of the research hypothesis.'))
    print('Readout-v3 execution/import integrity: PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references-only', action='store_true')
    args = parser.parse_args()
    if args.references_only:
        result = references_check()
        result.update(validator_sha256=rt.sha(__file__), evaluator_sha256=rt.sha(evaluation.__file__), created_unix=time.time())
        rt.save_json(rt.E / 'evaluation/REFERENCE_IMPORT_VALIDATION.json', result)
        print(json.dumps(dict(status=result['status'], checks=result['checks'])), flush=True)
    else:
        validate()
