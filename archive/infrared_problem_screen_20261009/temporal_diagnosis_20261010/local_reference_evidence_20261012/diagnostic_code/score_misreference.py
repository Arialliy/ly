#!/usr/bin/env python3
"""Isolated same-checkpoint mismatch scoring; main artifacts remain read-only."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
Q = BASE.parent
OUTPUT = BASE/'diagnostic_evaluation'
CONDITION = 'N2_misreference'
SEQUENCES = ['DJI_0026_2','DJI_0067_2']
sys.path.insert(0,str(BASE/'evaluation_code'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def load(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def identity(sequence):
    from verify_identity import verify_identity
    # The training identity is genuinely N2; the diagnostic is not a new trained variant.
    original_identity = verify_identity(BASE,'N2',sequence,BASE/'predictions/N2')
    sources = dict(original_identity['identity_source_sha256'])
    run = json.loads((BASE/'training/N2/run/RUN.json').read_text())
    prior_path = BASE/'predictions/N2'/sequence/'COMPLETE.json'
    prior = json.loads(prior_path.read_text())
    folder = BASE/'diagnostic_predictions'/CONDITION/sequence
    receipt_path = folder/'COMPLETE.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['sequence'] == sequence
    assert receipt['condition'] == receipt['variant'] == CONDITION
    assert receipt['training_variant'] == 'N2' and receipt['mode'] == 'relative'
    assert receipt['diagnostic'] == 'conditional_same_checkpoint_global_local_reference_mismatch'
    checkpoint = BASE/'training/N2/run/final_model.pth'
    assert Path(receipt['checkpoint']).resolve() == checkpoint.resolve()
    assert receipt['checkpoint_sha256'] == prior['checkpoint_sha256'] == run['final_checkpoint_sha256'] == sha(checkpoint)
    assert receipt['model_initial_state_sha256'] == receipt['model_final_state_sha256'] == run['final_model_state_sha256']
    assert receipt['frozen_all_model_state_unchanged'] is True and receipt['frozen_original_unchanged'] is True
    for key in ['reference_checkpoint_sha256','raw_sha256','annotations_sha256','model_config',
                'plan_sha256','selection_sha256','original_initial_state_sha256','original_final_state_sha256',
                'bn_initial_sha256','bn_final_sha256','normalization','seed','precision','official_starts',
                'symmetric_starts','diagnostic_starts','tile_core','tile_halo','parameters','label_config',
                'checkpoint_step','main_checkpoint_step','checkpoint_is_main']:
        assert receipt[key] == prior[key], ('Diagnostic changed common setup',key)
    for key in ['train_source_sha256','module_source_sha256','model_source_sha256','label_source_sha256',
                'top1_loss_source_sha256','common_loss_source_sha256','prior_train_source_sha256','batch_source_sha256']:
        assert receipt[key] == prior[key] == run[key],key
    assert receipt['model_labels_access'] is False and receipt['read_only_query_labels_access'] is False
    assert receipt['threshold'] == '>=0.5' and receipt['trainable_parameters'] == 0
    assert receipt['training_branch_parameter_count'] == 1529
    assert receipt['original_N2_complete_sha256'] == sha(prior_path)
    protocol = BASE/'PROTOCOL.md'
    freeze_path = BASE/'PROTOCOL_FREEZE.json'
    freeze = json.loads(freeze_path.read_text())
    assert receipt['protocol_sha256'] == freeze['protocol_sha256'] == sha(protocol)
    assert receipt['protocol_freeze_sha256'] == sha(freeze_path)
    assert '将μ和s共同循环平移' in protocol.read_text()
    definition = receipt['diagnostic_definition']
    assert definition['source_sha256'] == sha(protocol)
    assert definition['global_roll_yx'] == [256,320] and definition['full_frame_hw'] == [512,640]
    assert definition['rolled_channels'] == ['mu1','s1','mu4','s4']
    assert definition['recomputed_channels'] == ['E1','E4']
    assert definition['unchanged_channels'] == ['I','D1','m1','D4','m4']
    assert definition['tile_local_roll'] is False and definition['s_min'] == 1/47.4845
    check = receipt['zero_shift_equivalence']
    assert check['status'] == 'PASS' and check['evidence_exact_equal'] is True
    assert check['window_start'] == 0 and check['threshold05_disagreements'] == 0
    assert check['max_probability_error'] <= 2e-6
    assert sha(check['reference_window']) == check['reference_sha256']
    assert receipt['total_diagnostic_tile_model_calls'] == 160 and receipt['zero_shift_check_tile_model_calls'] == 20
    source = BASE/'diagnostic_code/infer_misreference.py'
    assert sha(source) == receipt['script_sha256']
    for path in [receipt_path,source,BASE/'DIAGNOSTIC_TRIGGER.md',Path(__file__),folder/'ZERO_SHIFT_CHECK.json']:
        sources[str(path)] = sha(path)
    return dict(diagnostic_identity_checked=True,training_variant='N2',checkpoint_step=512,
                checkpoint_sha256=sha(checkpoint),diagnostic=receipt['diagnostic'],
                identity_source_sha256=sources,same_checkpoint_all_state_unchanged=True,
                preregistered_protocol_sha256=sha(protocol),zero_shift_threshold_decisions_equal=True)


def score(sequence):
    validated = identity(sequence)
    frozen = Q/'top1_background_control_20261011/evaluation_code/evaluate.py'
    old = load(frozen,'misreference_original_scorer')
    old.E = OUTPUT
    old.ROOTS[CONDITION] = BASE/'diagnostic_predictions'/CONDITION
    old.score(CONDITION,sequence)
    path = OUTPUT/CONDITION/sequence/'COMPLETE.json'
    receipt = json.loads(path.read_text())
    receipt.update(validated,diagnostic_wrapper_sha256=sha(__file__),reused_scorer_sha256=sha(frozen))
    write_json(path,receipt)


def summary():
    assert not (OUTPUT/'SUMMARY_COMPLETE.json').exists()
    main_receipt = BASE/'evaluation/SUMMARY_COMPLETE.json'
    main_receipt_sha = sha(main_receipt)
    main_manifest = {str(p):sha(p) for p in (BASE/'evaluation').glob('*') if p.is_file()}
    sources = {str(main_receipt):main_receipt_sha,str(Path(__file__)):sha(__file__)}
    for seq in SEQUENCES:
        sources.update(identity(seq)['identity_source_sha256'])
    frozen = load(BASE/'evaluation_code/evaluate.py','misreference_frozen_summarizer')
    original_read_record = frozen.read_record
    frozen.record_root = lambda condition: (OUTPUT if condition == CONDITION else
                                Q/'top1_background_control_20261011/evaluation' if condition == 'release' else BASE/'evaluation')

    def read_record(condition,sequence):
        if condition == CONDITION:
            receipt = json.loads((OUTPUT/CONDITION/sequence/'COMPLETE.json').read_text())
            assert receipt['diagnostic_identity_checked'] is True and receipt['training_variant'] == 'N2'
            assert receipt['diagnostic_wrapper_sha256'] == sha(__file__)
            for path,expected in receipt['identity_source_sha256'].items():
                assert sha(path) == expected, ('Diagnostic identity source changed',path)
        return original_read_record(condition,sequence)

    frozen.read_record = read_record
    conditions = ['release','N0','N1','N2',CONDITION]
    frozen.summarize(conditions=conditions,output=OUTPUT)
    path = OUTPUT/'SUMMARY_COMPLETE.json'
    receipt = json.loads(path.read_text())
    receipt.update(new_models_evaluated=[],reused_main_conditions=['release','N0','N1','N2'],
                   diagnostic_conditions=[CONDITION],diagnostic_training_variant='N2',new_training_updates=0,
                   summary_wrapper_sha256=sha(__file__),summary_wrapper=str(Path(__file__).resolve()),
                   main_summary_sha256=main_receipt_sha,preregistered_conditional_diagnostic=True,
                   scope='Same final N2 checkpoint; globally mismatched local reference. Distribution intervention, not training ablation, deployable gain or causal proof.')
    receipt['source_sha256'].update(sources)
    write_json(path,receipt)
    assert all(sha(path) == value for path,value in main_manifest.items()), 'Main evaluation artifacts changed'
    object_rows = list(csv.DictReader((OUTPUT/'OBJECT_COMPARISON.csv').open()))
    target = [r for r in object_rows if r['sequence']=='DJI_0067_2' and int(r['frame'])==9 and int(r['track_id'])==7700514]
    assert len(target) == 2
    write_json(OUTPUT/'UNIQUE_LOW_RECOVERY_CHECK.json',dict(
        status='COMPLETE',object='DJI_0067_2 / frame9 / track7700514',
        source='This is the same single additional frozen-low object recovered by N2 versus N1 in both O/S, not two independent recoveries.',
        rows=[{key:row[key] for key in ['sequence','frame','source_frame','annotation_id','track_id','arrangement',
              'release_matched','N0_matched','N1_matched','N2_matched',CONDITION+'_matched',
              'release_box_max','N2_box_max',CONDITION+'_box_max']} for row in target],
        source_sha256={str(OUTPUT/'OBJECT_COMPARISON.csv'):sha(OUTPUT/'OBJECT_COMPARISON.csv')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',action='store_true')
    args=parser.parse_args()
    if args.summary:
        summary()
    else:
        for sequence in SEQUENCES:
            score(sequence)


if __name__=='__main__':
    main()
