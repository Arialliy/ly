"""Frozen component scoring for one History checkpoint under three state routes."""
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import runtime as rt
import cohorts

CONDITIONS = ['release', 'History', 'HistoryCurrent', 'HistoryOff']
NEW = ['HistoryCurrent', 'HistoryOff']
MODES = {'HistoryCurrent': 'current_only', 'HistoryOff': 'off'}
STEP = 2048
OUTPUT = rt.E / 'evaluation/step_2048'
SCORED = OUTPUT / 'scored'


def prediction_root(condition):
    base = rt.V1 if condition == 'History' else rt.E
    return base / 'predictions' / condition / 'step_2048'


def inference_identity(condition, sequence):
    """Bind both substitutions to the exact completed History state, without training."""
    assert condition in NEW and sequence in rt.SEQS
    path = prediction_root(condition) / sequence / 'COMPLETE.json'
    receipt = rt.read_json(path)
    checkpoint = rt.CHECKPOINT
    run_path = checkpoint.with_name('RUN.json')
    run = rt.read_json(run_path)
    assert receipt['status'] == 'COMPLETE' and run['status'] == 'complete'
    assert receipt['condition'] == condition and receipt['sequence'] == sequence
    assert receipt['mode'] == MODES[condition]
    assert receipt['checkpoint_step'] == run['final_step'] == STEP
    assert run['condition'] == 'History'
    assert receipt['checkpoint_sha256'] == rt.CHECKPOINT_SHA == rt.sha(checkpoint)
    assert receipt['training_run_sha256'] == rt.sha(run_path)
    assert receipt['initial_model_state_sha256'] == receipt['final_model_state_sha256'] == run['final_model_state_sha256']
    assert receipt['all_model_state_unchanged_during_inference']
    assert receipt['source_sha256'] == rt.source_hashes()
    assert receipt['official_starts'] == rt.PLANS['O'] and receipt['symmetric_starts'] == rt.PLANS['S']
    assert receipt['diagnostic_starts'] == rt.STARTS and receipt['normalization'] == [85.0799, 47.4845]
    assert receipt['raw_sha256'] == rt.sha(rt.DATA / sequence / 'raw_frames.npy')
    assert receipt['annotations_sha256'] == rt.sha(rt.DATA / sequence / 'annotations.json')
    sources = {str(p): rt.sha(p) for p in (path, checkpoint, run_path)}
    protocol = rt.E / 'PROTOCOL.md'
    assert protocol.is_file(), 'Freeze the diagnostic protocol before formal evaluation'
    assert receipt['protocol_sha256'] == rt.sha(protocol)
    sources[str(protocol)] = rt.sha(protocol)
    return sources


def evaluator():
    """Redirect paths and identities only; matching and continuity remain archived code."""
    v1rt = rt.old()
    with v1rt.import_aliases({'runtime': v1rt}):
        v1eval = rt.load_module('_history_effect_reference_evaluation', rt.V1 / 'code/evaluate.py')
    reference = v1eval.evaluator(STEP)
    oldrt = v1rt.frozen_f().runtime
    evaluation_rt = SimpleNamespace(**{k: v for k, v in vars(oldrt).items() if not k.startswith('__')})
    evaluation_rt.E = rt.E
    with v1rt.import_aliases({'runtime': evaluation_rt}):
        impl = rt.load_module('_history_effect_frozen_evaluation', v1rt.V2 / 'code/evaluate.py')
    impl.E, impl.OUTPUT, impl.SCORED = rt.E, OUTPUT, SCORED
    impl.CONDITIONS, impl.TRAINED = CONDITIONS, NEW
    impl.COHORTS = list(impl.COHORTS) + ['m1_low_common_recovered']
    original_read = impl.read_record

    def read_record(condition, sequence):
        if condition in ('release', 'History'):
            if condition == 'History': v1eval.inference_identity(condition, sequence, STEP)
            return reference.read_record(condition, sequence)
        assert condition in NEW
        return original_read(condition, sequence)
    impl.read_record = read_record

    def score(condition, sequence):
        if condition not in NEW:
            read_record(condition, sequence)
            return
        sources = impl.matcher_sources()
        sources.update(inference_identity(condition, sequence))
        output = SCORED / condition / sequence
        complete_path = output / 'COMPLETE.json'
        if complete_path.exists():
            read_record(condition, sequence)
            print({'status': 'REUSED', 'condition': condition, 'sequence': sequence}, flush=True)
            return
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Preserve incomplete scoring outputs: ' + str(output))
        scorer_path = impl.OLD / 'evaluation_code/evaluate.py'
        frozen = rt.load_module('_history_effect_original_component_scorer', scorer_path)
        frozen.E = SCORED
        frozen.ROOTS[condition] = prediction_root(condition)
        frozen.score(condition, sequence)
        receipt = rt.read_json(complete_path)
        for path in (scorer_path, Path(__file__), rt.E / 'code/runtime.py'):
            sources[str(path)] = rt.sha(path)
        receipt.update(wrapper_source_sha256=sources, wrapper_path=str(Path(__file__).resolve()),
            artifact_sha256={name: rt.sha(output / name) for name in ('scores.npz', 'component_areas.npz', 'FRAME_COUNTS.csv')},
            checkpoint_policy='same_History_2048_weights_no_training', checkpoint_step=STEP,
            checkpoint_sha256=rt.CHECKPOINT_SHA, mode=MODES[condition])
        rt.save(complete_path, receipt)
    impl.score = score

    def collect(conditions):
        manifest = cohorts.verify()
        frozen = rt.load_module('_history_effect_frozen_cohort_join', impl.PREVIOUS / 'evaluation_code/evaluate.py')
        frozen.CONDITIONS, frozen.NEW, frozen.COMPARISONS = conditions, NEW, conditions
        frozen.digest = lambda path: rt.sha(impl.remap(path))
        frozen.read_record = read_record
        frozen.record_root = lambda condition: (impl.OLD / 'evaluation' if condition == 'release' else
            rt.V1 / 'evaluation/step_2048/scored' if condition == 'History' else SCORED)
        data = frozen.collect(conditions)
        common = {cohorts.key(row) for row in cohorts.read_csv(cohorts.OUT / 'COMMON_RECOVERED_9.csv')}
        assert len(common) == manifest['m1_low_common_recovered'] == 9
        for row in data[-1]: row['m1_low_common_recovered'] = int(cohorts.key(row) in common)
        assert sum(row['m1_low_common_recovered'] for row in data[-1]) == 18
        frozen.COHORT_KEYS = list(frozen.COHORT_KEYS) + ['m1_low_common_recovered']
        data[3][str(cohorts.OUT / 'COMPLETE.json')] = rt.sha(cohorts.OUT / 'COMPLETE.json')
        data[3][str(rt.E / 'code/cohorts.py')] = rt.sha(rt.E / 'code/cohorts.py')
        return frozen, data
    impl.collect = collect

    def save_json(path, value):
        value = dict(value)
        if Path(path).name == 'SUMMARY_COMPLETE.json':
            value.update(fixed_main_step=STEP, evaluated_step=STEP, initial_step=STEP, additional_steps=0,
                no_training=True, checkpoint_sha256=rt.CHECKPOINT_SHA,
                requested_paired_comparisons=['History_vs_HistoryCurrent', 'History_vs_HistoryOff'],
                cohorts=dict(value['cohorts'], m1_low_common_recovered=9),
                prediction_sources={'History': str(prediction_root('History')),
                    **{condition: str(prediction_root(condition)) for condition in NEW}},
                interpretation='Same frozen History2048 model state with history, current_only, or M1 bypass. '
                'No training or checkpoint selection. Frozen actual-component diagnostics on two development clips, '
                'not official Pd/Fa. O/S are paired arrangements of the same objects; release anchors fixed cohorts. '
                'The common-recovered9 subgroup is conditioned on completed v1 results and is explanatory only; '
                'its response means do not estimate effects in the full target population.')
            # Publish COMPLETE only after the same-checkpoint supplements finish.
            path = Path(path).with_name('SUMMARY_PENDING.json')
        rt.save(path, value)
    evaluation_rt.save_json = save_json
    return impl


def summarize():
    impl = evaluator()
    complete_path = OUTPUT / 'SUMMARY_COMPLETE.json'
    if complete_path.exists():
        complete = rt.read_json(complete_path)
        assert complete['status'] == 'COMPLETE'
        for path, expected in complete['source_sha256'].items(): assert rt.sha(impl.remap(path)) == expected
        for name, expected in complete['output_sha256'].items(): assert rt.sha(OUTPUT / name) == expected
        print({'status': 'REUSED', 'summary': str(complete_path)}, flush=True)
        return
    for condition in NEW:
        for sequence in rt.SEQS: inference_identity(condition, sequence)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    impl.summarize(CONDITIONS)
    rows = impl.read_csv(OUTPUT / 'OBJECT_COMPARISON.csv')
    common = [row for row in rows if int(row['m1_low_common_recovered'])]
    assert len(common) == 18
    impl.write_csv(OUTPUT / 'COMMON_RECOVERED_9_MATCHES.csv', common)
    unique = []
    for reference in NEW + ['release']:
        for cohort in ('all', 'low_all_05', 'extreme_all_001', 'both_fixed_success', 'm1_low_common_recovered'):
            selected = [row for row in rows if cohort == 'all' or int(row[cohort])]
            gains, losses = {}, {}
            for plan in rt.PLANS:
                chosen = [row for row in selected if row['arrangement'] == plan]
                gains[plan] = {cohorts.key(row) for row in chosen if int(row['History_gain_vs_' + reference])}
                losses[plan] = {cohorts.key(row) for row in chosen if int(row['History_loss_vs_' + reference])}
            unique.append(dict(condition='History', reference=reference, cohort=cohort,
                gains_O=len(gains['O']), gains_S=len(gains['S']),
                unique_gains_either_arrangement=len(gains['O'] | gains['S']),
                gains_both_arrangements=len(gains['O'] & gains['S']),
                losses_O=len(losses['O']), losses_S=len(losses['S']),
                unique_losses_either_arrangement=len(losses['O'] | losses['S']),
                losses_both_arrangements=len(losses['O'] & losses['S'])))
    impl.write_csv(OUTPUT / 'UNIQUE_HISTORY_EFFECTS.csv', unique)
    for condition in NEW:
        changed = [row for row in rows if int(row['History_matched']) != int(row[condition + '_matched'])]
        impl.write_csv(OUTPUT / ('HISTORY_VS_' + condition.upper() + '_CHANGED_OBJECTS.csv'), changed, list(rows[0]))
    equivalence = []
    for sequence in rt.SEQS:
        off, off_areas, _, _ = impl.read_record('HistoryOff', sequence)
        release, release_areas, _, _ = impl.read_record('release', sequence)
        checks = {}
        for key in off:
            checks['scores/' + key] = bool(np.array_equal(off[key], release[key], equal_nan=True)
                if off[key].dtype.kind not in ('U', 'S') else np.array_equal(off[key], release[key]))
        checks.update({'areas/' + key: bool(np.array_equal(off_areas[key], release_areas[key])) for key in off_areas})
        equivalence.append({'sequence': sequence, 'all_arrays_exact': all(checks.values()), 'checks': checks})
    rt.save(OUTPUT / 'OFF_RELEASE_ARRAY_COMPARISON.json', {'status': 'COMPLETE', 'records': equivalence})
    complete = rt.read_json(OUTPUT / 'SUMMARY_PENDING.json')
    complete['source_sha256'][str(rt.E / 'PROTOCOL.md')] = rt.sha(rt.E / 'PROTOCOL.md')
    complete['output_sha256'] = {p.name: rt.sha(p) for p in sorted(OUTPUT.glob('*.csv'))}
    complete['output_sha256']['OFF_RELEASE_ARRAY_COMPARISON.json'] = rt.sha(OUTPUT / 'OFF_RELEASE_ARRAY_COMPARISON.json')
    rt.save(complete_path, complete)
    (OUTPUT / 'SUMMARY_PENDING.json').unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--sequence', choices=rt.SEQS)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.summary:
        summarize()
    else:
        if args.condition is None: parser.error('--condition is required for scoring')
        impl = evaluator()
        for sequence in [args.sequence] if args.sequence else rt.SEQS:
            impl.score(args.condition, sequence)
