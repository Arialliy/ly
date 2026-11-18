"""Reuse the frozen matcher, cohort join and trajectory analysis for paired M1."""
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import runtime as rt

CONDITIONS = ['release', 'Current', 'History']


def inference_identity(condition, sequence, step):
    path = rt.E / 'predictions' / condition / f'step_{step:04d}' / sequence / 'COMPLETE.json'
    receipt = rt.read_json(path)
    checkpoint = rt.E / 'training' / condition / 'run' / f'snapshot_{step:03d}.pth'
    run_path = checkpoint.with_name('RUN.json'); run = rt.read_json(run_path)
    assert receipt['status'] == 'COMPLETE' and run['status'] == 'complete'
    assert receipt['condition'] == run['condition'] == condition
    assert receipt['sequence'] == sequence and receipt['checkpoint_step'] == step
    assert run['final_step'] == receipt['main_checkpoint_step'] == rt.STEPS
    assert receipt['checkpoint_is_main'] == (step == rt.STEPS)
    assert receipt['checkpoint_sha256'] == rt.sha(checkpoint)
    assert receipt['training_run_sha256'] == rt.sha(run_path)
    assert receipt['source_sha256'] == run['source_sha256'] == rt.source_hashes()
    assert receipt['reference_checkpoint_sha256'] == run['release_checkpoint_sha256'] == rt.sha(rt.CP)
    assert receipt['protocol_sha256'] == run['protocol_sha256'] == rt.protocol_hash()
    assert receipt['official_starts'] == rt.O and receipt['symmetric_starts'] == rt.S
    assert receipt['diagnostic_starts'] == rt.STARTS and receipt['normalization'] == [85.0799, 47.4845]
    assert receipt['all_original_frozen'] and receipt['original_unchanged']
    assert receipt['all_model_state_unchanged_during_inference']
    assert receipt['initial_model_state_sha256'] == receipt['final_model_state_sha256']
    assert receipt['raw_sha256'] == rt.sha(rt.DATA / sequence / 'raw_frames.npy')
    assert receipt['annotations_sha256'] == rt.sha(rt.DATA / sequence / 'annotations.json')
    result = {str(p): rt.sha(p) for p in (path, checkpoint, run_path)}
    if (rt.E / 'PROTOCOL_FREEZE.json').exists():
        result[str(rt.E / 'PROTOCOL_FREEZE.json')] = rt.protocol_hash()
    return result


def evaluator(step=2048):
    """Redirect paths/identity only; preserve scorer and scientific definitions."""
    oldrt = rt.frozen_f().runtime
    evaluation_rt = SimpleNamespace(**{k: v for k, v in vars(oldrt).items() if not k.startswith('__')})
    evaluation_rt.E = rt.E
    with rt.import_aliases({'runtime': evaluation_rt}):
        impl = rt.load_module('_m1_frozen_evaluation', rt.V2 / 'code/evaluate.py')
    impl.E = rt.E
    impl.OUTPUT = rt.E / 'evaluation' / f'step_{step:04d}'
    impl.SCORED = impl.OUTPUT / 'scored'
    impl.CONDITIONS = CONDITIONS
    impl.TRAINED = rt.CONDITIONS
    impl.inference_identity = lambda condition, sequence: inference_identity(condition, sequence, step)
    old_score = impl.score

    def score(condition, sequence):
        if condition == 'release': return  # The already verified release score cache is reused.
        # The inherited score function reads E/predictions/{condition}; change
        # only the frozen scorer's root through a scoped loader interception.
        original_loader = evaluation_rt.load_module
        def loader(name, path):
            module = original_loader(name, path)
            if Path(path) == impl.OLD / 'evaluation_code/evaluate.py':
                class Roots(dict):
                    def update(self, mapping=None, **kwargs):
                        for key, value in dict(mapping or {}, **kwargs).items():
                            self[key] = (rt.E / 'predictions' / key / f'step_{step:04d}'
                                         if key in rt.CONDITIONS else value)
                module.ROOTS = Roots(module.ROOTS)
            return module
        evaluation_rt.load_module = loader
        try:
            old_score(condition, sequence)
        finally:
            evaluation_rt.load_module = original_loader
        path = impl.SCORED / condition / sequence / 'COMPLETE.json'
        receipt = rt.read_json(path)
        receipt['checkpoint_policy'] = 'fixed2048_main' if step == rt.STEPS else 'process_only'
        receipt['step'] = step; receipt['main_step'] = rt.STEPS
        rt.save_json(path, receipt)

    old_read_record = impl.read_record
    def read_record(condition, sequence):
        if condition != 'release': return old_read_record(condition, sequence)
        folder = impl.OLD / 'evaluation/release' / sequence
        receipt = rt.read_json(folder / 'COMPLETE.json')
        assert receipt['status'] == 'COMPLETE' and receipt['condition'] == 'release' and receipt['threshold'] == .5
        with np.load(folder / 'scores.npz', allow_pickle=False) as values:
            scores = {k: values[k] for k in values.files}
        with np.load(folder / 'component_areas.npz', allow_pickle=False) as values:
            areas = {k: values[k] for k in values.files}
        assert scores['plans'].tolist() == list(rt.PLANS)
        assert scores['frame_counts'].shape == (2, 160, 6)
        return scores, areas, receipt, {str(folder / name): rt.sha(folder / name)
            for name in ('COMPLETE.json', 'scores.npz', 'component_areas.npz', 'FRAME_COUNTS.csv')}
    impl.read_record = read_record

    def collect(conditions):
        frozen = rt.load_module('_m1_frozen_cohort_join', impl.PREVIOUS / 'evaluation_code/evaluate.py')
        frozen.CONDITIONS = conditions
        frozen.NEW = [c for c in conditions if c != 'release']
        frozen.COMPARISONS = conditions
        frozen.digest = lambda path: rt.sha(impl.remap(path))
        frozen.read_record = read_record
        frozen.record_root = lambda condition: impl.OLD / 'evaluation' if condition == 'release' else impl.SCORED
        return frozen, frozen.collect(conditions)
    impl.collect = collect
    impl.score = score

    # Correct only the inherited v2 experiment identity in final metadata.
    def save_json(path, value):
        value = dict(value)
        if Path(path).name == 'SUMMARY_COMPLETE.json':
            value.update(fixed_main_step=rt.STEPS, evaluated_step=step, checkpoint_is_main=step == rt.STEPS,
                initial_step=0, additional_steps=rt.STEPS,
                requested_paired_comparisons=['History_vs_Current', 'History_vs_release', 'Current_vs_release'],
                interpretation='Frozen actual-component diagnostic on two development clips, not official Pd/Fa. '
                'Release initialization, original parameters/BN frozen, same seed46 and F supervision, '
                'existing256items repeated8. O/S are paired arrangements of the same objects.')
        rt.save_json(path, value)
    evaluation_rt.save_json = save_json
    return impl


def summarize(step=2048):
    impl = evaluator(step)
    impl.OUTPUT.mkdir(parents=True, exist_ok=True)
    impl.summarize(CONDITIONS)
    rows = impl.read_csv(impl.OUTPUT / 'OBJECT_COMPARISON.csv')
    unique = []
    for condition, reference in [('Current', 'release'), ('History', 'release'), ('History', 'Current')]:
        for cohort in ('all', 'low_all_05', 'extreme_all_001', 'both_fixed_success'):
            chosen = [r for r in rows if cohort == 'all' or int(r[cohort])]
            gains = {}; losses = {}
            for plan in rt.PLANS:
                gains[plan] = {(r['sequence'], int(r['frame']), int(r['annotation_id'])) for r in chosen
                    if r['arrangement'] == plan and int(r[condition + '_gain_vs_' + reference])}
                losses[plan] = {(r['sequence'], int(r['frame']), int(r['annotation_id'])) for r in chosen
                    if r['arrangement'] == plan and int(r[condition + '_loss_vs_' + reference])}
            unique.append({'condition': condition, 'reference': reference, 'cohort': cohort,
                'gains_O': len(gains['O']), 'gains_S': len(gains['S']),
                'unique_gains_either_arrangement': len(gains['O'] | gains['S']),
                'gains_both_arrangements': len(gains['O'] & gains['S']),
                'losses_O': len(losses['O']), 'losses_S': len(losses['S']),
                'unique_losses_either_arrangement': len(losses['O'] | losses['S'])})
    impl.write_csv(impl.OUTPUT / 'UNIQUE_RECOVERY.csv', unique)
    changed = [r for r in rows if int(r['History_matched']) != int(r['Current_matched'])]
    impl.write_csv(impl.OUTPUT / 'HISTORY_VS_CURRENT_CHANGED_OBJECTS.csv', changed, list(rows[0]))
    complete_path = impl.OUTPUT / 'SUMMARY_COMPLETE.json'
    complete = rt.read_json(complete_path)
    complete['output_sha256'] = {p.name: rt.sha(p) for p in sorted(impl.OUTPUT.glob('*.csv'))}
    rt.save_json(complete_path, complete)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=rt.CONDITIONS)
    parser.add_argument('--sequence', choices=rt.SEQS)
    parser.add_argument('--step', type=int, choices=[128, 512, 1024, 2048], default=2048)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.summary: summarize(args.step)
    else:
        if args.condition is None: parser.error('--condition is required for scoring')
        for sequence in [args.sequence] if args.sequence else rt.SEQS:
            evaluator(args.step).score(args.condition, sequence)
