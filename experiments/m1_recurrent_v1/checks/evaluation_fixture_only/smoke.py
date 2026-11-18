"""Software integration fixture ONLY: both aliases point to archived RELEASE.

These files are NOT M1 predictions, experiments, or scientific evidence.
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parents[1]
sys.path.insert(0, str(FORMAL / 'code'))
import runtime as rt
import evaluate

fixture = HERE / 'artifacts'
fixture.mkdir(exist_ok=True)
(fixture / 'code').mkdir(exist_ok=True)
for name in ('evaluate.py', 'runtime.py'):
    target = fixture / 'code' / name
    if not target.exists(): target.symlink_to(FORMAL / 'code' / name)
for condition in rt.CONDITIONS:
    for sequence in rt.SEQS:
        target = fixture / 'predictions' / condition / 'step_2048' / sequence
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): target.symlink_to(rt.Q / 'stage1_goal/predictions/official' / sequence)
rt.E = fixture
# Deliberately bypass M1 provenance for this explicit release-data software fixture.
evaluate.inference_identity = lambda condition, sequence, step: {}
for condition in rt.CONDITIONS:
    for sequence in rt.SEQS:
        evaluate.evaluator().score(condition, sequence)
evaluate.summarize()
impl = evaluate.evaluator()
summary = impl.read_csv(impl.OUTPUT / 'SUMMARY.csv')
assert len(summary) == 6
for condition in evaluate.CONDITIONS:
    rows = [r for r in summary if r['condition'] == condition]
    assert [(int(r['M']), int(r['unmatched_components'])) for r in rows] == [(700,4015),(720,3967)]
unique = impl.read_csv(impl.OUTPUT / 'UNIQUE_RECOVERY.csv')
assert all(int(r['unique_gains_either_arrangement']) == 0 and int(r['unique_losses_either_arrangement']) == 0 for r in unique)
rt.save_json(HERE / 'RESULT.json', {'status': 'PASS', 'purpose': 'software integration fixture only',
    'M1_model_executed': False, 'all_three_aliases_use': 'archived release predictions',
    'matcher_and_cohort_pipeline': 'original reused functions', 'summary_rows': len(summary),
    'unique_recovery_rows': len(unique), 'formal_outputs_written': False})
print('EVALUATION_FIXTURE_PASS', flush=True)
