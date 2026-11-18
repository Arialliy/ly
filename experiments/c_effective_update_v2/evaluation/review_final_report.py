"""Check final prose table/covariates and all raw reliable peak coordinates."""
from collections import Counter
import csv
import json
from statistics import median
import review_pair_results as independent


def main():
    E = independent.E
    read, sha = independent.read, independent.sha
    report_path = E / 'DECISION.md'
    prose = report_path.read_text()
    plan, decision = read(E/'PAIR_PLAN.json'), read(E/'DECISION.json')
    reviewed = read(E/'evaluation/RESULT_REVIEW.json')
    assert reviewed['status'] == 'PASS'
    assert reviewed['source_sha256']['DECISION.json'] == sha(E/'DECISION.json')
    assert reviewed['source_sha256']['PAIRED_EFFECT.csv'] == sha(E/'PAIRED_EFFECT.csv')
    pairs = {p['pair_id']: read(E/'pairs'/f"pair_{p['pair_id']:02d}.json") for p in plan['pairs']}
    table_rows = [line for line in prose.splitlines() if line.startswith(('| 同源7对 |', '| 跨源2对 |'))]
    assert len(table_rows) == 12
    for row in table_rows:
        relation, condition, role, *numbers = [cell.strip() for cell in row.strip('|').split('|')]
        relation = {'同源7对': 'same_source', '跨源2对': 'cross_source'}[relation]
        condition = {'Full': 'full_F', 'No-Keep': 'no_keep', 'No-Track': 'no_track'}[condition]
        role = {'更新': 'update', '观察': 'observer'}[role]
        values = decision['aggregate']['all_qualified_9'][relation][condition][role]['weak']['values']
        for raw, key in zip(numbers, ('q_positive_delta', 'q_negative_delta', 'margin_delta', 'margin_advantage_vs_full')):
            assert abs(float(raw)-1000*values[key]) <= 5.01e-7
    assert '各列在对象层计算后分别汇总取中位数' in prose
    assert '不能将轨迹数直接相加' in prose
    counts, no_track_absolute = Counter(), []
    eps = read(E/'GATE_SPEC.json')['metric_epsilons']
    for pid, p in pairs.items():
        for role in ('update', 'observer'):
            baseline = {x['object_key']: x for x in p['baseline'][role]['reliable_objects']}
            full = {x['object_key']: x for x in p['conditions']['full_F']['roles'][role]['reliable_objects']}
            for condition in independent.CONDITIONS:
                for obj in p['conditions'][condition]['roles'][role]['reliable_objects']:
                    counts['raw_reliable_coordinate_comparisons'] += 1
                    assert obj['peak_xy'] == full[obj['object_key']]['peak_xy']
                    if obj['peak_xy'] != baseline[obj['object_key']]['peak_xy']:
                        counts[f'{condition}_{role}_changed_vs_initial'] += 1
            for (source, track), unit in independent.pair_units(p, role, 'weak', 'no_track').items():
                v = unit['values']
                if v['q_positive_delta'] > eps['q_positive'] and v['margin_delta'] > eps['margin']:
                    assert not independent.benefit(v, eps)
                    no_track_absolute.append({'pair_id': pid, 'role': role, 'track': track,
                        'delta_q_positive': v['q_positive_delta'], 'delta_margin': v['margin_delta'],
                        'margin_advantage_vs_full': v['margin_advantage_vs_full'],
                        'q_positive_advantage_vs_full': v['q_positive_advantage_vs_full']})
    with (E/'evaluation/COHORT_COVARIATES.csv').open(newline='') as stream:
        covariates = list(csv.DictReader(stream))
    assert len(covariates) == 9
    for row in covariates:
        p = pairs[int(row['pair_id'])]
        assert row['source_relation'] == p['source_relation']
        for role in ('update', 'observer'):
            b = p['baseline'][role]
            assert int(row[role+'_weak_context_count']) == len(b['weak_objects'])
            assert float(row[role+'_weak_bbox_area_median']) == median(o['bbox_area'] for o in b['weak_objects'])
            assert int(row[role+'_exposed_target_context_count']) == b['counts']['exposed_targets']
            assert int(row[role+'_reliable_context_count']) == len(b['reliable_objects'])
            assert row[role+'_visibility_status'] == ';'.join(sorted({x['visibility_status'] for x in b['weak_objects']}))
    reconstruction = read(E/'replay/RECONSTRUCTION.json')
    assert reconstruction['exact_count'] == 27
    assert max(len(x['ambiguous_coordinates']) for x in reconstruction['endpoints']) == 2
    assert max(x['combinations'] for x in reconstruction['endpoints']) == 35
    assert sum(x['attempted'] for x in reconstruction['endpoints']) == 157
    result = {'status': 'PASS', 'report_sha256': sha(report_path),
        'numeric_table_rows_checked': len(table_rows), 'numeric_table_unit': '1e-3 logit',
        'cohort_covariate_rows_checked': len(covariates),
        'coverage': {k: reviewed[k] for k in ('pair_count', 'trusted_weak_track_count', 'weak_source_count', 'weak_context_count', 'uncertain_reliable_contexts_retained')},
        'scope': {'new_optimizer_steps': 0, 'historical_exact_endpoints': 27, 'new_cross_source_observer_pairings': 2},
        'raw_reliable_peak_coordinate_checks': dict(counts),
        'reliable_condition_vs_full_position_differences': 0,
        'no_track_local_absolute_improvements_do_not_meet_relative_requirements': no_track_absolute,
        'nonlinear_source_median_explanation_present': True,
        'decision_and_denominators_match_raw_review': True,
        'source_sha256': {str(path.relative_to(E)): sha(path) for path in (report_path, E/'DECISION.json', E/'PAIRED_EFFECT.csv', E/'evaluation/COHORT_COVARIATES.csv', E/'evaluation/RESULT_REVIEW.json', E/'replay/RECONSTRUCTION.json')},
        'review_script_sha256': sha(__file__),
        'notes': ['Reliable peak positions can move versus F2304; all three conditions share those same positions. The report correctly restricts zero-difference wording to comparisons between conditions.',
                  'No-Track has four pair-track-role absolute q+/margin improvements; none beats Full sufficiently. The report correctly requires both absolute and relative benefit.',
                  'Thirty-nine is the union of trusted weak tracks; 3632 is the count of window-context exposures, not independent samples.']}
    (E/'evaluation/FINAL_REPORT_REVIEW.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'report_sha256': result['report_sha256'], 'raw_reliable_comparisons': counts['raw_reliable_coordinate_comparisons']}))


if __name__ == '__main__':
    main()
