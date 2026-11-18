#!/usr/bin/env python3
"""Pre-registered paired single-Adam-step effect gate; standard library only.

Reads 16 fixed pair receipts, never executes a model, gradient or optimizer.
Every statistic gives equal weight to the predetermined pairs, including failures
and unavailable observations. These pairs are not independent statistical trials.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

E = Path(__file__).resolve().parents[1]
ROLES = ('update', 'observer')
BASE_CONDITIONS = ('full_F', 'no_keep', 'no_track')
METRICS = ('weak_logit', 'weak_margin', 'weak_probability_margin',
           'hardbg_logit', 'reliable_logit', 'reliable_peak_shift')
COSTS = {'hardbg_logit': 1, 'reliable_logit': -1, 'reliable_peak_shift': 1}
EPS32 = 2 ** -23
GATE_SPEC = {
    'version': 1, 'pairs': 16, 'aggregation': 'equal pair weight; no dropped pairs',
    'primary': 'weak_margin = original weak target ROI mean logit minus original hard-background ROI logit',
    'secondary_required': 'own weak_logit delta median > epsilon; versus full_F median >= -2*epsilon',
    'noise_floor': 'epsilon_m=max(4*max_abs(baseline-repeat across16pairs/bothroles),8*float32_eps*max(1,median_abs_baseline)); pairwise change comparisons use2*epsilon',
    'position_cost_threshold': 'one original image pixel plus2*epsilon_position; null remains unavailable',
    'full_not_helping': 'update weak_margin delta median <= epsilon and positive pair count <= negative pair count (noise ties excluded from signs)',
    'removal_benefit': 'own margin and weak_logit delta medians > respective epsilon in both roles; margin advantage versus full_F median>2*epsilon and positive pairs >=10 update/>=9 observer; weak_logit advantage median>=-2*epsilon',
    'removal_cost': 'one fixed cost metric has update deterioration median above its pairwise threshold,with >=8 update pairs jointly showing margin advantage and this cost; observer cost median above measurement epsilon (position:one pixel+epsilon)',
    'multi_source_support': 'joint update pattern must include at least2 distinct predeclared source IDs; batch aggregates are not source-specific effects',
    'config_benefit': 'same benefit rule as removal,using config versus full_F',
    'config_avoidance': 'for a supported removal/cost path,config cost median versus full_F <= pairwise threshold and config improvement versus removal median > pairwise threshold in both roles; >=9/16 pairs in each role jointly show margin advantage and do not reproduce that cost',
    'config_new_cost_veto': 'any cost metric showing median deterioration above pairwise threshold and >=9/16 harmful pairs in either role blocks256-step launch',
    'execution': 'all required real parameter updates must be finite and strictly nonzero; declared config fallback is counted by reason/source and retained among all16 pairs,not an extra blocking gate',
    'missing': 'weak metrics require all16 pairs in each role; a missing cost measurement cannot support or clear that cost path and is not replaced by0',
    'interpretation': 'Descriptive pre-registered directional gate,not a significance test or proof of detected-target loss.ROI scores/peak shifts are proxies;negative gradient cosine alone never triggers.',
    'statuses': ['TRIGGER_NOT_SUPPORTED', 'CONFIG_SINGLE_STEP_AUTHORIZED',
                 'CONFIG_NOT_SUPPORTED', 'CONFIG_256_AUTHORIZED', 'EXECUTION_ISSUE'],
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def save_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def load_pairs():
    paths = sorted((E / 'pairs').glob('pair_[0-9][0-9].json'))
    require(len(paths) == 16, 'Exactly16 complete predeclared pair receipts required')
    pairs = [read_json(p) for p in paths]
    require(sorted(p['pair_id'] for p in pairs) == list(range(16)), 'Pair identities differ')
    pairs.sort(key=lambda p: p['pair_id'])
    indices = [p[k] for p in pairs for k in ('update_index', 'observer_index')]
    require(len(set(indices)) == 32, 'The32 original plan entries must not be reused')
    config_presence = ['config' in p['conditions'] for p in pairs]
    require(not any(config_presence) or all(config_presence), 'Partial config results must not enter the gate')
    for pair in pairs:
        require(pair['status'] == 'COMPLETE' and pair['copies_accumulated'] is False
                and pair['parameter_updates_per_copy'] == 1, 'Single independent update receipt incomplete')
        for role in ROLES:
            require(len(pair[role + '_sources']) == 2, 'Each pair role must preserve both declared sources')
            baseline, repeat = pair['baseline'][role], pair['repeat'][role]
            require(baseline['counts'] == repeat['counts'], 'Baseline/repeat cohort counts changed')
            for metric in METRICS:
                for values in [baseline, repeat]:
                    require(metric in values and (values[metric] is None or finite(values[metric])),
                            'Missing/nonfinite metric: ' + metric)
                require((baseline[metric] is None) == (repeat[metric] is None),
                        'Metric availability changed in baseline repeat')
            for condition in BASE_CONDITIONS + (('config',) if all(config_presence) else ()):
                require(condition in pair['conditions'], 'Incomplete update condition')
                current = pair['conditions'][condition]['roles'][role]
                require(current['counts'] == baseline['counts'], 'Frozen metric cohorts changed after update')
                require(all(k in current and (current[k] is None or finite(current[k])) for k in METRICS),
                        'Post-update metrics absent/nonfinite')
    return pairs, paths, all(config_presence)


def noise_floors(pairs):
    result = {}
    for metric in METRICS:
        observations = [(p['baseline'][r][metric], p['repeat'][r][metric]) for p in pairs for r in ROLES]
        good = [(a, b) for a, b in observations if finite(a) and finite(b)]
        if not good:
            result[metric] = dict(epsilon=None, available_pairs_roles=0)
            continue
        repeat_error = max(abs(a-b) for a, b in good)
        scale = max(1., statistics.median(abs(a) for a, _ in good))
        numerical_floor = 8 * EPS32 * scale
        result[metric] = dict(epsilon=max(4*repeat_error, numerical_floor),
            max_repeat_abs_difference=repeat_error, numerical_floor=numerical_floor,
            baseline_abs_scale=scale, available_pairs_roles=len(good))
    return result


def metric_delta(pair, condition, role, metric):
    before, after = pair['baseline'][role][metric], pair['conditions'][condition]['roles'][role][metric]
    return after-before if finite(before) and finite(after) else None


def relative_delta(pair, condition, reference, role, metric):
    left, right = metric_delta(pair, condition, role, metric), metric_delta(pair, reference, role, metric)
    return left-right if finite(left) and finite(right) else None


def summarize_values(values, tolerance):
    valid = [x for x in values if finite(x)]
    available = tolerance is not None
    return dict(expected_pairs=16, available_pairs=len(valid), complete=len(valid)==16 and available,
        mean=statistics.mean(valid) if valid else None, median=statistics.median(valid) if valid else None,
        minimum=min(valid) if valid else None, maximum=max(valid) if valid else None,
        positive=sum(x > tolerance for x in valid) if available else None,
        negative=sum(x < -tolerance for x in valid) if available else None,
        noise_ties=sum(abs(x) <= tolerance for x in valid) if available else None,
        tolerance=tolerance, values=values)


def summaries(pairs, conditions, noise):
    result = {}
    for condition in conditions:
        result[condition] = {}
        for role in ROLES:
            out = {}
            for metric in METRICS:
                epsilon = noise[metric]['epsilon']
                out[metric] = dict(
                    delta=summarize_values([metric_delta(p, condition, role, metric) for p in pairs], epsilon),
                    advantage_vs_full=summarize_values([relative_delta(p, condition, 'full_F', role, metric) for p in pairs],
                                                       2*epsilon if epsilon is not None else None))
            result[condition][role] = out
    return result


def threshold(noise, metric, pairwise=True):
    epsilon = noise[metric]['epsilon']
    return ((2 if pairwise else 1)*epsilon + (1. if metric=='reliable_peak_shift' else 0.)) if epsilon is not None else None


def benefit_check(condition, aggregate, noise):
    roles = {}
    for role, min_positive in [('update', 10), ('observer', 9)]:
        values = aggregate[condition][role]
        margin, weak = values['weak_margin']['delta'], values['weak_logit']['delta']
        advantage, weak_adv = values['weak_margin']['advantage_vs_full'], values['weak_logit']['advantage_vs_full']
        complete = all(x['complete'] for x in [margin, weak, advantage, weak_adv])
        checks = dict(complete=complete,
            own_margin_improves=complete and margin['median'] > noise['weak_margin']['epsilon'],
            own_weak_logit_improves=complete and weak['median'] > noise['weak_logit']['epsilon'],
            margin_advantage=complete and advantage['median'] > 2*noise['weak_margin']['epsilon'],
            margin_sign_support=complete and advantage['positive'] >= min_positive,
            weak_logit_not_worse_than_full=complete and weak_adv['median'] >= -2*noise['weak_logit']['epsilon'])
        roles[role] = dict(checks=checks, required_positive_pairs=min_positive, passed=all(checks.values()))
    return dict(passed=all(r['passed'] for r in roles.values()), roles=roles)


def cost_values(pairs, condition, reference, role, metric):
    values = [relative_delta(p, condition, reference, role, metric) for p in pairs]
    return [COSTS[metric]*v if finite(v) else None for v in values]


def sources_for(pairs, ids, role):
    return sorted({s for p in pairs if p['pair_id'] in ids for s in p[role+'_sources']})


def stage_one(pairs, aggregate, noise):
    full = aggregate['full_F']['update']['weak_margin']['delta']
    full_not_helping = full['complete'] and full['median'] <= noise['weak_margin']['epsilon'] and full['positive'] <= full['negative']
    removals, supported = {}, []
    for condition in ['no_keep', 'no_track']:
        benefit = benefit_check(condition, aggregate, noise)
        costs = {}
        for metric in COSTS:
            tolerance, observer_floor = threshold(noise, metric), threshold(noise, metric, pairwise=False)
            update = summarize_values(cost_values(pairs, condition, 'full_F', 'update', metric), tolerance)
            observer = summarize_values(cost_values(pairs, condition, 'full_F', 'observer', metric), tolerance)
            joint = [p['pair_id'] for p, harm in zip(pairs, update['values']) if finite(harm) and tolerance is not None
                and harm > tolerance and finite(relative_delta(p, condition, 'full_F', 'update', 'weak_margin'))
                and relative_delta(p, condition, 'full_F', 'update', 'weak_margin') > 2*noise['weak_margin']['epsilon']]
            source_ids = sources_for(pairs, joint, 'update')
            checks = dict(update_complete=update['complete'], observer_complete=observer['complete'],
                update_cost_median=update['complete'] and update['median'] > tolerance,
                same_pair_joint_pattern=len(joint) >= 8, multiple_sources=len(source_ids) >= 2,
                observer_cost_direction=observer['complete'] and observer['median'] > observer_floor)
            passed = full_not_helping and benefit['passed'] and all(checks.values())
            costs[metric] = dict(passed=passed, checks=checks, update=update, observer=observer,
                joint_update_pair_ids=joint, joint_update_source_ids=source_ids,
                cost_interpretation='score/position proxy change,not a demonstrated detection loss')
            if passed:
                supported.append(dict(removal=condition, cost_metric=metric))
        removals[condition] = dict(benefit=benefit, costs=costs)
    return dict(passed=bool(supported), full_not_stably_helping=full_not_helping,
        full_update_margin=full, removals=removals, supported_paths=supported)


def stage_two(pairs, aggregate, noise, initial_gate):
    benefit = benefit_check('config', aggregate, noise)
    new_costs, veto = {}, False
    for metric in COSTS:
        new_costs[metric] = {}
        for role in ROLES:
            stats = summarize_values(cost_values(pairs, 'config', 'full_F', role, metric), threshold(noise, metric))
            harmful = stats['complete'] and stats['median'] > threshold(noise, metric) and stats['positive'] >= 9
            new_costs[metric][role] = dict(consistent_new_cost=harmful, summary=stats)
            veto = veto or harmful
    paths = []
    for path in initial_gate['supported_paths']:
        removal, metric = path['removal'], path['cost_metric']
        tolerance = threshold(noise, metric)
        roles = {}
        for role in ROLES:
            harm = summarize_values(cost_values(pairs, 'config', 'full_F', role, metric), tolerance)
            avoided = summarize_values(cost_values(pairs, removal, 'config', role, metric), tolerance)
            joint = [p['pair_id'] for p, h in zip(pairs, harm['values']) if finite(h) and h <= tolerance
                and finite(relative_delta(p, 'config', 'full_F', role, 'weak_margin'))
                and relative_delta(p, 'config', 'full_F', role, 'weak_margin') > 2*noise['weak_margin']['epsilon']]
            checks = dict(complete=harm['complete'] and avoided['complete'],
                not_reproducing_cost=harm['complete'] and harm['median'] <= tolerance,
                better_than_removal=avoided['complete'] and avoided['median'] > tolerance,
                same_pair_joint_pattern=len(joint) >= 9)
            roles[role] = dict(checks=checks, passed=all(checks.values()), config_cost=harm,
                              removal_cost_avoided=avoided, joint_pair_ids=joint,
                              joint_source_ids=sources_for(pairs, joint, role))
        paths.append(dict(**path, roles=roles, passed=benefit['passed'] and all(r['passed'] for r in roles.values())))
    return dict(passed=initial_gate['passed'] and any(p['passed'] for p in paths) and not veto,
                benefit=benefit, paths=paths, consistent_new_cost_veto=veto, all_costs=new_costs)


def execution_check(pairs, conditions):
    issues, fallbacks, updates = [], [], {}
    for pair in pairs:
        for condition in conditions:
            record = pair['conditions'][condition]
            update = record['update']
            if not finite(update.get('l2')) or update['l2'] <= 0:
                issues.append(dict(pair_id=pair['pair_id'], condition=condition, reason='nonfinite_or_zero_actual_update'))
            require(record['optimizer_step_before'] == 2304 and record['optimizer_step_after_used'] == 2305,
                    'Wrong optimizer state/step in single-update receipt')
            if condition == 'config':
                operator = record['operator']
                require(operator is not None and operator['requested_mode'] == 'config', 'Missing operator receipt')
                reason = operator.get('fallback_reason')
                if reason is not None:
                    fallbacks.append(dict(pair_id=pair['pair_id'], reason=reason,
                        update_sources=pair['update_sources'], observer_sources=pair['observer_sources']))
    for condition in conditions:
        updates[condition] = {field: summarize_values(
            [p['conditions'][condition]['update'].get(field) for p in pairs], 0.)
            for field in ['l2', 'input_gradient_l2', 'cosine_full', 'norm_ratio_full', 'cosine_negative_input']}
    reasons = sorted({r['reason'] for r in fallbacks})
    return dict(passed=not issues, issues=issues, actual_updates=updates, fallback_pairs=fallbacks,
                fallback_total=len(fallbacks), fallback_counts_by_reason={k: sum(r['reason']==k for r in fallbacks) for k in reasons},
                fallback_note='Prespecified numerical/structural fallbacks remain in all16 pair statistics;no additional zero-fallback gate.',
                scope='Reads recorded actual parameter delta;does not infer Adam behavior from raw gradient cosines.')


def effect_rows(pairs, conditions, noise):
    rows = []
    for pair in pairs:
        for condition in conditions:
            record = pair['conditions'][condition]
            for role in ROLES:
                baseline, after = pair['baseline'][role], record['roles'][role]
                row = dict(pair_id=pair['pair_id'], condition=condition, role=role,
                    update_index=pair['update_index'], observer_index=pair['observer_index'],
                    sources=json.dumps(pair[role+'_sources'], ensure_ascii=False),
                    weak_count=baseline['counts']['weak'], reliable_count=baseline['counts']['reliable'],
                    hardbg_count=baseline['counts']['hardbg'], actual_update_l2=record['update']['l2'],
                    input_gradient_l2=record['update'].get('input_gradient_l2'),
                    actual_update_cosine_full=record['update'].get('cosine_full'),
                    actual_update_norm_ratio_full=record['update'].get('norm_ratio_full'),
                    actual_update_cosine_negative_input=record['update'].get('cosine_negative_input'),
                    grouped_gradient_dot_actual_update_json=json.dumps(record['update'].get('gradient_dot_delta', {})),
                    operator_fallback_reason=(record.get('operator') or {}).get('fallback_reason'))
                for metric in METRICS:
                    row[metric+'_before'] = baseline[metric]
                    row[metric+'_repeat'] = pair['repeat'][role][metric]
                    row[metric+'_after'] = after[metric]
                    row[metric+'_delta'] = metric_delta(pair, condition, role, metric)
                    row[metric+'_advantage_vs_full'] = relative_delta(pair, condition, 'full_F', role, metric)
                    row[metric+'_epsilon'] = noise[metric]['epsilon']
                row['gradient_diagnostics_json'] = json.dumps(pair.get('gradients', {}), ensure_ascii=False)
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-gate', action='store_true')
    args = parser.parse_args()
    if args.print_gate:
        print(json.dumps(GATE_SPEC, ensure_ascii=False, indent=2));return
    pairs, paths, has_config = load_pairs()
    conditions = BASE_CONDITIONS + (('config',) if has_config else ())
    noise = noise_floors(pairs)
    aggregate = summaries(pairs, conditions, noise)
    initial_gate = stage_one(pairs, aggregate, noise)
    candidate_gate = stage_two(pairs, aggregate, noise, initial_gate) if has_config else None
    execution = execution_check(pairs, conditions)
    if not execution['passed']:
        status = 'EXECUTION_ISSUE'
    elif not initial_gate['passed']:
        status = 'TRIGGER_NOT_SUPPORTED'
    elif not has_config:
        status = 'CONFIG_SINGLE_STEP_AUTHORIZED'
    else:
        status = 'CONFIG_256_AUTHORIZED' if candidate_gate['passed'] else 'CONFIG_NOT_SUPPORTED'
    rows = effect_rows(pairs, conditions, noise)
    table_path = E / 'TRAIN_UPDATE_EFFECT.csv'
    temporary = table_path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temporary.replace(table_path)
    sources = {str(p): sha(p) for p in paths}
    sources[str(Path(__file__))] = sha(__file__)
    for name in ['PROTOCOL.md', 'PROTOCOL_FREEZE.json']:
        if (E/name).exists():sources[str(E/name)] = sha(E/name)
    save_json(E / 'OPERATOR_CHECK.json', dict(status=status, created_unix=time.time(), gate_spec=GATE_SPEC,
        pairs=16, conditions=list(conditions), uses_development=False, no_significance_test=True,
        noise_floors=noise, aggregate=aggregate, stage_one=initial_gate, stage_two=candidate_gate,
        execution=execution, source_sha256=sources, effect_table_sha256=sha(table_path),
        scope='Single independent Adam updates from identical F2304 state;no accumulated16-step training.',
        followup='Only execute the next prespecified stage when authorized;never tune this gate or drop pairs based on measured results.'))
    print(json.dumps(dict(status=status, stage_one_passed=initial_gate['passed'],
        config_measured=has_config, supported_paths=initial_gate['supported_paths'])), flush=True)


if __name__ == '__main__':
    main()
