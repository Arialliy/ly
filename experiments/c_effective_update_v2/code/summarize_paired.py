"""Frozen object/track/source summaries for C/F effective-update v2.

Pure Python, no model construction and no optimizer calls.  Each removal,
source relation and cost path is evaluated independently on all qualified data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

E = Path(__file__).resolve().parents[1]
OLD = E.parent / 'c_effective_update_v1'
ROLES = ('update', 'observer')
CONDITIONS = ('full_F', 'no_keep', 'no_track')
COMMON = (6, 8, 9, 10, 12, 13, 15)
METRICS = {
    'weak': ('q_positive', 'q_negative', 'margin', 'peak_score', 'peak_shift'),
    'reliable_trusted': ('q_positive', 'peak_score', 'peak_shift'),
    'reliable_uncertain': ('q_positive', 'peak_score', 'peak_shift'),
    'background': ('mean_logit',),
}
VALUE_TYPES = ('before', 'after', 'delta', 'advantage_vs_full')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def median(values):
    values = list(values)
    return statistics.median(values) if values else None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def flat_values(before, after, full, cohort):
    values = {}
    for metric in METRICS[cohort]:
        b, a, f = before[metric], after[metric], full[metric]
        require(all(finite(v) for v in (b, a, f)), 'Nonfinite object readout: ' + metric)
        # m was already evaluated in original FP32; never recompute from doubles.
        values.update({metric + '_before': b, metric + '_after': a,
                       metric + '_delta': a - b,
                       metric + '_advantage_vs_full': (a - b) - (f - b)})
    return values


def averaged(rows, reducer=mean):
    require(bool(rows), 'Cannot aggregate an empty group')
    keys = list(rows[0]['values'])
    require(all(set(r['values']) == set(keys) for r in rows), 'Metric keys differ')
    return {k: reducer(r['values'][k] for r in rows) for k in keys}


def index_objects(objects, cohort):
    key_name = 'reference_id' if cohort == 'background' else 'object_key'
    result = {o[key_name]: o for o in objects}
    require(len(result) == len(objects), 'Duplicate context/object ID')
    return result


def validate_frozen_cohorts(pairs, plan):
    """Anchor every readout to the frozen cohort, not merely to other readouts.

    If the baseline and all conditions accidentally omitted the same objects,
    their mutual equality would not detect the altered source denominator.
    Missing or changed identities must fail before any gate is evaluated.
    """
    for pair in pairs:
        for role in ROLES:
            index = pair[role + '_index']
            entry = plan['entries'][str(index)]
            path = Path(entry['cohort_path'])
            require(sha(path) == entry['cohort_sha256'], 'Frozen cohort hash changed')
            cohort = read(path)
            counts = cohort['counts']
            require(cohort['index'] == index, 'Frozen cohort entry differs')
            require(entry['cohort_counts'] == counts, 'Manifest/cohort counts differ')
            require(counts['weak'] > 0 and counts['hardbg'] > 0, 'Pair role is ineligible')
            weak_indices = set(cohort['weak_target_indices'])
            expected_weak = {'entry=%d|target=%d' % (index, obj['target_index']): obj
                             for obj in cohort['target_objects'] if obj['target_index'] in weak_indices}
            expected_reliable = {'entry=%d|reliable=%d' % (index, ordinal): obj
                                 for ordinal, obj in enumerate(cohort['reliable_objects'])}
            expected_background = {'entry=%d|background=%d' % (index, bg): (bg, region)
                                  for bg, region in zip(cohort['background_indices'], cohort['background_regions'])}
            require(len(expected_weak) == len(weak_indices) == counts['weak'], 'Weak cohort identity/count mismatch')
            require(len(expected_reliable) == counts['reliable'], 'Reliable cohort identity/count mismatch')
            require(len(expected_background) == len(cohort['background_regions']) == counts['hardbg'],
                    'Background cohort identity/count mismatch')
            measurements = [pair['baseline'][role]] + [pair['conditions'][c]['roles'][role] for c in CONDITIONS]
            for measured in measurements:
                require(measured['counts'] == counts, 'Readout/cohort counts differ')
                for name, expected, kind in [('weak_objects', expected_weak, 'weak'),
                                              ('reliable_objects', expected_reliable, 'reliable_trusted')]:
                    current = index_objects(measured[name], kind)
                    require(set(current) == set(expected), 'Readout omits/adds frozen identities: ' + name)
                    for key, original in expected.items():
                        require(all(field in current[key] and current[key][field] == value
                                    for field, value in original.items()), 'Original object metadata changed: ' + key)
                current_background = index_objects(measured['background_objects'], 'background')
                require(set(current_background) == set(expected_background), 'Readout omits/adds frozen background IDs')
                for key, (bg, region) in expected_background.items():
                    obj = current_background[key]
                    require(obj['background_index'] == bg and obj['region'] == region,
                            'Fixed background ROI changed: ' + key)


def object_rows(pair, condition, role, cohort):
    list_name = 'background_objects' if cohort == 'background' else (
        'weak_objects' if cohort == 'weak' else 'reliable_objects')
    sets = [pair['baseline'][role][list_name],
            pair['conditions'][condition]['roles'][role][list_name],
            pair['conditions']['full_F']['roles'][role][list_name]]
    indexed = [index_objects(x, cohort) for x in sets]
    require(set(indexed[0]) == set(indexed[1]) == set(indexed[2]), 'Frozen objects changed')
    rows = []
    for key, before in indexed[0].items():
        track = before.get('scoped_track_key')
        if cohort == 'reliable_trusted' and track is None:
            continue
        if cohort == 'reliable_uncertain' and track is not None:
            continue
        if cohort == 'weak':
            require(track is not None, 'Untrusted weak identity cannot count as a trajectory')
        after, full = indexed[1][key], indexed[2][key]
        frame_key = before['source_background_frame_key'] if cohort == 'background' else before['source_object_frame_key']
        for current in (after, full):
            require(current.get('scoped_track_key') == before.get('scoped_track_key'), 'Track identity changed')
            current_key = current['source_background_frame_key'] if cohort == 'background' else current['source_object_frame_key']
            require(current_key == frame_key and current['source'] == before['source'], 'Physical identity changed')
        rows.append({'source': before['source'], 'track': track, 'frame_key': frame_key,
                     'object_key': key, 'values': flat_values(before, after, full, cohort)})
    return rows


def pair_units(pair, condition, role, cohort):
    """Context -> physical frame/ROI -> trusted trajectory or source proxy."""
    objects = object_rows(pair, condition, role, cohort)
    frames = defaultdict(list)
    for obj in objects:
        frames[(obj['source'], obj['track'], obj['frame_key'])].append(obj)
    grouped = defaultdict(list)
    for (source, track, frame_key), members in sorted(frames.items(), key=lambda x: str(x[0])):
        grouped[(source, track)].append({'values': averaged(members),
            'frame_key': frame_key, 'context_count': len(members)})
    units = []
    for (source, track), members in sorted(grouped.items(), key=lambda x: str(x[0])):
        units.append({'subset': 'pair', 'source_relation': pair['source_relation'],
            'pair_id': pair['pair_id'], 'condition': condition, 'role': role, 'cohort': cohort,
            'level': 'track' if track is not None else 'source_proxy',
            'source': source, 'track': track, 'pair_count': 1,
            'frame_count': len(members), 'context_count': sum(x['context_count'] for x in members),
            'values': averaged(members)})
    return units


def source_units(units):
    groups = defaultdict(list)
    for unit in units:
        groups[unit['source']].append(unit)
    return [{'source': source, 'track_count': sum(x['track'] is not None for x in members),
             'unit_count': len(members), 'values': averaged(members)}
            for source, members in sorted(groups.items())]


def balanced(units):
    sources = source_units(units)
    return {'source_count': len(sources), 'sources': sources,
            'values': averaged(sources, reducer=median) if sources else {}}


def subset_summary(units, ids, relation):
    selected = [u for u in units if u['pair_id'] in ids and
                (relation == 'all_relations_descriptive' or u['source_relation'] == relation)]
    groups = defaultdict(list)
    for unit in selected:
        # A trusted trajectory receives equal pair weight before source weighting.
        # Background/uncertain identities have no track; source proxy gets equal pair weight.
        groups[(unit['source'], unit['track'])].append(unit)
    combined = []
    for (source, track), members in sorted(groups.items(), key=lambda x: str(x[0])):
        require(len({m['pair_id'] for m in members}) == len(members), 'Repeated per-pair unit')
        combined.append({'source': source, 'track': track, 'pair_count': len(members),
            'pair_ids': sorted(m['pair_id'] for m in members),
            'frame_count': sum(m['frame_count'] for m in members),
            'context_count': sum(m['context_count'] for m in members),
            'values': averaged(members)})
    return {'pair_ids': sorted({x['pair_id'] for x in selected}),
            'track_count': sum(x['track'] is not None for x in combined),
            'context_count': sum(x['context_count'] for x in combined),
            'pair_frame_count': sum(x['frame_count'] for x in combined),
            'units': combined, **balanced(combined)}


def benefit(values, spec):
    ep = spec['metric_epsilons']
    checks = {
        'absolute_target_improves': values.get('q_positive_delta', -math.inf) > ep['q_positive'],
        'absolute_margin_improves': values.get('margin_delta', -math.inf) > ep['margin'],
        'margin_better_than_full': values.get('margin_advantage_vs_full', -math.inf) > 2 * ep['margin'],
        'target_not_worse_than_full': values.get('q_positive_advantage_vs_full', -math.inf) >= -2 * ep['q_positive'],
    }
    return {'passed': all(checks.values()), 'checks': checks}


def cost_receipt(units, cost):
    aggregate = balanced(units)
    key = cost['metric'] + '_advantage_vs_full'
    harm = cost['harm_sign'] * aggregate['values'][key] if aggregate['values'] else None
    sources = [{'source': s['source'], 'harm': cost['harm_sign'] * s['values'][key],
                'above_threshold': cost['harm_sign'] * s['values'][key] > cost['threshold']}
               for s in aggregate['sources']]
    return {'available': harm is not None, 'source_balanced_harm': harm,
            'threshold': cost['threshold'], 'passed': harm is not None and harm > cost['threshold'],
            'sources': sources, 'harmful_sources': [s['source'] for s in sources if s['above_threshold']]}


def first_stage(pairs, units, aggregates, spec):
    relations = sorted({p['source_relation'] for p in pairs})
    paths, supported = [], []
    for relation in relations:
        relation_pairs = [p for p in pairs if p['source_relation'] == relation]
        observer_sources = sorted({u['source'] for u in units if u['cohort'] == 'weak'
            and u['role'] == 'observer' and u['condition'] == 'full_F' and u['source_relation'] == relation})
        for removal in ('no_keep', 'no_track'):
            all_roles = {role: benefit(aggregates['all_qualified_9'][relation][removal][role]['weak']['values'], spec)
                         for role in ROLES}
            for cost_name, cost in spec['costs'].items():
                pair_details, supporting_tracks, supporting_observer_sources, supporting_pairs = [], set(), set(), []
                for pair in relation_pairs:
                    roles = {}
                    for role in ROLES:
                        subset = [u for u in units if u['pair_id'] == pair['pair_id'] and
                                  u['condition'] == removal and u['role'] == role]
                        weak = [u for u in subset if u['cohort'] == 'weak']
                        good = [u for u in weak if benefit(u['values'], spec)['passed']]
                        c = cost_receipt([u for u in subset if u['cohort'] == cost['cohort']], cost)
                        roles[role] = {'qualifying_weak_tracks': [u['track'] for u in good],
                            'qualifying_weak_sources': sorted({u['source'] for u in good}),
                            'all_weak_track_count': len(weak), 'cost': c,
                            'passed': bool(good) and c['passed']}
                    joint = all(r['passed'] for r in roles.values())
                    pair_details.append({'pair_id': pair['pair_id'], 'roles': roles, 'passed': joint})
                    if joint:
                        supporting_pairs.append(pair['pair_id'])
                        for r in roles.values():
                            supporting_tracks.update(r['qualifying_weak_tracks'])
                        supporting_observer_sources.update(roles['observer']['qualifying_weak_sources'])
                checks = {
                    'all_qualified_source_balanced_benefit_both_roles': all(r['passed'] for r in all_roles.values()),
                    'multiple_supporting_weak_tracks': len(supporting_tracks) >= 2,
                    'multiple_supporting_observer_sources': len(supporting_observer_sources) >= 2,
                    'strict_majority_eligible_observer_sources': len(supporting_observer_sources) > len(observer_sources) / 2,
                }
                result = {'source_relation': relation, 'removal': removal, 'cost': cost_name,
                    'passed': all(checks.values()), 'checks': checks,
                    'all_qualified_benefit': all_roles, 'eligible_observer_sources': observer_sources,
                    'supporting_pair_ids': supporting_pairs, 'supporting_weak_tracks': sorted(supporting_tracks),
                    'supporting_observer_sources': sorted(supporting_observer_sources), 'pairs': pair_details}
                paths.append(result)
                if result['passed']:
                    supported.append({'source_relation': relation, 'removal': removal, 'cost': cost_name})
    return {'passed': bool(supported), 'supported_paths': supported, 'paths': paths}


def execution(pairs):
    issues = []
    for field in ('initial_model_sha256', 'initial_optimizer_sha256', 'initial_rng_sha256'):
        if len({p.get(field) for p in pairs}) != 1 or any(not p.get(field) for p in pairs):
            issues.append({'reason': 'Initial states differ', 'field': field})
    for pair in pairs:
        if pair.get('status') != 'COMPLETE' or pair.get('new_optimizer_steps') != 0 or pair.get('historical_parameter_updates_per_copy') != 1:
            issues.append({'pair_id': pair['pair_id'], 'reason': 'Invalid independent replay receipt'})
        checks = pair.get('replay_checks', {})
        for field in ('all_reconstructed_model_hashes_exact', 'old_baselines_exact', 'old_update_metrics_exact',
                      'optimizer_untouched', 'no_backward', 'bn_frozen', 'rng_unchanged'):
            if checks.get(field) is not True:
                issues.append({'pair_id': pair['pair_id'], 'reason': 'Replay check absent or failed', 'field': field})
        if pair['reuse_mode'] == 'original_pair' and checks.get('old_observer_metrics_exact') is not True:
            issues.append({'pair_id': pair['pair_id'], 'reason': 'Original observer readouts do not match'})
        for condition in CONDITIONS:
            rec = pair['conditions'][condition]
            if not finite(rec['update']['l2']) or rec['update']['l2'] <= 0:
                issues.append({'pair_id': pair['pair_id'], 'condition': condition, 'reason': 'Historical update invalid'})
            if rec.get('reconstructed_model_hash_exact') is not True:
                issues.append({'pair_id': pair['pair_id'], 'condition': condition, 'reason': 'Endpoint hash not exact'})
            if rec.get('optimizer_step_before') != 2304 or rec.get('optimizer_step_after_used') != 2305:
                issues.append({'pair_id': pair['pair_id'], 'condition': condition, 'reason': 'Wrong historical optimizer steps'})
    return {'passed': not issues, 'issues': issues, 'new_optimizer_steps': sum(p.get('new_optimizer_steps', 0) for p in pairs),
            'historical_copies_reused': len(pairs) * 3, 'new_config_steps': 0, 'new_256_step_branches': 0}


def csv_rows(units, aggregates):
    rows = []
    for u in units:
        rows.append({k: v for k, v in u.items() if k != 'values'} | u['values'])
    for subset, by_relation in aggregates.items():
        for relation, by_condition in by_relation.items():
            for condition, by_role in by_condition.items():
                for role, by_cohort in by_role.items():
                    for cohort, result in by_cohort.items():
                        common = {'subset': subset, 'source_relation': relation, 'condition': condition,
                            'role': role, 'cohort': cohort, 'pair_count': len(result['pair_ids'])}
                        for unit in result['units']:
                            rows.append(common | {'level': 'cross_pair_track' if unit['track'] is not None else 'cross_pair_source_proxy',
                                'source': unit['source'], 'track': unit['track'], 'pair_count': unit['pair_count'],
                                'frame_count': unit['frame_count'], 'context_count': unit['context_count']} | unit['values'])
                        for source in result['sources']:
                            rows.append(common | {'level': 'source', 'source': source['source'],
                                'track_count': source['track_count']} | source['values'])
                        rows.append(common | {'level': 'source_balanced_median', 'source_count': result['source_count'],
                            'track_count': result['track_count'], 'frame_count': result['pair_frame_count'],
                            'context_count': result['context_count']} | result['values'])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-code', action='store_true')
    args = parser.parse_args()
    spec = read(E / 'GATE_SPEC.json')
    require(sha(spec['noise_source']) == spec['noise_source_sha256'], 'Old noise source changed')
    old = read(spec['noise_source'])
    require(spec['noise_floors'] == old['noise_floors'], 'Noise was re-estimated or altered')
    if args.check_code:
        print(json.dumps({'status': 'GATE_SPEC_VALID', 'gate_sha256': sha(E / 'GATE_SPEC.json')}))
        return
    plan = read(E / 'PAIR_PLAN.json')
    require([p['pair_id'] for p in plan['pairs']] == spec['registered_pair_ids'], 'Manifest differs from fixed gate')
    paths = [E / 'pairs' / ('pair_%02d.json' % p['pair_id']) for p in plan['pairs']]
    pairs = [read(p) for p in paths]
    for actual, declared in zip(pairs, plan['pairs']):
        for field in ('pair_id', 'update_index', 'observer_index', 'source_relation', 'reuse_mode'):
            require(actual[field] == declared[field], 'Pair identity changed: ' + field)
        require(set(actual['conditions']) == set(CONDITIONS), 'This stage only summarizes the three fixed existing interventions')
    validate_frozen_cohorts(pairs, plan)
    units = [u for p in pairs for c in CONDITIONS for r in ROLES for cohort in METRICS for u in pair_units(p, c, r, cohort)]
    aggregates = {}
    for subset, ids in [('common_old_7', set(COMMON)), ('all_qualified_9', {p['pair_id'] for p in pairs})]:
        relations = sorted({p['source_relation'] for p in pairs if p['pair_id'] in ids}) + ['all_relations_descriptive']
        aggregates[subset] = {relation: {c: {r: {cohort: subset_summary(
            [u for u in units if u['condition'] == c and u['role'] == r and u['cohort'] == cohort], ids, relation)
            for cohort in METRICS} for r in ROLES} for c in CONDITIONS} for relation in relations}
    gate = first_stage(pairs, units, aggregates, spec)
    executed = execution(pairs)
    status = 'EXECUTION_ISSUE' if not executed['passed'] else (
        'CONFIG_SINGLE_STEP_AUTHORIZED' if gate['passed'] else 'TRIGGER_NOT_SUPPORTED')
    rows = csv_rows(units, aggregates)
    first = ['subset', 'source_relation', 'pair_id', 'condition', 'role', 'cohort', 'level', 'source', 'track',
             'pair_count', 'source_count', 'track_count', 'frame_count', 'context_count']
    fields = first + sorted(set().union(*(set(row) for row in rows)) - set(first))
    table = E / 'PAIRED_EFFECT.csv'
    temporary = table.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(table)
    evidence = [Path(__file__), E / 'GATE_SPEC.json', E / 'PAIR_PLAN.json', E / 'PAIR_MANIFEST.csv',
                E / 'PROTOCOL.md', E / 'PROTOCOL_FREEZE.json', Path(spec['noise_source'])] + paths
    save(E / 'DECISION.json', {'status': status, 'gate_spec_sha256': sha(E / 'GATE_SPEC.json'),
        'pair_ids': [p['pair_id'] for p in pairs], 'common_old_pair_ids': list(COMMON),
        'new_pair_ids': [p['pair_id'] for p in pairs if p['pair_id'] not in COMMON],
        'new_optimizer_steps': executed['new_optimizer_steps'], 'execution': executed,
        'source_relation_counts': {rel: sum(p['source_relation'] == rel for p in pairs)
                                   for rel in sorted({p['source_relation'] for p in pairs})},
        'noise_reused_unchanged': spec['noise_floors'], 'aggregate': aggregates,
        'stage_one': gate, 'config_measured': False, 'candidate_gate': None,
        'uses_development': False, 'uses_confirmation': False,
        'limitations': ['Nine pairs exhaust this prequalified pool; 224 other training entries have unknown exact peak eligibility.',
            'A source-balanced directional gate is not statistical significance or evidence of independent-source generalization.',
            'Trusted weak trajectories and identity-uncertain reliable source proxies are reported separately.',
            'Repeated physical frames across distinct pairs are equal-pair trajectory estimates, not independent samples.',
            'No unsupported result establishes absence of conflict or ConFIG ineffectiveness.'],
        'effect_table_rows': len(rows), 'effect_table_sha256': sha(table),
        'source_sha256': {str(p): sha(p) for p in evidence}})
    print(json.dumps({'status': status, 'supported_paths': gate['supported_paths'],
                      'new_optimizer_steps': executed['new_optimizer_steps'], 'csv_rows': len(rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
