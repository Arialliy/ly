"""Independent standard-library audit of raw pair results; no execution imports."""
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

E = Path(__file__).resolve().parents[1]
OLD = E.parent / 'c_effective_update_v1'
METRICS = ('q_positive', 'q_negative', 'margin', 'peak_score', 'peak_shift')
CONDITIONS = ('full_F', 'no_keep', 'no_track')
SUFFIXES = ('before', 'after', 'delta', 'advantage_vs_full')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def avg(values):
    return {key: mean([x[key] for x in values]) for key in values[0]}


def check_close(a, b, label):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (label, a, b)


def objects(result, role, cohort, condition=None):
    record = result['baseline'][role] if condition is None else result['conditions'][condition]['roles'][role]
    if cohort == 'weak':
        return record['weak_objects']
    if cohort == 'background':
        return record['background_objects']
    return [obj for obj in record['reliable_objects']
            if (obj['scoped_track_key'] is not None) == (cohort == 'reliable_trusted')]


def pair_units(result, role, cohort, condition):
    metrics = METRICS if cohort == 'weak' else ('mean_logit',) if cohort == 'background' else ('q_positive', 'peak_score', 'peak_shift')
    key = 'reference_id' if cohort == 'background' else 'object_key'
    before = {o[key]: o for o in objects(result, role, cohort)}
    after = {o[key]: o for o in objects(result, role, cohort, condition)}
    full = {o[key]: o for o in objects(result, role, cohort, 'full_F')}
    assert before.keys() == after.keys() == full.keys()
    physical = defaultdict(list)
    context_counts = defaultdict(int)
    for k, obj in before.items():
        values = {}
        for metric in metrics:
            b, a, f = obj[metric], after[k][metric], full[k][metric]
            for suffix, value in zip(SUFFIXES, (b, a, a-b, a-f)):
                values[f'{metric}_{suffix}'] = value
        source = obj['source']
        track = obj.get('scoped_track_key') if cohort in ('weak', 'reliable_trusted') else None
        frame = obj['source_object_frame_key']
        assert (source, track, frame) == (after[k]['source'], after[k].get('scoped_track_key') if track else None,
                                          after[k]['source_object_frame_key'])
        physical[(source, track, frame)].append(values)
        context_counts[(source, track)] += 1
    tracks = defaultdict(list)
    for (source, track, frame), contexts in physical.items():
        tracks[(source, track)].append(avg(contexts))
    return {key: {'values': avg(frames), 'frame_count': len(frames),
                  'context_count': context_counts[key]} for key, frames in tracks.items()}


def benefit(values, eps):
    return (values['q_positive_delta'] > eps['q_positive']
            and values['margin_delta'] > eps['margin']
            and values['margin_advantage_vs_full'] > 2*eps['margin']
            and values['q_positive_advantage_vs_full'] >= -2*eps['q_positive'])


def main():
    plan = read(E / 'PAIR_PLAN.json')
    decision = read(E / 'DECISION.json')
    spec = read(E / 'GATE_SPEC.json')
    pairs = {p['pair_id']: read(E / 'pairs' / f"pair_{p['pair_id']:02d}.json") for p in plan['pairs']}
    originals = {p['pair_id']: read(OLD / 'pairs' / f"pair_{p['pair_id']:02d}.json") for p in read(OLD / 'PAIR_PLAN.json')['pairs']}
    old_entries = {p[role+'_index']: (p, role) for p in originals.values() for role in ('update', 'observer')}
    endpoint_audit = read(E / 'replay/ENDPOINT_AUDIT.json')
    assert endpoint_audit['status'] == 'PASS' and endpoint_audit['checked_endpoints'] == 27
    endpoint_sha = sha(E / 'replay/endpoints.npz')
    assert endpoint_sha == endpoint_audit['endpoints_npz_sha256']
    legacy_scalars_checked = 0
    weak_tracks, weak_sources = set(), set()
    uncertain_count, weak_contexts = 0, 0
    for pair_id, p in pairs.items():
        assert p['status'] == 'COMPLETE' and p['new_optimizer_steps'] == 0
        assert p['protocol_sha256'] == sha(E / 'PROTOCOL_FREEZE.json')
        assert p['exact_endpoint_archive_sha256'] == endpoint_sha
        original = originals[p['reuse_update_from_pair_id']]
        for k in ('initial_checkpoint_sha256', 'initial_model_sha256', 'initial_optimizer_sha256', 'initial_rng_sha256', 'parameter_names_in_vector_order'):
            assert p[k] == original[k], (pair_id, k)
        for role in ('update', 'observer'):
            old, old_role = old_entries[p[role+'_index']]
            for k, value in old['baseline'][old_role].items():
                assert p['baseline'][role][k] == value, (pair_id, role, 'baseline', k)
                legacy_scalars_checked += 1
            weak = p['baseline'][role]['weak_objects']
            weak_contexts += len(weak)
            weak_tracks.update(x['scoped_track_key'] for x in weak)
            weak_sources.update(x['source'] for x in weak)
            uncertain_count += sum(x['scoped_track_key'] is None for x in p['baseline'][role]['reliable_objects'])
            assert all(x['scoped_track_key'] is not None for x in weak)
        for condition in CONDITIONS:
            new, old = p['conditions'][condition], original['conditions'][condition]
            assert new['model_after_sha256'] == old['model_after_sha256']
            assert new['reconstructed_model_hash_exact'] and new['new_optimizer_steps'] == 0
            for k in ('update', 'optimizer_step_before', 'optimizer_step_after_used', 'optimizer_after_sha256'):
                assert new[k] == old[k]
            for role in ('update', 'observer') if pair_id in plan['common_old_pair_ids'] else ('update',):
                for k, value in old['roles'][role].items():
                    assert new['roles'][role][k] == value, (pair_id, condition, role, k)
                    legacy_scalars_checked += 1
    assert len(weak_tracks) == 39 and len(weak_sources) == 5 and uncertain_count == 655
    units = {(pid, condition, role, cohort): pair_units(p, role, cohort, condition)
             for pid, p in pairs.items() for condition in CONDITIONS for role in ('update', 'observer')
             for cohort in ('weak', 'reliable_trusted', 'reliable_uncertain', 'background')}
    with (E / 'PAIRED_EFFECT.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    csv_index = {(int(row['pair_id']), row['condition'], row['role'], row['cohort'], row['source'], row['track'] or None): row
                 for row in rows if row['subset'] == 'pair'}
    compared_cells = 0
    for (pid, condition, role, cohort), group in units.items():
        for (source, track), unit in group.items():
            row = csv_index[(pid, condition, role, cohort, source, track)]
            assert int(row['frame_count']) == unit['frame_count']
            assert int(row['context_count']) == unit['context_count']
            for key, value in unit['values'].items():
                check_close(value, float(row[key]), (pid, condition, role, cohort, source, track, key))
                compared_cells += 1
    weak_aggregate = []
    for subset, relationships in decision['aggregate'].items():
        allowed_ids = set(plan['common_old_pair_ids']) if subset == 'common_old_7' else set(pairs)
        for relation, conditions in relationships.items():
            chosen = [pid for pid, p in pairs.items() if pid in allowed_ids and (relation == 'all_relations_descriptive' or p['source_relation'] == relation)]
            for condition, roles in conditions.items():
                for role, cohorts in roles.items():
                    for cohort, target in cohorts.items():
                        per_track = defaultdict(list)
                        for pid in chosen:
                            for key, unit in units[(pid, condition, role, cohort)].items():
                                per_track[key].append(unit['values'])
                        merged = {key: avg(values) for key, values in per_track.items()}
                        source_data = defaultdict(list)
                        for (source, track), values in merged.items():
                            source_data[source].append(values)
                        source_values = {s: avg(values) for s, values in source_data.items()}
                        if not source_values:
                            continue
                        aggregate = {key: median([v[key] for v in source_values.values()]) for key in next(iter(source_values.values()))}
                        for key, value in aggregate.items():
                            check_close(value, target['values'][key], (subset, relation, condition, role, cohort, key))
                            compared_cells += 1
                        actual_sources = {x['source']: x for x in target['sources']}
                        assert set(actual_sources) == set(source_values)
                        for source, values in source_values.items():
                            for key, value in values.items():
                                check_close(value, actual_sources[source]['values'][key], (subset, relation, condition, role, cohort, source, key))
                                compared_cells += 1
                        if subset == 'all_qualified_9' and relation != 'all_relations_descriptive' and cohort == 'weak':
                            weak_aggregate.append({'relation': relation, 'condition': condition, 'role': role,
                                'source_count': len(source_values), 'track_count': len(merged),
                                'values': {k: v for k, v in aggregate.items() if k.startswith(('q_positive_', 'q_negative_', 'margin_'))},
                                'benefit': benefit(aggregate, spec['metric_epsilons'])})
    pair_benefit = []
    for relation in ('same_source', 'cross_source'):
        for condition in ('no_keep', 'no_track'):
            records = []
            for pid, p in pairs.items():
                if p['source_relation'] != relation:
                    continue
                qualifying = {role: sorted(track for (source, track), unit in units[(pid, condition, role, 'weak')].items()
                                     if benefit(unit['values'], spec['metric_epsilons']))
                              for role in ('update', 'observer')}
                records.append({'pair_id': pid, 'qualifying': qualifying,
                                'both_roles': all(qualifying.values())})
            pair_benefit.append({'relation': relation, 'condition': condition,
                                 'both_roles_count': sum(x['both_roles'] for x in records), 'pairs': records})
    cost_paths_checked = 0
    for path in decision['stage_one']['paths']:
        for item in path['pairs']:
            pid, condition, cost_name = item['pair_id'], path['removal'], path['cost']
            cost_spec = spec['costs'][cost_name]
            for role, target in item['roles'].items():
                qualifying = sorted(track for (source, track), unit in units[(pid, condition, role, 'weak')].items()
                                    if benefit(unit['values'], spec['metric_epsilons']))
                assert qualifying == target['qualifying_weak_tracks']
                by_source = defaultdict(list)
                for (source, track), unit in units[(pid, condition, role, cost_spec['cohort'])].items():
                    by_source[source].append(cost_spec['harm_sign'] * unit['values'][cost_spec['metric']+'_advantage_vs_full'])
                harms = {s: mean(v) for s, v in by_source.items()}
                harm = median(harms.values()) if harms else None
                assert bool(harms) == target['cost']['available']
                if harm is not None:
                    check_close(harm, target['cost']['source_balanced_harm'], (pid, condition, role, cost_name))
                    assert (harm > cost_spec['threshold']) == target['cost']['passed']
                assert bool(qualifying and harm is not None and harm > cost_spec['threshold']) == target['passed']
                cost_paths_checked += 1
            assert all(x['passed'] for x in item['roles'].values()) == item['passed']
    assert not any(x['both_roles_count'] for x in pair_benefit)
    assert decision['status'] == 'TRIGGER_NOT_SUPPORTED' and not decision['stage_one']['passed']
    assert decision['config_measured'] is False and decision['new_optimizer_steps'] == 0
    report = {'status': 'PASS', 'method': 'Independent Python standard-library reconstruction from raw objects; no import of production summarizer.',
              'pair_count': len(pairs), 'historical_endpoints': 27, 'new_optimizer_steps': 0,
              'legacy_scalar_or_collection_comparisons_exact': legacy_scalars_checked,
              'endpoint_archive_matches_independent_hash_audit': True,
              'trusted_weak_track_count': len(weak_tracks), 'weak_source_count': len(weak_sources),
              'weak_context_count': weak_contexts, 'uncertain_reliable_contexts_retained': uncertain_count,
              'csv_and_decision_numeric_cells_checked': compared_cells, 'numeric_comparison_absolute_tolerance': 1e-12,
              'cost_role_paths_checked': cost_paths_checked, 'weak_aggregate': weak_aggregate,
              'paired_benefit': pair_benefit, 'decision_supported': decision['status'],
              'source_sha256': {str(p.relative_to(E)): sha(p) for p in [E/'PAIR_PLAN.json', E/'PAIRED_EFFECT.csv', E/'DECISION.json', E/'GATE_SPEC.json', E/'PROTOCOL_FREEZE.json', E/'replay/ENDPOINT_AUDIT.json', Path(__file__)]},
              'pair_sha256': {str(pid): sha(E/'pairs'/f'pair_{pid:02d}.json') for pid in pairs}}
    (E/'evaluation/RESULT_REVIEW.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'pair_count', 'trusted_weak_track_count', 'csv_and_decision_numeric_cells_checked', 'cost_role_paths_checked')}))


if __name__ == '__main__':
    main()
