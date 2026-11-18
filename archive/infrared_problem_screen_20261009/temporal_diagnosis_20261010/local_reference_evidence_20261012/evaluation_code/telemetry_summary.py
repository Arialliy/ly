#!/usr/bin/env python3
"""Describe saved same-forward inputs; no model, probe, threshold or inference."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
PLANS = {'O': [0,36,72,108,120], 'S': [0,24,60,96,120]}
CONDITIONS = ['N0','N1','N2']
SEQUENCES = ['DJI_0026_2','DJI_0067_2']
COMMON = ['I','original_logit','delta_logit']
TEMPORAL = [name+str(delay) for delay in [1,4] for name in ['D','mu','s','E','abs_D','abs_E']]
METRICS = COMMON + TEMPORAL
PAIRED = ['target_minus_background_'+m for m in METRICS]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    assert rows, path
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(row, name):
    if name.startswith('abs_'):
        return abs(float(row[name[len('abs_'):]]))
    return float(row[name])


def valid(row, metric):
    if metric in COMMON:
        return True
    plain = metric.removeprefix('abs_')
    if row['_condition'] == 'N1' and not plain.startswith('D'):
        return False
    return float(row['m'+metric[-1]]) == 1.0


def outcome_groups(row):
    condition = row['condition']
    match = bool(int(row[condition+'_matched']))
    release = bool(int(row['release_matched']))
    low = bool(int(row['low_all_05']))
    n1, n2 = bool(int(row['N1_matched'])), bool(int(row['N2_matched']))
    return {
        'all_targets': True,
        'frozen_low_recovered': low and match,
        'frozen_low_unrecovered': low and not match,
        'release_success_retained': release and match,
        'release_success_lost': release and not match,
        'protected646_retained': bool(int(row['both_fixed_success'])) and match,
        'protected646_lost': bool(int(row['both_fixed_success'])) and not match,
        'N2_low_gain_vs_N1_shared_objects': low and n2 and not n1,
        'N2_low_loss_vs_N1_shared_objects': low and n1 and not n2,
    }


def aggregate(raw, background, metadata, condition, plan):
    starts = set(PLANS[plan])
    selected = [r for r in raw if int(r['window_start']) in starts]
    groups = collections.defaultdict(list)
    for row in selected:
        key = (row['sequence'], int(row['frame']), int(row['annotation_id']))
        groups[key].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        frozen = metadata[key+(plan,)]
        assert len({int(r['window_start']) for r in rows}) == len(rows)
        row = {k: frozen[k] for k in ['sequence','frame','source_frame','annotation_id','track_id','arrangement',
               'low_all_05','extreme_all_001','both_fixed_success','release_matched','N0_matched','N1_matched','N2_matched']}
        row.update(condition=condition, window_point_rows=len(rows),
                   query_y=int(rows[0]['y']), query_x=int(rows[0]['x']))
        assert all(int(r['y']) == row['query_y'] and int(r['x']) == row['query_x'] for r in rows)
        for delay in [1,4]:
            row['valid_reference_windows_d'+str(delay)] = sum(float(r['m'+str(delay)]) == 1 for r in rows)
        for metric in METRICS:
            values = [number(r,metric) for r in rows if valid(r,metric)]
            pairs = []
            for r in rows:
                bg = background[(r['sequence'],int(r['window_start']),int(r['frame']))]
                if valid(r,metric) and valid(bg,metric):
                    pairs.append(number(r,metric)-number(bg,metric))
            row[metric] = float(np.mean(values)) if values else ''
            row['target_minus_background_'+metric] = float(np.mean(pairs)) if pairs else ''
        result.append(row)
    return result


def background_aggregate(background, condition, plan):
    groups = collections.defaultdict(list)
    for (seq,start,frame), row in background.items():
        if start in PLANS[plan]:
            groups[seq,frame].append(row)
    result = []
    for (seq,frame), rows in sorted(groups.items()):
        item = dict(condition=condition, arrangement=plan, sequence=seq, frame=frame,
                    source_frame=int(rows[0]['source_frame']), window_point_rows=len(rows))
        for delay in [1,4]:
            item['valid_reference_windows_d'+str(delay)] = sum(float(r['m'+str(delay)]) == 1 for r in rows)
        for metric in METRICS:
            values = [number(r,metric) for r in rows if valid(r,metric)]
            item[metric] = float(np.mean(values)) if values else ''
        result.append(item)
    assert len(result) == 320
    return result


def distribution(rows, metrics, condition, plan, source_group, cohort, kind):
    result = []
    for metric in metrics:
        available = [row for row in rows if row[metric] != '']
        values = np.array([row[metric] for row in available], dtype=np.float64)
        assert np.isfinite(values).all()
        item = dict(condition=condition, arrangement=plan, source_group=source_group,
                    cohort=cohort, query_kind=kind, metric=metric,
                    observation_units=len(rows), valid_measure_units=len(available),
                    unique_frames=len({(r['sequence'],r['frame']) for r in rows}),
                    tracks=len({(r['sequence'],r['track_id']) for r in rows}) if kind=='target' else '',
                    clips=len({r['sequence'] for r in rows}),
                    window_point_rows=sum(r['window_point_rows'] for r in rows),
                    unit='target-frame, after window averaging' if kind=='target' else 'frame, after window averaging',
                    mean=float(values.mean()) if len(values) else '',
                    minimum=float(values.min()) if len(values) else '',
                    q10=float(np.quantile(values,.1)) if len(values) else '',
                    median=float(np.median(values)) if len(values) else '',
                    q90=float(np.quantile(values,.9)) if len(values) else '',
                    maximum=float(values.max()) if len(values) else '')
        result.append(item)
    return result


def runtime_costs(sources):
    rows = []
    for condition in CONDITIONS:
        runpath = BASE/'training'/condition/'run/RUN.json'
        run = json.loads(runpath.read_text())
        assert run['status'] == 'complete' and run['final_step'] == 512
        sources[str(runpath)] = sha(runpath)
        predictions = []
        for seq in SEQUENCES:
            path = BASE/'predictions'/condition/seq/'COMPLETE.json'
            record = json.loads(path.read_text())
            assert record['status'] == 'COMPLETE' and record['checkpoint_step'] == 512
            assert record['parameters'] == run['parameter_count']
            sources[str(path)] = sha(path)
            predictions.append(record)
        for plan, starts in PLANS.items():
            model_seconds = sum(record[plan+'_model_seconds'] for record in predictions)
            selected = [window for record in predictions for window in record['window_timing'] if window['start'] in starts]
            assert len(selected) == 10
            rows.append(dict(condition=condition, mode=run['mode'], arrangement=plan, seed=46, final_step=512,
                             parameter_count=run['parameter_count'], new_parameters=run['new_parameters'],
                             trainable_parameters=run['trainable_parameters'],
                             training_duration_seconds=run['duration_seconds'],
                             training_telemetry_seconds=run['telemetry_seconds'],
                             training_peak_allocated_bytes=run['max_cuda_memory_allocated_bytes'],
                             training_peak_reserved_bytes=run['max_cuda_memory_reserved_bytes'],
                             inference_frames=320, inference_clips=2, windows_for_arrangement=10,
                             tile_model_calls_for_arrangement=sum(w['tile_model_calls'] for w in selected),
                             inference_model_plus_cpu_copy_seconds=model_seconds,
                             inference_model_plus_cpu_copy_seconds_per_output_frame=model_seconds/320,
                             inference_passive_telemetry_seconds_for_arrangement=sum(w['provenance_seconds'] for w in selected),
                             inference_peak_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in predictions),
                             inference_peak_reserved_bytes=max(r['peak_cuda_reserved_bytes'] for r in predictions),
                             actual_unique_cached_windows=16,
                             all_cached_windows_wall_seconds=sum(r['elapsed_seconds'] for r in predictions),
                             all_cached_windows_telemetry_seconds=sum(r['telemetry_seconds'] for r in predictions),
                             timing_scope='Recorded actual jobs, no repeated benchmark; model includes CPU output copy; passive extraction and compressed writes separately excluded from plan model sum. O/S reuse overlapping windows.'))
    references = {row['arrangement']:row for row in rows if row['condition']=='N1'}
    for row in rows:
        reference = references[row['arrangement']]
        row['inference_model_plus_copy_ratio_vs_N1'] = row['inference_model_plus_cpu_copy_seconds']/reference['inference_model_plus_cpu_copy_seconds']
        row['training_wall_ratio_vs_N1'] = row['training_duration_seconds']/reference['training_duration_seconds']
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=BASE/'evaluation')
    parser.add_argument('--output', type=Path, default=BASE/'evaluation/telemetry')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    objects_path = args.evaluation/'OBJECT_COMPARISON.csv'
    object_rows = read_csv(objects_path)
    metadata = {(r['sequence'],int(r['frame']),int(r['annotation_id']),r['arrangement']):r for r in object_rows}
    assert len(metadata) == len(object_rows) == 10400
    sources = {str(objects_path): sha(objects_path)}
    all_objects, all_background, summaries = [], [], []
    coordinate_reference = None
    for condition in CONDITIONS:
        targets, background = [], {}
        coordinates = {}
        for seq in SEQUENCES:
            folder = BASE/'predictions'/condition/seq
            receipt_path, samples_path = folder/'COMPLETE.json', folder/'EVIDENCE_SAMPLES.csv'
            receipt = json.loads(receipt_path.read_text())
            assert receipt['status'] == 'COMPLETE' and receipt['condition'] == condition
            assert sha(samples_path) == receipt['telemetry_sha256']
            sources.update({str(receipt_path): sha(receipt_path), str(samples_path): sha(samples_path)})
            for row in read_csv(samples_path):
                assert row['sequence'] == seq
                row['_condition'] = condition
                for delay in [1,4]:
                    mask = float(row['m'+str(delay)])
                    assert mask in [0.0,1.0]
                    if mask == 0:
                        assert all(float(row[name+str(delay)]) == 0 for name in ['D','mu','s','E'])
                if row['kind'] == 'target_coarse_box_center':
                    targets.append(row)
                    key = (seq,int(row['window_start']),int(row['frame']),int(row['annotation_id']))
                else:
                    assert row['kind'] == 'original_global_hard_background'
                    key = (seq,int(row['window_start']),int(row['frame']))
                    assert key not in background
                    background[key] = row
                coordinates[key] = (int(row['y']),int(row['x']),float(row['original_logit']))
        if coordinate_reference is None:
            coordinate_reference = coordinates
        else:
            assert coordinates == coordinate_reference, 'Frozen original-score query selection differs by condition'
        for plan in PLANS:
            target = aggregate(targets,background,metadata,condition,plan)
            assert len(target) == 5200 and len({(r['sequence'],r['track_id']) for r in target}) == 130
            bg = background_aggregate(background,condition,plan)
            all_objects.extend(target); all_background.extend(bg)
            for group in ['pooled']+SEQUENCES:
                group_target = [r for r in target if group=='pooled' or r['sequence']==group]
                for cohort in outcome_groups(group_target[0]):
                    selected = [r for r in group_target if outcome_groups(r)[cohort]]
                    summaries.extend(distribution(selected,METRICS+PAIRED,condition,plan,group,cohort,'target'))
                group_bg = [r for r in bg if group=='pooled' or r['sequence']==group]
                summaries.extend(distribution(group_bg,METRICS,condition,plan,group,'all_frame_references','background'))
    write_csv(args.output/'EVIDENCE_OBJECT_DIAGNOSTIC.csv',all_objects)
    write_csv(args.output/'EVIDENCE_BACKGROUND_DIAGNOSTIC.csv',all_background)
    write_csv(args.output/'EVIDENCE_DISTRIBUTIONS.csv',summaries)
    write_csv(args.output/'RUNTIME_COSTS.csv',runtime_costs(sources))
    receipt = dict(status='COMPLETE',source_sha256=sources,script_sha256=sha(__file__),
                   target_arrangement_condition_rows=len(all_objects),background_arrangement_condition_rows=len(all_background),
                   original_query_coordinates_and_logits_equal_all_conditions=True,
                   aggregation='Average available covering-window measurements within each object-frame (or background frame), then equal-weight units. No invalid temporal reference contributes to D/mu/s/E statistics.',
                   absent_channels='N0 temporal channels absent and m=0: temporal statistics unavailable, not zero evidence. N1 mu/s/E are deliberately zero placeholders and excluded as unavailable rather than summarized as measured statistics.',
                   paired_background='Same window/current-frame original-score maximum outside valid annotation boxes. Subtract target minus this fixed background at each valid window before averaging. This is not a matched-size difficulty control.',
                   coordinate_limitation='Target query is coarse box center, not known true center or matched component peak. Background annotation exclusion does not establish ground-truth negative visibility. Neither query alone explains matching or background components.',
                   scope='Read-only saved-forward descriptive telemetry, grouped retrospectively by frozen final outcomes. No new forward, probe, selection, training or causal claim.',
                   zero_new_model_calls=True,zero_new_training=True,
                   artifact_sha256={p.name:sha(p) for p in sorted(args.output.glob('*.csv'))})
    (args.output/'COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','target_arrangement_condition_rows','background_arrangement_condition_rows','zero_new_model_calls']}))


if __name__=='__main__':
    main()
