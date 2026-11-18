#!/usr/bin/env python3
"""Apply frozen VT annotation prescreen. Python standard library; no inference/image IO."""
from pathlib import Path
import collections
import datetime
import gzip
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RULE_PATH = HERE / 'PRESCREEN_RULE.json'
RULE = json.loads(RULE_PATH.read_text())
RULE_SHA = hashlib.sha256(RULE_PATH.read_bytes()).hexdigest()
assert RULE_SHA == (HERE / 'PRESCREEN_RULE.sha256').read_text().split()[0]
assert RULE['rule_id'] == 'vt_annotation_motion_prescreen_v1'
for rel, expected in RULE['input_sha256'].items():
    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected, rel


def quantile(values, q):
    s = sorted(values)
    p = q * (len(s) - 1)
    lo, hi = math.floor(p), math.ceil(p)
    return s[lo] * (hi - p) + s[hi] * (p - lo) if hi != lo else s[lo]


def stats(values):
    return {'min': min(values), 'median': statistics.median(values), 'max': max(values)}


def clean(obj):
    if isinstance(obj, float):
        return round(obj, 8)
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(x) for x in obj]
    return obj


def dump(path, obj):
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False) + '\n')


def line(f, obj):
    f.write(json.dumps(clean(obj), separators=(',', ':'), ensure_ascii=False) + '\n')


def block_stats(rows):
    c = [r['center_xy'] for r in rows]
    vec = [(c[i+5][0]-c[i][0], c[i+5][1]-c[i][1]) for i in range(35)]
    speed = [math.hypot(x, y)/5 for x, y in vec]
    quarter = [math.dist(c[i], c[i+9])/9 for i in (0, 10, 20, 30)]
    net = [(c[-1][k]-c[0][k])/39 for k in (0, 1)]
    net_speed = math.hypot(*net)
    path = sum(math.hypot(*v) for v in vec)
    coherence = math.hypot(sum(v[0] for v in vec), sum(v[1] for v in vec))/path if path else 0.0
    med, p25, p90 = statistics.median(speed), quantile(speed, .25), quantile(speed, .9)
    low = p90 <= .15 and net_speed <= .15 and max(quarter) <= .20
    moving = med >= .50 and p25 >= .30 and sum(s >= .30 for s in quarter) >= 3
    directional = moving and coherence >= .70 and net_speed >= .30
    return {'start_frame': rows[0]['frame'], 'end_frame': rows[-1]['frame'],
            'lag5_speed_px_per_frame': {'p25': p25, 'median': med, 'p90': p90},
            'quarter_speed_px_per_frame': quarter, 'net_velocity_px_per_frame': net,
            'net_speed_px_per_frame': net_speed, 'lag5_coherence': coherence,
            'heading_degrees': math.degrees(math.atan2(net[1], net[0])) if net_speed else None,
            'low': low, 'moving': moving, 'directional_moving': directional}


def event_stats(blocks):
    events = []
    for i, (a, b) in enumerate(zip(blocks, blocks[1:])):
        tags = []
        sa, sb = a['lag5_speed_px_per_frame']['median'], b['lag5_speed_px_per_frame']['median']
        if a['low'] and b['moving']:
            tags.append('low_to_moving')
        if a['moving'] and b['low']:
            tags.append('moving_to_low')
        if a['moving'] and b['moving']:
            if sb >= 2 * sa and sb-sa >= .40:
                tags.append('acceleration')
            if sa >= 2 * sb and sa-sb >= .40:
                tags.append('deceleration')
        angle = None
        if a['directional_moving'] and b['directional_moving']:
            angle = abs((b['heading_degrees'] - a['heading_degrees'] + 180) % 360 - 180)
            if angle >= 45:
                tags.append('turn')
        if tags:
            events.append({'block_indices': [i, i+1], 'boundary_frame': b['start_frame'],
                           'tags': tags, 'later_over_earlier_speed_ratio': sb/sa if sa else None,
                           'robust_speed_change_px_per_frame': sb-sa, 'heading_angle_degrees': angle})
    for i in range(len(blocks)-2):
        a, b, c = blocks[i:i+3]
        tag = ('low_moving_low' if a['low'] and b['moving'] and c['low'] else
               'moving_low_moving' if a['moving'] and b['low'] and c['moving'] else None)
        if tag:
            events.append({'block_indices': [i, i+1, i+2], 'tags': [tag],
                           'boundary_frames': [b['start_frame'], c['start_frame']]})
    return events


started = datetime.datetime.now(datetime.timezone.utc).isoformat()
data = json.loads(gzip.decompress((ROOT / RULE['inputs'][0]).read_bytes()))
train = set((ROOT / RULE['inputs'][2]).read_text().split())
test = set((ROOT / RULE['inputs'][3]).read_text().split())
excluded = train | test
images = {im['id']: im for im in data['images']}
assert len(images) == len(data['images'])
videos = {v['id']: v for v in data['videos']}
categories = {c['id']: c['name'] for c in data['categories']}
sequences = sorted({im['file_name'].split('/')[0] for im in data['images']})
eligible_sequences = sorted(set(sequences) - excluded)
split = {'role': RULE['split']['role'], 'rule_sha256': RULE_SHA,
         'annotation_file': RULE['inputs'][0], 'annotation_sha256': RULE['input_sha256'][RULE['inputs'][0]],
         'annotation_archive_member': 'VT-Tiny-MOT/annotations/instances_01_train2017.json',
         'deep_pro_train_path': RULE['inputs'][2], 'deep_pro_test_path': RULE['inputs'][3],
         'deep_pro_train_ids': sorted(train), 'deep_pro_test_ids': sorted(test),
         'source_sequence_count': len(sequences), 'eligible_sequence_ids': eligible_sequences,
         'excluded_train_ids_present_in_source': sorted(set(sequences) & train),
         'excluded_test_ids_present_in_source': sorted(set(sequences) & test),
         'excluded_any_ids_present_in_source': sorted(set(sequences) & excluded),
         'source_url': json.loads((ROOT / RULE['inputs'][1]).read_text())['source_url'],
         'scene_independence_verified': False}
dump(HERE / 'SPLIT_PROVENANCE.json', split)

counts = collections.Counter()
grouped = collections.defaultdict(list)
seq_counts = {s: collections.Counter() for s in eligible_sequences}
raw_track_frame_counts = collections.Counter(
    (images[a['image_id']]['file_name'].split('/')[0], int(a['track_id']),
     int(Path(images[a['image_id']]['file_name']).stem))
    for a in data['annotations'] if 'track_id' in a)
raw_duplicate_keys = {k for k,v in raw_track_frame_counts.items() if v > 1}
counts['source_duplicate_track_frame_keys'] = len(raw_duplicate_keys)
counts['development_duplicate_track_frame_keys'] = sum(k[0] not in excluded for k in raw_duplicate_keys)
for a in data['annotations']:
    counts['source_annotations'] += 1
    im = images[a['image_id']]
    parts = im['file_name'].split('/')
    seq = parts[0]
    if seq in excluded:
        counts['annotations_excluded_by_manifest'] += 1
        continue
    counts['annotations_in_development_sequences'] += 1
    seq_counts[seq]['annotations'] += 1
    b = a.get('bbox', [])
    reason = None
    if 'track_id' in a and (seq, int(a['track_id']), int(Path(parts[-1]).stem)) in raw_duplicate_keys:
        reason = 'duplicate_track_frame_before_eligibility_filter'
    elif parts[1] != '01':
        reason = 'nonthermal_channel'
    elif a.get('ignore', 0) or a.get('iscrowd', 0):
        reason = 'ignored_or_crowd'
    elif 'track_id' not in a:
        reason = 'missing_track_id'
    elif len(b) != 4 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in b):
        reason = 'invalid_box'
    elif min(b[2:]) <= 0 or max(b[2:]) > 16:
        reason = 'box_outside_primary_scale'
    elif b[0] < 0 or b[1] < 0 or b[0]+b[2] > im['width'] or b[1]+b[3] > im['height']:
        reason = 'box_outside_image'
    if reason:
        counts['rejected_annotation_' + reason] += 1
        continue
    frame = int(Path(parts[-1]).stem)
    if frame != im.get('mot_frame_id', frame):
        counts['rejected_annotation_filename_mot_frame_disagreement'] += 1
        continue
    counts['eligible_primary_annotations_after_duplicate_filter'] += 1
    seq_counts[seq]['eligible_annotations_16px'] += 1
    if max(b[2:]) <= 12:
        counts['eligible_12px_annotations_after_duplicate_filter'] += 1
        seq_counts[seq]['eligible_annotations_12px'] += 1
    grouped[(seq, int(a['track_id']))].append({'frame': frame, 'image_id': im['id'],
        'annotation_id': a['id'], 'original_tracking_id': a['tracking_id'], 'category_id': a['category_id'],
        'video_id': im['video_id'], 'bbox_xywh': b, 'center_xy': [b[0]+b[2]/2, b[1]+b[3]/2],
        'file_name': im['file_name'], 'mot_frame_id': im.get('mot_frame_id'), 'compact_frame_id': im.get('frame_id')})

run_counts = collections.Counter()
window_counts = collections.Counter()
event_counts = collections.Counter()
event_counts_12 = collections.Counter()
representatives = []
seen_representatives = set()
selection = {160: None, 120: None}
queues = {160: [], 120: []}
queue_seen_sequences = {160: set(), 120: set()}
passed_sequences = {160: set(), 120: set()}
passed_tracks = {160: set(), 120: set()}
eligible_track_set = set()
run_manifest = []
fps_values = collections.Counter()

def process_run(seq, track, rows, run_index, trajectories, evaluated, candidates):
    if not rows:
        return
    n = len(rows)
    run_counts['primary_16px_runs_total'] += 1
    streak = 0
    for r in rows + [None]:
        if r is not None and max(r['bbox_xywh'][2:]) <= 12:
            streak += 1
        elif streak:
            run_counts['nested_12px_contiguous_runs_total'] += 1
            run_counts['nested_12px_contiguous_runs_' + ('at_least_160' if streak >= 160 else
                       '120_to_159' if streak >= 120 else 'below_120')] += 1
            streak = 0
    if n < 120:
        run_counts['primary_16px_runs_shorter_than_120'] += 1
        return
    length = 160 if n >= 160 else 120
    run_counts[f'primary_16px_runs_{length}_stratum'] += 1
    seq_counts[seq][f'runs_{length}_stratum'] += 1
    eligible_track_set.add((seq, track))
    run_id = f'{seq}__{track}__{rows[0]["frame"]:05d}__{rows[-1]["frame"]:05d}'
    video = videos[rows[0]['video_id']]
    fps = video.get('fps')
    fps_values[str(fps) if fps is not None else 'unknown'] += 1
    run_info = {'run_id': run_id, 'sequence': seq, 'track_id': track, 'source_video_id': video['id'],
                'source_video_name': video['name'], 'category_id': rows[0]['category_id'],
                'category_name': categories[rows[0]['category_id']], 'start_frame': rows[0]['frame'],
                'end_frame': rows[-1]['frame'], 'frame_count': n, 'fps': fps,
                'fps_source': f'annotation.videos[id={video["id"]}].fps' if fps is not None else 'unknown',
                'image_width': video['width'], 'image_height': video['height'],
                'max_box_side_px': max(max(r['bbox_xywh'][2:]) for r in rows),
                'length_stratum': length, 'trajectory_file': 'ELIGIBLE_TRAJECTORIES.jsonl.gz'}
    run_manifest.append(run_info)
    line(trajectories, dict(run_info, frames=rows))
    bs = [block_stats(rows[i:i+40]) for i in range(n-39)]
    raw_steps = [math.dist(a['center_xy'], b['center_xy']) for a,b in zip(rows, rows[1:])]
    side_jump = [any(abs(b['bbox_xywh'][k]-a['bbox_xywh'][k]) > max(4, .5*a['bbox_xywh'][k])
                     for k in (2,3)) for a,b in zip(rows, rows[1:])]
    for offset in range(n-length+1):
        clip = rows[offset:offset+length]
        blocks = [bs[offset+j] for j in range(0, length, 40)]
        ev = event_stats(blocks)
        sides = [max(r['bbox_xywh'][2:]) for r in clip]
        is12 = max(sides) <= 12
        ss = raw_steps[offset:offset+length-1]
        limit = max(8, 6*statistics.median(ss))
        reasons = []
        if max(ss) > limit:
            reasons.append('raw_center_step_jump')
        if any(side_jump[offset:offset+length-1]):
            reasons.append('box_side_jump')
        tags = sorted({tag for e in ev for tag in e['tags']})
        cid = f'{seq}__{track}__{clip[0]["frame"]:05d}__{clip[-1]["frame"]:05d}'
        base = {'candidate_id': cid, 'sequence': seq, 'track_id': track, 'run_id': run_id,
                'run_frame_offset': offset, 'start_frame': clip[0]['frame'], 'end_frame': clip[-1]['frame'],
                'frame_count': length, 'category_id': clip[0]['category_id'],
                'category_name': categories[clip[0]['category_id']], 'max_box_side_px': max(sides),
                'nested_12px_stratum': is12, 'raw_event_tags': tags, 'artifact_rejection_reasons': reasons,
                'passes': bool(tags) and not reasons}
        line(evaluated, base)
        window_counts[f'evaluated_{length}'] += 1
        if is12:
            window_counts[f'evaluated_{length}_12px'] += 1
        if reasons:
            window_counts[f'artifact_rejected_{length}'] += 1
            if tags:
                window_counts[f'event_windows_rejected_by_artifact_guard_{length}'] += 1
            for reason in reasons:
                window_counts[reason + f'_{length}'] += 1
        if not tags:
            window_counts[f'no_event_{length}'] += 1
        if not base['passes']:
            continue
        window_counts[f'passing_{length}'] += 1
        seq_counts[seq][f'passing_windows_{length}'] += 1
        if is12:
            window_counts[f'passing_{length}_12px'] += 1
        passed_sequences[length].add(seq)
        passed_tracks[length].add((seq, track))
        for tag in tags:
            event_counts[f'{length}:{tag}'] += 1
            if is12:
                event_counts_12[f'{length}:{tag}'] += 1
        candidate = dict(base, rule_sha256=RULE_SHA, event_tags=tags, events=ev, blocks=blocks,
                         fps=fps, fps_source=run_info['fps_source'],
                         elapsed_seconds=(length-1)/fps if fps else None,
                         source_video_id=video['id'], source_video_name=video['name'],
                         first_image_id=clip[0]['image_id'], last_image_id=clip[-1]['image_id'],
                         first_annotation_id=clip[0]['annotation_id'], last_annotation_id=clip[-1]['annotation_id'],
                         source_frame_ids='ELIGIBLE_TRAJECTORIES.jsonl.gz, run_id + run_frame_offset + frame_count',
                         box_width_px=stats([r['bbox_xywh'][2] for r in clip]),
                         box_height_px=stats([r['bbox_xywh'][3] for r in clip]),
                         max_box_side_distribution_px=stats(sides),
                         raw_center_step_px=stats(ss), raw_center_step_rejection_threshold_px=limit,
                         split_role=RULE['split']['role'], in_deeppro_train=False, in_deeppro_test=False,
                         camera_motion_verified=False, physical_motion_verified=False,
                         continuous_visibility_verified=False, model_failure_claimed=False)
        line(candidates, candidate)
        if selection[length] is None:
            selection[length] = dict(candidate, frames=clip)
        if seq not in queue_seen_sequences[length] and len(queues[length]) < 12:
            queues[length].append(dict(candidate, frames=clip))
            queue_seen_sequences[length].add(seq)
        for tag in tags:
            key = (seq, track, tag, length)
            if key not in seen_representatives:
                seen_representatives.add(key)
                representatives.append(dict(base, representative_event_tag=tag))


with gzip.open(HERE / 'ELIGIBLE_TRAJECTORIES.jsonl.gz', 'wt') as trajectories, \
     gzip.open(HERE / 'ALL_EVALUATED_WINDOWS.jsonl.gz', 'wt') as evaluated, \
     gzip.open(HERE / 'ALL_CANDIDATES.jsonl.gz', 'wt') as candidates:
    for (seq, track), rows in sorted(grouped.items()):
        counts['tracks_with_primary_size_annotations'] += 1
        duplicate_frames = {k for k,v in collections.Counter(r['frame'] for r in rows).items() if v > 1}
        counts['remaining_duplicate_track_frame_keys_after_pre_filter'] += len(duplicate_frames)
        rows = sorted((r for r in rows if r['frame'] not in duplicate_frames), key=lambda r:r['frame'])
        run, run_index = [], 0
        for row in rows:
            if run and (row['frame'] != run[-1]['frame']+1 or
                        row['category_id'] != run[-1]['category_id'] or row['video_id'] != run[-1]['video_id']):
                process_run(seq, track, run, run_index, trajectories, evaluated, candidates)
                run, run_index = [], run_index+1
            run.append(row)
        process_run(seq, track, run, run_index, trajectories, evaluated, candidates)
        if counts['tracks_with_primary_size_annotations'] % 250 == 0:
            print('tracks', counts['tracks_with_primary_size_annotations'], 'windows', sum(v for k,v in window_counts.items() if k in ('evaluated_160','evaluated_120')),
                  'passing160', window_counts['passing_160'], flush=True)

length_used = 160 if selection[160] else 120
selected = selection[length_used]
dump(HERE / 'ELIGIBLE_RUNS.json', run_manifest)
dump(HERE / 'EVENT_REPRESENTATIVES.json', representatives)
dump(HERE / 'SELECTED_CANDIDATE.json', {'selection_rule': RULE['selection'], 'candidate': selected,
    'status': 'annotation_candidate_for_raw_video_inspection' if selected else 'zero_candidates_under_frozen_rule',
    'limits': RULE['limits']})
dump(HERE / 'INSPECTION_QUEUE.json', {'rule': RULE['selection']['inspection_queue'],
    'length_stratum': length_used, 'candidates': queues[length_used], 'limits': RULE['limits']})
summary = {'status': 'annotation_candidates_only', 'started_utc': started,
    'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'rule_sha256': RULE_SHA,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_images': len(images), 'source_sequences': len(sequences), 'development_sequences': len(eligible_sequences),
    'counts': dict(counts), 'run_counts': dict(run_counts), 'window_counts': dict(window_counts),
    'tracks_with_120plus_continuous_16px_frames': len(eligible_track_set),
    'passing_sequence_counts': {str(k): len(v) for k,v in passed_sequences.items()},
    'passing_track_counts': {str(k): len(v) for k,v in passed_tracks.items()},
    'event_tag_window_counts': dict(event_counts), 'event_tag_window_counts_nested_12px': dict(event_counts_12),
    'event_representative_count': len(representatives), 'fps_metadata_values_by_eligible_run': dict(fps_values),
    'per_sequence_counts': {k: dict(v) for k,v in seq_counts.items()},
    'selected_candidate_id': selected['candidate_id'] if selected else None,
    'limits': RULE['limits'], 'no_images_downloaded': True, 'no_model_outputs_used': True}
dump(HERE / 'SCREEN_SUMMARY.json', summary)
print(json.dumps({k:summary[k] for k in ['status','counts','run_counts','window_counts','passing_sequence_counts','event_tag_window_counts','selected_candidate_id']}, indent=2))
