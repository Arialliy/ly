#!/usr/bin/env python3
"""Verify retained candidates, source IDs and deterministic selection independently of the scan."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import math

R = Path(__file__).resolve().parent
root = R.parent
rule = json.loads((R/'PRESCREEN_RULE.json').read_text())
sha = hashlib.sha256((R/'PRESCREEN_RULE.json').read_bytes()).hexdigest()
assert sha == (R/'PRESCREEN_RULE.sha256').read_text().split()[0]
summary = json.loads((R/'SCREEN_SUMMARY.json').read_text())
assert summary['rule_sha256'] == sha
assert summary['script_sha256'] == hashlib.sha256((R/'screen_annotations.py').read_bytes()).hexdigest()
for p, digest in rule['input_sha256'].items():
    assert hashlib.sha256((root/p).read_bytes()).hexdigest() == digest
excluded = set((root/rule['inputs'][2]).read_text().split()) | set((root/rule['inputs'][3]).read_text().split())
with gzip.open(R/'ELIGIBLE_TRAJECTORIES.jsonl.gz', 'rt') as f:
    runs = {x['run_id']: x for x in map(json.loads, f)}
with gzip.open(R/'ALL_CANDIDATES.jsonl.gz', 'rt') as f:
    candidates = list(map(json.loads, f))
with gzip.open(R/'ALL_EVALUATED_WINDOWS.jsonl.gz', 'rt') as f:
    evaluated = list(map(json.loads, f))
candidate_ids = [x['candidate_id'] for x in candidates]
assert len(candidate_ids) == len(set(candidate_ids))
assert set(candidate_ids) == {x['candidate_id'] for x in evaluated if x['passes']}
assert len(evaluated) == sum(max(0, x['frame_count']-(160 if x['frame_count']>=160 else 120)+1) for x in runs.values())
counts = collections.Counter()
for c in candidates:
    assert c['sequence'] not in excluded
    n = c['frame_count']
    run = runs[c['run_id']]
    clip = run['frames'][c['run_frame_offset']:c['run_frame_offset']+n]
    assert len(clip) == n and n in (120, 160)
    assert [x['frame'] for x in clip] == list(range(c['start_frame'], c['end_frame']+1))
    assert len({x['annotation_id'] for x in clip}) == n
    assert len({x['image_id'] for x in clip}) == n
    assert len({x['category_id'] for x in clip}) == 1
    sides = [max(x['bbox_xywh'][2:]) for x in clip]
    assert max(sides) <= 16 and max(sides) == c['max_box_side_px']
    assert c['nested_12px_stratum'] == (max(sides)<=12)
    assert c['first_annotation_id'] == clip[0]['annotation_id'] and c['last_annotation_id'] == clip[-1]['annotation_id']
    assert c['first_image_id'] == clip[0]['image_id'] and c['last_image_id'] == clip[-1]['image_id']
    assert c['fps'] == run['fps']
    assert len(c['blocks']) == n//40
    for i,b in enumerate(c['blocks']):
        assert b['start_frame'] == c['start_frame']+i*40 and b['end_frame'] == c['start_frame']+i*40+39
        assert len(b['quarter_speed_px_per_frame']) == 4
    assert c['event_tags'] and not c['artifact_rejection_reasons']
    counts[f'passing_{n}'] += 1
    if c['nested_12px_stratum']:
        counts[f'passing_{n}_12px'] += 1
for key,value in counts.items():
    assert summary['window_counts'][key] == value
primary = sorted([c for c in candidates if c['frame_count']==160],key=lambda c:(c['sequence'],c['track_id'],c['start_frame']))
fallback = sorted([c for c in candidates if c['frame_count']==120],key=lambda c:(c['sequence'],c['track_id'],c['start_frame']))
selected = json.loads((R/'SELECTED_CANDIDATE.json').read_text())['candidate']
expected = (primary or fallback)[0] if candidates else None
assert (selected['candidate_id'] if selected else None) == (expected['candidate_id'] if expected else None)
if selected:
    run = runs[selected['run_id']]
    assert selected['frames'] == run['frames'][selected['run_frame_offset']:selected['run_frame_offset']+selected['frame_count']]
queue = json.loads((R/'INSPECTION_QUEUE.json').read_text())['candidates']
seen = set()
expected_queue = []
for c in primary or fallback:
    if c['sequence'] not in seen and len(expected_queue)<12:
        seen.add(c['sequence'])
        expected_queue.append(c['candidate_id'])
assert [c['candidate_id'] for c in queue] == expected_queue
result = {'status':'PASS', 'rule_sha256':sha, 'checked_candidate_windows':len(candidates),
          'checked_evaluated_windows':len(evaluated), 'checked_eligible_runs':len(runs),
          'counts':dict(counts), 'source_hashes_match':True, 'split_disjoint':True,
          'source_id_references_resolve':True, 'selected_by_frozen_order':True,
          'all_evaluated_windows_retained':True, 'overlapping_candidates_are_not_independent_events':True,
          'selected_candidate_id':selected['candidate_id'] if selected else None}
(R/'MANIFEST_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
