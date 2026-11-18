#!/usr/bin/env python3
"""Check supplementary numerical claims in the completed root report."""
import csv
import hashlib
import json
import re
from collections import defaultdict,Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
V=HERE.parent
def read(name):
    with (V/name).open() as f:return list(csv.DictReader(f))
def key(r):return r['sequence'],int(r['frame']),int(r['annotation_id'])
objects=read('OBJECTS.csv');windows=read('WINDOWS.csv');control=read('control_summary/OBJECTS.csv')
assert len(objects)==21270
wg=defaultdict(list)
for r in windows:wg[key(r)].append(r)
groups=defaultdict(list)
for r in objects:
    if int(r['small_geometry']):groups[r['group']].append(r)
new=groups['additional_annotation_selected_development']
old=groups['prior_motion_selected_development']
assert len(new)==5200 and len(old)==1850
extra=[r for r in new if r['system_case']=='extra_window_only_opportunity']
assert len(extra)==104
original_internal=sum(any(int(w['original5']) and 8<=int(w['slot'])<=31 for w in wg[key(r)]) for r in extra)
success_internal=sum(any(int(w['matched']) and 8<=int(w['slot'])<=31 for w in wg[key(r)]) for r in extra)
assert original_internal==59 and success_internal==77
protected={key(r) for r in new if len(wg[key(r)])>=2 and int(r['fused_matched']) and all(int(w['matched']) for w in wg[key(r)])}
assert len(protected)==422
protected_losses={mode:sum(not int(r['control_matched']) for r in control if r['mode']==mode and key(r) in protected)
                  for mode in ['official5_probmean','symmetric5_probmax']}
assert all(x==0 for x in protected_losses.values())
tracks=[r for r in read('control_summary/TRACKS.csv') if r['mode']=='symmetric5_probmax']
track_states=Counter('gain' if int(r['net'])>0 else 'loss' if int(r['net'])<0 else 'zero' for r in tracks)
assert len(tracks)==130 and track_states==dict(gain=14,loss=13,zero=103)
expected=[(old,126,1,16),(new,163,3,24)]
for rows,flip,fusion_tracks,extra_tracks in expected:
    assert sum(int(r['fused_matched']) and int(r['response_cross_05']) for r in rows)==flip
    assert len({(r['sequence'],r['track_id']) for r in rows if r['system_case']=='original_window_match_lost_by_fusion'})==fusion_tracks
    assert len({(r['sequence'],r['track_id']) for r in rows if r['system_case']=='extra_window_only_opportunity'})==extra_tracks
assert sum(not int(r['fused_matched']) and int(r['any11_matched']) for r in old+new)==157
opportunity_keys={key(r) for r in old+new if not int(r['fused_matched']) and int(r['any11_matched'])}
evidence=read('evidence_context/WINDOW_EVIDENCE.csv')
evidence_objects=read('evidence_context/OBJECT_CONTEXT.csv')
assert len(evidence)==500 and len(evidence_objects)==157
assert {key(r) for r in evidence_objects}==opportunity_keys=={key(r) for r in evidence}
expected_window_keys={(key(r),int(w['window_start'])) for r in old+new if key(r) in opportunity_keys for w in wg[key(r)]}
actual_window_keys={(key(r),int(r['window_start'])) for r in evidence}
assert len(actual_window_keys)==len(evidence) and actual_window_keys==expected_window_keys
numerical_audit=json.loads((HERE/'SUMMARY.json').read_text())
official=next(r for r in numerical_audit['fixed_half'] if r['mode']=='official5_probmax' and r['group']=='pooled')
symmetric=next(r for r in numerical_audit['fixed_half'] if r['mode']=='symmetric5_probmax' and r['group']=='pooled')
assert official['positive_pixels_outside_boxes']==38368 and symmetric['positive_pixels_outside_boxes']==38641
plan=V.parent/'stage1_goal/CVPR2027_RESEARCH_PLAN.md'
link_checks={}
for path in [V/'REPORT.md',plan]:
    local=[x for x in re.findall(r'\]\(([^)]+)\)',path.read_text()) if not x.startswith(('http:','https:','#'))]
    assert all((path.parent/x.split('#')[0]).exists() for x in local)
    link_checks[str(path)]=len(local)
report=V/'REPORT.md'
out=dict(status='PASS',root_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
    checked_total_valid_objects=len(objects),protected=len(protected),protected_losses=protected_losses,
    extra_opportunity_original_internal_slot=original_internal,extra_opportunity_success_internal_slot=success_internal,
    symmetric_track_net_states=dict(track_states),small_opportunities_all_six=157,
    evidence_object_window_rows=len(evidence),positive_pixels_outside_boxes=[38368,38641],
    plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),local_link_checks=link_checks,
    ratio_checks=dict(opportunities_per_all=109/5200,opportunities_per_final_misses=109/4500,
        original_input_loss_per_all=5/5200,absolute_gain_percentage_points=100*20/5200,
        relative_gain_baseline_matches=100*20/700),
    inputs_sha256={name:hashlib.sha256((V/name).read_bytes()).hexdigest() for name in [
        'OBJECTS.csv','WINDOWS.csv','control_summary/OBJECTS.csv','control_summary/TRACKS.csv','REPORT.md','PROTOCOL.md',
        'evidence_context/WINDOW_EVIDENCE.csv','evidence_context/OBJECT_CONTEXT.csv',
        '../stage1_goal/CVPR2027_RESEARCH_PLAN.md']})
(HERE/'CLAIM_REVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
