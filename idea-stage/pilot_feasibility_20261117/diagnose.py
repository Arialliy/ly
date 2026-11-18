#!/usr/bin/env python3
"""CPU/std-library-only feasibility audit. Reads train/calibration annotations only."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = {}

def read(path):
    path = ROOT / path
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw.decode()

def segments(rows):
    runs = []
    for row in sorted(rows, key=lambda x: x['frame']):
        if not runs or row['frame'] != runs[-1][-1]['frame'] + 1 or row['source_frame'] != runs[-1][-1]['source_frame'] + 1:
            runs.append([])
        runs[-1].append(row)
    return runs

def dist(values):
    values = sorted(values)
    def q(p):
        if not values: return None
        v = (len(values)-1)*p; lo = int(v); hi = min(lo+1, len(values)-1)
        return values[lo] + (values[hi]-values[lo])*(v-lo)
    return {'n': len(values), 'min': min(values) if values else None, 'median': q(.5), 'p75': q(.75), 'p90': q(.9), 'p95': q(.95), 'max': max(values) if values else None, 'mean': sum(values)/len(values) if values else None, 'ge8': sum(v>=8 for v in values), 'ge16': sum(v>=16 for v in values), 'ge32': sum(v>=32 for v in values)}

selection = json.loads(read('results/references/DATA_SELECTION.json'))
allowed = {s: selection['splits'][s] for s in ('train', 'calibration')}
assert not set(sum(allowed.values(), [])).intersection(selection['splits']['confirmation'])
clip_results = []
for split, names in allowed.items():
    for name in names:
        data = json.loads(read(f'data/stage1/{name}/annotations.json'))
        images = sorted(data['images'], key=lambda x: x['frame_id'])
        imap = {im['id']: (i, int(im['frame_id'])) for i, im in enumerate(images)}
        tracks = defaultdict(list); excluded = 0
        for a in data['annotations']:
            x,y,w,h = a['bbox']
            if a.get('ignore',0) or a.get('iscrowd',0) or w<=0 or h<=0 or not all(math.isfinite(float(v)) for v in a['bbox']):
                excluded += 1; continue
            f,sf = imap[a['image_id']]
            tracks[str(a['track_id'])].append({'frame': f, 'source_frame': sf, 'cx': x+w/2, 'cy': y+h/2, 'annotation_id': a['id']})
        track_rows = []
        for tid, observations in sorted(tracks.items()):
            assert len({x['frame'] for x in observations}) == len(observations), (name,tid,'duplicate track/frame')
            obs_segments = segments(observations)
            lowruns = []; eligible = 0; stop_go_runs = 0
            for seg in obs_segments:
                states = []
                for i in range(5,len(seg)):
                    cur,prev = seg[i],seg[i-5]
                    low = math.hypot(cur['cx']-prev['cx'],cur['cy']-prev['cy']) < 1.0
                    states.append(dict(cur,low=low))
                eligible += len(states)
                runs = segments([x for x in states if x['low']])
                state_map = {x['frame']:x['low'] for x in states}
                for run in runs:
                    before = state_map.get(run[0]['frame']-1)
                    after = state_map.get(run[-1]['frame']+1)
                    is_stop_go = before is False and after is False
                    stop_go_runs += int(is_stop_go and len(run)>=8)
                    lowruns.append({'start':run[0]['frame'],'end':run[-1]['frame'],'length':len(run),'moving_before_and_after':is_stop_go})
            obs_frames = {x['frame'] for x in observations}
            appearance = [seg[0]['frame'] for seg in obs_segments if seg[0]['frame']>0]
            disappearance = [seg[-1]['frame'] for seg in obs_segments if seg[-1]['frame']<len(images)-1]
            windows = []
            for start in [0,36,72,108,120]:
                end = start+39
                count = sum(start<=f<=end for f in obs_frames)
                if count:
                    windows.append({'start':start,'observed_frames':count,'appearance_inside':sum(start<f<=end for f in appearance),'disappearance_inside':sum(start<=f<end for f in disappearance)})
            track_rows.append({'track_id':tid,'observed_frames':len(observations),'coverage_fraction_160':len(observations)/len(images),'continuous_observation_lengths':[len(x) for x in obs_segments], 'eligible_5lag_endpoints':eligible,'low_runs':lowruns,'stop_go_runs_ge8':stop_go_runs,'appearance_events_inside_clip':len(appearance),'disappearance_events_inside_clip':len(disappearance),'O_windows':windows})
        thresholds = {}
        for k in [8,16,32]:
            matching = [(t,r) for t in track_rows for r in t['low_runs'] if r['length']>=k]
            thresholds[str(k)] = {'tracks':len({t['track_id'] for t,r in matching}),'runs':len(matching),'endpoint_frames':sum(r['length'] for t,r in matching)}
        clip_results.append({'split':split,'sequence':name,'frames':len(images),'raw_annotations':len(data['annotations']),'excluded_annotations':excluded,'tracks':len(track_rows),'valid_target_frames':sum(t['observed_frames'] for t in track_rows),'track_coverage_distribution':dist([t['observed_frames'] for t in track_rows]),'eligible_5lag_endpoints':sum(t['eligible_5lag_endpoints'] for t in track_rows),'low_runs_thresholds':thresholds,'stop_go_runs_ge8':sum(t['stop_go_runs_ge8'] for t in track_rows),'tracks_with_appearance':sum(t['appearance_events_inside_clip']>0 for t in track_rows),'tracks_with_disappearance':sum(t['disappearance_events_inside_clip']>0 for t in track_rows),'appearance_events':sum(t['appearance_events_inside_clip'] for t in track_rows),'disappearance_events':sum(t['disappearance_events_inside_clip'] for t in track_rows),'O_track_windows':sum(len(t['O_windows']) for t in track_rows),'O_track_windows_with_appearance':sum(w['appearance_inside']>0 for t in track_rows for w in t['O_windows']),'O_track_windows_with_disappearance':sum(w['disappearance_inside']>0 for t in track_rows for w in t['O_windows']),'track_details':track_rows})

object_rows = list(csv.DictReader(read('results/tlfp_pilot_20261012/evaluation/OBJECT_COMPARISON.csv').splitlines()))
object_rows = [r for r in object_rows if r['arrangement']=='O']
assert set(r['sequence'] for r in object_rows) == set(allowed['calibration'])
keys = [(r['sequence'],r['frame'],r['annotation_id']) for r in object_rows]
assert len(set(keys))==len(keys)==5200
for r in object_rows:
    for k in ['frame','source_frame','release_matched','low_all_05','extreme_all_001']: r[k]=int(r[k])
cache_rows = list(csv.DictReader(read('results/tlfp_pilot_20261012/evaluation/CONTINUOUS_MISS_TRACKS.csv').splitlines()))
cache_rows = [r for r in cache_rows if r['condition']=='release' and r['arrangement']=='O' and r['cohort']=='all']
assert len(cache_rows)==130 and set(r['sequence'] for r in cache_rows)==set(allowed['calibration'])
cached = {(r['sequence'],r['track_id']):int(r['longest_miss_run']) for r in cache_rows}
grouped = defaultdict(list)
for r in object_rows: grouped[r['sequence'],r['track_id']].append(r)
miss_details = []
for key, rows in sorted(grouped.items()):
    obs = segments(rows); misses = segments([r for r in rows if not r['release_matched']])
    longest = max([len(x) for x in misses],default=0)
    assert longest==cached[key], (key,longest,cached[key])
    miss_details.append({'sequence':key[0],'track_id':key[1],'frames':len(rows),'matched':sum(r['release_matched'] for r in rows),'longest_miss':longest,'observation_segments':len(obs),'all_missed_segments':sum(not any(r['release_matched'] for r in seg) for seg in obs)})
cohorts = {}
for label, flag in [('all',None),('low','low_all_05'),('extreme','extreme_all_001')]:
    selected = [r for r in object_rows if flag is None or r[flag]]
    groups = defaultdict(list)
    for r in selected: groups[r['sequence'],r['track_id']].append(r)
    runs = [seg for rows in groups.values() for seg in segments(rows)]
    missed_runs = [seg for seg in runs if not any(r['release_matched'] for r in seg)]
    full_tracks = [key for key in groups if not any(r['release_matched'] for r in grouped[key])]
    cohorts[label] = {'target_frames':len(selected),'matched_frames':sum(r['release_matched'] for r in selected),'tracks_containing_cohort':len(groups),'contiguous_cohort_segments':len(runs),'all_missed_cohort_segments':len(missed_runs),'all_missed_cohort_segment_fraction':len(missed_runs)/len(runs) if runs else None,'cohort_segment_lengths':dist([len(x) for x in runs]),'containing_tracks_never_matched_in_full_strict_observation':len(full_tracks),'containing_tracks_never_matched_fraction':len(full_tracks)/len(groups) if groups else None,'cohort_frames_on_never_matched_full_strict_tracks':sum(len(groups[k]) for k in full_tracks)}

split_summary = {}
for split in allowed:
    cs=[c for c in clip_results if c['split']==split]
    split_summary[split]={'clips':len(cs),'tracks':sum(c['tracks'] for c in cs),'valid_target_frames':sum(c['valid_target_frames'] for c in cs),'eligible_5lag_endpoints':sum(c['eligible_5lag_endpoints'] for c in cs),'low_runs_thresholds':{str(k):{f:sum(c['low_runs_thresholds'][str(k)][f] for c in cs) for f in ['tracks','runs','endpoint_frames']} for k in [8,16,32]},'stop_go_runs_ge8':sum(c['stop_go_runs_ge8'] for c in cs),'appearance_events':sum(c['appearance_events'] for c in cs),'disappearance_events':sum(c['disappearance_events'] for c in cs),'O_track_windows_with_appearance':sum(c['O_track_windows_with_appearance'] for c in cs),'O_track_windows_with_disappearance':sum(c['O_track_windows_with_disappearance'] for c in cs)}
dwell_overlap = {}
for k in [8,16,32]:
    endpoints = {(c['sequence'],t['track_id'],f) for c in clip_results if c['split']=='calibration' for t in c['track_details'] for run in t['low_runs'] if run['length']>=k for f in range(run['start'],run['end']+1)}
    rr = [r for r in object_rows if (r['sequence'],r['track_id'],r['frame']) in endpoints]
    dwell_overlap[str(k)] = {'strict_frames':len(rr),'strict_tracks':len({(r['sequence'],r['track_id']) for r in rr}),'matched':sum(r['release_matched'] for r in rr),'low_frames':sum(r['low_all_05'] for r in rr),'low_tracks':len({(r['sequence'],r['track_id']) for r in rr if r['low_all_05']}),'extreme_frames':sum(r['extreme_all_001'] for r in rr),'extreme_tracks':len({(r['sequence'],r['track_id']) for r in rr if r['extreme_all_001']}),'per_clip':{name:{'strict_frames':sum(r['sequence']==name for r in rr),'low_frames':sum(r['low_all_05'] for r in rr if r['sequence']==name),'extreme_frames':sum(r['extreme_all_001'] for r in rr if r['sequence']==name)} for name in allowed['calibration']}}
limits=['5-frame net displacement means Euclidean distance center[t]-center[t-5], requiring 6 consecutive observations; low endpoint iff distance <1px. Run length counts consecutive qualifying endpoint frames (not the union of their preceding five frames).','All nonignore/noncrowd finite positive-area bounding boxes are used for motion feasibility, without strict-small/nonoverlap filtering. Track counts are clip+track-ID units, not independent physical entities or scenes.','Integer/half-pixel box centers, annotation quantization/jitter, reversal/circling, camera motion and unknown visibility/occlusion prevent treating low displacement as physical stationary truth. No motion compensation or image inspection performed.','Appearance/disappearance means annotation presence boundary inside the frozen 160-frame clip/window. It may reflect occlusion, missing annotation or truncation; not verified physical birth/death. Overlapping windows repeat events.','Low/extreme cohorts are defined by low released-model response, so their complete missing is expected/selection-conditioned. Cohort filtering creates gaps; full-strict track never-detection is separately reported.','Only release O once per annotation; strict observation may omit larger/overlapping portions of a physical track. No official Pd/Fa, no new prediction or model-success conclusion.','No confirmation annotation/image/prediction was read. DATA_SELECTION contains confirmation names only. No GPU library or model loaded.','False-alarm run lengths cannot be inferred from per-target matching tables without temporal association of unmatched components; they were not estimated.']
result={'created_utc':datetime.now(timezone.utc).isoformat(),'allowed_splits':allowed,'definition':{'lag_frames':5,'net_displacement_px_strict_upper_bound':1,'minimum_low_endpoint_run_lengths':[8,16,32],'O_window_starts':[0,36,72,108,120],'window_length':40},'split_summary':split_summary,'clips':clip_results,'calibration_dwell_strict_overlap':dwell_overlap,'release_O':{'target_frames':len(object_rows),'matched':sum(r['release_matched'] for r in object_rows),'tracks':len(miss_details),'longest_miss_distribution':dist([t['longest_miss'] for t in miss_details]),'tracks_never_matched':sum(t['matched']==0 for t in miss_details),'cohorts':cohorts,'track_details':miss_details,'cached_longest_miss_assertions':len(miss_details)},'limitations':limits,'source_sha256':SOURCES,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['# CPU feasibility audit: dwell/lifetime proxies','', 'Reproduce: `python3 '+str(Path(__file__))+'`','', 'No training, GPU inference, confirmation data, or historical-file edits. See RESULTS.json for complete per-track records, source hashes and limitations.','', '| Split | Tracks | Target frames | Low >=8 tracks/runs | Low >=16 tracks/runs | Low >=32 tracks/runs | Stop-go >=8 runs |','|---|---:|---:|---:|---:|---:|---:|']
for s,v in split_summary.items():
    cols=[f"{v['low_runs_thresholds'][str(k)]['tracks']}/{v['low_runs_thresholds'][str(k)]['runs']}" for k in [8,16,32]]
    lines.append(f"|{s}|{v['tracks']}|{v['valid_target_frames']}|{'|'.join(cols)}|{v['stop_go_runs_ge8']}|")
lines += ['', 'Release O longest miss distribution: '+json.dumps(result['release_O']['longest_miss_distribution']),f"Never matched within full strict observation: {result['release_O']['tracks_never_matched']}/130 tracks.",'', '## Limits','']+['- '+x for x in limits]
lines += ['', '## Development overlap with strict evaluation', '', json.dumps(dwell_overlap,ensure_ascii=False), '', 'In particular, all dwell >=16 endpoints contain only '+str(dwell_overlap['16']['low_tracks'])+' low-response track(s). This is data coverage, not causal evidence that dwelling causes a miss.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'split_summary':split_summary,'release_O':{k:v for k,v in result['release_O'].items() if k!='track_details'},'clips':[{k:v for k,v in c.items() if k not in ['track_details','track_coverage_distribution']} for c in clip_results]},ensure_ascii=False,indent=2))
