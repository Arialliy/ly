#!/usr/bin/env python3
"""Independent stdlib audit. Reads source CSV/annotations; imports no analysis code."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

Q = Path(__file__).resolve().parents[2]
S = Q / 'result_stratification'
OUT = Path(__file__).resolve().parent
SEQS = ['DJI_0028_2', 'DJI_0028_4', 'DJI_0032_5', 'DJI_0047_3']
INPUTS = set()
ERRORS = []


def read_csv(p):
    INPUTS.add(p)
    with p.open() as f:
        return list(csv.DictReader(f))


def read_json(p):
    INPUTS.add(p)
    return json.loads(p.read_text())


def yes(x):
    return str(x).lower() in ('1', 'true')


def check(kind, key, expected, actual):
    if isinstance(expected, float) and isinstance(actual, float):
        good = math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9)
    else:
        good = expected == actual
    if not good:
        ERRORS.append(dict(kind=kind, key=str(key), expected=expected, actual=actual))


def sb(x):
    return '<=8' if x <= 8 else '(8,12]' if x <= 12 else '(12,16]' if x <= 16 else '>16'


def intersect(a, b):
    x, y, w, h = a
    u, v, s, t = b
    return min(x + w, u + s) > max(x, u) and min(y + h, v + t) > max(y, v)


def summary(rows):
    n = len(rows)
    tr = set(r['tk'] for r in rows)
    return dict(target_frames=n, tracks=len(tr), sequences=len(set(r['sequence'] for r in rows)),
                matched_frames=sum(r['point'] for r in rows),
                response_frames=sum(not r['low'] for r in rows),
                low_box_response_frames=sum(r['low'] for r in rows),
                tracks_with_low_response=len(set(r['tk'] for r in rows if r['low'])),
                motion_valid_frames=sum(r['motion_valid'] for r in rows))


rows = []
contexts = {}
metadata = {}
annotation_counts = {}
for seq in SEQS:
    ap = (Q.parent / 'real_motion_stage2/selected_raw/annotations.json' if seq == 'DJI_0028_2'
          else next((Q / 'additional_clips').glob(seq + '*/annotations.json')))
    ann = read_json(ap)
    ims = {int(i['id']): i for i in ann['images']}
    cats = {int(c['id']): c['name'] for c in ann['categories']}
    aa = {int(a['id']): a for a in ann['annotations']}
    af = defaultdict(list)
    tf = defaultdict(set)
    identities = defaultdict(set)
    class_identities = defaultdict(set)
    for a in ann['annotations']:
        af[int(a['image_id'])].append(a)
        tf[int(a['track_id'])].add(int(ims[int(a['image_id'])]['frame_id']))
        identities[str(a.get('tracking_id'))].add(int(a['track_id']))
        class_identities[(cats[int(a['category_id'])],str(a.get('tracking_id')))].add(int(a['track_id']))
    for tid, frames in tf.items():
        annotation_counts[(seq, tid)] = len(frames)
    metadata[seq] = dict(raw_annotations=len(aa), tracks=len(tf),
        tracks_ge40=sum(len(x) >= 40 for x in tf.values()),
        tracks_with_missing_frames=sum(max(x)-min(x)+1 != len(x) for x in tf.values()),
        source_tracking_ids_with_multiple_track_ids={k: sorted(v) for k,v in identities.items() if len(v)>1},
        category_and_source_tracking_ids_with_multiple_track_ids={str(k):sorted(v) for k,v in class_identities.items() if len(v)>1})
    ctx = defaultdict(list)
    for c in read_csv(Q / 'window_offset_experiment/analysis' / seq / 'per_context_all_tracks.csv'):
        if float(c['threshold']) == .5:
            ctx[(seq, int(c['source_frame']), int(c['track_id']))].append(c)
    contexts.update(ctx)
    mdir = Q / 'motion' if seq == 'DJI_0028_2' else Q / 'motion' / seq
    read_json(mdir / 'PROTOCOL.json')
    contrast = {(int(c['raw_frame']), int(c['track_id'])): c for c in read_csv(mdir / 'TRACK_FRAME_DIAGNOSTICS.csv')}
    motion = {(int(c['raw_frame_from']), int(c['track_id'])): c for c in read_csv(mdir / 'LAG5_TRACK_MOTION.csv')}
    for r in read_csv(Q / 'analysis' / seq / 'per_track_frame.csv'):
        if float(r['threshold']) != .5:
            continue
        tid, frame = int(r['track_id']), int(r['source_frame'])
        key = (seq, frame, tid)
        a = aa[int(r['annotation_id'])]
        im = ims[int(a['image_id'])]
        x,y,w,h = a['bbox']
        partly = x < 0 or y < 0 or x+w > im['width'] or y+h > im['height']
        overlap = sum(intersect(a['bbox'], b['bbox']) for b in af[int(a['image_id'])] if b['id'] != a['id'])
        clean = not partly and overlap == 0
        check('partly_outside', key, partly, yes(r['partly_outside']))
        check('overlap_count', key, overlap, int(r['label_overlap_count']))
        c = contrast.get((frame, tid), {})
        m = motion.get((frame, tid), {})
        cv = yes(c.get('contrast_valid'))
        signed = float(c['box_mean']) - float(c['ring_mean']) if cv else None
        # The saved motion protocol defines a one-gray-level standard-deviation floor.
        std = abs(signed) / max(float(c['ring_std']), 1) if cv else None
        valid_motion = bool(m) and yes(m.get('camera_adjustment_valid')) and bool(m.get('adjusted_speed_px_per_frame'))
        speed = float(m['adjusted_speed_px_per_frame']) if valid_motion else None
        mb = 'invalid' if speed is None else '<=0.15' if speed <= .15 else '(0.15,0.5]' if speed <= .5 else '(0.5,1]' if speed <= 1 else '>1'
        rows.append(dict(sequence=seq, frame=frame, track_id=tid, tk=(seq,tid), key=key,
            local_frame=int(r['local_frame']), annotation_id=int(r['annotation_id']),
            category=cats[int(a['category_id'])], max_side=max(w,h), scale_bin=sb(max(w,h)),
            clean=clean, strict=clean and not yes(r['association_ambiguous']) and not any(yes(c['association_ambiguous']) for c in ctx[key]),
            point=int(r['matched']), low=float(r['box_max']) < .5, boxmax=float(r['box_max']),
            prediction_boxmean=float(r['box_mean']), contrast_valid=cv,
            image_boxmean=float(c['box_mean']) if cv else None, signed=signed, standardized=std,
            signed_contrast_bin='invalid' if signed is None else '<-10' if signed < -10 else '>10' if signed > 10 else '[-10,10]',
            absolute_standardized_contrast_bin='invalid' if std is None else '<0.5' if std < .5 else '[0.5,1)' if std < 1 else '>=1',
            motion_valid=valid_motion, motion_bin=mb, speed=speed, motion=m))

source = {r['key']:r for r in rows}
check('unique_source_rows', 'all', len(rows), len(source))
joined = [r for r in read_csv(S/'response/JOINED_TRACK_FRAME.csv') if float(r['threshold']) == .5]
for j in joined:
    key = (j['sequence'],int(j['source_frame']),int(j['track_id']))
    r = source[key]
    for a,b in [('clean','clean_inframe_no_gt_overlap'), ('strict','clean_plus_no_context_ambiguity'), ('low','low_box_response'),('motion_valid','motion_valid')]:
        check('joined_boolean', (key,b), r[a], yes(j[b]))
    for a,b in [('scale_bin','scale_bin'),('category','category'),('motion_bin','motion_bin'),('signed_contrast_bin','signed_contrast_bin'),('absolute_standardized_contrast_bin','absolute_standardized_contrast_bin')]:
        check('joined_label',(key,b),r[a],j[b])
    if r['contrast_valid']:
        for a,b in [('image_boxmean','image_box_mean'),('signed','signed_mean_contrast'),('standardized','absolute_standardized_mean_contrast')]:
            check('image_contrast',(key,b),r[a],float(j[b]))
    if r['motion']:
        check('motion_join_start',key,r['frame'],int(r['motion']['raw_frame_from']))
        check('motion_join_lag',key,5,int(r['motion']['raw_frame_to'])-r['frame'])

subsets = {'all_annotations': lambda r:True, 'clean_inframe_no_gt_overlap':lambda r:r['clean'], 'clean_plus_no_context_ambiguity':lambda r:r['strict']}
populations = {'all_tracks': lambda r: True, 'tracks_ge40_annotation_frames':lambda r:annotation_counts[r['tk']]>=40}
tables = {}
for fields in [(),('category',),('scale_bin',),('category','scale_bin'),('sequence','category'),('sequence','scale_bin'),('sequence','category','scale_bin'),('motion_bin',),('signed_contrast_bin',),('absolute_standardized_contrast_bin',)]:
    name = '__'.join(fields) if fields else 'overall'
    existing = [r for r in read_csv(S / 'response' / ('SUMMARY_'+name+'.csv')) if float(r['threshold']) == .5]
    lookup = {(r['population'],r['subset'],tuple(r[f] for f in fields)):r for r in existing}
    computed = []
    for pn,pf in populations.items():
        for sn,sf in subsets.items():
            groups = defaultdict(list)
            for r in rows:
                if pf(r) and sf(r):groups[tuple(r[f] for f in fields)].append(r)
            for g,rr in groups.items():
                values = summary(rr)
                target = lookup[(pn,sn,g)]
                for k,v in values.items():check('response_summary',(name,pn,sn,g,k),v,int(target[k]))
                if pn == 'all_tracks' and sn == 'clean_inframe_no_gt_overlap':
                    computed.append(dict(zip(fields,g),**values))
    tables[name] = computed

# Audit the subsequently reported exploratory combination using original joined sources.
focus_tables = {}
for fields in [(),('sequence',),('category',),('sequence','category','absolute_standardized_contrast_bin')]:
    name='__'.join(fields) if fields else 'overall'
    groups=defaultdict(list)
    for r in rows:
        if r['clean'] and r['max_side']<=16 and r['motion_valid'] and r['speed']>.5 and r['signed'] is not None and r['signed']>10:
            groups[tuple(r[f] for f in fields)].append(r)
    expected=read_csv(S/'response'/('FOCUS_'+name+'.csv'))
    lookup={tuple(r[f] for f in fields):r for r in expected if float(r['threshold'])==.5 and r['population']=='all_tracks' and r['subset']=='clean_inframe_no_gt_overlap'}
    focus_tables[name]=[]
    for key,rr in groups.items():
        values=summary(rr)
        for k,v in values.items():check('exploratory_focus',(name,key,k),v,int(lookup[key][k]))
        focus_tables[name].append(dict(zip(fields,key),**values))

# Runs are recomputed from original source frame numbers after the fixed subset filter.
# A missing annotation, excluded frame or non-low observation always breaks a run.
expected_runs = set()
run_stats = []
for pn,pf in populations.items():
    for sn,sf in subsets.items():
        tracks = defaultdict(list)
        for r in rows:
            if pf(r) and sf(r): tracks[r['tk']].append(r)
        runrows = []
        for tk,rr in tracks.items():
            current = []
            runs = []
            for r in sorted(rr,key=lambda r:r['frame']):
                if current and (not r['low'] or r['frame'] != current[-1]+1):
                    runs.append(current);current=[]
                if r['low']:current.append(r['frame'])
            if current:runs.append(current)
            for run in runs:
                expected_runs.add((pn,sn,tk[0],str(tk[1]),run[0],run[-1],len(run)))
            runrows.append(dict(sequence=tk[0],track_id=tk[1],category=rr[0]['category'],target_frames=len(rr),low=sum(r['low'] for r in rr),longest=max(map(len,runs),default=0),runs=len(runs)))
        run_stats.append(dict(population=pn,subset=sn,tracks=len(tracks),tracks_run_ge20=sum(r['longest']>=20 for r in runrows),tracks_run_ge40=sum(r['longest']>=40 for r in runrows),tracks_run_ge80=sum(r['longest']>=80 for r in runrows),track_details=runrows))
actual_runs = set((r['population'],r['subset'],r['sequence'],r['track_id'],int(r['start_source_frame']),int(r['end_source_frame']),int(r['frames'])) for r in read_csv(S/'response/LOW_RESPONSE_RUNS.csv') if float(r['threshold'])==.5)
check('run_set_missing','primary',[],sorted(expected_runs-actual_runs))
check('run_set_extra','primary',[],sorted(actual_runs-expected_runs))

# Re-select policies from upstream per-context observations, independently of choice CSV.
choices = []
existing_choices = {(r['sequence'],int(r['source_frame']),int(r['track_id'])):r for r in read_csv(S/'context/same_gt_frame_choices.csv') if float(r['threshold'])==.5}
for key,cc in contexts.items():
    r=source[key];j=existing_choices[key]
    starts=[int(c['window_start']) for c in cc]
    check('context_availability',key,sorted(s for s in range(0,121,12) if s<=r['local_frame']<s+40),sorted(starts))
    policies = dict(earliest=min(cc,key=lambda c:int(c['window_start'])),latest=max(cc,key=lambda c:int(c['window_start'])),center=min(cc,key=lambda c:(abs(int(c['slot'])-19.5),int(c['window_start']))))
    item = dict(r, multi=len(cc)>=2, context_strict=r['clean'] and not any(yes(c['association_ambiguous']) for c in cc),
                all_low=all(float(c['box_max'])<.5 for c in cc),all_point_negative=not any(yes(c['matched']) for c in cc),
                point_disagreement=len(set(c['matched'] for c in cc))>1,
                response_disagreement=len(set(float(c['box_max'])>=.5 for c in cc))>1,
                strong_response_crossing=min(float(c['box_max']) for c in cc)<.1 and max(float(c['box_max']) for c in cc)>=.9)
    for rule,c in policies.items():
        check('policy_start',(key,rule),int(c['window_start']),int(j[rule+'_start']))
        check('policy_slot',(key,rule),r['local_frame']-int(c['window_start']),int(j[rule+'_slot']))
        item[rule+'_point']=int(c['matched']);item[rule+'_response']=int(float(c['box_max'])>=.5)
        for metric in ('point','response'):check('policy_value',(key,rule,metric),item[rule+'_'+metric],int(j[rule+'_'+metric]))
    for a,b in [('all_low','all_contexts_boxmax_low'),('all_point_negative','all_contexts_point_negative'),('context_strict','strict'),('clean','clean')]:check('choice_boolean',(key,b),item[a],yes(j[b]))
    choices.append(item)

context_tables=[]
pair_tables=[]
pair_csv=read_csv(S/'context/pairwise_gain_loss.csv')
pair_lookup={(r['subset'],r['sequence'],r['scale'],r['a'],r['b'],r['metric']):r for r in pair_csv if float(r['threshold'])==.5 and r['stratum_kind']=='all'}
for subset,fun in [('all_multi',lambda r:True),('clean',lambda r:r['clean']),('strict',lambda r:r['context_strict'])]:
    for seq in ['ALL']+SEQS:
        for scale in ['all','<=12','<=16','<=8','(8,12]','(12,16]','>16']:
            rr=[r for r in choices if r['multi'] and fun(r) and (seq=='ALL' or r['sequence']==seq) and (scale=='all' or scale=='<=12' and r['max_side']<=12 or scale=='<=16' and r['max_side']<=16 or scale==r['scale_bin'])]
            if not rr:continue
            entry=dict(subset=subset,sequence=seq,scale=scale,gt_frames=len(rr),tracks=len(set(r['tk'] for r in rr)),point_disagreement=sum(r['point_disagreement'] for r in rr),response_disagreement=sum(r['response_disagreement'] for r in rr),all_low=sum(r['all_low'] for r in rr),all_point_negative=sum(r['all_point_negative'] for r in rr),point_sensitive_tracks=len(set(r['tk'] for r in rr if r['point_disagreement'])),response_sensitive_tracks=len(set(r['tk'] for r in rr if r['response_disagreement'])))
            context_tables.append(entry)
            entry['strong_response_crossing']=sum(r['strong_response_crossing'] for r in rr)
            entry['strong_response_crossing_tracks']=len(set(r['tk'] for r in rr if r['strong_response_crossing']))
            for a,b in [('earliest','latest'),('earliest','center'),('latest','center')]:
                for metric in ['point','response']:
                    av=[r[a+'_'+metric] for r in rr];bv=[r[b+'_'+metric] for r in rr]
                    gains=sum(not x and y for x,y in zip(av,bv));losses=sum(x and not y for x,y in zip(av,bv))
                    values=dict(gt_frames=len(rr),tracks=len(set(r['tk'] for r in rr)),a_positive=sum(av),b_positive=sum(bv),gains=gains,losses=losses,net=gains-losses)
                    target=pair_lookup[(subset,seq,scale,a,b,metric)]
                    for k,v in values.items():check('context_summary',(subset,seq,scale,a,b,metric,k),v,int(target[k]))
                    pair_tables.append(dict(subset=subset,sequence=seq,scale=scale,a=a,b=b,metric=metric,**values))

output=dict(primary_threshold=.5,independence='stdlib only; no response/context analysis module imported; counts rebuilt from upstream annotations, original per_track_frame.csv, per_context_all_tracks.csv, and motion diagnostics',
    source_rows=len(rows),metadata=metadata,verified_response_tables=tables,verified_exploratory_focus=focus_tables,low_response_runs=run_stats,
    context_sensitivity=context_tables,context_pairwise=pair_tables,
    checks=dict(error_count=len(ERRORS),errors=ERRORS,primary_low_run_records=len(expected_runs),joined_primary_rows=len(joined),policy_gt_frame_rows=len(choices)),
    inputs_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS)})
(OUT/'INDEPENDENT_CHECKS.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(dict(errors=len(ERRORS),first_errors=ERRORS[:8],overall=tables['overall'],metadata=metadata,runs=[{k:v for k,v in r.items() if k!='track_details'} for r in run_stats],context=[r for r in context_tables if r['sequence']=='ALL' and r['scale'] in ['all','<=12']]),indent=2))
