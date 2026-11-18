"""Freeze training-only, same-source/time/role legal-window replay schedules.

No images, new model predictions, development labels or intervention outcomes
are read. Known weak identities come exclusively from the completed v2 cohort.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]
V1 = ROOT / 'experiments/track_contrast_v1'
V2 = ROOT / 'experiments/c_effective_update_v2'
sys.path.insert(0, str(V1 / 'train_modules'))
import track_pair_sampler as old


def read(p): return json.loads(Path(p).read_text())
def save(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
def sha(p): return old.file_hash(p)
def rank(*args): return old.content_hash([46, *args])
def track_key(s, video, track): return f'{s}|video={video}|track={track}'


def known_objects():
    plan = read(V2/'PAIR_PLAN.json'); tracks = {}; objects = {}
    for entry_index, entry in plan['entries'].items():
        assert sha(entry['cohort_path']) == entry['cohort_sha256']
        cohort = read(entry['cohort_path']); weak = set(cohort['weak_target_indices'])
        for a in cohort['target_objects']:
            if a['target_index'] not in weak: continue
            s = a['sequence']; video = int(a['video_id']); track = int(a['track_id'])
            assert track not in plan['identity_audits'][s]['excluded_uncertain_track_ids']
            key = track_key(s, video, track)
            tracks[key] = {'sequence':s, 'video_id':video, 'track_id':track, 'scoped_track_key':key}
            objects[(s,a['annotation_id'])] = {'sequence':s, 'video_id':video, 'track_id':track,
                'annotation_id':a['annotation_id'], 'source_frame':a['source_frame'], 'scoped_track_key':key}
    assert len(tracks)==39 and len(objects)==2066
    return tracks, objects


def pixel_rect(box):
    x,y,w,h = box
    return (max(0,min(128,math.ceil(x-.5))), max(0,min(128,math.ceil(y-.5))),
            max(0,min(128,math.ceil(x+w-.5))), max(0,min(128,math.ceil(y+h-.5))))


def nonempty_label(annotations, crop):
    """Exact nonempty predicate of the frozen quantized-Gaussian support.

    Primary windows are additionally checked against all 512 historical label
    receipts. Sparse scalar rasterization avoids loading any images on CPU.
    """
    cy,cx = crop
    for frame in annotations:
        positive=[]; excluded=[]
        for a in frame:
            if not old.valid_box(a): continue
            if not a.get('ignore',0) and not a.get('iscrowd',0) and old.full_core(a,crop): positive.append(a)
            else:
                x,y,w,h=a['bbox']; excluded.append(pixel_rect((x-cx-2,y-cy-2,w+4,h+4)))
        for a in positive:
            x,y,w,h=a['bbox'];x0,y0,x1,y1=pixel_rect((x-cx,y-cy,w,h))
            for yy in range(y0,y1):
                for xx in range(x0,x1):
                    value=255*math.exp(-.5*(((yy+cy+.5-(y+h/2))/(h/6))**2+((xx+cx+.5-(x+w/2))/(w/6))**2))
                    if value>=1 and not any(a<=xx<c and b<=yy<d for a,b,c,d in excluded): return True
    return False


def longest_run(frames):
    best=current=0; prev=None
    for f in sorted(frames):
        current=current+1 if prev is not None and f==prev+1 else 1
        best=max(best,current);prev=f
    return best


def window_record(entry, wi, sequences, known):
    pi=(wi%4)//2; within=wi%2; role='primary' if wi<4 else 'auxiliary'
    pair=copy.deepcopy(entry['pairs'][pi]);crop=pair['crop_yx']
    if wi>=4:
        grid=[(y,x)for y in range(0,385,64)for x in range(0,513,64)]
        g=(entry['step']*17+pi*29+46)%63
        if list(grid[g])==crop:g=(g+1)%63
        crop=list(grid[g]);pair['crop_yx']=crop
    start=pair['starts'][within]; s=pair['sequence']; seq=sequences[s]
    positive=[]; weak_frames=defaultdict(list)
    for f in range(start,start+40):
        for a in seq.frames[f]:
            if old.valid_box(a) and not a.get('ignore',0) and not a.get('iscrowd',0) and old.full_core(a,crop):
                key=track_key(s,seq.video_ids[f],a['track_id'])
                positive.append({'key':key,'annotation_id':a['id'],'source_frame':seq.source_frames[f]})
        for track,a in seq.eligible[f].items():
            key=track_key(s,seq.video_ids[f],track)
            if key in known and old.full_core(a,crop):weak_frames[key].append(f)
    pair['starts']=[start]
    return {'original_index':entry['step'],'window':wi,'sequence':s,'role':role,'start':start,
        'crop_yx':crop,'pair':pair,'nonempty':nonempty_label(seq.frames[start:start+40],crop),
        'positive_objects':positive,'weak_frame_counts':{k:len(v)for k,v in weak_frames.items()},
        'eligible_weak_tracks':sorted(k for k,v in weak_frames.items()if longest_run(v)>=8)}


def build():
    if (E/'SCHEDULE.json').exists():raise FileExistsError('Preserve the frozen schedule; no outcome-driven reselection')
    plan=read(old.STAGE/'pilot/TRAIN_PLAN.json');sampler=old.TrackPairSampler();known,objects=known_objects()
    records=[window_record(e,w,sampler.sequences,known)for e in plan['entries']for w in range(8)]
    original_logs=[json.loads(x)for x in (old.STAGE.parent/'mutual_temporal_pilot_20261012/training/E0/run/TRAIN.jsonl').read_text().splitlines()]
    for i in range(512):
        for w in range(4):assert records[i*8+w]['nonempty']==(original_logs[i]['label_stats']['per_window'][w]['target_positive_pixels']>0),(i,w)
    order=list(range(256,512))+list(range(256)); weak_sources=sorted({v['sequence']for v in known.values()})
    pool=defaultdict(list)
    for r in records:
        if r['nonempty'] and r['eligible_weak_tracks']:pool[r['sequence'],r['role'],r['start']].append(r)
    schedule=[];slots=defaultdict(list);nonempty=Counter();original_exposure=Counter()
    for index,pi in enumerate(order):
        row={'additional_index':index,'original_index':pi,'windows':[]}
        for wi in range(8):
            r=records[pi*8+wi]; original_exposure.update(r['weak_frame_counts'])
            item={'window':wi,'donor_index':pi,'donor_window':wi,'replaced':False,'selected_track':None}
            row['windows'].append(item)
            if not r['nonempty'] or r['sequence']not in weak_sources:continue
            nonempty[r['sequence']]+=1
            donors=[d for d in pool[r['sequence'],r['role'],r['start']]if d['crop_yx']!=r['crop_yx']]
            if donors:slots[r['sequence']].append((index,wi,donors))
        schedule.append(row)
    selection_counts=Counter();donor_counts=Counter();source_audit={}
    for s in weak_sources:
        count=nonempty[s]//4
        ordered=sorted(slots[s],key=lambda a:rank('slot',s,a[0],a[1]))
        assert len(ordered)>=count
        chosen=ordered[:count]
        # One deterministic pass. Track balancing uses selected-track counts,
        # with original exposure then seed46 as tie breaks; no response used.
        for index,wi,donors in chosen:
            by_track=defaultdict(list)
            for d in donors:
                for key in d['eligible_weak_tracks']:by_track[key].append(d)
            key=min(by_track,key=lambda k:(selection_counts[k],original_exposure[k],rank('track',s,index,wi,k)))
            d=min(by_track[key],key=lambda d:(donor_counts[d['original_index'],d['window']],rank('donor',index,wi,key,d['original_index'],d['window'])))
            schedule[index]['windows'][wi].update(donor_index=d['original_index'],donor_window=d['window'],replaced=True,selected_track=key)
            selection_counts[key]+=1;donor_counts[d['original_index'],d['window']]+=1
        source_audit[s]={'nonempty_windows':nonempty[s],'same_time_role_donor_eligible_slots':len(ordered),
            'nonempty_slots_without_donor':nonempty[s]-len(ordered),'replacement_count':count,
            'replacement_fraction':count/nonempty[s],'rounding_remainder_mod4':nonempty[s]%4,
            'selected_track_counts':{k:selection_counts[k]for k in known if known[k]['sequence']==s}}
    assert all(selection_counts[k]>0 for k in known),'Every known track must receive an explicit balanced selection'
    fields=['additional_step','original_index','window','role','sequence','start','crop_y','crop_x','donor_index','donor_window','replaced','selected_track','nonempty','positive_object_frames','known_weak_object_frames','eligible_weak_tracks']
    exposures={};empty_patterns={};source_windows={};weak_actual={};window_counts={};objects_actual={}
    for condition in ('ORIGINAL','REPLAY'):
        counts=Counter();empties=[];sources=Counter();weak_counts=Counter();obj_counts=Counter();wins=Counter()
        with (E/f'{condition}_SCHEDULE.csv').open('w',newline='')as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for row in schedule:
                for item in row['windows']:
                    pi=row['original_index'];wi=item['window'];base=records[pi*8+wi]
                    donor=records[item['donor_index']*8+item['donor_window']] if condition=='REPLAY' else base
                    assert (base['sequence'],base['role'],base['start'],base['nonempty'])==(donor['sequence'],donor['role'],donor['start'],donor['nonempty'])
                    if not base['nonempty']:assert donor==base
                    sources[donor['sequence']]+=1
                    if not donor['nonempty']:empties.append((row['additional_index'],wi))
                    for p in donor['positive_objects']:counts[p['key']]+=1;obj_counts[p['key'],p['source_frame'],p['annotation_id']]+=1
                    wins.update(set(p['key']for p in donor['positive_objects']));weak_counts.update(donor['weak_frame_counts'])
                    writer.writerow(dict(additional_step=row['additional_index']+1,original_index=pi,window=wi,role=donor['role'],sequence=donor['sequence'],start=donor['start'],crop_y=donor['crop_yx'][0],crop_x=donor['crop_yx'][1],donor_index=donor['original_index'],donor_window=donor['window'],replaced=bool(condition=='REPLAY'and item['replaced']),selected_track=item['selected_track']if condition=='REPLAY'else'',nonempty=donor['nonempty'],positive_object_frames=len(donor['positive_objects']),known_weak_object_frames=sum(donor['weak_frame_counts'].values()),eligible_weak_tracks=';'.join(donor['eligible_weak_tracks'])))
        exposures[condition]=counts;empty_patterns[condition]=empties;source_windows[condition]=sources;weak_actual[condition]=weak_counts;window_counts[condition]=wins;objects_actual[condition]=obj_counts
    assert empty_patterns['ORIGINAL']==empty_patterns['REPLAY'] and source_windows['ORIGINAL']==source_windows['REPLAY']
    comparison=[]
    for key in sorted(set(exposures['ORIGINAL'])|set(exposures['REPLAY'])|set(known)):
        comparison.append({'level':'track','key':key,'known_weak':key in known,'original_exposure':exposures['ORIGINAL'][key],'replay_exposure':exposures['REPLAY'][key],'difference':exposures['REPLAY'][key]-exposures['ORIGINAL'][key], 'original_window_exposure':window_counts['ORIGINAL'][key], 'replay_window_exposure':window_counts['REPLAY'][key], 'original_strict_weak_frames':weak_actual['ORIGINAL'][key]if key in known else '', 'replay_strict_weak_frames':weak_actual['REPLAY'][key]if key in known else '', 'balanced_selections':selection_counts[key]if key in known else ''})
    for s in sorted(source_windows['ORIGINAL']):
        comparison.append({'level':'source_windows','key':s,'known_weak':s in weak_sources,'original_exposure':source_windows['ORIGINAL'][s],'replay_exposure':source_windows['REPLAY'][s],'difference':0})
        before=sum(1 for r in records if r['sequence']==s and not r['nonempty'])
        comparison.append({'level':'empty_windows','key':s,'known_weak':s in weak_sources,'original_exposure':before,'replay_exposure':before,'difference':0})
    # Per actual labeled object-frame exposure includes collateral objects.
    for k in sorted(set(objects_actual['ORIGINAL'])|set(objects_actual['REPLAY'])):
        a=objects_actual['ORIGINAL'][k];b=objects_actual['REPLAY'][k]
        comparison.append({'level':'object_frame','key':f'{k[0]}|frame={k[1]}|ann={k[2]}','known_weak':k[0]in known,'original_exposure':a,'replay_exposure':b,'difference':b-a})
    keys=['level','key','known_weak','original_exposure','replay_exposure','difference','original_window_exposure','replay_window_exposure','original_strict_weak_frames','replay_strict_weak_frames','balanced_selections']
    with(E/'EXPOSURE_COMPARISON.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(comparison)
    a=sum(weak_actual['ORIGINAL'].values());b=sum(weak_actual['REPLAY'].values());changed=sum(x['replaced']for r in schedule for x in r['windows'])
    assert b>a*1.01 and changed>0,'Exposure difference is negligible: do not train'
    result={'schema_version':1,'seed':46,'steps':512,'continuation_order':order,'known_weak_tracks':list(known.values()),'confirmed_weak_object_frames':list(objects.values()),'source_audit':source_audit,'entries':schedule,'window_pool':records,
        'rules':{'replacement_fraction':'floor(all nonempty primary+auxiliary windows in each known-weak source /4)','slot_rule':'seed46 hashed selection among slots with a different-crop existing donor in same source/start/role','balance_rule':'least explicit selected-track count; original strict-frame exposure then seed46 break ties; choose least-used donor then seed46','donor_eligibility':'known39 track has >=8 consecutive original strict eligible full-core annotations; no fresh model scoring','temporal_and_empty_policy':'all source/start/role preserved, all empty windows copied byte-for-byte','keep_cache':'reuse exact original donor-window cached teacher targets/pixels; no new keep selection','C_sampling':'original exact candidate selection algorithm, extended only to four single-start primary windows'},
        'exposure_gate':{'status':'PASS_NONTRIVIAL_EXPOSURE_CHANGE','changed_windows':changed,'total_windows':4096,'original_strict_weak_frame_exposures':a,'replay_strict_weak_frame_exposures':b,'ratio':b/a,'known_tracks_explicitly_selected':len(selection_counts),'known_tracks_increased':sum(weak_actual['REPLAY'][k]>weak_actual['ORIGINAL'][k]for k in known),'known_tracks_decreased':sum(weak_actual['REPLAY'][k]<weak_actual['ORIGINAL'][k]for k in known),'empty_positions_unchanged':True,'per_source_window_counts_unchanged':True},
        'sources':{str(p):sha(p)for p in [Path(__file__),V2/'PAIR_PLAN.json',old.STAGE/'pilot/TRAIN_PLAN.json',V1/'train_modules/track_pair_sampler.py',*(s.path for s in sampler.sequences.values())]},'csv_sha256':{p.name:sha(p)for p in [E/'ORIGINAL_SCHEDULE.csv',E/'REPLAY_SCHEDULE.csv',E/'EXPOSURE_COMPARISON.csv']}}
    save(E/'SCHEDULE.json',result)
    print(json.dumps({'gate':result['exposure_gate'],'source_audit':source_audit},ensure_ascii=False))


if __name__=='__main__':build()
