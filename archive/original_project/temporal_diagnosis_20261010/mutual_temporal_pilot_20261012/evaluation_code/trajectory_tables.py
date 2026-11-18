"""Fixed full-observation miss-run summaries, no selection by new outcome."""
import collections
import csv
import hashlib
import json
import numpy as np


def write_csv(path, rows):
    assert rows
    with path.open('w', newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def adjacent(a,b):
    return int(b['frame'])==int(a['frame'])+1 and int(b['source_frame'])==int(a['source_frame'])+1


def segments(rows,predicate):
    result=[];active=[];previous=None
    for r in rows:
        if previous is not None and not adjacent(previous,r):
            if active:result.append(active)
            active=[]
        if predicate(r):active.append(r)
        else:
            if active:result.append(active)
            active=[]
        previous=r
    if active:result.append(active)
    return result


def trajectory_tables(objects,conditions,comparisons,output,q):
    partitions=collections.defaultdict(list)
    for row in objects:
        partitions[row['arrangement'],row['sequence'],row['track_id']].append(row)
    tracks=[];release_runs=[]
    cohorts=['all','low_all_05','extreme_all_001','both_fixed_success','candidate_window_mixed']
    for (plan,seq,tid),raw in sorted(partitions.items()):
        raw=sorted(raw,key=lambda r:(r['frame'],r['source_frame']))
        for cohort in cohorts:
            rows=[r for r in raw if cohort=='all' or r[cohort]]
            if not rows:continue
            observations=segments(rows,lambda r:True)
            per_condition={c:segments(rows,lambda r,c=c:not r[c+'_matched']) for c in conditions}
            for c in conditions:
                miss=per_condition[c]
                row=dict(condition=c,arrangement=plan,sequence=seq,track_id=tid,cohort=cohort,
                         target_frames=len(rows),observation_segments=len(observations),
                         observation_longest=max(map(len,observations)),
                         observation_gap_pairs=sum(not adjacent(a,b) for a,b in zip(rows,rows[1:])),
                         missed_frames=sum(map(len,miss)),miss_run_count=len(miss),
                         longest_miss_run=max(map(len,miss),default=0),
                         miss_run_lengths=json.dumps([len(s) for s in miss],separators=(',',':')))
                for ref in comparisons:
                    lengths=[len(s) for s in per_condition[ref]]
                    row['longest_miss_delta_vs_'+ref]=row['longest_miss_run']-max(lengths,default=0)
                    row['missed_frames_delta_vs_'+ref]=row['missed_frames']-sum(lengths)
                tracks.append(row)
        # All release misses, not merely segments where some new model recovers.
        for index,seg in enumerate(segments(raw,lambda r:not r['release_matched'])):
            row=dict(arrangement=plan,sequence=seq,track_id=tid,release_miss_segment=index,
                     frame_start=seg[0]['frame'],frame_end=seg[-1]['frame'],
                     source_frame_start=seg[0]['source_frame'],source_frame_end=seg[-1]['source_frame'],
                     target_frames=len(seg),low_all_05_frames=sum(r['low_all_05'] for r in seg))
            for c in conditions:
                left=segments(seg,lambda r,c=c:not r[c+'_matched'])
                row[c+'_recovered_frames']=sum(r[c+'_matched'] for r in seg)
                row[c+'_remaining_miss_longest']=max(map(len,left),default=0)
                row[c+'_remaining_miss_runs']=len(left)
            release_runs.append(row)
    summaries=[]
    for c in conditions:
        for plan in ['O','S']:
            for group in ['pooled','DJI_0026_2','DJI_0067_2']:
                for cohort in cohorts:
                    ts=[r for r in tracks if r['condition']==c and r['arrangement']==plan and r['cohort']==cohort
                        and (group=='pooled' or r['sequence']==group)]
                    if not ts:continue
                    lengths=[r['longest_miss_run'] for r in ts]
                    row=dict(condition=c,arrangement=plan,group=group,cohort=cohort,
                             target_frames=sum(r['target_frames'] for r in ts),tracks=len(ts),
                             clips=len({r['sequence'] for r in ts}),missed_frames=sum(r['missed_frames'] for r in ts),
                             miss_run_count=sum(r['miss_run_count'] for r in ts),
                             longest_miss_track_mean=float(np.mean(lengths)),
                             longest_miss_track_median=float(np.median(lengths)),
                             longest_miss_track_q90=float(np.quantile(lengths,.9)),
                             longest_miss_max=max(lengths),
                             track_equal_miss_fraction=float(np.mean([r['missed_frames']/r['target_frames'] for r in ts])))
                    for ref in comparisons:
                        ds=[r['longest_miss_delta_vs_'+ref] for r in ts]
                        row['tracks_shorter_vs_'+ref]=sum(d<0 for d in ds)
                        row['tracks_same_vs_'+ref]=sum(d==0 for d in ds)
                        row['tracks_longer_vs_'+ref]=sum(d>0 for d in ds)
                        row['longest_delta_track_mean_vs_'+ref]=float(np.mean(ds))
                    summaries.append(row)
    for plan in ['O','S']:
        base=[r for r in tracks if r['condition']=='release' and r['arrangement']==plan and r['cohort']=='all']
        assert len(base)==130 and sum(r['target_frames'] for r in base)==5200
        assert sum(r['missed_frames'] for r in base)==(4500 if plan=='O' else 4480)
        assert sum(r['target_frames'] for r in release_runs if r['arrangement']==plan)==sum(r['missed_frames'] for r in base)
    write_csv(output/'CONTINUOUS_MISS_TRACKS.csv',tracks)
    write_csv(output/'CONTINUOUS_MISS_SUMMARY.csv',summaries)
    write_csv(output/'RELEASE_MISS_SEGMENTS.csv',release_runs)
    prior=q/'low_response_information_20261011/analysis/TRACK_SCHEMA.json'
    schema=dict(status='COMPLETE',prior_schema=str(prior),prior_schema_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
                full_observation_scope='Exactly all 5200 frozen strict-small target frames, 130 tracks, two clips per arrangement. No filtering by new match or recovery.',
                continuity='Both local frame and source frame must increase by exactly one. Missing labels, frame gaps and original selection holes break runs.',
                cohort_continuity='Cohort rows are separately filtered; excluded rows create local/source gaps and break runs. Cohort longest runs are not complete physical trajectory lengths.',
                release_segments='All maximal release-miss segments within the full frozen observation range, separately for O/S. The same segment boundaries are used for every new condition.',
                quantiles='Linear interpolation; tracks with observations and zero misses have longest miss 0. No-observation track/cohort combinations omitted.',
                artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [output/'CONTINUOUS_MISS_TRACKS.csv',output/'CONTINUOUS_MISS_SUMMARY.csv',output/'RELEASE_MISS_SEGMENTS.csv']})
    (output/'CONTINUOUS_MISS_SCHEMA.json').write_text(json.dumps(schema,indent=2)+'\n')
