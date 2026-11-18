"""Read-only integrity and actual supervision-exposure audit of frozen plans."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from replay_data import E,V1,V2,row,changed,schedule,cache_paths,read,samplers
from runtime import sha,save_json


def main():
    freeze=read(E/'PREPARATION_FREEZE.json')
    for p,h in freeze['files'].items():assert sha(E/p)==h,p
    for p,h in freeze['source_sha256'].items():assert sha(p)==h,p
    cplans=read(V1/'PAIR_PLAN.json')['entries'];fplans=read(V2/'SEGMENT_PLAN.json')['entries']
    known={x['scoped_track_key']for x in schedule()['known_weak_tracks']}
    def key(d):return f"{d['sequence']}|video={d['video_id']}|track={d['track_id']}"
    outputs={};cache_files={};validations=Counter()
    for condition in ('ORIGINAL','REPLAY'):
        totals=Counter();tracks={k:Counter()for k in known};sources={}
        for i in range(512):
            cp,fp=cache_paths(i,condition);pi=row(i)['original_index'];new=condition=='REPLAY'and changed(i)
            for p in (cp,fp):
                if str(p)not in cache_files:
                    receipt=read(p.with_suffix('.json'));assert receipt['sha256']==sha(p),p
                    cache_files[str(p)]=receipt['sha256']
            with np.load(cp,allow_pickle=False)as c:
                cm=json.loads(str(c['metadata_json']));cplan=cm['annotation_plan']if new else cplans[pi]
                pc=c['positive_pairs_C'];negative=c['negative_indices'];active=(negative>=0).all(1)
                assert np.array_equal(pc,np.asarray(cplan['positive_pairs_C'],np.int64).reshape(-1,2))
                assert np.array_equal(c['target_regions'],np.asarray(cplan['target_regions'],np.int64).reshape(-1,6))
                assert len(pc)==len(negative)
                totals['C_pairs_declared']+=len(pc);totals['C_pairs_effective']+=int(active.sum());totals['steps_without_effective_C']+=not active.any()
                totals['keep_reliable_pixels']+=int((c['keep_groups']==0).sum());totals['keep_background_pixels']+=int((c['keep_groups']==1).sum())
                assert np.isfinite(c['keep_probabilities']).all()
                assert np.isin(c['keep_groups'],[0,1]).all()
                for j,(a,b)in enumerate(pc):
                    da,db=cplan['descriptors'][int(a)],cplan['descriptors'][int(b)]
                    assert key(da)==key(db) and da['window']==db['window']
                    assert db['source_frame']-da['source_frame']in(1,4)
                    if key(da)in known:
                        tracks[key(da)]['C_pairs_declared']+=1
                        tracks[key(da)]['C_pairs_effective']+=int(active[j])
                if new:
                    for donor in cm['donor_keep']:
                        p=V1/'cache'/f"entry_{donor['donor_index']:03d}.npz";assert sha(p)==donor['cache_sha256']
                        with np.load(p,allow_pickle=False)as old:
                            take=old['keep_indices'][:,0]==donor['donor_window'];now=c['keep_indices'][:,0]==donor['new_window']
                            coordinates=old['keep_indices'][take].copy();coordinates[:,0]=donor['new_window']
                            for k,expected in [('keep_indices',coordinates),('keep_probabilities',old['keep_probabilities'][take]),('keep_groups',old['keep_groups'][take])]:assert np.array_equal(c[k][now],expected),(i,k)
                    validations['new_contrast_caches_checked']+=1
            with np.load(fp,allow_pickle=False)as f:
                fm=json.loads(str(f['metadata_json']));fplan=fm['annotation_plan']if 'annotation_plan'in fm else fplans[pi]
                segments=f['segment_indices'];ids=segments.flatten();neg=f['negative_indices'];valid=(neg>=0).any(1)
                assert segments.shape[1:]==(8,)and (not len(ids)or valid[ids].all())
                assert np.array_equal(f['target_regions'],np.asarray(fplan['target_regions'],np.int64).reshape(-1,6))
                for k in ('input_sha256','label_sha256','auxiliary_input_sha256'):assert fm[k]==cm[k],(condition,i,k)
                totals['FRAME_segments']+=len(segments);totals['FRAME_observations_with_multiplicity']+=len(ids)
                totals['FRAME_unique_window_observations']+=len(np.unique(ids));totals['FRAME_background_regions']+=len(f['background_regions'])
                totals['steps_without_FRAME']+=len(segments)==0
                counts=Counter(map(int,ids));unique=Counter()
                for j,n in counts.items():
                    k=key(fplan['descriptors'][j])
                    if k in known:tracks[k]['FRAME_observations_with_multiplicity']+=n;tracks[k]['FRAME_unique_window_observations']+=1
                for k in known:
                    descriptors=[d for d in fplan['descriptors']if key(d)==k]
                    if descriptors:tracks[k]['strict_annotation_window_frames']+=len(descriptors)
            totals['steps']+=1
        for metric in ('strict_annotation_window_frames','FRAME_observations_with_multiplicity','FRAME_unique_window_observations','C_pairs_declared','C_pairs_effective'):
            totals['known_weak_'+metric]=sum(v[metric]for v in tracks.values())
        expected=schedule()['exposure_gate']['original_strict_weak_frame_exposures'if condition=='ORIGINAL'else'replay_strict_weak_frame_exposures']
        assert totals['known_weak_strict_annotation_window_frames']==expected,(condition,expected,totals['known_weak_strict_annotation_window_frames'])
        outputs[condition]={'totals':dict(totals),'known_weak_track_exposure':{k:dict(v)for k,v in sorted(tracks.items())}}
    delta={k:outputs['REPLAY']['totals'].get(k,0)-v for k,v in outputs['ORIGINAL']['totals'].items()}
    per_track=[]
    for k in sorted(known):
        a=outputs['ORIGINAL']['known_weak_track_exposure'][k];b=outputs['REPLAY']['known_weak_track_exposure'][k]
        per_track.append({'track':k,'original':a,'replay':b,'delta':{name:b.get(name,0)-a.get(name,0)for name in sorted(set(a)|set(b))}})
    report={'status':'PASS','interpretation':'Counts are supervision exposures, often overlapping, not independent objects. Loss formulas, coefficients and reductions are unchanged; window replacement naturally changes these sampled counts.',
        'preparation_freeze_sha256':sha(E/'PREPARATION_FREEZE.json'),'schedule_sha256':sha(E/'SCHEDULE.json'),
        'conditions':outputs,'total_difference_replay_minus_original':delta,'per_track_difference':per_track,
        'validation':dict(validations),'cache_sha256':cache_files,'new_optimizer_updates':0,'new_weak_qualification':False}
    save_json(E/'CACHE_EXPOSURE.json',report)
    print(json.dumps({'status':'PASS','conditions':{k:v['totals']for k,v in outputs.items()},'delta':delta,'validation':dict(validations),'files':len(cache_files)}))


if __name__=='__main__':main()
