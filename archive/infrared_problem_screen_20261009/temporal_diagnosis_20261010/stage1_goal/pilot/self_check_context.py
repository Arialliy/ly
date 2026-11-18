#!/usr/bin/env python3
"""Synthetic-only context-summary verification; no real predictions are opened."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

import summarize_pilot as f
import summarize_context as c
from self_check_summary import fixture


def bundle_test():
    frames=[30,30,30,30,30,30,1]
    objects=[{'frame':t,'source_frame':100+t,'annotation_id':i+1000,'image_id':t+2000,
              'track_id':i,'category_id':3,'bbox':[10.+20*i,10.,4.,4.],'strict_small':1}
             for i,t in enumerate(frames)]
    keys=[f.object_key(o) for o in objects]
    covered=np.array([[start<=t<start+40 for t in frames] for start in c.STARTS])
    r=np.zeros_like(covered);q=np.zeros_like(covered)
    for i in [0,1,4,6]:r[:,i]=covered[:,i]
    r[0,2]=True;r[0,5]=True
    for i in [0,4,5,6]:q[:,i]=covered[:,i]
    q[1,3]=True
    def make(match):
        x=np.full(match.shape,np.nan);y=x.copy()
        for wi,oi in zip(*np.where(match)):x[wi,oi]=objects[oi]['bbox'][0]+2;y[wi,oi]=12
        return SimpleNamespace(threshold=.5,objects=objects,keys=keys,strict=np.ones(7,bool),
                covered=covered,matched=match,peak_x=x,peak_y=y,center_error=np.where(match,0.,np.nan))
    reference,candidate=make(r),make(q)
    def selected(match):
        return {'objects':objects,'keys':keys,'strict':np.ones(7,bool),'matched':np.array(match,np.uint8),
                'peak_x':np.zeros(7),'peak_y':np.zeros(7),'center_error':np.zeros(7),'component_area':np.ones(7)}
    rf=selected([1,0,1,0,1,1,1]);cf=selected([0,0,0,1,1,1,1])
    row,objects_out,windows,_=c.analyze_sequence('candidate','synthetic',reference,candidate,rf,cf)
    expected={'eligible_object_frames':6,'excluded_coverage_lt_2':1,'window_object_pairs':18,
              'reference_window_matches':11,'candidate_window_matches':10,'gained_window_matches':3,
              'lost_window_matches':4,'reference_any_to_candidate_none':2,
              'reference_none_to_candidate_any':1,'reference_all_to_candidate_all':2,
              'reference_all_to_candidate_none':1,'reference_partial_to_candidate_none':1,
              'reference_partial_to_candidate_all':1,'stable_original_fused_misses':1,
              'protected_denominator':2,'protected_candidate_fused_losses':1}
    for key,value in expected.items():assert row[key]==value,(key,row[key],value)
    assert row['protection_gate_status']=='failed'
    assert len(objects_out)==6 and len(windows)==11
    # Reorder the candidate's fused and window tensors: key-based alignment must preserve all counts.
    order=np.array([6,2,0,5,1,3,4])
    candidate2=SimpleNamespace(threshold=.5,objects=[objects[i] for i in order],keys=[keys[i] for i in order],
          strict=candidate.strict[order],**{k:getattr(candidate,k)[:,order] for k in ['covered','matched','peak_x','peak_y','center_error']})
    cf2={**cf,'objects':candidate2.objects,'keys':candidate2.keys,
         **{k:cf[k][order] for k in ['strict','matched','peak_x','peak_y','center_error','component_area']}}
    row2,_,_,_=c.analyze_sequence('candidate','synthetic',reference,candidate2,rf,cf2)
    for key,value in expected.items():assert row2[key]==value
    assert c.protection_gate(100,1)['passes']
    assert not c.protection_gate(99,1)['passes']
    assert c.protection_gate(0,0)['status']=='not_evaluable_empty_protected_set'


def relocate_fixture(directory):
    meta_path=directory/'official_fused_COMPLETE.json';npz_path=directory/'official_fused_matches.npz'
    meta=json.loads(meta_path.read_text());frames=[20,30,50]
    for obj,frame in zip(meta['objects'],frames):obj['frame']=frame;obj['source_frame']=frame+100
    with np.load(npz_path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    data['offsets']=np.array([sum(t<frame for t in frames) for frame in range(161)])
    old=data['frame_counts'];counts=np.zeros_like(old);counts[:,:,0]=np.diff(data['offsets'])[None,:]
    for ti in range(len(f.GRID)):
        for oi,frame in enumerate(frames):counts[ti,frame,1]=data['matched'][ti,oi]
        counts[ti,10,3]=old[ti,:,3].sum();counts[ti,:,2]=counts[ti,:,1]+counts[ti,:,3]
        counts[ti,:,4]=counts[ti,:,2];counts[ti,:,5]=counts[ti,:,3]
    data['frame_counts']=counts
    np.savez_compressed(npz_path,**data);f.write_json(meta_path,meta)


def context_fixture(root,variant,sequence,score):
    directory=root/variant/sequence;directory.mkdir(parents=True)
    selected=score.selected(.5)
    for start in c.STARTS:
        indices=np.array([i for i,obj in enumerate(score.objects) if start<=obj['frame']<start+40],dtype=int)
        objects=[score.objects[i] for i in indices];m=len(objects)
        offsets=np.array([sum(obj['frame']<frame for obj in objects) for frame in range(start,start+41)])
        counts=np.zeros((1,40,6),np.int64);counts[0,:,0]=np.diff(offsets)
        for ci,oi in enumerate(indices):
            local=score.objects[oi]['frame']-start;counts[0,local,1]+=selected['matched'][oi]
        counts[0,:,2]=counts[0,:,1];counts[0,:,4]=counts[0,:,1]
        arrays={k:selected[k][indices][None] for k in ['matched','peak_x','peak_y','center_error','component_area']}
        np.savez_compressed(directory/f'window_{start:03d}_matches.npz',**arrays,thresholds=np.array([.5]),
                  strict_small=score.strict[indices],frame_counts=counts,offsets=offsets,box_max=np.ones(m))
        f.write_json(directory/f'window_{start:03d}_COMPLETE.json',{'status':'COMPLETE','name':f'window_{start:03d}',
            'start':start,'end':start+40,'thresholds':[.5],'frame_count_fields':f.FRAME_FIELDS,
            'annotations_sha256':score.meta['annotations_sha256'],'prediction_sha256':'synthetic_window',
            'objects':objects})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    bundle_test()
    with tempfile.TemporaryDirectory(prefix='stage1_context_selfcheck_') as directory:
        root=Path(directory);evaluation=root/'evaluation';context=root/'context';output=root/'summary'
        selection=root/'selection.json';sel={'splits':{'calibration':['calA','calB'],'confirmation':['confirmationSynthetic']}}
        f.write_json(selection,sel)
        for variant in f.VARIANTS:
            for sequence in ['calA','calB','confirmationSynthetic']:
                fixture(evaluation,variant,sequence);relocate_fixture(evaluation/variant/sequence)
            score=f.FusedScore(evaluation,variant,'confirmationSynthetic')
            context_fixture(context,variant,'confirmationSynthetic',score)
        cli=SimpleNamespace(evaluation_root=evaluation,context_root=context,selection=selection,output=output,
                            calibration=None,fused_summary=None)
        with contextlib.redirect_stdout(io.StringIO()):f.calibrate(cli,sel);f.confirmation(cli,sel)
        score=f.FusedScore(evaluation,'official','confirmationSynthetic')
        b=c.WindowBundle(context,'official','confirmationSynthetic',.5,score)
        assert b.covered[:,0].sum()==2 and b.covered[:,1].sum()==3
        # Missing window and wrong threshold cannot be silently accepted.
        meta_path=context/'official/confirmationSynthetic/window_000_COMPLETE.json'
        original=meta_path.read_text();meta_path.unlink()
        try:c.WindowBundle(context,'official','confirmationSynthetic',.5,score)
        except ValueError:pass
        else:raise AssertionError('Missing window was accepted')
        meta_path.write_text(original);bad=json.loads(original);bad['thresholds']=[.4];f.write_json(meta_path,bad)
        try:c.WindowBundle(context,'official','confirmationSynthetic',.5,score)
        except ValueError:pass
        else:raise AssertionError('Unfrozen threshold was accepted')
        meta_path.write_text(original)
        with contextlib.redirect_stdout(io.StringIO()):c.run(cli)
        result=json.loads((output/'CONTEXT_CONFIRMATION.json').read_text())
        candidate=result['variants']['paired_max']
        assert candidate['counts']['protected_denominator']==2
        assert candidate['counts']['protected_candidate_fused_losses']==0
        assert candidate['counts']['reference_none_to_candidate_any']==1
        assert candidate['protection_gate']['passes']
        assert not result['prespecified_main_candidate_conditions']['both_prespecified_conditions_met']
        assert result['prespecified_main_candidate_conditions']['fused_condition_status']=='failed_fused_prerequisite'
        assert (output/'CONTEXT_OBJECTS.csv').exists()
        try:c.run(cli)
        except FileExistsError:pass
        else:raise AssertionError('Context result overwrite was accepted')
    report={'status':'passed','execution':'CPU synthetic fixtures only; no real calibration or confirmation predictions opened',
            'checks':['exactly eleven starts and frozen threshold enforcement','absolute-frame metadata and relative offsets',
                      'reference all/some/none strata','window gain/loss and collapse-through-consistency counts',
                      'stable original fused misses excluded from protection denominator',
                      'integer 1% boundary and empty-set failure','key-based reordering alignment',
                      'fused failure cannot pass despite successful protection','result overwrite rejected']}
    if args.output:f.write_json(args.output,report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
