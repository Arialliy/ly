#!/usr/bin/env python3
"""Synthetic-only test: never reads the real calibration/confirmation outputs."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import summarize_pilot as s


def fixture(root,variant,sequence):
    directory=root/variant/sequence;directory.mkdir(parents=True)
    objects=[{'frame':t,'source_frame':t+100,'image_id':1000+t,'track_id':7,'annotation_id':2000+t,
              'category_id':3,'bbox':[10.,10.,4.,4.],'strict_small':1} for t in [0,1,4]]
    offsets=np.array([sum(o['frame']<frame for o in objects) for frame in range(161)])
    n,nt=3,len(s.GRID)
    matched=np.zeros((nt,n),np.uint8);center=np.full((nt,n),np.nan,np.float32)
    px=center.copy();py=center.copy();areas=np.zeros((nt,n),np.int32)
    counts=np.zeros((nt,160,6),np.int64);rows=[]
    for ti,threshold in enumerate(s.GRID):
        # official budget=2 U at .5. Candidate .4/.5 tie on strict matches and U;
        # higher .5 must win. At >.5 strict matches fall; below .4 violates budget.
        if variant=='official':which=[1,0,1];u=2
        elif threshold<.4:which=[1,1,1];u=3
        elif threshold<=.5:which=[1,1,1];u=2
        else:which=[1,0,0];u=0
        matched[ti]=which
        counts[ti,:,0]=np.diff(offsets)
        for i,o in enumerate(objects):
            frame=o['frame'];counts[ti,frame,1]=which[i]
            if which[i]:center[ti,i]=0.;px[ti,i]=12.;py[ti,i]=12.;areas[ti,i]=1
        counts[ti,10,3]=u;counts[ti,:,2]=counts[ti,:,1]+counts[ti,:,3]
        counts[ti,:,4]=counts[ti,:,2];counts[ti,:,5]=counts[ti,:,3]
        totals=counts[ti].sum(0)
        rows.append({'threshold':threshold,'frames':160,'gt':3,'matched':sum(which),
                     'strict_small_gt':3,'strict_small_matched':sum(which),
                     **{k:int(v) for k,v in zip(s.FRAME_FIELDS[2:],totals[2:])}})
    s.write_csv(directory/'official_fused_curves.csv',rows)
    np.savez_compressed(directory/'official_fused_matches.npz',thresholds=np.array(s.GRID),matched=matched,
                        center_error=center,peak_x=px,peak_y=py,component_area=areas,
                        box_max=np.ones(n),frame_counts=counts,strict_small=np.ones(n,bool),offsets=offsets)
    s.write_json(directory/'official_fused_COMPLETE.json',{'status':'COMPLETE','name':'official_fused',
                 'start':0,'end':160,'thresholds':s.GRID,'objects':objects,'frame_count_fields':s.FRAME_FIELDS,
                 'annotations_sha256':'synthetic_same_labels','prediction_sha256':variant+'synthetic'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    curves=[{'threshold':.1,'frames':200,'unmatched_components':20,'strict_small_matched':12},
            {'threshold':.2,'frames':200,'unmatched_components':15,'strict_small_matched':12},
            {'threshold':.3,'frames':200,'unmatched_components':15,'strict_small_matched':12}]
    assert s.choose_threshold(curves,10,100)['threshold']==.3
    assert s.choose_threshold(curves,0,100) is None
    assert s.longest_miss([{'source_frame':i} for i in [0,1,4,5,6]],[False]*5)['longest_miss_run']==3
    assert s.longest_miss([{'source_frame':i} for i in [0,1,2]],[True,True,True])['longest_miss_run']==0
    with tempfile.TemporaryDirectory(prefix='stage1_summary_selfcheck_') as directory:
        root=Path(directory);evaluation=root/'evaluation';output=root/'summary'
        selection=root/'selection.json'
        s.write_json(selection,{'splits':{'calibration':['calA','calB'],'confirmation':['confirmSynthetic']}})
        data=json.loads(selection.read_text())
        for variant in s.VARIANTS:
            for sequence in ['calA','calB','confirmSynthetic']:fixture(evaluation,variant,sequence)
        cli=SimpleNamespace(evaluation_root=evaluation,selection=selection,output=output,calibration=None)
        s.calibrate(cli,data)
        frozen=json.loads((output/'CALIBRATION.json').read_text())
        assert frozen['budget']['unmatched_components']==4 and frozen['budget']['frames']==320
        assert all(x['threshold']==.5 for x in frozen['variants'].values())
        try:s.calibrate(cli,data)
        except FileExistsError:pass
        else:raise AssertionError('Calibration overwrite was accepted')
        official=s.FusedScore(evaluation,'official','confirmSynthetic').selected(.5)
        candidate=s.FusedScore(evaluation,'paired_max','confirmSynthetic').selected(.5)
        order=[2,0,1]
        reordered={k:([candidate[k][i] for i in order] if k in ['objects','keys'] else candidate[k][order])
                   for k in ['objects','keys','strict','matched','center_error','peak_x','peak_y','component_area']}
        comparison=s.compare(official,reordered,'strict_small')
        assert comparison['gained']==1 and comparison['lost']==0 and comparison['net']==1
        bad=copy.deepcopy(reordered);bad['objects'][0]['bbox']=[11.,10.,4.,4.]
        try:s.compare(official,bad,'all')
        except ValueError:pass
        else:raise AssertionError('Label mismatch was accepted')
        s.confirmation(cli,data)
        summary=json.loads((output/'FUSED_CONFIRMATION.json').read_text())
        assert summary['variants']['paired_max']['comparisons']['official']['strict_small']['gained']==1
        assert summary['fused_prerequisite']['status']=='failed_fused_prerequisite'  # equals mean, so no claimed advantage.
        assert not summary['fused_prerequisite']['stage1_positive_decision']
        assert summary['context_interface']['status']=='pending_window_scores'
        assert (output/'CONFIRMATION_OBJECTS.csv').exists()
        assert (output/'CONFIRMATION_TRACKS.csv').exists()
    result={'status':'passed','execution':'CPU synthetic score fixtures only; no real confirmation outcomes read',
            'checks':['exact rational budget and tie-breaking','no-feasible-point failure','immutable calibration',
                      'official threshold remains 0.5','object/frame alignment after reordering','label mismatch rejection',
                      'longest missed run breaks at annotation gaps','gain/loss counts','fused-only cannot pass stage1',
                      'four descriptive tables and aggregate results']}
    if args.output:s.write_json(args.output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
