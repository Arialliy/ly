#!/usr/bin/env python3
"""CPU-only checks of loss/gradient, box masks and deterministic plan semantics."""
import argparse
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

import losses
import train_pilot as trainer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=losses.self_test()
    a={'id':1,'bbox':[100,100,10,10],'track_id':1,'ignore':0,'iscrowd':0}
    large={'id':2,'bbox':[120,120,20,20],'track_id':2,'ignore':0,'iscrowd':0}
    ignored={'id':3,'bbox':[150,150,5,5],'track_id':3,'ignore':1,'iscrowd':0}
    partial={'id':4,'bbox':[60,75,20,20],'track_id':4,'ignore':0,'iscrowd':0}
    annotations=[[] for _ in range(160)]
    annotations[20]=[a,large,ignored,partial]
    strict=[[] for _ in range(160)];strict[20]=[a]
    frames=np.broadcast_to(np.arange(160,dtype=np.uint8)[:,None,None],(160,512,640))
    sequences={name:SimpleNamespace(name=name,frames=frames,annotations=annotations,strict=strict,
                 track_samples={'1':[(20,a)]},fingerprint={'synthetic':True},
                 crop_limits=trainer.SequenceData.crop_limits) for name in ['seqA','seqB']}
    pairs=[{'sequence':name,'crop_yx':[64,64],'starts':[0,12],'anchor_frame':20} for name in sequences]
    entry={'pairs':pairs,'extra_specs':trainer.extra_specs_for_pairs(pairs,sequences)}
    arrays,pos,mask,specs=trainer.build_batch(entry,sequences)
    assert arrays.shape==(4,1,40,128,128)
    np.testing.assert_array_equal(arrays[0,0,20],arrays[1,0,8])
    assert [len(x) for x in pos]==[2,2,2,2]  # large included; ignored/partial excluded.
    assert all(s['slots']==[20,8] and s['rect']==[36,36,46,46] for s in specs)
    assert not mask[:,:,0:9,:].any() and not mask[:,:,-9:,:].any()
    assert not mask[:,:,:,0:9].any() and not mask[:,:,:,-9:].any()
    assert mask[0,0,10,10]  # Clean core remains a background example.
    assert not mask[0,20,34:48,34:48].any()  # small box + 2px.
    assert not mask[0,20,54:78,54:78].any()  # large box + 2px.
    assert not mask[0,20,84:93,84:93].any()  # ignored box remains protected.
    assert not mask[0,20,9:33,9:18].any()  # partial box is protected in crop.
    for strategy in losses.STRATEGIES:
        logits=torch.zeros((4,40,128,128),requires_grad=True)
        total,parts=losses.pilot_loss(logits,pos,torch.from_numpy(mask),specs,strategy,[[0,2],[1,3]])
        torch.testing.assert_close(total,torch.tensor(3*math.log(2)),rtol=1e-6,atol=1e-6)
        total.backward()
        assert torch.isfinite(logits.grad).all()
        assert torch.count_nonzero(logits.grad[:,:,0:9,:])==0
        # Positive selected regions have negative gradients; clean background has positive.
        assert logits.grad[0,20,36:46,36:46].sum()<0 and logits.grad[0,0,10,10]>0
    rng=np.random.default_rng(46)
    for n in [2,3,4,10,50]:
        pairs_random=trainer.random_matching(n,rng)
        assert sorted(i for pair in pairs_random for i in pair)==list(range(2*n))
        assert all(a//2!=b//2 for a,b in pairs_random)
    with tempfile.TemporaryDirectory(prefix='stage1_plan_selfcheck_') as directory:
        root=Path(directory);selection=root/'selection.json'
        trainer.save_json(selection,{'splits':{'train':['seqA','seqB']}})
        config=SimpleNamespace(seed=46,steps=8,selection=selection,plan=root/'plan1.json')
        trainer.prepare_plan(config,sequences)
        first=json.loads(config.plan.read_text())
        config.plan=root/'plan2.json';trainer.prepare_plan(config,sequences)
        assert first==json.loads(config.plan.read_text())
        assert first['counts']['interior']==8 and first['counts']['boundary']==8
        assert first['counts']['target_centered']==12 and first['counts']['random_crop']==4
        for entry in first['entries']:
            assert len(set(p['sequence'] for p in entry['pairs']))==2
            for p in entry['pairs']:
                s0,s1=p['starts'];t=p['anchor_frame']
                assert 0<=s0<s1<=120 and s1<=t<s0+40
                if p['stratum']=='interior':assert 8<=t-s0<=31 and 8<=t-s1<=31
                if p['target_centered_crop']:assert trainer.full_core(a,p['crop_yx'])
            trainer.build_batch(entry,sequences)
    result['checks']+=['same absolute frame across two windows',
                       'base positives include all legal sizes; ignore/partial excluded',
                       'background protects all boxes, including ignored/partial, plus margin',
                       '9px core exclusion', 'composite MIL loss and gradient direction',
                       'random matching without same-target repair',
                       'deterministic plan and exact sampling proportions']
    result['execution']='CPU synthetic semantics only; no model training or real-data optimization'
    if args.output:trainer.save_json(args.output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
