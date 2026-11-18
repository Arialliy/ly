"""Cache training-only release-teacher masks/mining for common A/B/C exposure."""
import argparse
import json
import time
import numpy as np
import torch
from runtime import E,Q,CP,MODEL_SOURCE,setup,load_model,load_module,sha,save_json
from data import inputs,batch_data,window_annotations
from region_contrast_loss import mine_background_regions
from reliable_keep_loss import actual_reliable_boxes,build_keep_cache


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=3);p.add_argument('--limit',type=int,default=512)
    a=p.parse_args();setup();state=inputs();model,_=load_model()
    matcher=load_module('track_keep_actual_matcher',Q/'analysis/analyze.py')
    pair_path=E/'PAIR_PLAN.json';pair_plan=json.loads(pair_path.read_text())['entries']
    sources={str(path):sha(path) for path in [pair_path,CP,MODEL_SOURCE,E/'code/prepare.py',E/'code/data.py',
        E/'train_modules/region_contrast_loss.py',E/'train_modules/reliable_keep_loss.py',Q/'analysis/analyze.py']}
    results=[];start=time.perf_counter()
    for index in range(a.shard,min(512,a.limit),a.shards):
        path=E/'cache'/f'entry_{index:03d}.npz'
        receipt=path.with_suffix('.json')
        if path.exists():
            prior=json.loads(receipt.read_text())
            assert prior['sources']==sources and prior['sha256']==sha(path)
            results.append(prior);continue
        batch=batch_data(index,state);plan=pair_plan[index]
        with torch.inference_mode():
            # Identical microbatch shape for teacher and formal student.
            probabilities=torch.cat([model(torch.from_numpy(batch['x'][i:i+2]).cuda())[1].sigmoid().cpu()
                                     for i in range(0,8,2)]).numpy()
        regions=np.asarray(plan['target_regions'],dtype=np.int64).reshape(-1,6)
        pc=np.asarray(plan['positive_pairs_C'],dtype=np.int64).reshape(-1,2)
        pb=np.asarray(plan['positive_pairs_B'],dtype=np.int64).reshape(-1,2)
        assert np.array_equal(pc[:,0],pb[:,0])
        assert sorted(pc[:,1].tolist())==sorted(pb[:,1].tolist())
        anchors=regions[pc[:,0]]
        bg,neg=mine_background_regions(probabilities,batch['background'],anchors,
                                      4+anchors[:,0],local_radius=32)
        invalid_units={plan['pair_metadata'][i]['swap_unit_id'] for i in range(len(pc))
                       if not bool((neg[i]>=0).all())}
        for i,item in enumerate(plan['pair_metadata']):
            if item['swap_unit_id'] in invalid_units:neg[i,:]=-1
        # Keep the complete paired permutation. A pair lacking either negative
        # contributes a finite zero in both variants (common fixed denominator).
        frames=window_annotations(batch,state)
        coords=[];values=[];groups=[];keep_stats=[]
        for wi in range(8):
            masks=actual_reliable_boxes(probabilities[wi],frames[wi],matcher,
                                       threshold=.5,core_margin=9,exclusion_margin=2)
            keep=build_keep_cache(probabilities[wi],masks['reliable_target_mask'],
                batch['background'][wi],masks['forbidden_mask'],max_background=1024,
                background_max_probability=.1,seed=46+index*8+wi)
            c=keep['coordinates'];coords.append(np.column_stack([np.full(len(c),wi),c]))
            values.append(keep['teacher_probabilities']);groups.append(keep['groups'])
            keep_stats.append({'match':masks['stats'],'selection':keep['stats']})
        meta={'index':index,'input_sha256':batch['input_sha256'],'label_sha256':batch['label_sha256'],
            'auxiliary_input_sha256':batch['auxiliary_input_sha256'],
            'aux_entry':batch['aux_entry'],'pairs':len(pc),'pairs_with_two_negatives':int((neg>=0).all(1).sum()),
            'keep_stats':keep_stats,'sources':sources}
        np.savez_compressed(path,target_regions=regions,positive_pairs_B=pb,positive_pairs_C=pc,
            background_regions=bg,negative_indices=neg,
            keep_indices=np.concatenate(coords).astype(np.int64),
            keep_probabilities=np.concatenate(values).astype(np.float32),
            keep_groups=np.concatenate(groups).astype(np.int64),metadata_json=np.array(json.dumps(meta)))
        meta['sha256']=sha(path);save_json(receipt,meta);results.append(meta)
        if len(results)%8==0 or len(results)==1:
            print(json.dumps({'shard':a.shard,'entry':index,'done':len(results),
                              'pairs':len(pc),'elapsed':time.perf_counter()-start}),flush=True)
    assert all(sha(path)==value for path,value in sources.items())
    save_json(E/'cache'/f'SHARD_{a.shard}.json',{'status':'complete','sources':sources,
        'indices':[r['index'] for r in results],'elapsed_seconds':time.perf_counter()-start})

if __name__=='__main__':main()
