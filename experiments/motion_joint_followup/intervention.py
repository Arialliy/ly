"""Frozen D joint-only zero/permutation tests, using the two original development sources."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.stats import spearmanr
import torch

HERE = Path(__file__).resolve().parent
FACTORIAL = HERE.parent / 'motion_joint_factorial'
sys.path.insert(0,str(FACTORIAL))
from evaluate import evaluate_method, peaks
from experiment import model_trainables, motion_window
from model import FactorialDetector, load_official_detector
from motion_evidence import extract_evidence
from protocol import CENTERS, DEV, SEED, frame_path, frame_record, metadata, setup
from readout import contacts
from extract import labels as split_boxes

MODES=('normal','zero','permute')


def span_assignment():
    """A repeatable, nonidentity cyclic permutation at every neighboring velocity."""
    rng=np.random.default_rng(SEED)
    shift=rng.choice((1,2),size=(5,5))
    shift[2,2]=0
    return torch.tensor((np.arange(3)[:,None,None]+shift)%3,dtype=torch.long,device='cuda')


def intervene(ev,assignment,mode):
    # Shape [N,2 exchanges,2 hypotheses,3 spans,5,5 velocities,3,3 patch].
    p=ev['patches']
    rows=p[:,:,:,1:]
    if mode=='zero':
        changed=torch.zeros_like(rows)
    elif mode=='permute':
        index=assignment.view(1,1,1,3,5,5,1,1).expand_as(rows)
        changed=torch.gather(rows,3,index)
        # Do not move a patch between cells with different valid frame counts at the borders.
        valid=ev['valid'][:,:,:,1:]
        eligible=(valid==valid[:,:,:,:1]).all(dim=3,keepdim=True)
        changed=torch.where(eligible[...,None,None],changed,rows)
    else:
        raise ValueError(mode)
    changed[:,:,:,:,2,2]=rows[:,:,:,:,2,2]
    return {**ev,'patches':torch.cat((p[:,:,:,:1],changed),dim=3)}


def eligible_ids(record):
    meta=metadata(record['sequence'])
    image_id,=[im['id'] for im in meta['images'] if im['frame_id']==record['index']]
    anns=[a for a in meta['annotations'] if a['image_id']==image_id and split_boxes([a])[0]]
    assert [a['bbox'] for a in anns]==record['targets']
    return [(record['sequence'],record['index'],int(a['id'])) for a in anns]


def unique_bounds(observations,frames,threshold):
    possible,forced=set(),set()
    total=0
    for (xy,scores),rec in zip(observations,frames):
        ids=eligible_ids(rec)
        active=contacts(xy[scores>=threshold],rec['targets']) if threshold is not None else np.zeros((0,len(ids)),bool)
        cols=np.flatnonzero(active.any(axis=0))
        graph=active[:,cols]
        graph=graph[graph.any(axis=1)]
        def maximum(a):
            if min(a.shape)==0:
                return 0
            return int(np.count_nonzero(maximum_bipartite_matching(csr_matrix(a),perm_type='column')>=0))
        n=maximum(graph)
        total+=n
        possible.update(ids[k] for k in cols)
        if len(cols)==n:
            forced.update(ids[k] for k in cols)
        else:
            forced.update(ids[k] for j,k in enumerate(cols) if maximum(np.delete(graph,j,axis=1))<n)
    return total,forced,possible


def compare_ids(reference,changed):
    n,f,p=reference
    m,g,q=changed
    return dict(reference_tp=n,modified_tp=m,reference_ambiguous_ids=sorted(p-f),
                modified_ambiguous_ids=sorted(q-g),both_guaranteed=sorted(f&g),
                both_possible_upper_bound=sorted(p&q),only_reference_guaranteed=sorted(f-q),
                only_changed_guaranteed=sorted(g-p),both_identity_sets_unique=f==p and g==q)


@torch.inference_mode()
def main():
    setup()
    release=load_official_detector('cuda')
    models={}
    for arm in 'CD':
        model=FactorialDetector(release,arm).eval()
        weights=torch.load(FACTORIAL/'runs/models'/f'{arm}.pt',weights_only=True,map_location='cpu')
        params=model_trainables(model)
        if set(weights)!=set(params):
            raise RuntimeError('Adapter keys differ')
        for k,v in params.items():
            v.copy_(weights[k])
        models[arm]=model
    assignment=span_assignment()
    observations={mode:[] for mode in MODES}
    records=[]
    summaries={mode:[] for mode in MODES[1:]}
    for seq in DEV:
        for idx in CENTERS:
            with np.load(frame_path('backbone',seq,idx)) as a:
                features=torch.from_numpy(a['features']).cuda().reshape(-1,32)
            with np.load(frame_path('predictions',seq,idx)) as a:
                references={k:torch.from_numpy(a[k]).cuda().flatten() for k in 'CD'}
            window=motion_window(seq,idx)
            h,w=window.shape[-2:]
            outs={mode:torch.empty_like(references['D']) for mode in MODES[1:]}
            parity=torch.zeros((),device='cuda')
            control=torch.zeros((),device='cuda')
            for start in range(0,h*w,1024):
                stop=min(start+1024,h*w)
                flat=torch.arange(start,stop,device='cuda')
                xy=torch.stack((flat%w,flat//w),dim=1)
                ev=extract_evidence(window,xy,'split')
                feat=features[start:stop]
                output={k:m.forward_points(feat,ev) for k,m in models.items()}
                for k in 'CD':
                    parity=torch.maximum(parity,(output[k]-references[k][start:stop]).abs().max())
                for mode in outs:
                    x=intervene(ev,assignment,mode)
                    outs[mode][start:stop]=models['D'].forward_points(feat,x)
                    control=torch.maximum(control,(models['C'].forward_points(feat,x)-output['C']).abs().max())
            if parity.item()>1e-5 or control.item()!=0:
                raise RuntimeError(f'Native parity {parity} or changed C {control}')
            original=references['D'].cpu().numpy().reshape(h,w)
            observations['normal'].append(peaks(original))
            for mode,out in outs.items():
                y=out.cpu().numpy().reshape(h,w)
                differences=np.abs(y-original)
                ref_xy=observations['normal'][-1][0]
                yy,xx=ref_xy[:,1],ref_xy[:,0]
                ranks=spearmanr(original[yy,xx],y[yy,xx]).statistic
                summaries[mode].append(dict(sequence=seq,frame=idx,pixels=h*w,
                    median_abs_logit_change=float(np.median(differences)),
                    frac_abs_logit_change_gt_1e_4=float((differences>1e-4).mean()),
                    spearman_on_all_fixed_reference_peaks=float(ranks),
                    native_map_max_error=parity.item(),negative_control_c_max_error=control.item()))
                observations[mode].append(peaks(y))
            records.append(frame_record(seq,idx))
            print(seq,idx,'frozen all-pixel interventions complete',flush=True)
    groups={'all_dev':list(range(len(records)))}
    groups.update({seq:[i for i,f in enumerate(records) if f['sequence']==seq] for seq in DEV})
    metrics={m:evaluate_method(records,v,groups) for m,v in observations.items()}
    # Both independently selected <=1-FP budgets and strictly unchanged D threshold are informative.
    normal=metrics['normal']['all_dev']['froc']['1']
    reference=unique_bounds(observations['normal'],records,normal['logit_threshold'])
    assert reference[0]==normal['tp']
    comparisons={}
    for mode in MODES[1:]:
        own=metrics[mode]['all_dev']['froc']['1']
        changed=unique_bounds(observations[mode],records,own['logit_threshold'])
        assert changed[0]==own['tp']
        same_threshold=unique_bounds(observations[mode],records,normal['logit_threshold'])
        comparisons[mode]={'own_one_fp_budget':compare_ids(reference,changed),
                           'original_D_threshold':compare_ids(reference,same_threshold)}
    result=dict(sources=list(DEV),centers=list(CENTERS),scope='development diagnostic, no fitting',
                interventions='Zero joint-only patch values; independently cycle the three spans by one or two slots per off-center velocity where all fractions match. No changes to other inputs.',
                span_assignment_zero_based=assignment.cpu().tolist(),methods=metrics,
                continuous_and_ranking=summaries,identities_at_primary=comparisons)
    dest=HERE/'runs/intervention.json'
    temp=dest.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temp.replace(dest)


if __name__=='__main__':
    main()
