"""Deterministic CPU implementation tests, independent scalar edge oracle."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from module import RejectableTemporalAggregation, bidirectional_probabilities

CHECKS = []

def check(name, condition, **detail):
    CHECKS.append(dict(name=name, passed=bool(condition), **detail))
    if not condition:
        raise AssertionError(name + ': ' + str(detail))


def close(name, actual, expected, atol=1e-10, rtol=1e-10):
    a = actual.detach().cpu().numpy() if torch.is_tensor(actual) else np.asarray(actual)
    b = expected.detach().cpu().numpy() if torch.is_tensor(expected) else np.asarray(expected)
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    check(name, np.allclose(a, b, atol=atol, rtol=rtol), maximum_absolute_error=error)


def oracle_probabilities(scores, radius, null):
    """No tensor shifts: explicitly enumerate pixels and bidirectional edges."""
    n, K, h, w = scores.shape
    offsets = [(dy, dx) for dy in range(-radius, radius+1) for dx in range(-radius, radius+1)]
    fwd, rev = np.zeros_like(scores), np.zeros_like(scores)
    nf, nr = np.empty((n,h,w)), np.empty((n,h,w))
    reverse_raw = np.full_like(scores, -np.inf)
    for b in range(n):
        for y in range(h):
            for x in range(w):
                for k, (dy, dx) in enumerate(offsets):
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < h and 0 <= xx < w:
                        reverse_raw[b,K-1-k,yy,xx] = scores[b,k,y,x]
    for b in range(n):
        for y in range(h):
            for x in range(w):
                for raw, prob, out_null in ((scores,fwd,nf),(reverse_raw,rev,nr)):
                    logits = np.append(raw[b,:,y,x], null)
                    exps = np.exp(logits-np.max(logits))
                    probs = exps/exps.sum()
                    prob[b,:,y,x], out_null[b,y,x] = probs[:-1],probs[-1]
    reverse_edges = np.zeros_like(scores)
    for b in range(n):
        for y in range(h):
            for x in range(w):
                for k, (dy, dx) in enumerate(offsets):
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < h and 0 <= xx < w:
                        reverse_edges[b,k,y,x] = rev[b,K-1-k,yy,xx]
    return fwd,rev,nf,nr,reverse_edges


def scalar_aggregate(features, descriptors, radius, null, mode):
    b,c,t,h,w=features.shape
    offsets=[(dy,dx) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)]
    out=np.zeros_like(features)
    for f in range(t):
        refs=[s for s in (f-1,f+1) if 0<=s<t]
        for s_orig in refs:
            s=f if mode=='current' else s_orig
            scores=np.full((b,len(offsets),h,w),-np.inf)
            for bi in range(b):
                for y in range(h):
                    for x in range(w):
                        for k,(dy,dx) in enumerate(offsets):
                            yy,xx=y+dy,x+dx
                            if 0<=yy<h and 0<=xx<w:
                                scores[bi,k,y,x]=np.dot(descriptors[bi,:,f,y,x],descriptors[bi,:,s,yy,xx])/.1
            pf,_,_,_,reverse_edge=oracle_probabilities(scores,radius,null)
            weights=pf if mode=='single' else pf*reverse_edge
            if mode=='mass':
                weights=pf*weights.sum(1,keepdims=True)/np.maximum(pf.sum(1,keepdims=True),np.finfo(float).tiny)
            for bi in range(b):
                for y in range(h):
                    for x in range(w):
                        for k,(dy,dx) in enumerate(offsets):
                            yy,xx=y+dy,x+dx
                            if 0<=yy<h and 0<=xx<w:
                                out[bi,:,f,y,x]+=weights[bi,k,y,x]*(features[bi,:,s,yy,xx]-features[bi,:,f,y,x])/len(refs)
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(761)
    rng=np.random.default_rng(825)
    for radius,h,w in [(0,2,3),(1,3,4),(2,3,4),(4,2,1)]:
        K=(2*radius+1)**2
        scores=rng.normal(size=(2,K,h,w))
        offsets=[(dy,dx) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)]
        for y in range(h):
            for x in range(w):
                for k,(dy,dx) in enumerate(offsets):
                    if not (0<=y+dy<h and 0<=x+dx<w):scores[:,k,y,x]=-np.inf
        result=bidirectional_probabilities(torch.tensor(scores),radius,torch.tensor(.37,dtype=torch.float64))
        oracle=oracle_probabilities(scores,radius,.37)
        for name,a,e in zip(('forward','reverse','forward_null','reverse_null','reverse_on_forward'),result,oracle):
            close(f'r{radius}_{h}x{w}_{name}',a,e)
        close(f'r{radius}_forward_sum',result[0].sum(1)+result[2],1.)
        close(f'r{radius}_reverse_sum',result[1].sum(1)+result[3],1.)
        check(f'r{radius}_masked_edges',bool((result[0][~torch.tensor(np.isfinite(scores))]==0).all()))
    uniform=torch.zeros(1,9,5,5,dtype=torch.float64)
    for y in range(5):
        for x in range(5):
            for k,(dy,dx) in enumerate([(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1)]):
                if not (0<=y+dy<5 and 0<=x+dx<5):uniform[0,k,y,x]=-math.inf
    pf,pr,nf,nr,re=bidirectional_probabilities(uniform,1,torch.tensor(0.))
    close('uniform_interior_single_mass',pf[:,:,2,2].sum(),.9)
    close('uniform_interior_mutual_mass',(pf*re)[:,:,2,2].sum(),.09)
    close('uniform_corner_null',nf[0,0,0],.2)
    close('uniform_corner_mass',pf[0,:,0,0].sum(),.8)
    empty=torch.full((1,9,2,2),-math.inf,dtype=torch.float64)
    pf,pr,nf,nr,re=bidirectional_probabilities(empty,1,torch.tensor(0.))
    close('no_valid_candidate_forward_null',nf,1.)
    close('no_valid_candidate_reverse_null',nr,1.)
    close('no_valid_candidate_weights',pf*re,0.)
    scores=uniform.clone()
    scores[0,5,2,2]=7.25 # explicitly designated x=(2,2)->y=(2,3)
    pf,pr,nf,nr,re=bidirectional_probabilities(scores,1,torch.tensor(-2.))
    check('artificial_known_edge_forward',float(pf[0,5,2,2])>.99)
    check('artificial_known_edge_reverse',float(re[0,5,2,2])>.99)
    check('reverse_probability_not_forward_transpose',not torch.allclose(re,pf))

    base=RejectableTemporalAggregation(C=3,descriptor_dim=2,radius=2,temporal_chunk_size=2).double()
    features=torch.randn(2,3,5,4,5,dtype=torch.float64)
    for mode in ('single','mutual','mass','current','off'):
        base.set_mode(mode)
        close(mode+'_initial_zero_exact',base(features),features,atol=0,rtol=0)
    check('single_mutual_identical_state_keys',list(base.state_dict())==['null_logit','descriptor.weight','output_projection.weight'])
    check('fixed_parameter_count',sum(p.numel() for p in base.parameters())==1+3*2+3*3)
    with torch.no_grad():
        base.output_projection.weight.copy_(torch.eye(3,dtype=torch.float64).reshape(3,3,1,1,1))
    features=torch.randn(1,3,4,3,4,dtype=torch.float64)
    desc=torch.nn.functional.normalize(base.descriptor(features),p=2,dim=1,eps=1e-6).detach().numpy()
    for mode in ('single','mutual','mass','current'):
        base.set_mode(mode)
        expected=features.numpy()+scalar_aggregate(features.numpy(),desc,2,float(base.null_logit),mode)
        close(mode+'_full_scalar_oracle',base(features),expected,atol=1e-10)
        check(mode+'_finite_logging',all(math.isfinite(v) for v in base.last_stats.values() if isinstance(v,float)))
        check(mode+'_mass_range',0<=base.last_stats['nonnull_mass']<=1+1e-10)
        check(mode+'_null_range',0<=base.last_stats['null_probability']<=1+1e-10)
    for mode in ('single','mutual','mass','current'):
        base.set_mode(mode)
        one=torch.randn(1,3,1,2,3,dtype=torch.float64)
        close(mode+'_T1_identity',base(one),one,atol=0,rtol=0)
        zeros=torch.zeros(1,3,2,1,1,dtype=torch.float64)
        close(mode+'_tiny_zero_finite',base(zeros),zeros,atol=0,rtol=0)
    base.set_mode('mutual')
    with torch.no_grad():base.null_logit.fill_(1000.)
    close('large_null_identity',base(features),features,atol=0,rtol=0)
    with torch.no_grad():base.null_logit.fill_(0.)

    for mode in ('single','mutual'):
        torch.manual_seed(27)
        grad_model=RejectableTemporalAggregation(C=3,descriptor_dim=2,radius=1,mode=mode).double()
        x=torch.randn(1,3,4,3,4,dtype=torch.float64,requires_grad=True)
        target=torch.randn_like(x)
        loss=((grad_model(x)-target)**2).sum()
        loss.backward()
        check(mode+'_output_starts_gradient',float(grad_model.output_projection.weight.grad.norm())>0)
        close(mode+'_descriptor_initial_gradient_zero',grad_model.descriptor.weight.grad,0.,atol=0,rtol=0)
        close(mode+'_null_initial_gradient_zero',grad_model.null_logit.grad,0.,atol=0,rtol=0)
        with torch.no_grad():grad_model.output_projection.weight.add_(-.01*grad_model.output_projection.weight.grad)
        grad_model.zero_grad(set_to_none=True)
        ((grad_model(x)-target)**2).sum().backward()
        check(mode+'_descriptor_gradient_after_update',float(grad_model.descriptor.weight.grad.norm())>1e-10)
        check(mode+'_null_gradient_after_update',abs(float(grad_model.null_logit.grad))>1e-10)
        check(mode+'_all_gradients_finite',all(p.grad is not None and torch.isfinite(p.grad).all() for p in grad_model.parameters()))
        unchunked=copy.deepcopy(grad_model)
        unchunked.temporal_chunk_size=999
        unchunked.checkpoint_chunks=False
        grad_model.zero_grad(set_to_none=True)
        unchunked.zero_grad(set_to_none=True)
        xc=x.detach().clone().requires_grad_()
        xu=x.detach().clone().requires_grad_()
        yc,yu=grad_model(xc),unchunked(xu)
        close(mode+'_checkpoint_chunk_output',yc,yu,atol=1e-12)
        (yc*target).sum().backward()
        (yu*target).sum().backward()
        close(mode+'_checkpoint_chunk_input_gradient',xc.grad,xu.grad,atol=1e-10)
        for (n,p),(n2,p2) in zip(grad_model.named_parameters(),unchunked.named_parameters()):
            close(mode+'_checkpoint_chunk_parameter_gradient_'+n,p.grad,p2.grad,atol=1e-10)
    # Nonzero correction: exact feature-space halo 2r for mutual/mass, using
    # a spatial tile whose artificial boundaries stay outside the central ROI.
    tile_model=copy.deepcopy(base).eval()
    wide=torch.randn(1,3,3,17,19,dtype=torch.float64)
    for mode in ('single','mutual','mass','current'):
        tile_model.set_mode(mode)
        whole=tile_model(wide)
        tile=tile_model(wide[:,:,:,2:15,3:17])
        close(mode+'_nonzero_feature_tile_2r',whole[:,:,:,6:11,7:13],tile[:,:,:,4:9,4:10],atol=1e-10)
    sparse_input=torch.randn(2,3,4,5,6,dtype=torch.float64)
    sparse_queries=[[0,0,0,0],[0,1,2,3],[1,3,4,5],[1,2,0,5]]
    sparse=tile_model.sparse_query_weights(sparse_input,sparse_queries,chunk_size=2)
    desc=torch.nn.functional.normalize(tile_model.descriptor(sparse_input),dim=1,eps=1e-6)
    check('sparse_one_row_per_query',len(sparse)==len(sparse_queries))
    for i,(row,(b,t,y,x)) in enumerate(zip(sparse,sparse_queries)):
        check(f'sparse_{i}_legal_refs',[r['ref_slot'] for r in row['references']]==[s for s in (t-1,t+1) if 0<=s<4])
        for ref in row['references']:
            s=ref['ref_slot']
            scores=tile_model._scores(desc[b:b+1,:,t],desc[b:b+1,:,s])
            pf,pr,nf,nr,re=bidirectional_probabilities(scores,2,tile_model.null_logit)
            prefix=f'sparse_{i}_ref{s}'
            close(prefix+'_forward',ref['forward'],pf[0,:,y,x])
            close(prefix+'_reverse',ref['reverse_on_forward'],re[0,:,y,x])
            close(prefix+'_mutual',ref['mutual'],(pf*re)[0,:,y,x])
            close(prefix+'_null',ref['null_probability'],nf[0,y,x])
            check(prefix+'_all_detached_python',all(isinstance(v,float) for v in ref['mutual']))
    custom=tile_model.sparse_query_weights(sparse_input,[dict(slot=1,y=2,x=3,object_key='kept')])
    check('sparse_preserves_metadata',custom[0]['object_key']=='kept')
    check('sparse_empty_queries',tile_model.sparse_query_weights(sparse_input,[])==[])
    result={'status':'PASS','checks_passed':len(CHECKS),'checks':CHECKS,
            'module_sha256':hashlib.sha256(Path(__file__).with_name('module.py').read_bytes()).hexdigest(),
            'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'torch_version':torch.__version__,'device':'cpu','seed':761,
            'scope':'module-level exact scalar probability/aggregation, state, gradients, chunking, nonzero feature tiling; not detection efficacy'}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','checks_passed':len(CHECKS)}))

if __name__=='__main__':main()
