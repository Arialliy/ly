"""Local integration checks; distinct from the supplied 24 reference tests."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]
sys.path.insert(0, str(E))
from train_modules.temporal_risk import (pool_logit_regions, temporal_risk_loss,
    temporal_miss_risk, readout_gradient_diagnostics)
from train_modules.segments import build_segments, Observation, visible_segments


def example(target=-5., background=2., dtype=torch.float64, device="cpu"):
    z = torch.zeros((8, 4, 6, 6), dtype=dtype, device=device)
    z[0, :, :2, :2] = target
    z[0, :, 3:5, 3:5] = background
    z[4, :, 3:5, 3:5] = background-1
    z.requires_grad_(True)
    targets = [[0, t, 0, 0, 2, 2] for t in range(4)]
    bg = [[w, t, 3, 3, 5, 5] for t in range(4) for w in (0, 4)]
    neg = [[2*t, 2*t+1] for t in range(4)]
    return z, targets, bg, neg, [[0, 1, 2, 3]], [0]


def test_roi_mean_exact_values_and_additive_overlapping_pixel_gradients():
    z = torch.arange(2*3*6*6, dtype=torch.float64).reshape(2, 3, 6, 6).requires_grad_()
    boxes = [[0, 0, 0, 0, 3, 3], [0, 0, 1, 1, 4, 4], [1, 2, 2, 0, 4, 5]]
    got = pool_logit_regions(z, boxes)
    direct = torch.stack([z[b, t, y0:y1, x0:x1].mean() for b,t,x0,y0,x1,y1 in boxes])
    assert torch.equal(got, direct)
    g, = torch.autograd.grad(got.sum(), z, retain_graph=True)
    ref, = torch.autograd.grad(direct.sum(), z)
    assert torch.equal(g, ref)
    assert g[0, 0, 1, 1] == 2/9


def test_actual_roi_means_match_reference_core_and_fixed_eight_window_reduction():
    args = example()
    out = temporal_risk_loss(*args)
    qplus = torch.full((1, 4), -5., dtype=torch.float64)
    qminus = torch.full((1, 4), 2., dtype=torch.float64)
    expected = temporal_miss_risk(qplus,qminus,torch.ones((1,4),dtype=torch.bool))
    assert torch.allclose(out['loss_gap'], expected.per_segment[0]/8)
    expected_frame = torch.nn.functional.softplus(-qplus)+torch.nn.functional.softplus(qminus)
    assert torch.allclose(out['loss_frame'], expected_frame.mean()/8)
    assert torch.equal(out['per_window_gap'][1:], torch.zeros(7,dtype=torch.float64))
    assert out['window_segment_counts'].tolist() == [1,0,0,0,0,0,0,0]
    assert out['diagnostics']['observations_exposed'] == 4


@pytest.mark.parametrize('target,bg', [(-1000.,1000.),(-1000.,-1000.),(1000.,-1000.),(1000.,1000.)])
@pytest.mark.parametrize('loss_key', ['loss_gap','loss_frame'])
def test_extreme_actual_logits_are_finite_and_wholly_weak_targets_have_gradients(target,bg,loss_key):
    args = example(target,bg,dtype=torch.float32)
    out = temporal_risk_loss(*args)
    assert torch.isfinite(out[loss_key])
    gradients = readout_gradient_diagnostics(out, loss_key)
    out[loss_key].backward()
    assert torch.isfinite(args[0].grad).all()
    assert torch.isfinite(gradients['target_readout_gradient']).all()
    assert torch.isfinite(gradients['background_readout_gradient']).all()
    if target < 0:
        assert (gradients['target_readout_gradient'] < 0).all()
        assert gradients['target_readout_gradient'].abs().sum() > .1


def test_real_logit_gradient_directions_and_hard_background_max_branch():
    args = example()
    out = temporal_risk_loss(*args)
    diagnostics = readout_gradient_diagnostics(out)
    assert (diagnostics['target_readout_gradient'] < 0).all()
    assert (diagnostics['background_readout_gradient'] > 0).all()
    out['loss_gap'].backward()
    assert (args[0].grad[0,:,:2,:2] < 0).all()
    assert (args[0].grad[0,:,3:5,3:5] > 0).all()
    assert args[0].grad[4].abs().sum() == 0  # Lower global mean is not selected.


def test_global_only_keeps_target_but_missing_both_cannot_bridge_segment():
    z, targets, bg, neg, seg, windows = example()
    neg[1][0] = -1
    out = temporal_risk_loss(z,targets,bg,neg,seg,windows)
    assert out['valid_observations'].all()
    assert out['q_negative'][1] == 1
    assert out['diagnostics']['local_missing_exposures'] == 1
    out['loss_gap'].backward()
    assert z.grad[4,1,3:5,3:5].sum() > 0
    neg[1][1] = -1
    with pytest.raises(ValueError, match='no valid background'):
        temporal_risk_loss(z,targets,bg,neg,seg,windows)


def test_empty_data_and_padded_rows_do_not_change_detection_gradient_or_denominator():
    z = torch.randn(8,4,6,6,requires_grad=True)
    out = temporal_risk_loss(z, [], [], np.empty((0,2),int),np.empty((0,8),int),[])
    assert out['loss_gap'] == out['loss_frame'] == 0
    det = z.square().mean()
    got, = torch.autograd.grad(det+out['loss_gap']+out['loss_frame'], z, retain_graph=True)
    want, = torch.autograd.grad(det, z)
    assert torch.equal(got,want)
    args = example()
    active = temporal_risk_loss(*args)
    mixed = temporal_risk_loss(*args[:4],[[0,1,2,3],[-1,-1,-1,-1]],[0,0])
    assert mixed['segment_active'].tolist() == [True,False]
    assert torch.equal(mixed['loss_gap'],active['loss_gap'])
    assert torch.equal(mixed['loss_frame'],active['loss_frame'])


def test_diagnostics_count_overlap_multiset_and_window_means_not_global_segment_mean():
    z, targets, bg, neg, _, _ = example()
    with torch.no_grad(): z[0,0,:2,:2] = 1
    segments = [[0,1,-1,-1],[0,2,3,-1]]
    out = temporal_risk_loss(z,targets,bg,neg,segments,[0,0])
    assert out['diagnostics']['observations_exposed'] == 5
    assert out['diagnostics']['observations_unique_exposed'] == 4
    assert out['diagnostics']['q_positive_mean'].item() == pytest.approx((1-5+1-5-5)/5)
    assert torch.allclose(out['loss_gap'],out['per_segment_gap'].mean()/8)
    assert torch.allclose(out['loss_frame'],out['per_segment_frame'].mean()/8)
    targets.append([1,0,0,0,2,2]);bg.append([1,0,3,3,5,5]);neg.append([8,-1])
    other=temporal_risk_loss(z,targets,bg,neg,segments+[[4,-1,-1,-1]],[0,0,1])
    expected=(other['per_segment_gap'][:2].mean()+other['per_segment_gap'][2])/8
    assert torch.allclose(other['loss_gap'],expected)
    assert not torch.allclose(other['loss_gap'],other['per_segment_gap'].mean()/8)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_half_input_pool_and_core_promote_before_mean_and_log_operations(dtype):
    args = example(-1000,1000,dtype=dtype)
    out = temporal_risk_loss(*args)
    assert out['loss_gap'].dtype == torch.float32
    out['loss_gap'].backward()
    assert torch.isfinite(args[0].grad).all() and args[0].grad.abs().sum()>0


def test_adapter_rejects_wrong_window_padding_size_and_coordinate_contract():
    z,targets,bg,neg,seg,windows=example()
    with pytest.raises(ValueError,match='fixed batch'):
        temporal_risk_loss(z,targets,bg,neg,seg,windows,num_windows=4)
    with pytest.raises(ValueError,match='suffix padding'):
        temporal_risk_loss(z,targets,bg,neg,[[0,-1,2,-1]],windows)
    with pytest.raises(ValueError,match='crosses input windows'):
        temporal_risk_loss(z,targets,bg,neg,seg,[1])
    bad = np.asarray(targets,dtype=float);bad[0,2]=.1
    with pytest.raises(ValueError,match='integer'):
        temporal_risk_loss(z,bad,bg,neg,seg,windows)
    bg[0][-1]=6
    with pytest.raises(ValueError,match='same slot and ROI size'):
        temporal_risk_loss(z,targets,bg,neg,seg,windows)


def descriptor(frame,window=0):
    return {'window':window,'sequence':'train','video_id':1,'track_id':7,
            'frame':frame,'source_frame':100+frame,'slot':frame}


def test_segment_adapter_matches_supplied_reference_without_tail_and_adds_one_tail():
    rows=[descriptor(i)for i in range(11)]
    got=build_segments(rows,length=8,stride=4,tail_alignment=False)
    expected=visible_segments([Observation('train', '7',100+i,i)for i in range(11)],length=8,step=1,stride=4)
    assert got['segment_indices'] == [list(x)for x in expected]
    tail=build_segments(rows,length=8,stride=4,tail_alignment=True)
    assert tail['segment_indices'] == [list(range(8)),list(range(3,11))]
    assert tail['coverage']['tail_extra_observations'] == 3


def test_segment_adapter_never_bridges_missing_background_or_source_frame_discontinuity():
    rows=[descriptor(i)for i in range(17)]
    valid=[i!=8 for i in range(17)]
    split=build_segments(rows,valid,length=8,stride=4)
    assert split['segment_indices'] == [list(range(8)),list(range(9,17))]
    for row in rows[8:]:row['source_frame']+=3
    broken=build_segments(rows,length=8,stride=4)
    assert all(not(min(seg)<8<=max(seg))for seg in broken['segment_indices'])


def test_actual_original_c_checkpoint_head_and_tpro_receive_gradients_without_state_change():
    torch.manual_seed(46);torch.set_num_threads(2)
    sys.path.insert(0,str(E/'code'))
    from runtime import load_model, freeze_for_training, INIT_CP, tensor_hash, frozen_state, bn_state
    model,_=load_model(INIT_CP,device='cpu')
    _,count=freeze_for_training(model)
    assert count==14593
    state=tensor_hash(model.state_dict());frozen=tensor_hash(frozen_state(model));bn=tensor_hash(bn_state(model))
    x=torch.randn(1,1,40,16,16)
    feature,z=model(x)
    original=z.detach().clone()
    targets=[[0,t,2,2,4,4]for t in range(8)]
    bg=[[0,t,10,10,12,12]for t in range(8)]
    out=temporal_risk_loss(z.repeat(8,1,1,1),targets,bg,[[i,-1]for i in range(8)],[list(range(8))],[0])
    out['loss_gap'].backward()
    for prefix in ('TPro.','conv_out1.','conv_out2.'):
        gradients=[p.grad for n,p in model.named_parameters()if n.startswith(prefix) and p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all()for g in gradients)
        assert sum(g.abs().sum()for g in gradients)>0
    assert all(p.grad is None for n,p in model.named_parameters()if n.startswith(('conv_in.','layer1.')))
    assert tensor_hash(model.state_dict())==state
    assert tensor_hash(frozen_state(model))==frozen and tensor_hash(bn_state(model))==bn
    with torch.no_grad(): assert torch.equal(model(x)[1],original)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU launch requires separate coordination')
def test_cuda_reference_extremes_match_cpu_and_overlapping_gather_is_deterministic():
    torch.use_deterministic_algorithms(True)
    results=[]
    for _ in range(2):
        args=example(-1000,1000,dtype=torch.float32,device='cuda')
        out=temporal_risk_loss(*args[:4],[[0,1,2,3],[1,2,3,-1]],[0,0])
        out['loss_gap'].backward()
        results.append((out['loss_gap'].detach().cpu(),args[0].grad.cpu()))
    assert torch.equal(results[0][0],results[1][0]) and torch.equal(results[0][1],results[1][1])
    args=example(-1000,1000,dtype=torch.float32)
    out=temporal_risk_loss(*args[:4],[[0,1,2,3],[1,2,3,-1]],[0,0])
    out['loss_gap'].backward()
    assert torch.allclose(results[0][0],out['loss_gap'],atol=2e-5,rtol=1e-6)
    assert torch.allclose(results[0][1],args[0].grad,atol=2e-6,rtol=1e-5)
