import pytest
import torch
import torch.nn.functional as F
from temporal_risk import temporal_miss_risk
from segments import Observation, visible_segments


def test_agrees_with_direct_formula():
    a = torch.tensor([[-2., 0., 2.], [1., -1., .3]], dtype=torch.float64)
    b = torch.tensor([[-1., .5, -2.], [1., -2., 0.]], dtype=torch.float64)
    valid = torch.ones_like(a, dtype=torch.bool)
    expected = -torch.log1p(-(1 - a.sigmoid() * (-b).sigmoid()).prod(-1))
    assert torch.allclose(temporal_miss_risk(a,b,valid).loss, expected, atol=1e-12)


def test_single_frame_is_two_bce_terms():
    a = torch.tensor([[-2.], [3.]], dtype=torch.float64)
    b = torch.tensor([[1.], [-4.]], dtype=torch.float64)
    out = temporal_miss_risk(a,b,torch.ones_like(a,dtype=torch.bool)).loss
    assert torch.allclose(out, (F.softplus(-a)+F.softplus(b)).squeeze(1), atol=1e-12)


def test_gradient_directions():
    a = torch.tensor([[-8.,-7.,-9.]], requires_grad=True)
    b = torch.tensor([[1.,2.,.5]], requires_grad=True)
    loss = temporal_miss_risk(a,b,torch.ones_like(a,dtype=torch.bool),reduction='mean').loss
    loss.backward()
    assert torch.all(a.grad < 0)
    assert torch.all(b.grad > 0)


@pytest.mark.parametrize('target,bg', [(-1000.,-5.), (-1000.,1000.), (1000.,-1000.), (1000.,1000.)])
def test_extreme_logits_finite(target,bg):
    a = torch.full((2,4),target,requires_grad=True)
    b = torch.full((2,4),bg,requires_grad=True)
    loss = temporal_miss_risk(a,b,torch.ones_like(a,dtype=torch.bool),reduction='mean').loss
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    if target < 0:
        assert a.grad.abs().sum() > .1


def test_padding_does_not_change_result_or_gradients():
    a = torch.tensor([[-2., -3., 99., 99.]], requires_grad=True)
    b = torch.tensor([[1.,2.,-99.,-99.]], requires_grad=True)
    v = torch.tensor([[True,True,False,False]])
    out = temporal_miss_risk(a,b,v,reduction='sum').loss
    small = temporal_miss_risk(a[:,:2],b[:,:2],v[:,:2],reduction='sum').loss
    assert torch.allclose(out,small)
    out.backward()
    assert torch.count_nonzero(a.grad[:,2:]) == 0
    assert torch.count_nonzero(b.grad[:,2:]) == 0


def test_empty_segments_return_differentiable_zero():
    a = torch.ones(3,4,requires_grad=True)
    b = torch.ones(3,4,requires_grad=True)
    out = temporal_miss_risk(a,b,torch.zeros_like(a,dtype=torch.bool),reduction='mean')
    assert out.loss == 0 and not out.active.any()
    out.loss.backward()
    assert torch.count_nonzero(a.grad)==0 and torch.count_nonzero(b.grad)==0


def test_mixed_empty_mean_uses_active_only():
    a=torch.zeros(2,3)
    b=torch.zeros(2,3)
    v=torch.tensor([[True,True,True],[False,False,False]])
    r=temporal_miss_risk(a,b,v,reduction='mean')
    ref=temporal_miss_risk(a[:1],b[:1],v[:1],reduction='mean')
    assert torch.allclose(r.loss,ref.loss)


def test_half_computes_float32_and_has_gradients():
    a=torch.full((1,3),-9.,dtype=torch.float16,requires_grad=True)
    b=torch.full((1,3),-2.,dtype=torch.float16,requires_grad=True)
    out=temporal_miss_risk(a,b,torch.ones_like(a,dtype=torch.bool),reduction='mean')
    assert out.loss.dtype == torch.float32
    out.loss.backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum()>0


def test_gradcheck():
    a=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    b=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    v=torch.tensor([[True,True,True],[True,True,False]])
    assert torch.autograd.gradcheck(lambda x,y:temporal_miss_risk(x,y,v,reduction='sum').loss,(a,b))


def test_same_scalar_shift_not_a_free_solution():
    v=torch.ones(1,3,dtype=torch.bool)
    def f(c):
        return temporal_miss_risk(torch.full((1,3),c),torch.full((1,3),c),v,reduction='sum').loss
    assert f(10.) > f(0.)
    assert f(-10.) > f(0.)


def test_one_easy_frame_can_dominate_explicit_limitation():
    v=torch.ones(1,4,dtype=torch.bool)
    b=torch.full((1,4),-10.)
    a=torch.full((1,4),-10.)
    before=temporal_miss_risk(a,b,v,reduction='sum').loss
    a[0,2]=10.
    after=temporal_miss_risk(a,b,v,reduction='sum').loss
    assert after < before / 100


def test_permutation_invariance_within_segment_explicit_limitation():
    a=torch.tensor([[-2.,3.,-1.]])
    b=torch.tensor([[1.,-3.,0.]])
    v=torch.ones_like(a,dtype=torch.bool)
    p=[2,0,1]
    assert torch.allclose(temporal_miss_risk(a,b,v).loss,temporal_miss_risk(a[:,p],b[:,p],v).loss)


def test_model_path_has_real_head_gradients():
    head=torch.nn.Conv3d(8,1,1)
    features=torch.randn(1,8,4,2,2,requires_grad=True)
    z=head(features)[:,0]
    pos=z[:,:,0,0]
    neg=z[:,:,1,1]
    result=temporal_miss_risk(pos,neg,torch.ones_like(pos,dtype=torch.bool),reduction='mean')
    result.loss.backward()
    assert head.weight.grad.abs().sum()>0 and features.grad.abs().sum()>0


def test_invalid_shapes_rejected():
    with pytest.raises(ValueError):
        temporal_miss_risk(torch.zeros(2,3),torch.zeros(2,4),torch.ones(2,3,dtype=torch.bool))


def test_nonfinite_valid_rejected():
    with pytest.raises(ValueError):
        temporal_miss_risk(torch.tensor([[float('nan')]]),torch.zeros(1,1),torch.ones(1,1,dtype=torch.bool))


def test_never_groups_across_sources_with_same_track_id():
    obs=[Observation(v,'0',f,i) for i,(v,f) in enumerate([('a',0),('a',1),('b',0),('b',1)])]
    assert visible_segments(obs,length=2,step=1)==[(0,1),(2,3)]


def test_missing_frame_breaks_segment():
    obs=[Observation('a','0',f,i) for i,f in enumerate([0,1,3,4])]
    assert visible_segments(obs,length=2,step=1)==[(0,1),(2,3)]


def test_invisible_frame_breaks_segment():
    obs=[Observation('a','0',i,i,i!=2) for i in range(5)]
    assert visible_segments(obs,length=2,step=1)==[(0,1),(3,4)]


def test_actual_frame_cadence_not_row_distance():
    obs=[Observation('a','0',f,i) for i,f in enumerate([10,12,14,18])]
    assert visible_segments(obs,length=3,step=2)==[(0,1,2)]


def test_duplicate_key_rejected():
    with pytest.raises(ValueError):
        visible_segments([Observation('a','x',0,0),Observation('a','x',0,1)],length=2,step=1)


def test_invalid_horizon_rejected():
    with pytest.raises(ValueError):
        visible_segments([],length=1,step=1)
