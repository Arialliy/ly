import pytest
import torch
import torch.nn.functional as F
from readout_weighting import readout_terms, equal_mean_per_group


@pytest.mark.parametrize('mode', ['unit', 'confidence', 'matched_confidence'])
def test_shapes_and_positive_unchanged(mode):
    p = torch.tensor([[-10., 1.], [2., -2.]], requires_grad=True)
    n = torch.tensor([[-3., 0., 3.], [-1., 1., 2.]], requires_grad=True)
    t = readout_terms(p, n, mode=mode)
    assert t.positive.shape == p.shape and t.negative.shape == n.shape
    torch.testing.assert_close(t.positive, F.binary_cross_entropy_with_logits(p, torch.ones_like(p), reduction='none'))
    assert not t.negative_weight.requires_grad


def test_unit_is_previous_frame_loss():
    p = torch.tensor([[-10., 0., 10.]])
    n = torch.tensor([[-9., 0., 9.]])
    t = readout_terms(p, n)
    torch.testing.assert_close(t.positive + t.negative, F.softplus(-p) + F.softplus(n))


def test_confidence_negative_matches_published_kernel():
    p = torch.tensor([[-10., 0.]], dtype=torch.float64)
    n = torch.tensor([[-5., 0., 5.]], dtype=torch.float64)
    t = readout_terms(p, n, mode='confidence')
    expected = n.sigmoid().detach().pow(2.) * F.binary_cross_entropy_with_logits(n, torch.zeros_like(n), reduction='none')
    torch.testing.assert_close(t.negative, expected, rtol=1e-12, atol=1e-12)


def test_gradient_uses_detached_weight():
    p = torch.tensor([[-10.]], requires_grad=True, dtype=torch.float64)
    n = torch.tensor([[-3., 0., 3.]], requires_grad=True, dtype=torch.float64)
    t = readout_terms(p, n, mode='confidence')
    gp, gn = torch.autograd.grad(t.positive.sum() + t.negative.sum(), (p, n))
    torch.testing.assert_close(gp, p.sigmoid() - 1)
    torch.testing.assert_close(gn, n.sigmoid().pow(3))


def test_weak_positive_not_confidence_downweighted():
    p = torch.tensor([[-20., -10., -7.]], requires_grad=True)
    n = torch.tensor([[0.]], requires_grad=True)
    t = readout_terms(p, n, mode='confidence')
    grad, = torch.autograd.grad(t.positive.sum(), p)
    assert torch.all(grad < -0.999)


def test_detach_changes_gradient_vs_focal_like_expression():
    z = torch.tensor([[1.]], requires_grad=True, dtype=torch.float64)
    a = readout_terms(z.new_zeros((1,0)), z, mode='confidence').negative.sum()
    da, = torch.autograd.grad(a, z)
    b = (z.sigmoid().pow(2) * F.softplus(z)).sum()
    db, = torch.autograd.grad(b, z)
    assert db.item() > da.item()


def test_rescaled_confidence_matches_unit_logit_gradient_sum_not_distribution():
    p = torch.zeros(2,1,dtype=torch.float64)
    z1 = torch.tensor([[-4., 0., 2.], [-2., 1., 3.]], dtype=torch.float64, requires_grad=True)
    z2 = z1.detach().clone().requires_grad_()
    x = readout_terms(p, z1, mode='unit')
    y = readout_terms(p, z2, mode='matched_confidence')
    gx, = torch.autograd.grad(equal_mean_per_group(x.negative,x.negative_valid).sum(), z1)
    gy, = torch.autograd.grad(equal_mean_per_group(y.negative,y.negative_valid).sum(), z2)
    torch.testing.assert_close(gx.sum(1), gy.sum(1), rtol=1e-12, atol=1e-12)
    assert not torch.allclose(gx, gy)


def test_one_negative_matched_equals_unit():
    p = torch.zeros(3,1)
    n = torch.tensor([[-2.],[0.],[2.]])
    a = readout_terms(p,n,mode='unit')
    b = readout_terms(p,n,mode='matched_confidence')
    torch.testing.assert_close(a.negative, b.negative)


def test_gamma_zero_equals_unit():
    p = torch.zeros(2,1)
    n = torch.tensor([[-10.,0.,10.],[1.,2.,3.]])
    a = readout_terms(p,n,mode='unit')
    b = readout_terms(p,n,mode='confidence',gamma=0)
    torch.testing.assert_close(a.negative,b.negative)


@pytest.mark.parametrize('dtype', [torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_extreme_values_are_finite(dtype):
    p = torch.tensor([[-80.,80.]],dtype=dtype,requires_grad=True)
    n = torch.tensor([[-80.,80.]],dtype=dtype,requires_grad=True)
    x = readout_terms(p,n,mode='confidence')
    assert x.positive.dtype in (torch.float32,torch.float64)
    loss=x.positive.sum()+x.negative.sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all() and torch.isfinite(n.grad).all()


def test_masked_nan_padding_and_gradient():
    p = torch.tensor([[1.,float('nan')]],requires_grad=True)
    n = torch.tensor([[2.,float('inf')]],requires_grad=True)
    mask = torch.tensor([[True,False]])
    x = readout_terms(p,n,mode='confidence',positive_valid=mask,negative_valid=mask)
    (x.positive.sum()+x.negative.sum()).backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(n.grad).all()
    assert p.grad[0,1]==0 and n.grad[0,1]==0


def test_empty_background_group_contributes_zero():
    p = torch.zeros(1,1,requires_grad=True)
    n = torch.empty(1,0,requires_grad=True)
    t=readout_terms(p,n,mode='matched_confidence')
    loss=equal_mean_per_group(t.negative,t.negative_valid).sum()+t.positive.sum()
    loss.backward()
    assert torch.isfinite(loss) and n.grad.shape==(1,0)


def test_all_invalid_background_has_no_gradient():
    p=torch.zeros(1,1,requires_grad=True)
    n=torch.randn(1,3,requires_grad=True)
    t=readout_terms(p,n,mode='confidence',negative_valid=torch.zeros_like(n,dtype=torch.bool))
    t.negative.sum().backward()
    torch.testing.assert_close(n.grad,torch.zeros_like(n))


def test_gradient_through_real_linear_readout():
    head=torch.nn.Linear(8,1)
    f=torch.randn(2,4,8,requires_grad=True)
    z=head(f).squeeze(-1)
    t=readout_terms(z[:,:2],z[:,2:],mode='confidence')
    (t.positive.mean()+t.negative.mean()).backward()
    assert f.grad.abs().sum()>0 and head.weight.grad.abs().sum()>0


@pytest.mark.parametrize('kwargs', [{'mode':'bad'},{'gamma':-1},{'gamma':float('nan')}])
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        readout_terms(torch.zeros(1,1),torch.zeros(1,1),**kwargs)


def test_nonfinite_valid_logits_rejected():
    with pytest.raises(ValueError):
        readout_terms(torch.tensor([[float('nan')]]),torch.zeros(1,1))


def test_group_mismatch_rejected():
    with pytest.raises(ValueError):
        readout_terms(torch.zeros(1,1),torch.zeros(2,1))


def test_no_inference_parameters():
    p=torch.randn(2,3)
    n=torch.randn(2,4)
    pb=p.clone(); nb=n.clone()
    readout_terms(p,n,mode='confidence')
    assert torch.equal(p,pb) and torch.equal(n,nb)


@pytest.mark.parametrize('level', [-80., -40., 0., 40.])
def test_rescaled_equal_scores_equal_unit_weights(level):
    p=torch.zeros(1,1)
    n=torch.full((1,8),level)
    x=readout_terms(p,n,mode='matched_confidence')
    torch.testing.assert_close(x.negative_weight,torch.ones_like(n),rtol=1e-4,atol=1e-4)


def test_rescaled_larger_probability_has_larger_weight():
    p=torch.zeros(1,1)
    n=torch.tensor([[-5.,0.,5.]])
    x=readout_terms(p,n,mode='matched_confidence')
    assert x.negative_weight[0,0]<x.negative_weight[0,1]<x.negative_weight[0,2]
