import copy
import pytest
import torch
from state_update import BoundedStateUpdate, convex_state_update

torch.set_num_threads(1)


def inputs(c=4, h=11, w=13, dtype=torch.float32):
    gen = torch.Generator().manual_seed(712)
    return tuple(torch.randn(2, c, h, w, generator=gen, dtype=dtype) for _ in range(3))


@pytest.mark.parametrize('scope', ['original','frame','pixel'])
@pytest.mark.parametrize('shape', [(3,5),(10,12),(1,1)])
def test_shape(scope, shape):
    data = inputs(h=shape[0], w=shape[1])
    model = BoundedStateUpdate(4,8,scope)
    y,g = model(*data)
    assert y.shape == data[0].shape
    assert g.shape == (2,1,*shape)
    assert torch.isfinite(y).all()


def test_original_exact():
    a,r,x = inputs()
    y,g = BoundedStateUpdate(4,8)(a,r,x,scope='original')
    assert y is r
    assert torch.count_nonzero(g) == 0


def test_initial_half_and_equal_controls():
    a,r,x = inputs()
    model = BoundedStateUpdate(4,8)
    yp,gp = model(a,r,x,scope='pixel')
    yf,gf = model(a,r,x,scope='frame')
    torch.testing.assert_close(gp, torch.full_like(gp,.5),rtol=0,atol=0)
    torch.testing.assert_close(yf,yp,rtol=0,atol=0)
    torch.testing.assert_close(yf, .5*a+.5*r,rtol=0,atol=0)
    assert not torch.allclose(yp,r)  # no false original-model equivalence claim


@pytest.mark.parametrize('gate_value', [0., 0.25, 1.])
def test_convex_bounds_and_endpoints(gate_value):
    a,r,_ = inputs()
    g = torch.full((2,1,11,13),gate_value)
    out=convex_state_update(a,r,g)
    assert torch.all(out >= torch.minimum(a,r)-1e-6)
    assert torch.all(out <= torch.maximum(a,r)+1e-6)
    if gate_value==0: torch.testing.assert_close(out,r,rtol=0,atol=0)
    if gate_value==1: torch.testing.assert_close(out,a,rtol=0,atol=0)


def test_identical_sources():
    a,_,_=inputs()
    g=torch.rand(2,1,11,13)
    y=convex_state_update(a,a,g)
    torch.testing.assert_close(y,a)


def test_frame_control_preserves_mean_of_same_gate():
    a,r,x=inputs()
    model=BoundedStateUpdate(4,8)
    torch.nn.init.normal_(model.gate_net[-1].weight,std=.2)
    _,gp=model(a,r,x,scope='pixel')
    _,gf=model(a,r,x,scope='frame')
    torch.testing.assert_close(gf, gp.mean((-2,-1),keepdim=True).expand_as(gp))
    assert (gp-gf).abs().max()>0


def test_gate_gradient_warm_start():
    torch.manual_seed(81)
    a,r,x=inputs()
    model=BoundedStateUpdate(4,8)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    y,_=model(a,r,x)
    y.square().mean().backward()
    assert model.gate_net[-1].weight.grad.abs().sum()>0
    assert model.gate_net[0].weight.grad.abs().sum()==0
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    y,_=model(a,r,x); y.square().mean().backward()
    assert model.gate_net[0].weight.grad.abs().sum()>0


def test_gradient_reaches_states_and_descriptor():
    a,r,x=(t.requires_grad_() for t in inputs())
    model=BoundedStateUpdate(4,8)
    torch.nn.init.normal_(model.gate_net[-1].weight,std=.1)
    y,_=model(a,r,x)
    y.square().mean().backward()
    for t in (a,r,x):
        assert t.grad is not None and torch.isfinite(t.grad).all()
        assert t.grad.abs().sum()>0


def test_pixel_tiling_with_halo_two():
    a,r,x=inputs(h=17,w=29)
    model=BoundedStateUpdate(4,8)
    torch.nn.init.normal_(model.gate_net[-1].weight,std=.15)
    full,g_full=model(a,r,x)
    parts=[]
    for left,right in [(0,7),(7,16),(16,29)]:
        lo=max(0,left-2); hi=min(29,right+2)
        out,_=model(a[...,lo:hi],r[...,lo:hi],x[...,lo:hi])
        parts.append(out[...,left-lo:right-lo])
    torch.testing.assert_close(torch.cat(parts,-1),full,atol=1e-6,rtol=1e-5)


def test_batch_separation():
    a,r,x=inputs()
    model=BoundedStateUpdate(4,8)
    torch.nn.init.normal_(model.gate_net[-1].weight,std=.1)
    full,_=model(a,r,x)
    chunks=[model(a[i:i+1],r[i:i+1],x[i:i+1])[0] for i in range(2)]
    torch.testing.assert_close(full,torch.cat(chunks))


def test_state_dict_same_control_capacity():
    m=BoundedStateUpdate(16,16,'pixel')
    control=copy.deepcopy(m); control.scope='frame'
    assert sum(p.numel() for p in m.parameters())==2017
    assert m.state_dict().keys()==control.state_dict().keys()
    for k in m.state_dict(): assert torch.equal(m.state_dict()[k],control.state_dict()[k])


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_dtype_and_finiteness(dtype):
    model=BoundedStateUpdate(4,8).to(dtype)
    y,g=model(*inputs(dtype=dtype))
    assert y.dtype==dtype and torch.isfinite(y).all()
    assert (g>=0).all() and (g<=1).all()


def test_bad_inputs():
    a,r,x=inputs()
    m=BoundedStateUpdate(4,8)
    with pytest.raises(ValueError): m(a,r[:,:,:-1],x)
    with pytest.raises(ValueError): m(a,r,x,scope='bad')
    with pytest.raises(ValueError): BoundedStateUpdate(0)
    with pytest.raises(ValueError): BoundedStateUpdate(4,scope='bad')
    with pytest.raises(TypeError): m(a.long(),r.long(),x.long())
    with pytest.raises(ValueError): m(a,r.double(),x)


def test_rolling_recurrence_chunks_preserve_explicit_state():
    # Synthetic caller integration only: not an actual PDA/TSFM test.
    torch.manual_seed(50)
    seq=torch.randn(1,7,4,9,9)
    model=BoundedStateUpdate(4,8)
    torch.nn.init.normal_(model.gate_net[-1].weight,std=.1)
    def run(frames,prev=None):
        outs=[]
        for xt in frames.unbind(1):
            if prev is None: prev=xt
            else:
                aligned=.95*prev
                candidate=torch.relu(.25*aligned+.75*xt)
                prev,_=model(aligned,candidate,xt)
            outs.append(prev)
        return torch.stack(outs,1),prev
    full,_=run(seq)
    first,state=run(seq[:,:3]); second,_=run(seq[:,3:],state)
    torch.testing.assert_close(full,torch.cat((first,second),1),rtol=0,atol=0)
    repeated,_=run(seq)
    torch.testing.assert_close(full,repeated,rtol=0,atol=0)
