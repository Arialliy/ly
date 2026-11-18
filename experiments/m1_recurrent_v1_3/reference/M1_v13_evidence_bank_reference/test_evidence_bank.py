import pytest
import torch
from torch import nn
from evidence_bank import start_window, read_then_write, scan_features


def align_ref(a, x):
    return a


def fuse_mean(a, x):
    return (a + x) / 2


def fuse_ref(a, x):
    return a


def scalar_video(values, requires_grad=False):
    return torch.tensor(values, dtype=torch.float64).reshape(1, 1, -1, 1, 1).requires_grad_(requires_grad)


def run(v, **kwargs):
    return scan_features(v, align=align_ref, fuse=fuse_mean, **kwargs)


@pytest.mark.parametrize('capacity', [1, 2, 4])
def test_fifo_contains_source_observations_not_readouts(capacity):
    x = scalar_video([0, 10, 20, 30, 40, 50])
    _, state = run(x, capacity=capacity)
    assert [e.step_index for e in state.entries] == list(range(6-capacity, 6))
    assert [e.feature.item() for e in state.entries] == [10*i for i in range(6-capacity, 6)]


def test_manual_history_read():
    y, _ = run(scalar_video([0, 10, 20, 30, 40]), capacity=2)
    torch.testing.assert_close(y.flatten(), torch.tensor([0, 5, 12.5, 22.5, 32.5], dtype=torch.float64))


def test_current_control_keeps_all_pair_operations():
    class Counter:
        def __init__(self): self.n = 0
        def __call__(self, a, x): self.n += 1; return a
    a, f = Counter(), Counter()
    x = scalar_video([1, 2, 3, 4, 5, 6])
    y, _ = scan_features(x, align=a, fuse=f, capacity=3, mode='current_only')
    assert a.n == f.n == 12
    torch.testing.assert_close(y, x)


def test_first_frame_does_not_call_pair_operators():
    def bad(a, x): raise AssertionError('unexpected pair call')
    x = scalar_video([2])
    y, state = scan_features(x, align=bad, fuse=bad)
    torch.testing.assert_close(y, x)
    assert len(state.entries) == 1


def test_weak_and_negative_values_are_written_without_threshold():
    x = scalar_video([-1e-8, 0, 1e-12])
    _, state = run(x)
    assert len(state.entries) == 3
    assert state.entries[0].feature.item() == -1e-8
    assert state.entries[2].feature.item() == 1e-12


def test_same_current_different_history_changes_history_read():
    a, _ = run(scalar_video([0, 1]))
    b, _ = run(scalar_video([10, 1]))
    assert a[:, :, -1].item() != b[:, :, -1].item()


def test_same_current_different_history_does_not_change_control():
    a, _ = run(scalar_video([0, 1]), mode='current_only')
    b, _ = run(scalar_video([10, 1]), mode='current_only')
    torch.testing.assert_close(a[:, :, -1], b[:, :, -1])


def test_no_current_is_written_before_reading():
    y, _ = scan_features(scalar_video([2, 100]), align=align_ref, fuse=fuse_ref)
    assert y[:, :, -1].item() == 2


def test_evicted_observation_cannot_affect_later_read_for_fixed_features():
    a, _ = run(scalar_video([0, 1, 2, 3, 4]), capacity=2)
    b, _ = run(scalar_video([100, 1, 2, 3, 4]), capacity=2)
    torch.testing.assert_close(a[:, :, -1], b[:, :, -1])


def test_chunk_forward_equivalence():
    x = torch.randn(2, 3, 9, 5, 4, dtype=torch.float64)
    whole, _ = run(x, capacity=4, window_id='v0')
    state = None
    parts = []
    for lo, hi in [(0, 2), (2, 6), (6, 9)]:
        y, state = run(x[:, :, lo:hi], capacity=4, window_id='v0', state=state)
        parts.append(y)
    torch.testing.assert_close(whole, torch.cat(parts, dim=2))


def test_chunk_gradient_equivalence():
    x = torch.randn(1, 2, 7, 3, 3, dtype=torch.float64, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    y, _ = run(x, capacity=3)
    y.square().sum().backward()
    p, state = run(x2[:, :, :2], capacity=3)
    q, state = run(x2[:, :, 2:], capacity=3, state=state)
    torch.cat([p, q], 2).square().sum().backward()
    torch.testing.assert_close(x.grad, x2.grad)


def test_history_gradient_reaches_stored_observation():
    x = scalar_video([1, 2, 3], requires_grad=True)
    y, _ = run(x, capacity=2)
    y[:, :, -1].sum().backward()
    assert x.grad[:, :, 0].abs().sum() > 0


def test_control_last_output_has_no_new_bank_gradient_to_past_features():
    x = scalar_video([1, 2, 3], requires_grad=True)
    y, _ = run(x, capacity=2, mode='current_only')
    y[:, :, -1].sum().backward()
    assert x.grad[:, :, :-1].abs().sum() == 0
    assert x.grad[:, :, -1].abs().sum() > 0


def test_cache_is_not_changed_by_inplace_pair_operator():
    def bad_align(a, x): a.add_(10); x.mul_(0); return a
    x = scalar_video([1, 2, 3])
    original = x.clone()
    _, state = scan_features(x, align=bad_align, fuse=fuse_mean)
    torch.testing.assert_close(x, original)
    assert [e.feature.item() for e in state.entries] == [1, 2, 3]


def test_pair_module_parameters_receive_gradients():
    conv = nn.Conv2d(4, 2, 1).double()
    def learned(a, x): return conv(torch.cat([a, x], 1))
    x = torch.randn(1, 2, 3, 4, 4, dtype=torch.float64, requires_grad=True)
    y, _ = scan_features(x, align=align_ref, fuse=learned)
    y.square().sum().backward()
    assert conv.weight.grad is not None and conv.weight.grad.abs().sum() > 0


def test_known_spatial_shift_uses_source_not_mutated_readout():
    first = torch.zeros(1, 1, 5, 5)
    first[:, :, 2, 1] = 1
    current = torch.zeros_like(first)
    current[:, :, 2, 2] = 0.1
    st = start_window('shift')
    _, st = read_then_write(first, st, window_id='shift', step_index=0, align=align_ref, fuse=fuse_ref)
    y, _ = read_then_write(current, st, window_id='shift', step_index=1,
        align=lambda a, x: torch.roll(a, shifts=1, dims=-1), fuse=fuse_ref)
    assert y[0, 0, 2, 2] == 1
    assert st.entries[0].feature[0, 0, 2, 1] == 1


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtype_shape_and_finite(dtype):
    x = torch.randn(2, 2, 5, 3, 4).to(dtype)
    y, _ = run(x)
    assert y.shape == x.shape and y.dtype == dtype
    assert torch.isfinite(y).all()


def test_cross_window_state_rejected():
    _, state = run(scalar_video([1]), window_id='a')
    with pytest.raises(ValueError, match='different'):
        run(scalar_video([2]), window_id='b', state=state)


def test_nonsequential_step_rejected():
    st = start_window('a')
    with pytest.raises(ValueError, match='nonsequential'):
        read_then_write(torch.ones(1,1,1,1), st, window_id='a', step_index=1, align=align_ref, fuse=fuse_mean)


def test_capacity_change_rejected():
    _, state = run(scalar_video([1]), capacity=2)
    with pytest.raises(ValueError, match='capacity'):
        run(scalar_video([2]), capacity=3, state=state)


def test_invalid_capacity_rejected():
    with pytest.raises(ValueError): start_window('a', 0)


def test_empty_window_rejected():
    with pytest.raises(ValueError): run(torch.empty(1,1,0,2,2))


def test_shape_changing_operator_rejected():
    with pytest.raises(ValueError, match='shape'):
        scan_features(torch.ones(1,1,2,4,4), align=lambda a,x:a[:,:,:2,:], fuse=fuse_mean)


def test_new_window_resets_history():
    x = scalar_video([1, 2, 3])
    a, _ = run(x, window_id='first')
    b, _ = run(x, window_id='second')
    torch.testing.assert_close(a, b)
