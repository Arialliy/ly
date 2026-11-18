"""M1 contracts, with an explicitly test-only recurrent cell on CPU.

These tests establish the adapter's state, control, freeze and learning semantics.
They do not certify official PDA/TSFM or its custom CUDA deformable convolution.
Real-operator integration is reported separately by the M1 runtime preflight.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters.window_recurrent_adapter import (  # noqa: E402
    FrozenDeepProWithAdapter,
    WindowRecurrentAdapter,
)


class TestOnlyRecurrentCell(nn.Module):
    """Small trainable recurrence for state tests; never an RFR substitute."""

    __test__ = False

    def __init__(self, channels=16):
        super().__init__()
        self.history_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.5))
        self.current_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.25))
        self.bias = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.calls = []

    def forward(self, previous, current):
        self.calls.append((previous.detach().clone(), current.detach().clone()))
        return self.history_scale * previous + self.current_scale * current + self.bias


def adapter(*, active=False, chunk=None):
    torch.manual_seed(46)
    module = WindowRecurrentAdapter(
        state_channels=16,
        recurrent_cell=TestOnlyRecurrentCell(),
        temporal_chunk_size=chunk,
    )
    if active:
        with torch.no_grad():
            # Expose only the recurrent state in output channels 0..15.
            module.in_projection.weight.zero_()
            if module.in_projection.bias is not None:
                module.in_projection.bias.zero_()
            module.out_projection.weight.zero_()
            module.out_projection.bias.zero_()
            for channel in range(16):
                module.in_projection.weight[channel, channel, 0, 0] = 1
                module.out_projection.weight[channel, channel + 16, 0, 0] = 1
    return module


def inputs(batch=2, time=5, height=4, width=6):
    generator = torch.Generator().manual_seed(4646)
    shape = (batch, 32, time, height, width)
    return torch.randn(shape, generator=generator), torch.randn(shape, generator=generator)


@pytest.mark.parametrize("mode", ["history", "current_only"])
@pytest.mark.parametrize("time", [1, 5, 40])
def test_zero_output_projection_is_exact_identity_without_skipping_cell(mode, time):
    module = adapter()
    pre, original = inputs(batch=1, time=time, height=2, width=3)
    result = module(pre, original, mode=mode)
    assert torch.equal(result, original)
    assert len(module.recurrent_cell.calls) == time - 1
    assert torch.count_nonzero(module.out_projection.weight) == 0
    assert torch.count_nonzero(module.out_projection.bias) == 0
    # Only the interface is zero initialized; the evidence projection is live.
    assert torch.count_nonzero(module.in_projection.weight) > 0


def test_first_frame_uses_projected_current_without_running_recurrence():
    module = adapter(active=True)
    pre, original = inputs(time=1)
    result = module(pre, original)
    torch.testing.assert_close(result[:, :16], original[:, :16] + pre[:, :16])
    assert torch.equal(result[:, 16:], original[:, 16:])
    assert not module.recurrent_cell.calls


def test_current_control_executes_same_cell_count_but_removes_history_dependence():
    module = adapter(active=True)
    pre, original = inputs()
    history = module(pre, original, mode="history")
    history_calls = module.recurrent_cell.calls.copy()
    module.recurrent_cell.calls.clear()
    current = module(pre, original, mode="current_only")
    current_calls = module.recurrent_cell.calls.copy()
    assert len(history_calls) == len(current_calls) == pre.shape[2] - 1
    for previous, now in current_calls:
        assert torch.equal(previous, now)
    assert any(not torch.equal(previous, now) for previous, now in history_calls)
    assert torch.equal(history[:, :, 0], current[:, :, 0])

    changed = pre.clone()
    changed[:, :, 0] += 2
    current_changed = module(changed, original, mode="current_only")
    history_changed = module(changed, original, mode="history")
    assert torch.equal(current[:, :, 1:], current_changed[:, :, 1:])
    assert not torch.equal(history[:, :, -1], history_changed[:, :, -1])


@pytest.mark.parametrize("mode", ["history", "current_only"])
def test_batch_members_and_successive_windows_are_isolated(mode):
    module = adapter(active=True)
    pre, original = inputs()
    expected = module(pre, original, mode=mode)
    separate = torch.cat([
        module(pre[index:index + 1], original[index:index + 1], mode=mode)
        for index in range(pre.shape[0])
    ])
    torch.testing.assert_close(expected, separate, atol=0, rtol=0)
    unrelated_pre, unrelated_original = inputs(batch=1, time=3, height=3, width=5)
    module(unrelated_pre + 11, unrelated_original, mode=mode)
    actual = module(pre, original, mode=mode)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("mode", ["history", "current_only"])
def test_training_and_inference_have_identical_state_rules(mode):
    module = adapter(active=True)
    pre, original = inputs()
    module.train()
    train_result = module(pre, original, mode=mode)
    training_calls = module.recurrent_cell.calls.copy()
    module.recurrent_cell.calls.clear()
    module.eval()
    with torch.no_grad():
        inference_result = module(pre, original, mode=mode)
    assert torch.equal(train_result, inference_result)
    assert len(training_calls) == len(module.recurrent_cell.calls)
    for train_call, eval_call in zip(training_calls, module.recurrent_cell.calls):
        assert all(torch.equal(a, b) for a, b in zip(train_call, eval_call))


@pytest.mark.parametrize("mode", ["history", "current_only"])
@pytest.mark.parametrize("chunk", [1, 2, 4, 40])
def test_temporal_chunks_preserve_state_outputs_and_backpropagation(mode, chunk):
    full = adapter(active=True)
    chunked = copy.deepcopy(full)
    pre, original = inputs(time=7)
    full_input = pre.clone().requires_grad_()
    chunked_input = pre.clone().requires_grad_()
    expected = full(full_input, original, mode=mode)
    actual = chunked(chunked_input, original, mode=mode, temporal_chunk_size=chunk)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    # The final frame must backpropagate across chunk boundaries in history mode.
    expected[:, :, -1].square().mean().backward()
    actual[:, :, -1].square().mean().backward()
    torch.testing.assert_close(chunked_input.grad, full_input.grad, atol=1e-6, rtol=1e-6)
    if mode == "history":
        assert torch.count_nonzero(chunked_input.grad[:, :, 0]) > 0
    else:
        assert torch.count_nonzero(chunked_input.grad[:, :, :-1]) == 0
    for (name, full_parameter), (other_name, chunk_parameter) in zip(
        full.named_parameters(), chunked.named_parameters()
    ):
        assert name == other_name
        assert full_parameter.grad is not None, name
        assert chunk_parameter.grad is not None, name
        torch.testing.assert_close(chunk_parameter.grad, full_parameter.grad, atol=1e-6, rtol=1e-5)


def test_constructor_chunking_does_not_carry_state_between_forward_calls():
    module = adapter(active=True, chunk=2)
    pre, original = inputs()
    result = module(pre, original)
    assert torch.equal(result, module(pre, original, temporal_chunk_size=1))
    module(pre * 2, original)
    assert torch.equal(result, module(pre, original))


@pytest.mark.parametrize("mode", ["history", "current_only"])
def test_activation_checkpointing_preserves_outputs_and_all_gradients(mode):
    ordinary = adapter(active=True, chunk=2).train()
    checkpointed = copy.deepcopy(ordinary)
    checkpointed.checkpoint_recurrence = True
    pre, original = inputs(time=5)
    ordinary_input = pre.clone().requires_grad_()
    checkpointed_input = pre.clone().requires_grad_()
    expected = ordinary(ordinary_input, original, mode=mode)
    actual = checkpointed(checkpointed_input, original, mode=mode)
    assert torch.equal(expected, actual)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(checkpointed_input.grad, ordinary_input.grad, atol=0, rtol=0)
    for (name, parameter), (other_name, other_parameter) in zip(
        ordinary.named_parameters(), checkpointed.named_parameters()
    ):
        assert name == other_name
        assert parameter.grad is not None and other_parameter.grad is not None, name
        torch.testing.assert_close(other_parameter.grad, parameter.grad, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["history", "current_only"])
def test_zero_initialization_learns_output_then_projection_and_cell(mode):
    module = adapter()
    pre = torch.linspace(0.1, 1.0, 32 * 4 * 3 * 4).reshape(1, 32, 4, 3, 4)
    original = torch.ones_like(pre)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.02)
    before = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}

    module(pre, original, mode=mode).square().mean().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        if name.startswith("out_projection."):
            assert parameter.grad.abs().sum() > 0, name
        else:
            assert torch.count_nonzero(parameter.grad) == 0, name
    optimizer.step()
    assert not torch.equal(module.out_projection.weight, before["out_projection.weight"])
    for name, parameter in module.named_parameters():
        if not name.startswith("out_projection."):
            assert torch.equal(parameter, before[name]), name

    optimizer.zero_grad(set_to_none=True)
    after_first = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
    module(pre, original, mode=mode).square().mean().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in module.named_parameters():
        assert not torch.equal(parameter, after_first[name]), name


class TestOnlyTPro(nn.Module):
    __test__ = False

    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm3d(32)

    def forward(self, features):
        return self.norm(features.permute(0, 3, 4, 1, 2))


class TestOnlyDeepPro(nn.Module):
    """Preserves the frozen model's interface, tensor order and BatchNorm sites."""

    __test__ = False

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv3d(1, 8, 1)
        self.layer1 = nn.Sequential(nn.Conv3d(8, 32, 1), nn.BatchNorm3d(32))
        self.TPro = TestOnlyTPro()
        self.conv_out1 = nn.Sequential(nn.Conv3d(32, 8, 1), nn.BatchNorm3d(8))
        self.conv_out2 = nn.Conv3d(8, 1, 1)

    def forward(self, frames):
        features = self.layer1(self.conv_in(frames))
        tpro = self.TPro(features.permute(0, 3, 4, 1, 2))
        output = self.conv_out1(tpro)
        return output, self.conv_out2(output).squeeze(1)


@pytest.mark.parametrize("mode", ["history", "current_only"])
def test_wrapper_preserves_outputs_and_frozen_weights_bn_while_adapter_learns(mode):
    torch.manual_seed(46)
    baseline = TestOnlyDeepPro().eval()
    original_state = {name: value.clone() for name, value in baseline.state_dict().items()}
    module = adapter()
    wrapped = FrozenDeepProWithAdapter(baseline, module, mode=mode)
    frames = torch.randn(2, 1, 4, 3, 5)
    with torch.no_grad():
        expected = baseline(frames)
    wrapped.train()
    assert wrapped.training and module.training
    assert all(not submodule.training for submodule in baseline.modules())
    assert all(not parameter.requires_grad for parameter in baseline.parameters())
    actual = wrapped(frames)
    assert len(actual) == len(expected) == 2
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    before_output = module.out_projection.weight.detach().clone()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    actual[1].square().mean().backward()
    assert module.out_projection.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in baseline.parameters())
    optimizer.step()
    assert not torch.equal(module.out_projection.weight, before_output)
    assert all(torch.equal(value, baseline.state_dict()[name]) for name, value in original_state.items())
    wrapped.eval()
    wrapped.train()
    assert all(not submodule.training for submodule in baseline.modules())


@pytest.mark.parametrize("mode", ["history", "current_only"])
def test_equal_initialization_pair_has_distinct_storage_and_strict_roundtrip(mode):
    module = adapter(active=True)
    clone = copy.deepcopy(module)
    for (name, parameter), (other_name, other_parameter) in zip(
        module.named_parameters(), clone.named_parameters()
    ):
        assert name == other_name and torch.equal(parameter, other_parameter)
        assert parameter.data_ptr() != other_parameter.data_ptr()
    clone.load_state_dict(module.state_dict(), strict=True)
    pre, original = inputs()
    assert torch.equal(module(pre, original, mode=mode), clone(pre, original, mode=mode))
    missing = dict(module.state_dict())
    missing.pop("out_projection.bias")
    with pytest.raises(RuntimeError):
        clone.load_state_dict(missing, strict=True)


def test_invalid_mode_cannot_silently_change_the_experimental_control():
    module = adapter()
    pre, original = inputs()
    with pytest.raises(ValueError):
        module(pre, original, mode="reset_and_skip")


@pytest.mark.parametrize("bad_shape", [(1, 16, 4, 3, 5), (1, 32, 3, 5), (1, 32, 0, 3, 5)])
def test_rejects_wrong_topology_and_empty_windows(bad_shape):
    module = adapter()
    with pytest.raises(ValueError):
        module(torch.ones(bad_shape), torch.ones(bad_shape))


def test_rejects_tpro_output_with_different_time_axis():
    module = adapter()
    pre, original = inputs()
    with pytest.raises(ValueError):
        module(pre, original[:, :, :-1])


@pytest.mark.parametrize("chunk", [0, -1])
def test_rejects_nonpositive_temporal_chunk(chunk):
    module = adapter()
    pre, original = inputs()
    with pytest.raises(ValueError):
        module(pre, original, temporal_chunk_size=chunk)
