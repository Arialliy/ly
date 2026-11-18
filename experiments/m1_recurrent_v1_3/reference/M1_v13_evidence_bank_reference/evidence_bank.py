"""A bounded, source-indexed feature bank for an M1 research control.

Independent reference code. This file contains no RFR, DCN, StreamVGGT,
DeepPro-Plus implementation, weights, labels, detector, or loss function.

Contract:
  * The caller passes projected features [B,C,T,H,W], not probabilities.
  * Every supplied frame is valid. Padding/invalid frames must be handled by
    the project caller, not silently interpreted as observations.
  * align(reference, current) and fuse(aligned, current) keep [B,C,H,W].
  * Read features are NEVER written back. Stored entries are current features
    before the new bank's alignment/fusion, in the coordinates of their source.
  * FIFO bounds source index age, not the temporal receptive field of a backbone.
  * Training keeps the autograd graph unless the caller explicitly changes the
    protocol. Eviction bounds the live cache, not all saved backward activations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor

Mode = Literal["history", "current_only"]
PairOperator = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class Entry:
    step_index: int
    feature: Tensor


@dataclass(frozen=True)
class BankState:
    window_id: str
    next_index: int
    capacity: int
    entries: tuple[Entry, ...]


def start_window(window_id: str, capacity: int = 4) -> BankState:
    """Start one window; caller must not reuse this state across windows/videos."""
    if not isinstance(window_id, str) or not window_id:
        raise ValueError("window_id must be a nonempty string")
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
        raise ValueError("capacity must be an integer >= 1")
    return BankState(window_id, 0, capacity, ())


def _check_feature(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must be a Tensor [B,C,H,W]")
    if not value.is_floating_point() or any(d < 1 for d in value.shape):
        raise ValueError(f"{name} needs a floating dtype and positive dimensions")


def _check_compatible(value: Tensor, reference: Tensor, name: str) -> None:
    _check_feature(value, name)
    if value.shape != reference.shape:
        raise ValueError(f"{name} changed feature shape")
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f"{name} changed feature device/dtype; handle AMP explicitly")


def read_then_write(
    current: Tensor,
    state: BankState,
    *,
    window_id: str,
    step_index: int,
    align: PairOperator,
    fuse: PairOperator,
    mode: Mode = "history",
) -> tuple[Tensor, BankState]:
    """Read at most K ORIGINAL past observations, then append this observation.

    current_only executes the same number of pair operators, substituting the
    current feature for each stored historical entry. It does not bypass align
    or fuse and does not describe the entire detector as a single-frame model.

    Clones protect cached inputs from accidental in-place operations and keep
    autograd attached. They incur memory/copy cost which must be profiled.
    """
    _check_feature(current, "current")
    if not isinstance(state, BankState):
        raise TypeError("state must be BankState")
    if mode not in ("history", "current_only"):
        raise ValueError("mode must be history or current_only")
    if state.window_id != window_id:
        raise ValueError("state belongs to a different video/window")
    if step_index != state.next_index:
        raise ValueError("nonsequential index; start a new window or continue exactly")
    if len(state.entries) > state.capacity:
        raise ValueError("corrupted state: too many entries")
    expected = list(range(max(0, step_index - state.capacity), step_index))
    if [e.step_index for e in state.entries] != expected:
        raise ValueError("corrupted state: source indices do not match the FIFO horizon")

    # This is a pre-write snapshot: neither readout nor pairwise output goes in.
    stored_current = current.clone()
    if not state.entries:
        readout = current.clone()  # same first-frame convention as original M1
    else:
        total: Tensor | None = None
        for entry in state.entries:
            _check_compatible(entry.feature, current, "cached feature")
            reference = entry.feature if mode == "history" else current
            aligned = align(reference.clone(), current.clone())
            _check_compatible(aligned, current, "alignment output")
            candidate = fuse(aligned, current.clone())
            _check_compatible(candidate, current, "fusion output")
            # Accumulate in FP32 for low-precision inputs, without detaching.
            value = candidate.float() if candidate.dtype in (torch.float16, torch.bfloat16) else candidate
            total = value if total is None else total + value
        assert total is not None
        readout = (total / len(state.entries)).to(current.dtype)

    new_entries = (state.entries + (Entry(step_index, stored_current),))[-state.capacity:]
    new_state = BankState(window_id, step_index + 1, state.capacity, new_entries)
    return readout, new_state


def scan_features(
    features: Tensor,
    *,
    align: PairOperator,
    fuse: PairOperator,
    mode: Mode = "history",
    capacity: int = 4,
    window_id: str = "window",
    state: BankState | None = None,
) -> tuple[Tensor, BankState]:
    """Convenience scan. Supply returned state for an internal time chunk.

    This scans already-computed features. Chunking the raw-video backbone also
    requires its original temporal/spatial halo and numerical equivalence tests.
    """
    if not isinstance(features, Tensor) or features.ndim != 5:
        raise ValueError("features must be [B,C,T,H,W]")
    if features.shape[2] < 1:
        raise ValueError("empty time chunk")
    if state is None:
        state = start_window(window_id, capacity)
    elif state.capacity != capacity:
        raise ValueError("cannot change capacity inside a window")
    outputs = []
    for t in range(features.shape[2]):
        readout, state = read_then_write(
            features[:, :, t], state,
            window_id=window_id, step_index=state.next_index,
            align=align, fuse=fuse, mode=mode,
        )
        outputs.append(readout)
    return torch.stack(outputs, dim=2), state
