"""New M1 from scratch: source bank -> PDA -> source relations -> detector.

Only the detector baseline may load released weights. Every M1/M2 parameter is
new; this module imports no historical M1 implementation or checkpoint.
"""
from __future__ import annotations

from functools import partial
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from source_relational_reducer import SourceRelationalReducer

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
    if not isinstance(window_id, str) or not window_id:
        raise ValueError("A nonempty window identifier is required")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity != 4:
        raise ValueError("This prototype fixes FIFO capacity at four")
    return BankState(window_id, 0, capacity, ())


class SelectiveEvidenceReader(nn.Module):
    """M2: H0 + sigmoid(G(Z))*P(Z), Z=phi([H0,X,E]); owns no state."""

    def __init__(self):
        super().__init__()
        self.joint = nn.Sequential(nn.Conv2d(64, 32, 1), nn.GELU())
        self.delta = nn.Conv2d(32, 32, 1)
        self.gate = nn.Conv2d(32, 32, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, original: Tensor, current: Tensor, evidence: Tensor) -> Tensor:
        z = self.joint(torch.cat((original, current, evidence), dim=1))
        return original + torch.sigmoid(self.gate(z)) * self.delta(z)


class SourceRelationsAdapter(nn.Module):
    """One local FIFO4 per ordered window; no state survives a forward call.

Inputs/outputs: [B,32,T,H,W]. Chunking traverses already-computed F/H and
preserves the same bank and autograd graph. It does not change the baseline's
temporal access. Current substitutes X_t BEFORE PDA for every occupied slot.
"""

    MODES = ("history", "current_only")
    REDUCTIONS = ("relations", "early_merge")
    READOUTS = ("simple", "selective")

    @staticmethod
    def _validate_chunk_size(value):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError("temporal_chunk_size must be a positive integer or None")

    def __init__(self, *, pda=None, in_projection=None, reduction_mode="relations",
                 readout="simple", temporal_chunk_size=None,
                 checkpoint_recurrence=True, native_pda=None):
        super().__init__()
        if reduction_mode not in self.REDUCTIONS or readout not in self.READOUTS:
            raise ValueError("Unsupported reduction or readout")
        self._validate_chunk_size(temporal_chunk_size)
        self.capacity = 4
        self.state_channels = 16
        self.reduction_mode = reduction_mode
        self.readout = readout
        self.temporal_chunk_size = temporal_chunk_size
        self.checkpoint_recurrence = bool(checkpoint_recurrence)
        self.enabled = True
        self._window_serial = 0
        if pda is None:
            from .pda import build_pda
            pda = build_pda()
            native_pda = True
        self.native_pda = True if native_pda is None else bool(native_pda)
        # PDA is a referenced operator, not a previously trained M1 state.
        self.recurrent_cell = nn.Module()
        self.recurrent_cell.pda = pda
        self.in_projection = in_projection if in_projection is not None else nn.Conv2d(32, 16, 1)
        self.reducer = SourceRelationalReducer(16, 8, 16)
        if readout == "simple":
            self.out_projection = nn.Conv2d(32, 32, 1)
            nn.init.zeros_(self.out_projection.weight)
            nn.init.zeros_(self.out_projection.bias)
        else:
            self.reader = SelectiveEvidenceReader()

    def align(self, source, current):
        if source.shape != current.shape or current.ndim != 4 or current.shape[1] != 16:
            raise ValueError("PDA inputs must have identical [B,16,H,W] shape")
        if source.device != current.device or source.dtype != current.dtype:
            raise ValueError("PDA inputs must share dtype and device")
        if self.native_pda:
            if not current.is_cuda:
                raise RuntimeError("Native RFR PDA requires CUDA; no substitute operator is enabled")
            if current.dtype not in (torch.float32, torch.float64):
                raise TypeError("Native RFR PDA requires float32/float64; AMP is not supported")
            if current.shape[-2] % 4 or current.shape[-1] % 4:
                raise ValueError("Native PDA requires spatial dimensions divisible by four")
        args = (source.contiguous(), current.contiguous())
        if self.checkpoint_recurrence and self.training and torch.is_grad_enabled():
            return checkpoint(self.recurrent_cell.pda, *args, use_reentrant=False)
        return self.recurrent_cell.pda(*args)

    def read_then_write(self, current: Tensor, state: BankState, *, window_id: str,
                        step_index: int, mode: str) -> tuple[Tensor, BankState]:
        """Expose all PDA outputs to one set reducer before any source pooling."""
        if not isinstance(current, Tensor) or current.ndim != 4 or current.shape[1] != 16:
            raise ValueError("Current source must have shape [B,16,H,W]")
        if min(current.shape) <= 0 or not current.is_floating_point():
            raise ValueError("Current source must be a nonempty floating point tensor")
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise ValueError("Source index must be a nonnegative integer")
        if mode not in self.MODES:
            raise ValueError("Invalid history mode")
        if not isinstance(state, BankState) or state.capacity != self.capacity:
            raise ValueError("Bank must use the fixed capacity four")
        if state.window_id != window_id or state.next_index != step_index:
            raise ValueError("Bank belongs to another window or time index")
        expected = list(range(max(0, step_index - self.capacity), step_index))
        if [entry.step_index for entry in state.entries] != expected:
            raise ValueError("Bank must contain unique ordered FIFO source indices")
        # Protect the stored source from in-place behavior of external operators.
        stored_current = current.clone()
        sources = []
        for entry in state.entries:
            source = entry.feature if mode == "history" else current
            if source.shape != current.shape or source.device != current.device or source.dtype != current.dtype:
                raise ValueError("Cached source shape/device/dtype changed inside a window")
            aligned = self.align(source.clone(), current.clone())
            if aligned.shape != current.shape or aligned.dtype != current.dtype or aligned.device != current.device:
                raise ValueError("PDA must preserve source shape/device/dtype")
            sources.append(aligned)
        aligned_sources = torch.stack(sources, dim=1) if sources else current.new_empty(
            (current.shape[0], 0, *current.shape[1:]))
        reduce_sources = partial(self.reducer, mode=self.reduction_mode)
        if self.checkpoint_recurrence and self.training and torch.is_grad_enabled():
            evidence = checkpoint(reduce_sources, current, aligned_sources, use_reentrant=False)
        else:
            evidence = reduce_sources(current, aligned_sources)
        entries = (state.entries + (Entry(step_index, stored_current),))[-self.capacity:]
        return evidence, BankState(window_id, step_index + 1, self.capacity, entries)

    def forward(self, pre_tpro_features, original_tpro_features, mode="history",
                temporal_chunk_size=None):
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode: {mode}")
        f, h = pre_tpro_features, original_tpro_features
        if f.ndim != 5 or f.shape[1] != 32 or f.shape != h.shape:
            raise ValueError("F and H must have identical [B,32,T,H,W] shapes")
        if min(f.shape) <= 0 or f.device != h.device or f.dtype != h.dtype:
            raise ValueError("F/H must be nonempty and share dtype/device")
        if not f.is_floating_point() or not h.is_floating_point():
            raise TypeError("F/H must be floating point")
        chunk = self.temporal_chunk_size if temporal_chunk_size is None else temporal_chunk_size
        self._validate_chunk_size(chunk)
        if not self.enabled:
            return h
        chunk = f.shape[2] if chunk is None else chunk
        self._window_serial += 1
        window_id = f"source_relations:{self._window_serial}"
        state = start_window(window_id, self.capacity)
        outputs = []
        for start in range(0, f.shape[2], chunk):
            for t in range(start, min(start + chunk, f.shape[2])):
                current = self.in_projection(f[:, :, t])
                evidence, state = self.read_then_write(current, state, window_id=window_id,
                                                       step_index=t, mode=mode)
                if self.readout == "simple":
                    adapted = h[:, :, t] + self.out_projection(torch.cat((current, evidence), dim=1))
                else:
                    adapted = self.reader(h[:, :, t], current, evidence)
                outputs.append(adapted)
        return torch.stack(outputs, dim=2)


class DeepProWithSourceRelations(nn.Module):
    """Keep the baseline frontend, unique TPro, prediction head and raw logits.

    The current prototype requires equal input/output temporal coordinates.
    It rejects a different TPro time length instead of guessing a mapping.
    No sigmoid, threshold, window fusion or tracking is added to the model.
    """

    def __init__(self, base, adapter: SourceRelationsAdapter, *, mode="history",
                 freeze_baseline=True):
        super().__init__()
        if mode not in adapter.MODES:
            raise ValueError("Invalid history mode")
        for name in ("conv_in", "layer1", "TPro", "conv_out1", "conv_out2"):
            if not isinstance(getattr(base, name, None), nn.Module):
                raise TypeError(f"Baseline is missing module {name}")
        self.base = base
        self.adapter = adapter
        self.mode = mode
        self.freeze_baseline = bool(freeze_baseline)
        if self.freeze_baseline:
            self.base.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_baseline:
            self.base.eval()
        return self

    def frontend(self, seq_imgs):
        f = self.base.layer1(self.base.conv_in(seq_imgs))
        h = self.base.TPro(f.permute(0, 3, 4, 1, 2))
        if f.shape != h.shape:
            raise ValueError("M1 requires matched F/H time indices; seqlen != out_len needs an explicit mapping")
        return f, h

    def forward(self, seq_imgs):
        if self.freeze_baseline:
            with torch.no_grad():
                f, h = self.frontend(seq_imgs)
        else:
            f, h = self.frontend(seq_imgs)
        adapted = self.adapter(f, h, mode=self.mode)
        # Frozen parameters do not stop the gradient from logits to M1/M2.
        features = self.base.conv_out1(adapted)
        logits = self.base.conv_out2(features).squeeze(dim=1)
        return features, logits
