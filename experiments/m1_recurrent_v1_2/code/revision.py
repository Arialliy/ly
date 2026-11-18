"""Integrate the supplied, unmodified bounded-update core into the pinned RFR cell."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import torch
from torch import nn

from experiments.m1_recurrent_v1.upstream.rfr_cell import RFRCell

REFERENCE = Path(__file__).resolve().parents[1] / 'reference/M1_v12_state_update_reference/state_update.py'
_spec = importlib.util.spec_from_file_location('_m12_supplied_state_update', REFERENCE)
_reference = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _reference
_spec.loader.exec_module(_reference)
BoundedStateUpdate = _reference.BoundedStateUpdate


class BoundedRFRCell(RFRCell):
    """Keep original PDA/TSFM module objects and add only the supplied update core.

    The unchanged adapter owns M0=X0, window-local state, temporal chunks and
    non-reentrant activation checkpointing. This cell stores no recurrent state.
    """
    def __init__(self, original):
        nn.Module.__init__(self)
        if type(original) is not RFRCell:
            raise TypeError('Install exactly once into the original v1 RFRCell')
        self.channels = original.channels
        self.pda, self.tsfm = original.pda, original.tsfm
        self.state_update = BoundedStateUpdate(channels=16, hidden=16, scope='pixel')
        parameter = next(original.parameters())
        self.state_update.to(device=parameter.device, dtype=parameter.dtype)
        self.training = original.training
        self.state_update.train(original.training)
        self._gate_statistics = None

    def begin_gate_statistics(self):
        """Enable forward-only detached summaries; caller ends BEFORE backward."""
        if self._gate_statistics is not None:
            raise RuntimeError('Gate statistics already enabled')
        self._gate_statistics = {'calls': 0, 'pixels': 0, 'sum': None, 'square_sum': None,
                                 'minimum': None, 'maximum': None, 'histogram': None}

    def _observe_gate(self, retention):
        record = self._gate_statistics
        if record is None: return
        with torch.no_grad():
            value = retention.detach().float()
            total = value.sum(dtype=torch.float64)
            squares = value.double().square().sum()
            minimum, maximum = value.amin(), value.amax()
            histogram = torch.histc(value, bins=10, min=0.0, max=1.0)
            record['calls'] += 1
            record['pixels'] += value.numel()
            if record['sum'] is None:
                record.update(sum=total, square_sum=squares, minimum=minimum,
                              maximum=maximum, histogram=histogram)
            else:
                record['sum'] += total
                record['square_sum'] += squares
                record['minimum'] = torch.minimum(record['minimum'], minimum)
                record['maximum'] = torch.maximum(record['maximum'], maximum)
                record['histogram'] += histogram

    def end_gate_statistics(self):
        """Disable before checkpoint backward recomputation to avoid double counts."""
        record, self._gate_statistics = self._gate_statistics, None
        if record is None: raise RuntimeError('Gate statistics were not enabled')
        if not record['pixels']: return {'calls': 0, 'pixels': 0}
        count = record['pixels']
        mean = float(record['sum']) / count
        variance = max(0.0, float(record['square_sum']) / count - mean * mean)
        return {'calls': record['calls'], 'pixels': count, 'mean': mean,
                'std': variance ** 0.5, 'minimum': float(record['minimum']), 'maximum': float(record['maximum']),
                'histogram_edges': [index / 10 for index in range(11)],
                'histogram_counts': [int(value) for value in record['histogram'].cpu().tolist()],
                'scope': 'All training forward gate pixels; no label conditioning; excludes t0 and backward recomputation'}

    def forward(self, previous, current):
        if previous.shape != current.shape or current.ndim != 4 or current.shape[1] != 16:
            raise ValueError('previous/current must share [B,16,H,W] shape')
        if current.shape[-2] % 4 or current.shape[-1] % 4:
            raise ValueError('Official PDA requires H/W divisible by four')
        if previous.device != current.device or previous.dtype != current.dtype:
            raise ValueError('previous/current device and dtype must match')
        if not current.is_cuda or current.dtype not in (torch.float32, torch.float64):
            raise TypeError('Official DCNv2 requires CUDA FP32/FP64; no AMP')
        aligned = self.pda(previous.contiguous(), current.contiguous())
        candidate = self.tsfm(aligned, current)
        state, retention = self.state_update(aligned, candidate, current, scope='pixel')
        self._observe_gate(retention)
        return state


def install_revision(model, gate_seed=46):
    """Preserve every old tensor/key/object; initialize only the supplied gate."""
    original = model.adapter.recurrent_cell
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    old_parameters = dict(model.named_parameters())
    # The reference initializes on CPU. Seed its CPU generator without consuming
    # the caller's CPU RNG or perturbing any CUDA generator.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(gate_seed)
        replacement = BoundedRFRCell(original)
    model.adapter.recurrent_cell = replacement
    after = model.state_dict()
    assert set(before).issubset(after)
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    now_parameters = dict(model.named_parameters())
    assert all(now_parameters[name] is parameter for name, parameter in old_parameters.items())
    added = {name: parameter for name, parameter in now_parameters.items() if name not in old_parameters}
    assert len(added) == 6 and sum(parameter.numel() for parameter in added.values()) == 2017
    assert all(name.startswith('adapter.recurrent_cell.state_update.') for name in added)
    assert all(torch.count_nonzero(value) == 0 for value in replacement.state_update.gate_net[-1].parameters())
    return {'reference_path': str(REFERENCE), 'scope': 'pixel', 'gate_seed': gate_seed,
        'gate_initial_retention': 0.5, 'first_frame': 'M0=X0 unchanged',
        'old_state_keys_and_values_unchanged': True, 'old_parameter_objects_unchanged': True,
        'added_parameter_names': list(added), 'added_parameter_tensors': 6, 'added_parameters': 2017,
        'initial_output_equal_to_old_model_claimed': False,
        'update': 'state_update(aligned=PDA(previous,X), candidate=TSFM(aligned,X), current=X, scope=pixel)'}


install_state_update = install_revision
