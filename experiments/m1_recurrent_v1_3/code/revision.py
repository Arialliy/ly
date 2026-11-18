"""M1 source bank: reuse the supplied core and all existing learned tensors."""
from pathlib import Path
import sys
import torch
from torch.utils.checkpoint import checkpoint
from models.adapters.window_recurrent_adapter import WindowRecurrentAdapter

REFERENCE = Path(__file__).resolve().parents[1]/'reference/M1_v13_evidence_bank_reference'
sys.path.insert(0, str(REFERENCE))
from evidence_bank import scan_features, read_then_write, start_window
import evidence_bank as bank


class WindowEvidenceBankAdapter(WindowRecurrentAdapter):
    """One local bank per forward, one ordered window per batch row.

    Only the organization of historical inputs changes. Native PDA/TSFM and
    projections retain their module objects, parameter names and trained values.
    Temporal chunks operate on precomputed F/H; they share one bank with no
    detach. Full-spatial PDA/TSFM remains mandatory for inference.
    """

    def _pair_operator(self, operator, reference, current):
        if self.checkpoint_recurrence and self.training and torch.is_grad_enabled():
            return checkpoint(operator, reference, current, use_reentrant=False)
        return operator(reference, current)

    def align(self, reference, current):
        return self._pair_operator(self.recurrent_cell.pda, reference, current)

    def fuse(self, aligned, current):
        return self._pair_operator(self.recurrent_cell.tsfm, aligned, current)

    def forward(self, pre_tpro_features, original_tpro_features, mode='history',
                temporal_chunk_size=None):
        if mode not in self.MODES:
            raise ValueError(f'invalid mode: {mode}')
        f, h = pre_tpro_features, original_tpro_features
        if f.ndim != 5 or f.shape[1] != 32 or f.shape != h.shape:
            raise ValueError('F and H must have identical [B,32,T,H,W] shapes')
        if min(f.shape[0], f.shape[2], f.shape[3], f.shape[4]) <= 0:
            raise ValueError('batch, time and spatial dimensions must be nonempty')
        if f.device != h.device or f.dtype != h.dtype:
            raise ValueError('F and H must share dtype and device')
        chunk = self.temporal_chunk_size if temporal_chunk_size is None else temporal_chunk_size
        self._validate_chunk_size(chunk)
        chunk = f.shape[2] if chunk is None else chunk
        self._window_serial += 1
        window_id = f'forward:{self._window_serial}:ordered_rows:{tuple(range(f.shape[0]))}'
        bank = None
        outputs = []
        for start in range(0, f.shape[2], chunk):
            stop = min(start + chunk, f.shape[2])
            projected = torch.stack([self.in_projection(f[:, :, t])
                                     for t in range(start, stop)], dim=2)
            reads, bank = scan_features(projected, align=self.align, fuse=self.fuse,
                mode=mode, capacity=self.capacity, window_id=window_id, state=bank)
            for offset, t in enumerate(range(start, stop)):
                update = self.out_projection(torch.cat((projected[:, :, offset],
                                                       reads[:, :, offset]), dim=1))
                outputs.append(h[:, :, t] + update)
        # Neither source tensors nor the BankState escape this forward call.
        return torch.stack(outputs, dim=2)


def install_revision(model):
    adapter = model.adapter
    if type(adapter) is not WindowRecurrentAdapter:
        raise TypeError('Install only on the strict-loaded original v1.0 adapter')
    adapter.__class__ = WindowEvidenceBankAdapter
    adapter.capacity = 4
    adapter._window_serial = 0
    return model
