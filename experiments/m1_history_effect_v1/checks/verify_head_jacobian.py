"""One CPU cross-check of trace's analytic derivative using actual fixed weights."""
from pathlib import Path
import hashlib
import importlib.util
import json
import torch
from torch import nn

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]
CHECKPOINT = ROOT / 'experiments/m1_recurrent_v1/training/History/run/snapshot_2048.pth'
SOURCE = E / 'code/trace.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_out1 = nn.Sequential(nn.Conv3d(32, 8, 1), nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.conv_out2 = nn.Conv3d(8, 1, 1)

    def forward(self, value):
        return self.conv_out2(self.conv_out1(value))


def main():
    torch.manual_seed(46)
    torch.set_num_threads(1)
    checkpoint_hash = sha(CHECKPOINT)
    assert checkpoint_hash == 'eda063496757f368d7442a7fac99a64f7b33ef0414ef99d006716d77a6dcb725'
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)['model_state_dict']
    head = Head().eval()
    subset = {key.removeprefix('base.'): value for key, value in state.items()
              if key.startswith(('base.conv_out1.', 'base.conv_out2.'))}
    head.load_state_dict(subset, strict=True)
    head.double().requires_grad_(False)
    spec = importlib.util.spec_from_file_location('_checked_trace', SOURCE)
    trace = importlib.util.module_from_spec(spec); spec.loader.exec_module(trace)
    value = torch.randn(4, 32, 1, 1, 1, dtype=torch.float64, requires_grad=True)
    preactivation = head.conv_out1[1](head.conv_out1[0](value)).detach().reshape(4, 8)
    analytic = trace.head_input_jacobian(head, preactivation)
    automatic = torch.autograd.grad(head(value).sum(), value)[0].reshape(4, 32)
    maximum_error = float((analytic - automatic).abs().max())
    assert maximum_error < 1e-12, maximum_error
    delta = torch.randn_like(value) * 1e-6
    with torch.no_grad():
        next_preactivation = head.conv_out1[1](head.conv_out1[0](value + delta)).reshape(4, 8)
        unchanged_masks = ((preactivation > 0) == (next_preactivation > 0)).all(1)
        assert bool(unchanged_masks.all())
        finite_delta = (head(value + delta) - head(value)).reshape(4)
        linear_delta = (analytic * delta.reshape(4, 32)).sum(1)
        secant_error = float((finite_delta - linear_delta).abs().max())
        assert secant_error < 1e-10, secant_error
    assert all(parameter.grad is None for parameter in head.parameters())
    assert sha(CHECKPOINT) == checkpoint_hash
    record = {'status': 'PASS', 'device': 'cpu', 'dtype': 'float64', 'points': 4,
              'checkpoint_sha256': checkpoint_hash, 'trace_source_sha256': sha(SOURCE),
              'analytic_vs_autograd_max_absolute_error': maximum_error,
              'same_relu_mask_finite_difference_max_absolute_error': secant_error,
              'jacobian_nonzero_points': int((analytic.norm(dim=1) > 0).sum()),
              'checkpoint_unchanged': True, 'training_or_gpu_workload': False,
              'scope': 'Computational correctness using actual History2048 head weights; synthetic feature points are not empirical target evidence.'}
    output = E / 'checks/HEAD_JACOBIAN_CPU.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
