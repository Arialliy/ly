"""Finite seeded runtime witness; no detector training or evaluation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(46)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from haar_reference import SpatialLFPReference
    from framewise_adapter import FramewiseFeatureAdapter

    model = FramewiseFeatureAdapter(SpatialLFPReference(channels=4), channels=4, frame_chunk=3).to(args.device)
    with torch.no_grad():
        model.scale.fill_(0.1)
    x = torch.randn(2, 4, 5, 17, 19, device=args.device, requires_grad=True)
    y = model(x)
    y.square().mean().backward()
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = {name: float(p.grad.abs().max()) if p.grad is not None else None for name, p in model.named_parameters()}
    assert all(v is not None and v > 0 for v in grads.values()), grads
    receipt = {
        'status': 'PASS', 'seed': 46, 'device': args.device,
        'device_name': torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU',
        'shape': list(y.shape), 'max_abs': float(y.detach().abs().max()),
        'output_sha256': hashlib.sha256(y.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        'parameter_gradient_maxima': grads,
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'pytest', 'PyWavelets', 'pytorch-wavelets']},
        'scope': 'seeded forward/backward dispatch only, no detector effectiveness claim',
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2) + '\n')
    print('WITNESS', json.dumps(receipt))


if __name__ == '__main__':
    main()
