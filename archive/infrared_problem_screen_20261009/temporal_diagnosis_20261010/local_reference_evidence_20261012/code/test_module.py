"""Focused CPU formula/interface tests, including an independent NumPy oracle."""
from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from module import (ANNULUS_COUNT, ANNULUS_OFFSETS, BRANCH_SPATIAL_RADIUS,
                    CHANNEL_NAMES, FrozenEvidenceDetector, LocalReferenceEvidence,
                    S_MIN, exact_even_median)

CHECKS = []


def check(name, condition, **details):
    CHECKS.append(dict(name=name, passed=bool(condition), **details))
    if not condition:
        raise AssertionError(f'{name}: {details}')


def close(name, actual, expected, atol=1e-6, rtol=1e-6):
    a = actual.detach().cpu().numpy() if torch.is_tensor(actual) else np.asarray(actual)
    b = expected.detach().cpu().numpy() if torch.is_tensor(expected) else np.asarray(expected)
    error = float(np.max(np.abs(a - b))) if a.size else 0.0
    check(name, np.allclose(a, b, atol=atol, rtol=rtol), maximum_absolute_error=error)


class ToyOriginal(nn.Module):
    """Shape-compatible frozen baseline with a BN state to protect."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 8, (1, 3, 3), padding=(0, 1, 1))
        self.bn = nn.BatchNorm3d(8)
        self.head = nn.Conv3d(8, 1, 1)

    def forward(self, image):
        features = self.bn(self.conv(image)).relu()
        return features, self.head(features)[:, 0]


def numpy_annulus_oracle(difference):
    """Independent membership enumeration and NumPy continuous-even median."""
    shape = difference.shape
    flat = difference.reshape(-1, *shape[-2:])
    mu = np.empty_like(flat)
    scale = np.empty_like(flat)
    for n in range(len(flat)):
        for y in range(flat.shape[1]):
            for x in range(flat.shape[2]):
                values = []
                for yy in range(y - 7, y + 8):
                    for xx in range(x - 7, x + 8):
                        if abs(yy - y) <= 5 and abs(xx - x) <= 5:
                            continue
                        iy = min(max(yy, 0), flat.shape[1] - 1)
                        ix = min(max(xx, 0), flat.shape[2] - 1)
                        values.append(flat[n, iy, ix])
                values = np.asarray(values)
                mu[n, y, x] = np.median(values)
                scale[n, y, x] = np.median(np.abs(values - mu[n, y, x]))
    return mu.reshape(shape), scale.reshape(shape)


def formula_tests():
    check('annulus_membership_104', ANNULUS_COUNT == 104 and len(set(ANNULUS_OFFSETS)) == 104
          and all(5 < max(abs(y), abs(x)) <= 7 for y, x in ANNULUS_OFFSETS))
    sample = torch.arange(104, dtype=torch.float64).reshape(1, 1, 104)
    close('even_median_averages_two_central_values', exact_even_median(sample), [[[51.5]]], 0, 0)
    evidence = LocalReferenceEvidence('relative', statistics_chunk_frames=2)
    rng = np.random.default_rng(46)
    data = rng.integers(-13, 19, size=(2, 5, 8)).astype(np.float64)
    actual_mu, actual_s = evidence.statistics(torch.from_numpy(data))
    expected_mu, expected_s = numpy_annulus_oracle(data)
    close('numpy_annulus_median_including_image_edges', actual_mu, expected_mu, 0, 0)
    close('numpy_annulus_mad_including_image_edges', actual_s, expected_s, 0, 0)
    evidence.statistics_chunk_frames = 1
    single_mu, single_s = evidence.statistics(torch.from_numpy(data))
    close('chunking_does_not_change_median', single_mu, actual_mu, 0, 0)
    close('chunking_does_not_change_mad', single_s, actual_s, 0, 0)
    image = torch.arange(6, dtype=torch.float64)[None, None, :, None, None].expand(1, 1, 6, 17, 19) * 1.75
    channels = evidence.construct_evidence(image)
    close('uniform_temporal_change_cancels_relative_signal', channels[:, [4, 9]], 0, 0, 0)
    close('uniform_temporal_change_preserves_signed_difference', channels[:, 1, 1:], 1.75, 0, 0)
    check('uniform_temporal_change_validity_is_honest',
          channels[:, 5, 0].count_nonzero() == 0 and torch.all(channels[:, 5, 1:] == 1)
          and channels[:, 10, :4].count_nonzero() == 0 and torch.all(channels[:, 10, 4:] == 1))
    d1 = torch.zeros(1, 23, 23)
    d2 = torch.zeros_like(d1)
    d1[0, 11, 11] = d2[0, 11, 11] = 2
    for i, (dy, dx) in enumerate(ANNULUS_OFFSETS):
        sign = 1 if i % 2 else -1
        d1[0, 11 + dy, 11 + dx] = sign
        d2[0, 11 + dy, 11 + dx] = 4 * sign
    mu1, s1 = evidence.statistics(d1)
    mu2, s2 = evidence.statistics(d2)
    center = (0, 11, 11)
    close('dispersion_intervention_holds_center_difference', d1[center], d2[center], 0, 0)
    close('dispersion_intervention_holds_local_median', mu1[center], mu2[center], 0, 0)
    close('dispersion_intervention_changes_mad_from_1_to_4', torch.stack((s1[center], s2[center])), [1, 4], 0, 0)
    close('dispersion_intervention_reduces_relative_magnitude',
          torch.stack(((d1[center]-mu1[center])/s1[center], (d2[center]-mu2[center])/s2[center])), [2, .5], 0, 0)
    zero = evidence.construct_evidence(torch.zeros(2, 1, 2, 3, 5))
    check('zero_difference_and_missing_reference_are_finite', torch.isfinite(zero).all())
    close('entire_missing_delay4_tuple_zero', zero[:, 6:], 0, 0, 0)
    close('missing_delay1_frame_zero', zero[:, 1:6, 0], 0, 0, 0)
    close('zero_relative_evidence_zero', zero[:, [1, 2, 3, 4, 6, 7, 8, 9]], 0, 0, 0)
    dnegative = torch.arange(6., 0., -1.)[None, None, :, None, None].expand(1, 1, 6, 3, 4)
    negative = evidence.construct_evidence(dnegative)
    close('difference_sign_preserved', negative[:, 1, 1:], -1, 0, 0)


def mode_tests():
    image = torch.randn(2, 1, 6, 7, 9, requires_grad=True)
    states = []
    for mode in ('current', 'difference', 'relative'):
        torch.manual_seed(46)
        model = LocalReferenceEvidence(mode)
        states.append(copy.deepcopy(model.state_dict()))
        values = model.construct_evidence(image)
        close(f'{mode}_retains_input', values[:, 0], image[:, 0], 0, 0)
        check(f'{mode}_raw_statistics_detached', not values.requires_grad)
        check(f'{mode}_telemetry_axis_order', tuple(model.last_input.shape) == (2, 6, 11, 7, 9)
              and tuple(model.last_statistics) == CHANNEL_NAMES)
        if mode == 'current':
            close('current_all_ten_temporal_channels_zero', values[:, 1:], 0, 0, 0)
        if mode == 'difference':
            close('difference_extra_reference_channels_zero', values[:, [2, 3, 4, 7, 8, 9]], 0, 0, 0)
            close('difference_delta1_exact', values[:, 1, 1:], image[:, 0, 1:] - image[:, 0, :-1], 0, 0)
            close('difference_delta4_exact', values[:, 6, 4:], image[:, 0, 4:] - image[:, 0, :-4], 0, 0)
        check(f'{mode}_same_1529_trainable_parameters', sum(p.numel() for p in model.parameters()) == 1529)
    check('all_modes_exact_same_initial_parameters',
          all(torch.equal(states[0][name], state[name]) for state in states[1:] for name in states[0]))


def integration_and_gradient_tests():
    torch.manual_seed(46)
    baseline = ToyOriginal().eval()
    frozen_state = copy.deepcopy(baseline.state_dict())
    wrapper = FrozenEvidenceDetector(baseline, 'relative').train()
    x = torch.randn(1, 1, 5, 13, 17)
    with torch.no_grad():
        h0, z0 = baseline(x)
    h, z = wrapper(x)
    close('zero_output_projection_exact_original_logits', z, z0, 0, 0)
    close('original_feature_interface_exact', h, h0, 0, 0)
    check('wrapper_train_does_not_unfreeze_original_bn_or_parameters',
          wrapper.training and wrapper.evidence.training and not wrapper.original.training
          and all(not m.training for m in wrapper.original.modules())
          and all(not p.requires_grad for p in wrapper.original.parameters()))
    optimizer = torch.optim.SGD(wrapper.evidence.parameters(), lr=.1)
    target = z0 + .7 + x[:, 0]
    loss = (z - target).square().mean()
    loss.backward()
    output_grad = float(wrapper.evidence.output.weight.grad.abs().sum())
    encoder_grad_first = sum(float(p.grad.abs().sum()) for p in wrapper.evidence.encoder.parameters() if p.grad is not None)
    check('output_projection_gets_first_step_gradient', output_grad > 0, gradient_l1=output_grad)
    check('zero_projection_gates_first_encoder_gradient', encoder_grad_first == 0, gradient_l1=encoder_grad_first)
    check('original_has_no_gradient', all(p.grad is None for p in baseline.parameters()))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _, z = wrapper(x)
    (z - target).square().mean().backward()
    encoder_grad_second = sum(float(p.grad.abs().sum()) for p in wrapper.evidence.encoder.parameters() if p.grad is not None)
    check('encoder_gets_second_step_gradient', encoder_grad_second > 0, gradient_l1=encoder_grad_second)
    optimizer.step()
    check('frozen_original_parameters_and_bn_buffers_exactly_unchanged',
          all(torch.equal(frozen_state[key], value) for key, value in baseline.state_dict().items()))
    wrapper.correction_enabled = False
    _, disabled = wrapper(x)
    close('disable_trained_correction_restores_original_output', disabled, z0, 0, 0)
    close('disabled_telemetry_delta_zero', wrapper.last_delta, 0, 0, 0)


def tiled_nonzero_test():
    torch.manual_seed(46)
    wrapper = FrozenEvidenceDetector(ToyOriginal().eval(), 'relative', statistics_chunk_frames=2).eval()
    with torch.no_grad():
        wrapper.evidence.output.weight.normal_(0, .2)
        wrapper.evidence.output.bias.fill_(.1)
        image = torch.randn(1, 1, 5, 29, 37)
        _, direct = wrapper(image)
        direct_delta = wrapper.last_delta.clone()
        tiled = torch.empty_like(direct)
        h, w = image.shape[-2:]
        core, halo = 16, BRANCH_SPATIAL_RADIUS
        for y in range(0, h, core):
            for x in range(0, w, core):
                end_y, end_x = min(y + core, h), min(x + core, w)
                top, left = max(0, y - halo), max(0, x - halo)
                bottom, right = min(h, end_y + halo), min(w, end_x + halo)
                _, patch = wrapper(image[..., top:bottom, left:right])
                tiled[..., y:end_y, x:end_x] = patch[..., y-top:end_y-top, x-left:end_x-left]
    check('tiling_witness_has_nonzero_correction', float(direct_delta.abs().max()) > 0)
    close('nonzero_correction_direct_equals_tiled_with_halo9', tiled, direct, 2e-6, 2e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.time()
    report = dict(status='RUNNING', torch_version=torch.__version__, seed=46,
                  device='cpu', checks=CHECKS)
    try:
        formula_tests()
        mode_tests()
        integration_and_gradient_tests()
        tiled_nonzero_test()
        report.update(status='PASS', number_checks=len(CHECKS), elapsed_seconds=time.time()-started)
    except Exception as exc:
        report.update(status='FAIL', exception=repr(exc), number_checks=len(CHECKS), elapsed_seconds=time.time()-started)
        raise
    finally:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
