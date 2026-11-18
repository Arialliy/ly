"""Synthetic reducer checks, not evidence of detection or learned recovery.

The original ten prototype checks are retained and strengthened. Semantic
checks explicitly document the mean-only control's inactive difference block
and limitations of the source summary rather than asserting extra information.
Run from the repository root with Python's unittest runner or pytest.
"""
from itertools import combinations
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_relational_reducer import SourceRelationalReducer


MODES = ("relations", "early_merge")


class ProductOnly(nn.Module):
    """An explicit operator for the original algebraic example only."""

    def forward(self, descriptor):
        return descriptor[:, descriptor.shape[1] * 2 // 3:]


class TestReducer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(19)
        self.net = SourceRelationalReducer()
        self.x = torch.randn(2, 16, 5, 7)
        self.a = torch.randn(2, 4, 16, 5, 7)

    def test_shape_and_finite(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                y = self.net(self.x, self.a, mode=mode)
                self.assertEqual(tuple(y.shape), (2, 16, 5, 7))
                self.assertEqual(y.dtype, self.x.dtype)
                self.assertEqual(y.device, self.x.device)
                self.assertTrue(torch.isfinite(y).all())

    def test_empty_memory(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                y = self.net(self.x, self.a[:, :0], mode=mode)
                torch.testing.assert_close(y[:, :8], self.net.value(self.x))
                self.assertEqual(y[:, 8:].count_nonzero().item(), 0)

    def test_single_source_no_pair_relation(self):
        valid = torch.tensor([[True, False, False, False], [False, False, True, False]])
        expected = self.net.value(torch.stack((self.a[0, 0], self.a[1, 2])))
        for mode in MODES:
            with self.subTest(mode=mode):
                y = self.net(self.x, self.a, valid, mode=mode)
                torch.testing.assert_close(y[:, :8], expected)
                self.assertEqual(y[:, 8:].count_nonzero().item(), 0)

    def test_padding_invariance_including_nan(self):
        valid = torch.tensor([[True, True, False, False], [True, True, False, False]])
        changed = self.a.clone()
        changed[:, 2] = float("nan")
        changed[:, 3] = float("inf")
        for mode in MODES:
            with self.subTest(mode=mode):
                torch.testing.assert_close(
                    self.net(self.x, self.a, valid, mode=mode),
                    self.net(self.x, changed, valid, mode=mode),
                )
                # Padding must also agree with actually removing these slots.
                torch.testing.assert_close(
                    self.net(self.x, changed, valid, mode=mode),
                    self.net(self.x, self.a[:, :2], mode=mode),
                )

    def test_all_invalid_fallback(self):
        valid = torch.zeros(2, 4, dtype=torch.bool)
        invalid = torch.full_like(self.a, float("nan"))
        for mode in MODES:
            with self.subTest(mode=mode):
                torch.testing.assert_close(
                    self.net(self.x, invalid, valid, mode=mode),
                    self.net(self.x, self.a[:, :0], mode=mode),
                )

    def test_source_permutation(self):
        valid = torch.tensor([[True, True, True, False], [True, False, True, True]])
        permutation = torch.tensor([2, 0, 3, 1])
        for mode in MODES:
            with self.subTest(mode=mode):
                torch.testing.assert_close(
                    self.net(self.x, self.a, valid, mode=mode),
                    self.net(self.x, self.a[:, permutation], valid[:, permutation], mode=mode),
                )

    def test_identical_sources_modes_coincide(self):
        repeated = self.a[:, :1].expand(-1, 4, -1, -1, -1)
        torch.testing.assert_close(
            self.net(self.x, repeated, mode="relations"),
            self.net(self.x, repeated, mode="early_merge"),
        )

    def test_gradient_and_invalid_source_isolation(self):
        a = self.a.clone()
        a[:, 3] = float("nan")
        a.requires_grad_()
        valid = torch.tensor([[True, True, True, False], [True, True, True, False]])
        self.net(self.x, a, valid).square().mean().backward()
        self.assertIsNotNone(a.grad)
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(a.grad[:, :3].abs().sum().item(), 0)
        self.assertEqual(a.grad[:, 3].count_nonzero().item(), 0)
        for name, p in self.net.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_equal_mean_distinct_cross_source_relation(self):
        # Specifying the operator establishes algebra, not that it is learned.
        net = SourceRelationalReducer(in_channels=8)
        net.value = nn.Identity()
        net.relation = ProductOnly()
        current = torch.zeros(1, 8, 1, 1)
        isolated = torch.tensor([0.9, 0.0, 0.0]).reshape(1, 3, 1, 1, 1).expand(1, 3, 8, 1, 1)
        repeated = torch.full_like(isolated, 0.3)
        yi = net(current, isolated)
        yr = net(current, repeated)
        torch.testing.assert_close(yi[:, :8], yr[:, :8])
        torch.testing.assert_close(yi[:, 8:], torch.zeros_like(yi[:, 8:]))
        torch.testing.assert_close(yr[:, 8:], torch.full_like(yr[:, 8:], 0.09))
        torch.testing.assert_close(
            net(current, isolated, mode="early_merge"),
            net(current, repeated, mode="early_merge"),
        )

    def test_validation(self):
        cases = (
            (ValueError, (self.x, self.a), {"mode": "other"}),
            (TypeError, (None, self.a), {}),
            (TypeError, (self.x, []), {}),
            (ValueError, (self.x[0], self.a), {}),
            (ValueError, (self.x, self.a[0]), {}),
            (ValueError, (self.x[:, :15], self.a[:, :, :15]), {}),
            (ValueError, (self.x, self.a[:1]), {}),
            (ValueError, (self.x, self.a[:, :, :, :, :6]), {}),
            (ValueError, (self.x[:, :, :0], self.a[:, :, :, :0]), {}),
            (ValueError, (self.x, self.a.double()), {}),
            (TypeError, (self.x.long(), self.a.long()), {}),
            (ValueError, (self.x, torch.empty_like(self.a, device="meta")), {}),
            (TypeError, (self.x, self.a, [[True] * 4] * 2), {}),
            (ValueError, (self.x, self.a, torch.ones(2, 4)), {}),
            (ValueError, (self.x, self.a, torch.ones(2, 3, dtype=torch.bool)), {}),
            (ValueError, (self.x, self.a, torch.ones(2, 4, dtype=torch.bool, device="meta")), {}),
        )
        for index, (exception, args, kwargs) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(exception):
                self.net(*args, **kwargs)

    def test_default_parameter_count_and_architecture(self):
        self.assertEqual(sum(p.numel() for p in self.net.parameters()), 672)
        self.assertIsInstance(self.net.value[0], nn.Conv2d)
        self.assertIsInstance(self.net.value[1], nn.Tanh)
        self.assertIsInstance(self.net.relation[0], nn.Conv2d)
        self.assertIsInstance(self.net.relation[1], nn.GELU)
        self.assertIsInstance(self.net.relation[2], nn.Conv2d)
        self.assertEqual(self.net.relation[0].weight.shape, (16, 24, 1, 1))

    def test_constructor_validation(self):
        for field in ("in_channels", "value_channels", "hidden_channels"):
            for value, exception in ((0, ValueError), (-1, ValueError), (2.5, TypeError), (True, TypeError)):
                with self.subTest(field=field, value=value), self.assertRaises(exception):
                    SourceRelationalReducer(**{field: value})

    def test_none_mask_means_all_sources_valid(self):
        valid = torch.ones(2, 4, dtype=torch.bool)
        for mode in MODES:
            with self.subTest(mode=mode):
                torch.testing.assert_close(
                    self.net(self.x, self.a, mode=mode),
                    self.net(self.x, self.a, valid, mode=mode),
                )

    def test_mixed_batch_matches_per_sample_formula(self):
        # Independent per-sample reference drops invalid slots before encoding.
        net = self.net.double()
        current = torch.randn(4, 16, 2, 3, dtype=torch.float64)
        aligned = torch.randn(4, 4, 16, 2, 3, dtype=torch.float64)
        valid = torch.tensor([
            [False, False, False, False],
            [False, True, False, False],
            [True, False, False, True],
            [True, True, True, True],
        ])
        aligned[~valid] = float("nan")
        for mode in MODES:
            expected = []
            for row in range(4):
                sources = aligned[row, valid[row]]
                if sources.shape[0] == 0:
                    mean = net.value(current[row:row + 1])[0]
                    relation = torch.zeros_like(mean)
                else:
                    values = net.value(sources)
                    mean = values.mean(dim=0)
                    pairs = []
                    for j, k in combinations(range(len(values)), 2):
                        a, b = (values[j], values[k]) if mode == "relations" else (mean, mean)
                        descriptor = torch.cat((a + b, (a - b).abs(), a * b), dim=0)
                        pairs.append(net.relation(descriptor[None])[0])
                    relation = torch.stack(pairs).mean(dim=0) if pairs else torch.zeros_like(mean)
                expected.append(torch.cat((mean, relation), dim=0))
            with self.subTest(mode=mode):
                y = net(current, aligned, valid, mode=mode)
                self.assertEqual(y.dtype, torch.float64)
                torch.testing.assert_close(y, torch.stack(expected), rtol=1e-10, atol=1e-12)

    def test_relation_network_call_counts_match(self):
        valid = torch.tensor([[True, True, False, False], [False, False, False, False]])
        for mode in MODES:
            calls = []
            handle = self.net.relation.register_forward_hook(lambda *args: calls.append(1))
            try:
                self.net(self.x, self.a, valid, mode=mode)
            finally:
                handle.remove()
            with self.subTest(mode=mode):
                # Every pair of storage slots is called, even if masked out.
                self.assertEqual(len(calls), 6)

    def test_early_merge_is_one_mean_only_function(self):
        values = self.net.value(self.a.flatten(0, 1)).reshape(2, 4, 8, 5, 7)
        mean = values.mean(dim=1)
        expected_relation = self.net.relation(torch.cat((2 * mean, torch.zeros_like(mean), mean.square()), dim=1))
        expected = torch.cat((mean, expected_relation), dim=1)
        torch.testing.assert_close(self.net(self.x, self.a, mode="early_merge"), expected)

    def test_early_merge_difference_weights_receive_zero_data_gradient(self):
        a = self.a.clone().requires_grad_()
        self.net(self.x, a, mode="early_merge").square().mean().backward()
        grad = self.net.relation[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertEqual(grad[:, 8:16].count_nonzero().item(), 0)
        self.assertGreater(grad[:, :8].abs().sum().item(), 0)
        self.assertGreater(grad[:, 16:].abs().sum().item(), 0)
        self.assertGreater(a.grad.abs().sum().item(), 0)
        # Zero data gradient is expected; do not mistake nominal parameter
        # matching or repeated calls for equal effective capacity.

    def test_identical_sources_do_not_identify_count_above_one(self):
        source = self.a[:, :1]
        for mode in MODES:
            reference = self.net(self.x, source.expand(-1, 2, -1, -1, -1), mode=mode)
            for count in (3, 5):
                with self.subTest(mode=mode, source_count=count):
                    torch.testing.assert_close(
                        self.net(self.x, source.expand(-1, count, -1, -1, -1), mode=mode),
                        reference,
                    )

    def test_actual_nonlinear_network_can_distinguish_some_equal_means(self):
        # Construct real Conv/Tanh/Conv/GELU/Conv weights, without replacing
        # layers. This proves representability of this example, not learning.
        net = SourceRelationalReducer().double()
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.zero_()
            net.value[0].weight[:, 0, 0, 0] = 1
            net.relation[0].weight[0, 16, 0, 0] = 1
            net.relation[2].weight[:, 0, 0, 0] = 1
        current = torch.zeros(1, 16, 1, 1, dtype=torch.float64)
        isolated = torch.zeros(1, 3, 16, 1, 1, dtype=torch.float64)
        isolated[0, :, 0, 0, 0] = torch.atanh(torch.tensor([0.9, 0.0, 0.0], dtype=torch.float64))
        repeated = torch.zeros_like(isolated)
        repeated[0, :, 0, 0, 0] = torch.atanh(torch.tensor(0.3, dtype=torch.float64))
        yi = net(current, isolated)
        yr = net(current, repeated)
        torch.testing.assert_close(yi[:, :8], yr[:, :8])
        self.assertGreater((yi[:, 8:] - yr[:, 8:]).abs().min().item(), 0.01)
        torch.testing.assert_close(
            net(current, isolated, mode="early_merge"),
            net(current, repeated, mode="early_merge"),
        )
        self.assertEqual(sum(p.numel() for p in net.parameters()), 672)

    def test_pair_descriptor_does_not_preserve_cross_channel_source_assignment(self):
        net = self.net.double()
        with torch.no_grad():
            net.value[0].weight.zero_()
            net.value[0].bias.zero_()
            for channel in range(8):
                net.value[0].weight[channel, channel, 0, 0] = 1
        current = torch.zeros(1, 16, 1, 1, dtype=torch.float64)
        separate = torch.zeros(1, 2, 16, 1, 1, dtype=torch.float64)
        together = torch.zeros_like(separate)
        activation = torch.atanh(torch.tensor(0.8, dtype=torch.float64))
        separate[0, 0, 0, 0, 0] = activation
        separate[0, 1, 1, 0, 0] = activation
        together[0, 0, :2, 0, 0] = activation
        torch.testing.assert_close(net(current, separate), net(current, together))

    def test_current_fallback_gradient_only_for_samples_without_history(self):
        current = self.x.clone().requires_grad_()
        aligned = self.a.clone().requires_grad_()
        valid = torch.tensor([[False, False, False, False], [True, True, False, False]])
        self.net(current, aligned, valid).square().mean().backward()
        self.assertGreater(current.grad[0].abs().sum().item(), 0)
        self.assertEqual(current.grad[1].count_nonzero().item(), 0)
        self.assertEqual(aligned.grad[0].count_nonzero().item(), 0)
        self.assertGreater(aligned.grad[1, :2].abs().sum().item(), 0)
        self.assertEqual(aligned.grad[1, 2:].count_nonzero().item(), 0)

    def test_cpu_autocast_preserves_input_dtype_including_empty_history(self):
        valid = torch.tensor([[False, False, False, False], [True, True, False, False]])
        for mode in MODES:
            with self.subTest(mode=mode), torch.autocast("cpu", dtype=torch.bfloat16):
                output = self.net(self.x, self.a, valid, mode=mode)
                empty = self.net(self.x, self.a[:, :0], mode=mode)
                fallback = self.net.value(self.x).float()
                self.assertEqual(output.dtype, torch.float32)
                self.assertEqual(empty.dtype, torch.float32)
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(empty[:, :8], fallback)
                torch.testing.assert_close(output[0], empty[0])
                self.assertEqual(empty[:, 8:].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
