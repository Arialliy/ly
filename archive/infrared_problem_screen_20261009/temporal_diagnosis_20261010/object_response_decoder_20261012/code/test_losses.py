#!/usr/bin/env python3
"""Finite CPU semantic and gradient checks; no model training or dev data."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from losses import decoder_loss


def target(t=2, h=3, w=4, observations=None, eligible=None):
    observations = [] if observations is None else observations
    valid = np.ones((t, h, w), dtype=np.bool_)
    background = valid.copy()
    obs = []
    for frame, identity, indices in observations:
        indices = np.asarray(indices, dtype=np.int64)
        key = ("training_only_synthetic", 3, str(identity))
        obs.append({"frame": frame, "track_key": key,
                    "annotation_id": frame*100 + int(identity),
                    "support_indices": indices})
        background[frame].ravel()[indices] = False
    return {"valid": valid, "background": background,
            "negative_eligible": np.ones(t, dtype=np.bool_) if eligible is None
            else np.asarray(eligible, dtype=np.bool_), "observations": obs,
            "track_keys": tuple(sorted({o["track_key"] for o in obs}))}


def explicit_oracle(a, masks, targets, assignments):
    """Independent dense per-object BCE oracle, used only on tiny arrays."""
    zero = a.flatten()[0]*0 + masks.flatten()[0]*0
    positives_e, positives_m, instances = [], [], []
    assigned = set()
    by_id = {int(o["annotation_id"]): o for o in targets["observations"]}
    for match in assignments:
        ti, qi = match["frame"], match["query"]
        obs = by_id[match["annotation_id"]]
        support = torch.zeros(masks.shape[-2:], dtype=torch.bool)
        support.flatten()[torch.as_tensor(obs["support_indices"])] = True
        positives_e.append(F.softplus(-a[ti, qi]))
        positives_m.append(F.softplus(-masks[ti, qi][support]).mean())
        assigned.add((ti, qi))
        if targets["negative_eligible"][ti]:
            negative = torch.as_tensor(targets["valid"][ti]) & ~support
            if bool(negative.any()):
                instances.append(F.softplus(masks[ti, qi][negative]).mean())
    negatives_e, backgrounds = [], []
    for ti in range(a.shape[0]):
        if not targets["negative_eligible"][ti]:
            continue
        background = torch.as_tensor(targets["background"][ti])
        for qi in range(a.shape[1]):
            if (ti, qi) in assigned:
                continue
            negatives_e.append(F.softplus(a[ti, qi]))
            if bool(background.any()):
                backgrounds.append(F.softplus(masks[ti, qi][background]).mean())
    terms = [torch.stack(values).mean() if values else zero for values in
             (positives_e, negatives_e, positives_m, instances, backgrounds)]
    return torch.stack(terms).sum()


class DirectLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(46)
        torch.use_deterministic_algorithms(True)

    def test_explicit_value_and_gradient_frame(self):
        self._oracle("frame")

    def test_explicit_value_and_gradient_track(self):
        self._oracle("track")

    def _oracle(self, mode):
        labels = target(observations=[(0, 1, [0, 1]), (0, 2, [5]),
                                      (1, 1, [1, 2]), (1, 2, [5, 6])])
        a = torch.randn(2, 4, requires_grad=True)
        masks = torch.randn(2, 4, 3, 4, requires_grad=True)
        actual, metrics, assignments = decoder_loss(a, masks, labels, mode)
        original_grad = torch.autograd.grad(actual, (a, masks))
        aa = a.detach().clone().requires_grad_()
        mm = masks.detach().clone().requires_grad_()
        expected = explicit_oracle(aa, mm, labels, assignments)
        expected_grad = torch.autograd.grad(expected, (aa, mm))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        for left, right in zip(original_grad, expected_grad):
            torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-7)
        self.assertEqual(metrics["object_frames"], 4)
        self.assertEqual(metrics["existence_negative_query_frames"], 4)
        self.assertEqual(metrics["instance_negative_object_frames"], 4)

    def test_overlapping_objects_keep_instance_semantics(self):
        labels = target(t=1, observations=[(0, 1, [0, 1]), (0, 2, [1, 2])])
        a = torch.randn(1, 3, requires_grad=True)
        masks = torch.randn(1, 3, 3, 4, requires_grad=True)
        actual, _, assignments = decoder_loss(a, masks, labels)
        expected = explicit_oracle(a, masks, labels, assignments)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertFalse(labels["background"][0].ravel()[:3].any())

    def test_extremely_low_positive_logits_have_direct_gradient(self):
        labels = target(t=1, observations=[(0, 1, [0])])
        a = torch.full((1, 1), -20.0, requires_grad=True)
        masks = torch.full((1, 1, 3, 4), -20.0, requires_grad=True)
        value, _, _ = decoder_loss(a, masks, labels)
        value.backward()
        self.assertLess(float(a.grad[0, 0]), -0.99)
        self.assertLess(float(masks.grad[0, 0, 0, 0]), -0.99)
        self.assertTrue(bool(torch.isfinite(masks.grad).all()))

    def test_empty_window_provides_negative_gradient(self):
        labels = target()
        a = torch.full((2, 3), 20.0, requires_grad=True)
        masks = torch.full((2, 3, 3, 4), 20.0, requires_grad=True)
        value, metrics, assignments = decoder_loss(a, masks, labels, "track")
        value.backward()
        self.assertTrue(bool((a.grad > 0).all()))
        self.assertTrue(bool((masks.grad > 0).all()))
        self.assertEqual(assignments, [])
        self.assertEqual(metrics["existence_negative_query_frames"], 6)
        self.assertEqual(metrics["object_frames"], 0)

    def test_all_unknown_window_finite_connected_zero(self):
        labels = target(eligible=[False, False])
        a = torch.randn(2, 3, requires_grad=True)
        masks = torch.randn(2, 3, 3, 4, requires_grad=True)
        value, metrics, _ = decoder_loss(a, masks, labels)
        value.backward()
        self.assertEqual(float(value), 0.0)
        self.assertEqual(int(torch.count_nonzero(a.grad)), 0)
        self.assertEqual(int(torch.count_nonzero(masks.grad)), 0)
        self.assertEqual(metrics["existence_negative_query_frames"], 0)

    def test_missing_observation_does_not_become_absence(self):
        labels = target(observations=[(0, 1, [0])], eligible=[True, False])
        for mode in ("frame", "track"):
            a = torch.zeros(2, 2, requires_grad=True)
            masks = torch.zeros(2, 2, 3, 4, requires_grad=True)
            value, metrics, _ = decoder_loss(a, masks, labels, mode)
            value.backward()
            self.assertEqual(metrics["existence_negative_query_frames"], 1)
            self.assertEqual(int(torch.count_nonzero(a.grad[1])), 0)
            self.assertEqual(int(torch.count_nonzero(masks.grad[1])), 0)

    def test_frame_assignments_switch_track_assignment_constant(self):
        labels = target(observations=[(0, 1, [0]), (0, 2, [11]),
                                      (1, 1, [0]), (1, 2, [11])])
        a = torch.zeros(2, 2, requires_grad=True)
        masks = torch.full((2, 2, 3, 4), -8.0)
        masks[0, 0, 0, 0] = 8
        masks[0, 1, 2, 3] = 8
        masks[1, 0, 2, 3] = 8
        masks[1, 1, 0, 0] = 8
        masks.requires_grad_()
        frame_value, frame_metrics, frames = decoder_loss(a, masks, labels, "frame")
        track_value, track_metrics, tracks = decoder_loss(a, masks, labels, "track")
        frame_map = {(r["frame"], r["track_key"][-1]): r["query"] for r in frames}
        track_map = {(r["frame"], r["track_key"][-1]): r["query"] for r in tracks}
        self.assertNotEqual(frame_map[0, "1"], frame_map[1, "1"])
        self.assertEqual(track_map[0, "1"], track_map[1, "1"])
        self.assertEqual(track_map[0, "2"], track_map[1, "2"])
        self.assertLess(float(frame_value), float(track_value))
        self.assertEqual(frame_metrics["existence_negative_query_frames"],
                         track_metrics["existence_negative_query_frames"])

    def test_input_object_order_does_not_change_assignment(self):
        labels = target(observations=[(0, 1, [0]), (0, 2, [11]),
                                      (1, 1, [1]), (1, 2, [10])])
        other = copy.deepcopy(labels)
        other["observations"].reverse()
        a = torch.randn(2, 3)
        masks = torch.randn(2, 3, 3, 4)
        for mode in ("frame", "track"):
            _, m1, r1 = decoder_loss(a, masks, labels, mode)
            _, m2, r2 = decoder_loss(a, masks, other, mode)
            self.assertEqual(r1, r2)
            self.assertEqual(m1["assignment_sha256"], m2["assignment_sha256"])
            self.assertEqual(m1["cost_sha256"], m2["cost_sha256"])

    def test_track_capacity_overflow_fails(self):
        labels = target(observations=[(0, 1, [0]), (1, 2, [11])],
                        eligible=[False, False])
        a, masks = torch.zeros(2, 1), torch.zeros(2, 1, 3, 4)
        decoder_loss(a, masks, labels, "frame")
        with self.assertRaisesRegex(ValueError, "whole-window tracks"):
            decoder_loss(a, masks, labels, "track")

    def test_same_frame_capacity_overflow_fails(self):
        labels = target(t=1, observations=[(0, 1, [0]), (0, 2, [11])])
        with self.assertRaisesRegex(ValueError, "frame objects"):
            decoder_loss(torch.zeros(1, 1), torch.zeros(1, 1, 3, 4), labels)

    def test_duplicate_same_frame_identity_fails(self):
        labels = target(t=1, observations=[(0, 1, [0]), (0, 1, [11])])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            decoder_loss(torch.zeros(1, 3), torch.zeros(1, 3, 3, 4), labels)

    def test_incorrect_unknown_negative_eligibility_fails(self):
        labels = target(observations=[(0, 1, [0])], eligible=[True, True])
        with self.assertRaisesRegex(ValueError, "observe every"):
            decoder_loss(torch.zeros(2, 3), torch.zeros(2, 3, 3, 4), labels)

    def test_background_cannot_include_target_support(self):
        labels = target(t=1, observations=[(0, 1, [0])])
        labels["background"][0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "physical background"):
            decoder_loss(torch.zeros(1, 3), torch.zeros(1, 3, 3, 4), labels)

    def test_no_negative_pixels_is_finite(self):
        labels = target(t=1, h=1, w=1, observations=[(0, 1, [0])])
        a = torch.zeros(1, 2, requires_grad=True)
        masks = torch.zeros(1, 2, 1, 1, requires_grad=True)
        value, metrics, _ = decoder_loss(a, masks, labels)
        value.backward()
        self.assertTrue(bool(torch.isfinite(value)))
        self.assertEqual(metrics["instance_negative_object_frames"], 0)
        self.assertEqual(metrics["background_negative_query_frames"], 0)
        self.assertEqual(metrics["existence_negative_query_frames"], 1)

    def test_nonfinite_assignment_cost_rejected(self):
        labels = target(t=1, observations=[(0, 1, [0])])
        with self.assertRaises(FloatingPointError):
            decoder_loss(torch.full((1, 2), float("nan")), torch.zeros(1, 2, 3, 4), labels)

    def test_nonfinite_empty_window_loss_rejected(self):
        labels = target(t=1)
        with self.assertRaises(FloatingPointError):
            decoder_loss(torch.zeros(1, 2), torch.full((1, 2, 3, 4), float("inf")), labels)

    def test_mask_negative_is_not_diluted_by_extra_unused_queries(self):
        labels = target(t=1, observations=[(0, 1, [0])])
        a = torch.tensor([[5., -5.]])
        masks = torch.full((1, 2, 3, 4), -3.)
        masks[0, 0, 0, 0] = 4
        _, m2, _ = decoder_loss(a, masks, labels)
        aa = torch.cat((a, torch.full((1, 8), -5.)), dim=1)
        mm = torch.cat((masks, torch.full((1, 8, 3, 4), -3.)), dim=1)
        _, m10, _ = decoder_loss(aa, mm, labels)
        self.assertAlmostEqual(m2["mask_instance_negative"], m10["mask_instance_negative"], places=7)


if __name__ == "__main__":
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DirectLossTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"tests": result.testsRun, "failures": len(result.failures),
                      "errors": len(result.errors), "status": "PASS" if result.wasSuccessful() else "FAIL"}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
