"""CPU contracts against the actual released DeepPro-Plus checkpoint.

TestOnlyPDA is a deterministic injected test operator. These tests validate the
baseline/new-M1 interface and gradients, not native CUDA deformable alignment or
detection quality. No training job, dataset evaluation, or result is generated.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from experiments.m1_source_relations_v1.code import runtime
from experiments.m1_source_relations_v1.code.adapter import (
    DeepProWithSourceRelations,
    SourceRelationsAdapter,
)


class TestOnlyPDA(nn.Module):
    """Small differentiable CPU test double; never a production fallback."""

    def __init__(self):
        super().__init__()
        self.source_scale = nn.Parameter(torch.linspace(0.65, 0.85, 16).view(1, 16, 1, 1))
        self.calls = 0

    def forward(self, source, current):
        self.calls += 1
        return self.source_scale * source + 0.25 * current


class ReleasedBaselineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            cls.base, cls.metadata = runtime.load_baseline(device='cpu')
            generator = torch.Generator(device='cpu').manual_seed(314159)
            cls.images = torch.randn((1, 1, 40, 4, 4), generator=generator)
            with torch.no_grad():
                cls.reference_features, cls.reference_logits = cls.base(cls.images)
                cls.reference_f = cls.base.layer1(cls.base.conv_in(cls.images))
                cls.reference_h = cls.base.TPro(cls.reference_f.permute(0, 3, 4, 1, 2))
        except BaseException:
            torch.set_num_threads(cls.previous_threads)
            raise

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_model(self, *, reduction='relations', readout='simple', mode='history'):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(46)
            adapter = SourceRelationsAdapter(
                pda=TestOnlyPDA(), native_pda=False, reduction_mode=reduction,
                readout=readout, checkpoint_recurrence=False,
            )
        model = DeepProWithSourceRelations(copy.deepcopy(self.base), adapter,
                                           mode=mode, freeze_baseline=True)
        return model.eval()

    def assert_has_nonzero_finite_gradient(self, module, label):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        self.assertTrue(gradients, label + ': no parameter received a gradient')
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients),
                        label + ': nonfinite gradient')
        self.assertTrue(any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients),
                        label + ': every gradient is zero')

    def test_loads_exact_release_and_original_dimensions(self):
        self.assertEqual(self.metadata['baseline_checkpoint_sha256'], runtime.CHECKPOINT_SHA)
        self.assertEqual(self.metadata['baseline_source_sha256'], runtime.MODEL_SOURCE_SHA)
        self.assertTrue(self.metadata['weights_only'])
        self.assertFalse(self.metadata['optimizer_state_restored'])
        self.assertEqual(runtime.tensor_hash(self.base.state_dict()), self.metadata['baseline_state_sha256'])
        self.assertEqual(self.metadata['baseline_parameters'], 70913)
        self.assertEqual(tuple(self.reference_f.shape), (1, 32, 40, 4, 4))
        self.assertEqual(tuple(self.reference_h.shape), (1, 32, 40, 4, 4))
        self.assertEqual(tuple(self.reference_features.shape), (1, 8, 40, 4, 4))
        self.assertEqual(tuple(self.reference_logits.shape), (1, 40, 4, 4))
        self.assertEqual(len(self.base.TPro.QK_heads), 8)
        for head in self.base.TPro.QK_heads:
            self.assertEqual((head.in_features, head.out_features), (40, 40))

    def test_zero_initialization_matches_release_for_all_controls(self):
        # This compares both outputs of the published detector, including raw
        # logits, and covers simple versus selective readout initialization.
        for reduction in ('relations', 'early_merge'):
            for readout in ('simple', 'selective'):
                for mode in ('history', 'current_only'):
                    with self.subTest(reduction=reduction, readout=readout, mode=mode):
                        model = self.make_model(reduction=reduction, readout=readout, mode=mode)
                        with torch.no_grad():
                            features, logits = model(self.images)
                        self.assertTrue(torch.equal(features, self.reference_features))
                        self.assertTrue(torch.equal(logits, self.reference_logits))
                        self.assertEqual(runtime.tensor_hash(model.base.state_dict()),
                                         self.metadata['baseline_state_sha256'])

    def test_preserves_frontend_and_calls_the_unique_tpro_once(self):
        model = self.make_model()
        observed = {'tpro_calls': 0}

        def capture_tpro(module, args, output):
            observed['tpro_calls'] += 1
            observed['tpro_input'] = args[0].detach().clone()
            observed['tpro_output'] = output.detach().clone()

        def capture_adapter(module, args):
            observed['adapter_f'] = args[0].detach().clone()
            observed['adapter_h'] = args[1].detach().clone()

        handles = [model.base.TPro.register_forward_hook(capture_tpro),
                   model.adapter.register_forward_pre_hook(capture_adapter)]
        try:
            with torch.no_grad():
                features, logits = model(self.images)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(observed['tpro_calls'], 1)
        self.assertEqual(tuple(observed['tpro_input'].shape), (1, 4, 4, 32, 40))
        self.assertTrue(torch.equal(observed['tpro_input'], self.reference_f.permute(0, 3, 4, 1, 2)))
        self.assertTrue(torch.equal(observed['tpro_output'], self.reference_h))
        self.assertTrue(torch.equal(observed['adapter_f'], self.reference_f))
        self.assertTrue(torch.equal(observed['adapter_h'], self.reference_h))
        self.assertEqual(tuple(logits.shape), (1, 40, 4, 4))
        self.assertTrue(torch.equal(features, self.reference_features))

    def test_bypass_skips_new_path_even_with_nonzero_output_weights(self):
        for readout in ('simple', 'selective'):
            with self.subTest(readout=readout):
                model = self.make_model(readout=readout)
                projection = model.adapter.out_projection if readout == 'simple' else model.adapter.reader.delta
                with torch.no_grad():
                    projection.weight.fill_(0.125)
                    projection.bias.fill_(0.25)
                model.adapter.enabled = False

                def reject_projection(module, args):
                    self.fail('Bypass must not execute M1 input projection')

                handle = model.adapter.in_projection.register_forward_pre_hook(reject_projection)
                try:
                    with torch.no_grad():
                        features, logits = model(self.images)
                finally:
                    handle.remove()
                self.assertEqual(model.adapter.recurrent_cell.pda.calls, 0)
                self.assertTrue(torch.equal(features, self.reference_features))
                self.assertTrue(torch.equal(logits, self.reference_logits))

    def test_detection_loss_reaches_new_modules_while_baseline_stays_frozen(self):
        for readout in ('simple', 'selective'):
            with self.subTest(readout=readout):
                model = self.make_model(readout=readout)
                model.train()
                self.assertTrue(model.adapter.training)
                self.assertFalse(model.base.training)
                for module in model.base.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        self.assertFalse(module.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in model.base.parameters()))
                before = runtime.tensor_hash(model.base.state_dict())
                # A zero final residual blocks upstream gradients by design.
                # Open that path explicitly without an optimizer/training job.
                projection = model.adapter.out_projection if readout == 'simple' else model.adapter.reader.delta
                with torch.no_grad():
                    values = torch.linspace(-0.025, 0.025, projection.weight.numel())
                    projection.weight.copy_(values.reshape_as(projection.weight))
                    projection.bias.fill_(0.01)
                _, logits = model(self.images)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                self.assert_has_nonzero_finite_gradient(model.adapter.in_projection, 'new input projection')
                self.assert_has_nonzero_finite_gradient(model.adapter.reducer, 'new source reducer')
                self.assert_has_nonzero_finite_gradient(model.adapter.recurrent_cell.pda, 'injected test-only PDA')
                self.assert_has_nonzero_finite_gradient(projection, 'new output projection')
                if readout == 'selective':
                    self.assert_has_nonzero_finite_gradient(model.adapter.reader.gate, 'new M2 gate')
                self.assertTrue(all(parameter.grad is None for parameter in model.base.parameters()))
                self.assertEqual(runtime.tensor_hash(model.base.state_dict()), before)
                self.assertEqual(before, self.metadata['baseline_state_sha256'])


if __name__ == '__main__':
    unittest.main()
