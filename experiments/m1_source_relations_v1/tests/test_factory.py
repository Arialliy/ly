"""CPU construction with the real fresh native-PDA factory, without any forward.

These checks exercise native import, checkpoint provenance and initialization.
They do not test the CUDA kernel, detector accuracy, training or dataset access.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from experiments.m1_source_relations_v1.code import runtime


class FreshNativeFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.loads = []
        cls.models = {}
        cls.metadata = {}
        actual_load = torch.load

        def release_only(path, *args, **kwargs):
            resolved = Path(path).resolve()
            if resolved != runtime.CHECKPOINT.resolve():
                raise AssertionError('Factory tried to load a non-release checkpoint: ' + str(path))
            if kwargs.get('weights_only') is not True or kwargs.get('map_location') != 'cpu':
                raise AssertionError('The release must use weights_only=True and CPU loading')
            cls.loads.append(resolved)
            return actual_load(path, *args, **kwargs)

        try:
            # A factory must construct native modules, never execute them. This
            # rejects every nn.Module forward, including accidental CUDA calls.
            with mock.patch('torch.load', side_effect=release_only), mock.patch.object(
                    nn.Module, '__call__', side_effect=AssertionError('Model construction executed a forward')):
                for condition in runtime.CONDITIONS:
                    model, metadata = runtime.load_initial(condition, device='cpu', seed=46)
                    cls.models[condition] = model
                    cls.metadata[condition] = metadata
        except BaseException:
            torch.set_num_threads(cls.previous_threads)
            raise

    @classmethod
    def tearDownClass(cls):
        cls.models.clear()
        torch.set_num_threads(cls.previous_threads)

    def test_six_factories_read_only_the_official_release(self):
        self.assertEqual(len(self.models), 6)
        self.assertEqual(self.loads, [runtime.CHECKPOINT.resolve()] * 6)
        for condition, model in self.models.items():
            with self.subTest(condition=condition):
                metadata = self.metadata[condition]
                self.assertEqual(metadata['baseline_checkpoint_sha256'], runtime.CHECKPOINT_SHA)
                self.assertFalse(metadata['optimizer_state_restored'])
                self.assertFalse(metadata['training_started'])
                self.assertFalse(metadata['native_GPU_execution_verified'])
                self.assertTrue(model.adapter.native_pda)
                self.assertEqual(type(model.adapter.recurrent_cell.pda).__module__,
                                 '_deeppro_m1_source_relations_rfr.pda')
                self.assertTrue(all(parameter.device.type == 'cpu' for parameter in model.parameters()))
                self.assertEqual(model.mode, runtime.CONFIGS[condition]['mode'])
                self.assertEqual(model.adapter.reduction_mode, runtime.CONFIGS[condition]['reduction_mode'])
                self.assertEqual(model.adapter.readout, runtime.CONFIGS[condition]['readout'])

    def test_matched_conditions_start_from_identical_states(self):
        for suffix in ('', 'M2'):
            conditions = [name + suffix for name in ('CurrentRelations', 'HistoryEarlyMerge', 'HistoryRelations')]
            reference = self.metadata[conditions[0]]
            for condition in conditions[1:]:
                with self.subTest(condition=condition):
                    current = self.metadata[condition]
                    for key in ('new_parameter_sha256', 'new_state_sha256', 'model_state_sha256'):
                        self.assertEqual(reference[key], current[key], key)
                    reference_state = self.models[conditions[0]].state_dict()
                    current_state = self.models[condition].state_dict()
                    self.assertEqual(reference_state.keys(), current_state.keys())
                    self.assertTrue(all(torch.equal(value, current_state[name])
                                        for name, value in reference_state.items()))

    def test_simple_and_m2_share_exact_initial_pda_projection_and_reducer(self):
        for name in ('CurrentRelations', 'HistoryEarlyMerge', 'HistoryRelations'):
            simple = self.models[name].adapter
            selective = self.models[name + 'M2'].adapter
            for label, left, right in (
                    ('PDA', simple.recurrent_cell.pda, selective.recurrent_cell.pda),
                    ('input projection', simple.in_projection, selective.in_projection),
                    ('source reducer', simple.reducer, selective.reducer)):
                with self.subTest(condition=name, component=label):
                    self.assertEqual(runtime.tensor_hash(left.state_dict()), runtime.tensor_hash(right.state_dict()))
                    self.assertIsNot(left, right)
                    self.assertEqual(list(left.state_dict()), list(right.state_dict()))
                    for key, value in left.state_dict().items():
                        self.assertTrue(torch.equal(value, right.state_dict()[key]))

    def test_seed_changes_only_new_parameters(self):
        for condition in ('HistoryRelations', 'HistoryRelationsM2'):
            with self.subTest(condition=condition):
                model, metadata = runtime.load_initial(condition, device='cpu', seed=47)
                same_seed, same_metadata = runtime.load_initial(condition, device='cpu', seed=46)
                original = self.metadata[condition]
                self.assertNotEqual(metadata['new_parameter_sha256'], original['new_parameter_sha256'])
                self.assertEqual(same_metadata['new_parameter_sha256'], original['new_parameter_sha256'])
                for result in (metadata, same_metadata):
                    self.assertEqual(result['baseline_state_sha256'], original['baseline_state_sha256'])
                self.assertEqual(runtime.tensor_hash(model.base.state_dict()),
                                 runtime.tensor_hash(self.models[condition].base.state_dict()))
                del model, same_seed

    def test_repeated_construction_restores_callers_cpu_rng(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261015)
            before = torch.get_rng_state().clone()
            for condition, seed in (('HistoryRelations', 46), ('HistoryEarlyMergeM2', 47),
                                    ('CurrentRelations', 46)):
                model, _ = runtime.load_initial(condition, device='cpu', seed=seed)
                self.assertTrue(torch.equal(torch.get_rng_state(), before), condition)
                del model

    def test_original_baseline_and_heads_are_unchanged(self):
        base, metadata = runtime.load_baseline('cpu')
        reference_heads = {name: value for name, value in base.state_dict().items()
                           if name.startswith(('conv_out1.', 'conv_out2.'))}
        self.assertEqual(metadata['baseline_parameters'], 70913)
        for condition, model in self.models.items():
            with self.subTest(condition=condition):
                self.assertEqual(sum(parameter.numel() for parameter in model.base.parameters()), 70913)
                self.assertEqual(runtime.tensor_hash(model.base.state_dict()), metadata['baseline_state_sha256'])
                heads = {name: value for name, value in model.base.state_dict().items()
                         if name.startswith(('conv_out1.', 'conv_out2.'))}
                self.assertEqual(runtime.tensor_hash(heads), runtime.tensor_hash(reference_heads))
                self.assertTrue(all(not parameter.requires_grad for parameter in model.base.parameters()))
                model.train()
                self.assertFalse(model.base.training)
                self.assertTrue(all(not module.training for module in model.base.modules()
                                    if isinstance(module, nn.modules.batchnorm._BatchNorm)))
                model.eval()
                projection = (model.adapter.out_projection if model.adapter.readout == 'simple'
                              else model.adapter.reader.delta)
                self.assertTrue(torch.equal(projection.weight, torch.zeros_like(projection.weight)))
                self.assertTrue(torch.equal(projection.bias, torch.zeros_like(projection.bias)))


if __name__ == '__main__':
    unittest.main()
