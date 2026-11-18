"""Actual released DeepPro-Plus tests (small spatial tensors, real 40-frame TPro)."""
from __future__ import annotations
import copy
import io
import unittest
import torch
import train_joint as r

SOURCE=r.Q.parent/'official_deeppro'
CHECKPOINT=SOURCE/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'


def released():
    base=r.load_module('joint_test_training_base',r.Q/'top1_background_control_20261011/code/train_pilot.py')
    original,path=base.load_model(SOURCE,CHECKPOINT)
    assert r.sha(path)==r.EXPECTED['model_source']
    assert r.sha(CHECKPOINT)==r.EXPECTED['checkpoint']
    return original.eval()


class ActualPlusJointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r.deterministic_setup()
        cls.original=released()
        cls.x=torch.randn(1,1,40,17,19)

    def test_exact_trainable_names_counts_and_optimizer_groups(self):
        for variant,count in [('H',273),('HL',373)]:
            with self.subTest(variant=variant):
                model=r.wrap_model(copy.deepcopy(self.original),variant)
                expected={'original.'+n for n in r.HEAD_NAMES}
                if variant=='HL':expected|={'feature_refiner.scale','feature_refiner.operator.sigma','feature_refiner.operator.attention.weight'}
                actual={n for n,p in model.named_parameters() if p.requires_grad}
                self.assertEqual(expected,actual)
                self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),count)
                groups=model.parameter_groups()
                self.assertEqual(groups[0]['lr'],1e-5)
                if variant=='HL':self.assertEqual(groups[1]['lr'],1e-4)
                ids=[id(p) for g in groups for p in g['params']]
                self.assertEqual(len(ids),len(set(ids)))
                self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})

    def test_train_eval_both_freeze_all_original_BN(self):
        for variant in r.VARIANTS:
            model=r.wrap_model(copy.deepcopy(self.original),variant)
            for training in (True,False,True):
                model.train(training)
                self.assertFalse(model.original.training)
                for mod in model.modules():
                    if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):
                        self.assertFalse(mod.training)
                        self.assertTrue(all(not p.requires_grad for p in mod.parameters()))

    def test_initial_outputs_exact_and_strict_roundtrip(self):
        with torch.no_grad():baseline=self.original(self.x)
        for variant in r.VARIANTS:
            model=r.wrap_model(copy.deepcopy(self.original),variant).eval()
            with torch.no_grad():actual=model(self.x)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(baseline,actual)))
            memory=io.BytesIO();torch.save(model.state_dict(),memory);memory.seek(0)
            state=torch.load(memory,weights_only=True)
            restored=r.wrap_model(copy.deepcopy(self.original),variant).eval()
            restored.load_state_dict(state,strict=True)
            with torch.no_grad():again=restored(self.x)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,again)))
            state.pop('original.conv_out2.bias')
            with self.assertRaises(RuntimeError):restored.load_state_dict(state,strict=True)

    def test_updates_only_whitelist_and_bypass_is_adapted_head(self):
        for variant in r.VARIANTS:
            model=r.wrap_model(copy.deepcopy(self.original),variant).train()
            fixed={k:v.detach().cpu().clone() for k,v in r.frozen_original_state(model).items()}
            before={k:p.detach().clone() for k,p in model.named_parameters()}
            optimizer=torch.optim.Adam(model.parameter_groups(),weight_decay=1e-4)
            logits=model(self.x)[1]
            logits.square().mean().backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
            if variant=='HL':
                self.assertNotEqual(float(model.feature_refiner.scale.grad),0)
                self.assertTrue(all(torch.count_nonzero(p.grad)==0 for p in model.feature_refiner.operator.parameters()))
            optimizer.step()
            r.assert_frozen(model,fixed)
            for name in r.HEAD_NAMES:
                self.assertFalse(torch.equal(dict(model.named_parameters())['original.'+name],before['original.'+name]),name)
            model.set_inference_mode('bypass')
            with torch.no_grad():
                bypass=model(self.x); adapted=model.original(self.x); baseline=self.original(self.x)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(bypass,adapted)))
            self.assertFalse(torch.equal(bypass[1],baseline[1]))

    def test_LFP_gradients_traverse_frozen_TPro_when_gate_nonzero(self):
        model=r.wrap_model(copy.deepcopy(self.original),'HL').train()
        with torch.no_grad():
            model.feature_refiner.scale.fill_(.1)
            model.feature_refiner.operator.attention.weight.normal_(std=.02)
        model(self.x)[1].square().mean().backward()
        for name,p in model.feature_refiner.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()),name)
            self.assertGreater(float(p.grad.norm()),0,name)
        self.assertTrue(all(p.grad is None for p in model.original.TPro.parameters()))

    def test_pair_has_independent_head_and_adapter_parameters(self):
        adapter=r.wrap_model(copy.deepcopy(self.original),'HL').feature_refiner
        from joint_adaptation import build_pair
        head,joint=build_pair(self.original,adapter)
        self.assertTrue(all(a.data_ptr()!=b.data_ptr() for a,b in zip(head.original.parameters(),joint.original.parameters())))
        self.assertTrue(torch.equal(head.original.conv_out2.weight,joint.original.conv_out2.weight))
        self.assertNotEqual(adapter.scale.data_ptr(),joint.feature_refiner.scale.data_ptr())


if __name__=='__main__':unittest.main(verbosity=2)
