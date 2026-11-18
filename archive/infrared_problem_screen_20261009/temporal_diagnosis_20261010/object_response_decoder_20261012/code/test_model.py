"""Bounded implementation checks; fake backbone by default, no formal training.

Run with the frozen Torch environment. Optional --device cuda additionally
exercises the same checks on CUDA. Actual released-backbone gates belong to the
runner so checkpoint loading stays independent of this architecture module.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from model import ObjectResponseDecoder, xy_encoding


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv3d(1, 8, 3, padding=1),
                                      nn.BatchNorm3d(8), nn.ReLU())
        self.head = nn.Conv3d(8, 1, 1)

    def forward(self, raw):
        H0 = self.features(raw)
        return H0, self.head(H0).squeeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(46)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(46)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    results = []

    def check(name, condition, **detail):
        if not bool(condition):
            raise AssertionError(f'{name}: {detail}')
        results.append(dict(name=name, passed=True, **detail))

    def close(name, actual, expected, atol=1e-5, rtol=1e-5):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        check(name, True, max_absolute_error=float((actual - expected).abs().max()))

    model = ObjectResponseDecoder(FakeBackbone(), spatial_frame_chunk=2).to(device).train()
    raw = torch.randn(1, 1, 4, 28, 36, device=device)
    origin, shape = (7, 11), (51, 73)
    original_state = {key: value.clone() for key, value in model.original.state_dict().items()}
    check('original_eval_under_wrapper_train', not model.original.training)
    check('original_all_parameters_frozen', all(not p.requires_grad for p in model.original.parameters()))
    output = model(raw, temporal_enabled=True, origin=origin, full_shape=shape)
    check('feature_shape', output['E'].shape == (1, 4, 32, 28, 36))
    check('query_shape', output['q'].shape == (1, 4, 100, 32))
    check('presence_shape', output['a'].shape == (1, 4, 100))
    check('original_logit_shape', output['z0'].shape == (1, 4, 28, 36))
    with torch.no_grad():
        H0, old_logit = model.original(raw)
        close('original_path_exact', output['z0'], old_logit, atol=0, rtol=0)
        E_tiled, z_tiled = model.tiled_features(raw, core=12, halo=4, origin=origin, full_shape=shape)
        close('nonzero_full_tiled_feature_equivalence', E_tiled, output['E'])
        close('nonzero_full_tiled_original_logit_equivalence', z_tiled, output['z0'])
        q_tiled = model.decode(E_tiled, temporal_enabled=True)
        close('full_tiled_global_query_equivalence', q_tiled, output['q'], atol=3e-5, rtol=1e-5)
        masks = model.mask_logits(output['E'], output['q'])
        chunk_masks = model.mask_logits(output['E'], output['q'], spatial_chunk=73)
        close('dense_mask_spatial_chunk_equivalence', masks, chunk_masks)
        dense_probability = (output['a'].sigmoid()[..., None, None] * masks.sigmoid()).max(dim=2).values
        stream = model.response_map(output['E'], output['q'], query_chunk=13,
                                    spatial_chunk=127, frame_chunk=3)
        close('dense_streaming_probability_equivalence', stream, dense_probability)
        tile_stream = model.response_map(E_tiled, q_tiled, query_chunk=11,
                                         spatial_chunk=139, frame_chunk=1)
        close('full_tiled_complete_probability_equivalence', stream, tile_stream, atol=3e-5)
        check('full_tiled_half_threshold_equivalence', torch.equal(stream >= 0.5, tile_stream >= 0.5),
              foreground_pixels=int((stream >= 0.5).sum()))
        old_chunk = model.spatial_frame_chunk
        model.spatial_frame_chunk = 1
        alternate_q = model.decode(output['E'], temporal_enabled=True)
        close('frame_attention_chunk_equivalence', alternate_q, output['q'], atol=3e-5)
        model.spatial_frame_chunk = old_chunk

        isolated = output['E'].clone()
        changed = isolated.clone()
        changed[:, 0] += torch.randn_like(changed[:, 0]) * 3
        q0 = model.decode(isolated, temporal_enabled=False)
        q0_changed = model.decode(changed, temporal_enabled=False)
        close('R0_new_decoder_has_no_cross_frame_communication', q0[:, 1:], q0_changed[:, 1:], atol=0, rtol=0)
        q1 = model.decode(isolated, temporal_enabled=True)
        q1_changed = model.decode(changed, temporal_enabled=True)
        difference = float((q1[:, 1:] - q1_changed[:, 1:]).abs().max())
        check('R1_temporal_communication_active', difference > 1e-6, max_change=difference)

    zero_content = copy.deepcopy(model)
    with torch.no_grad():
        for p in zero_content.raw_encoder.parameters():
            p.zero_()
        for p in zero_content.project.parameters():
            p.zero_()
        whole = zero_content.features(torch.zeros(1, 1, 1, 51, 73, device=device),
                                      torch.zeros(1, 8, 1, 51, 73, device=device), full_shape=shape)
        crop = zero_content.features(raw[:, :, :1], H0[:, :, :1], origin=origin, full_shape=shape)
        close('crop_position_encoding_is_full_image_subfield',
              crop, whole[..., 7:35, 11:47], atol=0, rtol=0)
        shifted = zero_content.features(raw[:, :, :1], H0[:, :, :1], origin=(8, 12), full_shape=shape)
        check('absolute_origin_changes_position_encoding', float((crop - shifted).abs().max()) > 0.01)
        anchors = model.anchors_yx
        check('100_unique_full_image_anchor_positions', len(torch.unique(anchors, dim=0)) == 100)
        check('anchor_centers_inside_image', bool(((anchors > 0) & (anchors < 1)).all()))

    for temporal, label in ((False, 'R0'), (True, 'R1')):
        model.zero_grad(set_to_none=True)
        small = model(raw[..., :12, :16], temporal_enabled=temporal,
                      origin=origin, full_shape=shape)
        logits = model.mask_logits(small['E'], small['q'])
        target_presence = torch.zeros_like(small['a'])
        target_presence[..., 0] = 1
        target_masks = torch.zeros_like(logits)
        target_masks[..., 0, 5:8, 6:9] = 1
        loss = F.binary_cross_entropy_with_logits(small['a'], target_presence)
        loss = loss + F.binary_cross_entropy_with_logits(logits, target_masks)
        loss.backward()
        check(f'{label}_direct_presence_gradient', model.presence_head.weight.grad is not None and
              float(model.presence_head.weight.grad.abs().max()) > 0)
        check(f'{label}_direct_mask_gradient', model.mask_head.weight.grad is not None and
              float(model.mask_head.weight.grad.abs().max()) > 0)
        check(f'{label}_new_raw_path_gradient', model.raw_encoder[0].weight.grad is not None and
              float(model.raw_encoder[0].weight.grad.abs().max()) > 0)
        check(f'{label}_query_gradient', model.query_embed.grad is not None and
              float(model.query_embed.grad.abs().max()) > 0)
        check(f'{label}_finite_new_gradients', all(torch.isfinite(p.grad).all()
              for n, p in model.named_parameters() if p.grad is not None))
        check(f'{label}_original_no_gradient', all(p.grad is None for p in model.original.parameters()))
        temporal_grad = [p.grad for layer in model.layers for p in layer.temporal_attention.parameters()]
        check(f'{label}_temporal_gradient_matches_switch',
              all(g is not None for g in temporal_grad) if temporal else all(g is None for g in temporal_grad))
    for key, value in model.original.state_dict().items():
        close(f'original_state_unchanged_{key}', value, original_state[key], atol=0, rtol=0)

    model.zero_grad(set_to_none=True)
    empty = model(raw[..., :8, :8], temporal_enabled=True, origin=origin, full_shape=shape)
    empty['a'].retain_grad()
    F.binary_cross_entropy_with_logits(empty['a'], torch.zeros_like(empty['a'])).backward()
    check('empty_window_presence_has_negative_supervision', bool((empty['a'].grad > 0).all()))
    report = dict(status='PASS', device=str(device), checks=results,
                  optimizer_steps=0, test_scope='synthetic fake-backbone implementation only',
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  config=model.config,
                  source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__), Path(__file__).with_name('model.py'))})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
