import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

E = Path(__file__).resolve().parents[1]
B = E.parents[1]
sys.path.insert(0, str(E))
from train_modules.region_contrast_loss import (pool_regions, region_contrast_loss,
                                                mine_background_regions)
from train_modules.reliable_keep_loss import (actual_reliable_boxes, build_keep_cache,
                                               reliable_keep_loss)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pool_exact_mean_and_overlapping_gradient():
    torch.manual_seed(46)
    a = torch.randn(2, 8, 4, 8, 9, requires_grad=True)
    regions = np.array([[0, 0, 1, 2, 5, 6], [0, 0, 3, 3, 7, 7], [1, 3, 0, 0, 2, 3]])
    got = pool_regions(a, regions)
    direct = F.normalize(torch.stack([a[b, :, t, y0:y1, x0:x1].mean((-2, -1))
                                     for b, t, x0, y0, x1, y1 in regions]), dim=1, eps=1e-8)
    assert torch.allclose(got, direct, atol=2e-7, rtol=2e-6)
    weights = torch.randn_like(got)
    g1, = torch.autograd.grad((got*weights).sum(), a, retain_graph=True)
    g2, = torch.autograd.grad((direct*weights).sum(), a)
    assert torch.allclose(g1, g2, atol=2e-7, rtol=2e-6)
    with pytest.raises(ValueError):
        pool_regions(a, [[0, 0, -.1, 0, 2, 2]])


def test_contrast_full_pair_denominator_and_no_implicit_target_negatives():
    torch.manual_seed(4)
    features = torch.randn(1, 8, 5, 8, 8, requires_grad=True)
    regions = np.array([[0, 0, 0, 0, 2, 2], [0, 1, 0, 0, 2, 2], [0, 2, 0, 0, 2, 2]])
    bg = np.array([[0, 0, 4, 4, 6, 6], [0, 0, 6, 6, 8, 8]])
    pair = np.array([[0, 1]])
    full, _ = region_contrast_loss(features, regions, pair, bg, [[0, 1]])
    half, stats = region_contrast_loss(features, regions, [[0, 1], [0, 2]], bg, [[0, 1], [0, -1]])
    assert torch.allclose(half, full/2)
    assert stats['pairs_requested'] == 2 and stats['pairs_used'] == 1
    # Adding an unused real target changes neither denominator nor negatives.
    without_extra, _ = region_contrast_loss(features, regions[:2], pair, bg, [[0, 1]])
    assert torch.equal(full, without_extra)


@pytest.mark.parametrize('pairs, negatives', [([], np.empty((0, 2), int)),
                                              ([[0, 1]], [[-1, -1]]),
                                              ([[0, 1]], [[0, -1]])])
def test_empty_auxiliary_does_not_change_detection_gradient(pairs, negatives):
    torch.manual_seed(5)
    features = torch.randn(1, 8, 2, 4, 4, requires_grad=True)
    target = [[0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 1]]
    bg = [[0, 0, 3, 3, 4, 4]]
    aux, _ = region_contrast_loss(features, target, pairs, bg, negatives)
    assert torch.isfinite(aux) and aux.item() == 0
    det = features.square().mean()
    a, = torch.autograd.grad(det+aux, features, retain_graph=True)
    b, = torch.autograd.grad(det, features)
    assert torch.equal(a, b)


def test_background_mining_excludes_every_masked_target_and_uses_same_size():
    p = np.zeros((2, 2, 20, 20), np.float32)
    valid = np.ones_like(p, bool)
    # All-category targets, ignored region and strongest false legal candidate
    # are represented by the caller's unified forbidden/valid mask.
    valid[:, :, 6:12, 6:12] = False
    valid[1, 0, 0:4, 0:4] = False
    p[:, :, 6:12, 6:12] = 1
    p[1, 0, 0:4, 0:4] = 1
    p[0, 0, 3:5, 7:9] = .8
    p[1, 0, 14:16, 14:16] = .9
    anchors = [[0, 0, 7, 7, 9, 9]]
    regions, idx = mine_background_regions(p, valid, anchors, [1], local_radius=5)
    assert (idx >= 0).all()
    assert regions[idx[0, 0]].tolist() == [0, 0, 7, 3, 9, 5]
    assert regions[idx[0, 1]].tolist() == [1, 0, 14, 14, 16, 16]
    for b, t, x0, y0, x1, y1 in regions:
        assert valid[b, t, y0:y1, x0:x1].all()
        assert (x1-x0, y1-y0) == (2, 2)
    _, empty = mine_background_regions(p, np.zeros_like(valid), anchors, [1])
    assert (empty == -1).all()


def test_keep_actual_component_match_not_bbox_max_and_invalid_slot():
    matcher = load_module('frozen_component_matcher_for_loss_test',
                          B/'sources/temporal_diagnosis_20261010/analysis/analyze.py')
    prob = np.zeros((2, 40, 40), np.float32)
    # One reliable isolated target and one missed target with a high pixel
    # attached to an outside peak: bbox max alone would misclassify the latter.
    prob[0, 13, 13] = .9
    prob[0, 25, 22:29] = .7
    prob[0, 25, 29] = .95
    anns = [[{'id': 1, 'track_id': 1, 'bbox': [12, 12, 4, 4]},
             {'id': 2, 'track_id': 2, 'bbox': [23, 23, 4, 4]},
             {'id': 3, 'track_id': 3, 'bbox': [30, 30, 3, 3], 'ignore': 1}],
            [{'id': 4, 'track_id': 4, 'bbox': [np.nan, 2, 2, 2]}]]
    result = actual_reliable_boxes(prob, anns, matcher)
    assert result['matched_annotation_ids'] == [[1], []]
    assert result['reliable_target_mask'][0, 12:16, 12:16].all()
    assert not result['reliable_target_mask'][0, 23:27, 23:27].any()
    assert result['forbidden_mask'][0, 21:29, 21:29].all()
    assert result['forbidden_mask'][1].all()


def test_keep_cache_never_distills_missed_targets_or_teacher_false_positives():
    prob = np.full((2, 8, 8), .01, np.float32)
    target = np.zeros_like(prob, bool); target[0, 1:3, 1:3] = True
    forbidden = np.zeros_like(prob, bool); forbidden[0, 2:5, 2:5] = True
    valid_bg = ~target
    prob[1, 4:6, 4:6] = .8
    prob[1, 0, :4] = .09
    one = build_keep_cache(prob, target, valid_bg, forbidden, max_background=10)
    two = build_keep_cache(prob, target, valid_bg, forbidden, max_background=10)
    assert all(np.array_equal(one[k], two[k]) for k in ('coordinates', 'groups', 'teacher_probabilities'))
    coord = tuple(one['coordinates'].T)
    assert not forbidden[coord].any()
    assert np.all(one['teacher_probabilities'][one['groups'] == 1] <= .1)
    assert one['stats']['target_pixels'] == 3
    assert one['stats']['background_hard_pixels'] == 5
    assert len(np.unique(one['coordinates'], axis=0)) == len(one['coordinates'])


def test_keep_fixed_window_group_reduction_teacher_detached_and_empty():
    logits = torch.zeros(2, 1, 2, 2, requires_grad=True)
    teacher = torch.tensor([1., 0.], requires_grad=True)
    loss, _ = reliable_keep_loss(logits, [[0, 0, 0, 0], [0, 0, 0, 1]], teacher, [0, 1])
    assert loss.item() == .125  # (.5*(.25+.25) + 0)/2 windows
    loss.backward()
    assert teacher.grad is None
    assert logits.grad[0, 0, 0, 0] < 0 and logits.grad[0, 0, 0, 1] > 0
    assert logits.grad[1].abs().sum() == 0
    empty, _ = reliable_keep_loss(logits, [], [], [])
    assert torch.isfinite(empty) and empty.item() == 0


def test_contrast_reaches_original_tpro_and_head_and_preserves_forward():
    torch.manual_seed(46)
    torch.set_num_threads(2)
    source = B/'archive/infrared_problem_screen_20261009/official_deeppro'
    sys.path.insert(0, str(source))
    original = load_module('original_deeppro_for_loss_test',
                           source/'networks/models/DeepPro-Plus.py').detector(1, 40, 40).eval()
    for name, p in original.named_parameters():
        p.requires_grad_(name.startswith(('TPro.', 'conv_out1.', 'conv_out2.')))
    for module in original.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for p in module.parameters():
                p.requires_grad_(False)
    x = torch.randn(1, 1, 40, 16, 16)
    feature, logits = original(x)
    assert feature.shape == (1, 8, 40, 16, 16)
    before = logits.detach().clone()
    state = {k: v.clone() for k, v in original.state_dict().items()}
    loss, _ = region_contrast_loss(feature,
        [[0, 0, 2, 2, 5, 5], [0, 4, 4, 4, 7, 7]], [[0, 1]],
        [[0, 0, 10, 10, 13, 13], [0, 0, 1, 10, 4, 13]], [[0, 1]])
    loss.backward()
    assert original.conv_out1[0].weight.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum().item() for n, p in original.named_parameters()
               if n.startswith('TPro.') and p.grad is not None) > 0
    assert all(p.grad is None for n, p in original.named_parameters()
               if n.startswith(('conv_in.', 'layer1.')))
    with torch.no_grad():
        assert torch.equal(original(x)[1], before)
    assert all(torch.equal(state[k], v) for k, v in original.state_dict().items())
