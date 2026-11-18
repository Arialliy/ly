"""S1 geometry gate. No detector, training, checkpoint, or test-set evaluation.

CPU operator checks are runnable without the release images. A complete gate
also needs a real training batch, the explicitly chosen formal velocity grid,
and an actual CUDA memory-budget measurement. Missing requirements are BLOCKED,
never silently counted as PASS. Exit codes: 0 PASS, 1 FAIL, 2 BLOCKED.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import unittest

import numpy as np
import torch
from torch.nn import functional as F
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.path.insert(0, str(PROJECT))

from experiments.irstd_independent_v1.contracts import ClipBatch, SceneMeta
from experiments.irstd_independent_v1.geometry import (
    MissingHaloError, make_velocity_grid, sample_clip_features, sample_context,
)
from experiments.irstd_independent_v1.geometry.tiles import dense_queries, plan_tiles
from experiments.irstd_independent_v1.checks.s1_runtime import (
    check_real_batch, profile_sampler, verify_s0,
)

METRICS: dict = {}
DEVICE = "cpu"
ATOL, RTOL = 1e-5, 1e-4


def reference_bilinear(features, offsets, queries, velocities, valid=None,
                       image_hw=None, origin_xy=None, source_hw=None):
    """Independent scalar NumPy enumeration, with no production grid helpers.

    Accumulates in float64 and returns an explicit positive-support validity.
    This intentionally does not use grid_sample or a vectorized Torch formula.
    """
    f = np.asarray(features, dtype=np.float64)
    bsz, times, channels, height, width = f.shape
    q = np.asarray(queries, dtype=np.float64)
    if q.ndim == 2:
        q = np.repeat(q[None], bsz, axis=0)
    v = np.asarray(velocities, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    if valid is None:
        valid = np.ones((bsz, times, 1, height, width), dtype=bool)
    if image_hw is None:
        image_hw = [(height, width)] * bsz
    if origin_xy is None:
        origin_xy = [(0, 0)] * bsz
    if source_hw is None:
        source_hw = image_hw
    values = np.zeros((bsz, q.shape[1], channels, times, *v.shape[:2]))
    mask = np.zeros((bsz, q.shape[1], times, *v.shape[:2]), dtype=bool)
    for b in range(bsz):
        ih, iw = image_hw[b]
        sh, sw = source_hw[b]
        ox, oy = origin_xy[b]
        for n, (px, py) in enumerate(q[b]):
            if not (0 <= px <= iw - 1 and 0 <= py <= ih - 1):
                continue
            for t, dt in enumerate(offsets[b]):
                for iy in range(v.shape[0]):
                    for ix in range(v.shape[1]):
                        x = float(px + dt * v[iy, ix, 0])
                        y = float(py + dt * v[iy, ix, 1])
                        if not (0 <= x <= iw - 1 and 0 <= y <= ih - 1):
                            continue
                        x, y = x - ox, y - oy
                        if not (0 <= x <= sw - 1 and 0 <= y <= sh - 1):
                            raise RuntimeError("Reference encountered missing tile halo")
                        x0, y0 = int(np.floor(x)), int(np.floor(y))
                        dx, dy = x - x0, y - y0
                        support = [(x0, y0, (1-dx)*(1-dy)),
                                   (x0+1, y0, dx*(1-dy)),
                                   (x0, y0+1, (1-dx)*dy),
                                   (x0+1, y0+1, dx*dy)]
                        support = [(xx, yy, w) for xx, yy, w in support if w > 0]
                        if not all(bool(valid[b, t, 0, yy, xx]) for xx, yy, _ in support):
                            continue
                        mask[b, n, t, iy, ix] = True
                        for xx, yy, w in support:
                            values[b, n, :, t, iy, ix] += w * f[b, t, :, yy, xx]
    return values, mask


def cpu(tensor):
    return tensor.detach().cpu().numpy()


def fixture_clip(sizes=((9, 11), (7, 8)), channels=2):
    """Only an in-memory ClipBatch contract fixture; not a release-data test."""
    h, w = max(s[0] for s in sizes), max(s[1] for s in sizes)
    frames = torch.rand(len(sizes), 17, 1, h, w, device=DEVICE)
    features = torch.rand(len(sizes), 17, channels, h, w, device=DEVICE)
    mask = torch.zeros_like(frames, dtype=torch.bool)
    metas = []
    for b, (ih, iw) in enumerate(sizes):
        mask[b, :, :, :ih, :iw] = True
        metas.append(SceneMeta(f"fixture{b}", tuple(range(1, 18)), (ih, iw),
                               (0, 0), (0, 0, iw, ih)))
    clip = ClipBatch(frames, torch.arange(-8, 9, device=DEVICE).float()[None].repeat(len(sizes), 1),
                     mask, 8, tuple(f"fixture{b}/00009" for b in range(len(sizes))), tuple(metas))
    return clip, features


class GeometryChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(46)

    def close(self, actual, expected, atol=None, rtol=None):
        atol=ATOL if atol is None else atol
        rtol=RTOL if rtol is None else rtol
        torch.testing.assert_close(actual, torch.as_tensor(expected, device=actual.device,
                                                        dtype=actual.dtype), atol=atol, rtol=rtol)

    def test_01_velocity_axis_and_units(self):
        grid = make_velocity_grid([-1.5, 0, .5], [-2, 0, 1], device=DEVICE)
        expected = [[[x, y] for x in [-1.5, 0, .5]] for y in [-2, 0, 1]]
        self.close(grid, expected, atol=0, rtol=0)
        self.assertEqual(tuple(grid.shape), (3, 3, 2))
        for bad in ([1, 2], [0, 0], [0, -1], [float("nan"), 0], []):
            with self.assertRaises((ValueError, TypeError)):
                make_velocity_grid(bad, [0], device=DEVICE)

    def test_02_independent_scalar_reference(self):
        f = torch.rand(2, 3, 2, 11, 13, device=DEVICE)
        times = torch.tensor([[-3, -1, 2], [-2, 1, 4]], device=DEVICE)
        q = torch.tensor([[0, 0], [12, 10], [5.25, 4.125], [8, 7], [1, 9]], device=DEVICE)
        v = make_velocity_grid([-.5, 0, .75], [-.25, 0, .5], device=DEVICE)
        mask = torch.ones(2, 3, 1, 11, 13, dtype=torch.bool, device=DEVICE)
        mask[0, 1, 0, 4, 5] = False
        got = sample_context(f, times, q, v, mask, query_chunk=2, velocity_chunk=4)
        ref, ok = reference_bilinear(cpu(f), cpu(times), cpu(q), cpu(v), cpu(mask))
        self.close(got.values, ref)
        self.assertTrue(np.array_equal(cpu(got.sample_valid), ok))
        self.assertTrue(np.array_equal(cpu(got.velocity_valid), ok.all(axis=2)))
        METRICS["scalar_reference"] = {"values_compared": int(ref.size),
            "valid_samples": int(ok.sum()), "max_absolute_error": float(np.max(np.abs(cpu(got.values)-ref)))}

    def test_03_analytic_plane_and_actual_time_offsets(self):
        yy, xx = torch.meshgrid(torch.arange(15, device=DEVICE), torch.arange(17, device=DEVICE), indexing="ij")
        f = torch.stack([.01*xx + .03*yy + .1*t for t in range(3)])[None, :, None]
        times = torch.tensor([[-3, 1, 2]], device=DEVICE)
        q = torch.tensor([[8.25, 7.5]], device=DEVICE)
        v = make_velocity_grid([-.5, 0, .5], [-.25, 0, .25], device=DEVICE)
        got = sample_context(f, times, q, v)
        for t, dt in enumerate([-3, 1, 2]):
            wanted = .01*(q[0, 0]+dt*v[..., 0])+.03*(q[0, 1]+dt*v[..., 1])+.1*t
            self.close(got.values[0, 0, 0, t], wanted)
        self.assertTrue(bool(got.sample_valid.all()))

    def test_04_known_moving_bright_and_dark_points(self):
        grid = make_velocity_grid([-1, -.5, 0, .5, 1], [-1, -.5, 0, .5, 1], device=DEVICE)
        times = torch.tensor([[-4, -2, 1, 4]], device=DEVICE)
        yy, xx = torch.meshgrid(torch.arange(41, device=DEVICE), torch.arange(41, device=DEVICE), indexing="ij")
        q = torch.tensor([[20., 20.]], device=DEVICE)
        cases = [(-1., 0.), (0., 1.), (.5, -.5), (0., 0.), (-.5, -1.), (1., 1.)]
        for vx, vy in cases:
            blob = torch.stack([torch.exp(-((xx-20-dt*vx)**2+(yy-20-dt*vy)**2)/2)
                                for dt in [-4, -2, 1, 4]])[None, :, None]
            for sign in [1, -1]:
                f = .5 + .4*sign*blob
                out = sample_context(f, times, q, grid)
                response = sign*out.values[0, 0, 0].mean(0)
                chosen = grid.reshape(-1, 2)[response.flatten().argmax()]
                self.close(chosen, [vx, vy], atol=0, rtol=0)
        METRICS["known_motion_cases"] = len(cases)*2

    def test_05_zero_velocity_preserves_time_and_excludes_center(self):
        clip, f = fixture_clip()
        f = f.detach().requires_grad_()
        grid = make_velocity_grid([0], [0], device=DEVICE)
        q = torch.tensor([[2, 3], [5, 4]], device=DEVICE)
        out = sample_clip_features(f, clip, q, grid)
        keep = list(range(8)) + list(range(9, 17))
        self.assertEqual(tuple(out.values.shape), (2, 2, 2, 16, 1, 1))
        self.close(out.time_offsets, [list(range(-8, 0))+list(range(1, 9))]*2, atol=0, rtol=0)
        self.close(out.values[:, 0, :, :, 0, 0], f[:, keep, :, 3, 2].permute(0, 2, 1), atol=0, rtol=0)
        out.values.sum().backward()
        self.assertEqual(int(torch.count_nonzero(f.grad[:, 8])), 0)
        self.assertGreater(int(torch.count_nonzero(f.grad[:, keep])), 0)

    def test_06_heterogeneous_padding_is_not_an_observation(self):
        clip, f = fixture_clip()
        v = make_velocity_grid([-.25, 0, .25], [-.25, 0, .25], device=DEVICE)
        q = torch.tensor([[7, 6], [10, 8], [3.25, 4.25]], device=DEVICE)
        a = sample_clip_features(f, clip, q, v)
        poisoned = torch.where(clip.pixel_valid, f, torch.full_like(f, 1e6)).requires_grad_()
        b = sample_clip_features(poisoned, clip, q, v)
        self.close(a.values, b.values, atol=0, rtol=0)
        self.assertFalse(bool(b.sample_valid[1, 1].any()))
        b.values.sum().backward()
        invalid = ~clip.pixel_valid.expand_as(poisoned)
        self.assertEqual(int(torch.count_nonzero(poisoned.grad[invalid])), 0)

    def test_07_fp32_integer_pixel_padding_poison_regression(self):
        # W=323 exposes normalized-grid round-trip errors at integer pixels.
        w = 323
        f = torch.full((w, 1, 1, 1, w), 1e6, device=DEVICE)
        valid = torch.zeros_like(f, dtype=torch.bool)
        rows = torch.arange(w, device=DEVICE)
        f[rows, 0, 0, 0, rows] = 1
        valid[rows, 0, 0, 0, rows] = True
        f.requires_grad_()
        q = torch.stack([rows, torch.zeros_like(rows)], -1).float()[:, None]
        out = sample_context(f, torch.zeros(w, 1, device=DEVICE), q,
                             make_velocity_grid([0], [0], device=DEVICE), valid)
        self.assertTrue(bool(out.sample_valid.all()))
        self.close(out.values, torch.ones_like(out.values), atol=0, rtol=0)
        out.values.sum().backward()
        self.assertEqual(int(torch.count_nonzero(f.grad[~valid])), 0)
        METRICS["fp32_integer_padding_regression"] = {"queries": w, "max_error": 0.0,
                                                      "invalid_pixel_nonzero_gradients": 0}

    def test_08_positive_weight_footprint_and_mask_holes(self):
        f = torch.arange(25, device=DEVICE).float().reshape(1, 1, 1, 5, 5)
        mask = torch.ones_like(f, dtype=torch.bool)
        mask[..., 2, 3] = False
        q = torch.tensor([[2., 2.], [2.25, 2.], [2., 2.25], [4., 4.]], device=DEVICE)
        out = sample_context(f, torch.zeros(1, 1, device=DEVICE), q,
                             make_velocity_grid([0], [0], device=DEVICE), mask)
        self.assertEqual(out.sample_valid.flatten().tolist(), [True, False, True, True])
        self.close(out.values.flatten(), [12, 0, 13.25, 24], atol=0, rtol=0)

    def test_09_true_image_boundary_and_zero_or_one_valid_path(self):
        f = torch.ones(1, 2, 1, 3, 3, device=DEVICE)
        times = torch.tensor([[-1, 1]], device=DEVICE)
        q = torch.tensor([[0, 0], [-1, 0], [2, 2]], device=DEVICE)
        out = sample_context(f, times, q, make_velocity_grid([-1, 0, 1], [-1, 0, 1], device=DEVICE))
        self.assertEqual(out.velocity_valid.sum((-1, -2)).tolist(), [[1, 0, 1]])
        self.assertTrue(bool(torch.isfinite(out.values).all()))
        self.assertEqual(int(torch.count_nonzero(out.values[:, 1])), 0)

    def test_10_missing_source_halo_is_an_error(self):
        f = torch.ones(1, 1, 1, 4, 4, device=DEVICE)
        args = dict(features=f, time_offsets=torch.ones(1, 1, device=DEVICE),
                    velocities_xy=make_velocity_grid([0, 2], [0], device=DEVICE),
                    image_hw=(10, 10), origin_xy=(3, 3), source_hw=(4, 4))
        with self.assertRaises(MissingHaloError):
            sample_context(queries_xy=torch.tensor([[6, 4]], device=DEVICE), **args)
        # In contrast, leaving the true image is a normal invalid observation.
        args.update(origin_xy=(6, 3))
        out = sample_context(queries_xy=torch.tensor([[9, 4]], device=DEVICE), **args)
        self.assertEqual(out.sample_valid.flatten().tolist(), [True, False])

    def test_11_chunked_values_masks_and_gradients(self):
        f = torch.rand(1, 3, 2, 9, 11, device=DEVICE, requires_grad=True)
        times = torch.tensor([[-3, 1, 2]], device=DEVICE)
        q = dense_queries((0, 0, 11, 9), device=DEVICE)
        v = make_velocity_grid([-.5, 0, .5], [-.25, 0, .25], device=DEVICE)
        a = sample_context(f, times, q, v, query_chunk=1000, velocity_chunk=100)
        ga, = torch.autograd.grad(a.values.square().sum(), f)
        b = sample_context(f, times, q, v, query_chunk=7, velocity_chunk=2)
        gb, = torch.autograd.grad(b.values.square().sum(), f)
        self.close(a.values, b.values, atol=0, rtol=0)
        self.assertTrue(torch.equal(a.sample_valid, b.sample_valid))
        self.assertTrue(torch.equal(a.velocity_valid, b.velocity_valid))
        self.close(ga, gb)
        METRICS["chunk_gradient_max_error"] = float((ga-gb).abs().max())

    def test_12_feature_gradient_independent_scatter(self):
        f = torch.rand(1, 1, 2, 5, 6, dtype=torch.float64, device=DEVICE, requires_grad=True)
        q = torch.tensor([[1.25, 2.5], [3.5, 1.25]], dtype=torch.float64, device=DEVICE)
        out = sample_context(f, torch.zeros(1, 1, device=DEVICE), q,
                             make_velocity_grid([0], [0], dtype=torch.float64, device=DEVICE))
        out.values.sum().backward()
        expected = torch.zeros_like(f)
        # Hand-enumerated bilinear weights for both locations, all channels.
        for x, y, weight in [(1,2,.375),(2,2,.125),(1,3,.375),(2,3,.125),
                             (3,1,.375),(4,1,.375),(3,2,.125),(4,2,.125)]:
            expected[0, 0, :, y, x] += weight
        self.close(f.grad, expected, atol=0, rtol=0)

    def test_13_feature_finite_difference_gradcheck(self):
        # CPU double is intentional; CUDA atomic addition is not bitwise deterministic.
        f = torch.rand(1, 2, 1, 5, 6, dtype=torch.float64, requires_grad=True)
        q = torch.tensor([[2.125, 2.375]], dtype=torch.float64)
        v = make_velocity_grid([-.25, 0, .25], [0], dtype=torch.float64)
        times = torch.tensor([[-1., 2.]], dtype=torch.float64)
        passed = torch.autograd.gradcheck(lambda x: sample_context(x, times, q, v).values,
                                         (f,), eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True)
        self.assertTrue(passed)

    def test_14_invalid_and_empty_samples_have_zero_gradient(self):
        f = torch.rand(1, 1, 1, 3, 4, device=DEVICE, requires_grad=True)
        grid = make_velocity_grid([0], [0], device=DEVICE)
        out = sample_context(f, torch.zeros(1, 1, device=DEVICE),
                             torch.tensor([[-1, -1]], device=DEVICE), grid)
        out.values.sum().backward()
        self.assertEqual(int(torch.count_nonzero(f.grad)), 0)
        f.grad = None
        empty = sample_context(f, torch.zeros(1, 1, device=DEVICE), torch.empty(0, 2, device=DEVICE), grid)
        self.assertEqual(tuple(empty.values.shape), (1, 0, 1, 1, 1, 1))
        empty.values.sum().backward()
        self.assertEqual(int(torch.count_nonzero(f.grad)), 0)

    def test_15_tile_partition_and_declared_halo(self):
        times = [-3, 1, 2]
        v = [[-1, -.5], [0, 0], [.5, 1.5]]
        tiles = plan_tiles((31, 43), (7, 9), times, v, encoder_radius=2,
                           contrast_radius=3, decoder_radius=2, interp_radius=1)
        counts = np.zeros((31, 43), dtype=int)
        for tile in tiles:
            x0, y0, x1, y1 = tile.core_xyxy
            counts[y0:y1, x0:x1] += 1
            expected_q = (max(0,x0-2), max(0,y0-2), min(43,x1+2), min(31,y1+2))
            self.assertEqual(tile.query_xyxy, expected_q)
            qx0,qy0,qx1,qy1 = expected_q
            self.assertEqual(tile.input_xyxy, (max(0,qx0-7),max(0,qy0-9),min(43,qx1+7),min(31,qy1+9)))
            q = dense_queries(tile.query_xyxy)
            self.assertEqual(q.tolist(), [[float(x),float(y)] for y in range(qy0,qy1) for x in range(qx0,qx1)])
        self.assertTrue(np.all(counts == 1))
        METRICS["tile_partition"] = {"tiles": len(tiles), "pixels_covered_once": int(counts.size)}

    def test_16_full_image_and_tile_sampling_match(self):
        f = torch.rand(1, 3, 2, 23, 29, device=DEVICE)
        times = torch.tensor([[-3, 1, 2]], device=DEVICE)
        v = make_velocity_grid([-.5, 0, .5], [-.25, 0, .25], device=DEVICE)
        all_q = dense_queries((0,0,29,23), device=DEVICE)
        full = sample_context(f, times, all_q, v)
        full_values = full.values.reshape(1,23,29,2,3,3,3)
        full_valid = full.sample_valid.reshape(1,23,29,3,3,3)
        maxerr = 0.0
        for tile in plan_tiles((23,29),(7,9),cpu(times[0]),cpu(v.reshape(-1,2))):
            x0,y0,x1,y1 = tile.input_xyxy
            out = sample_context(f[...,y0:y1,x0:x1], times,
                dense_queries(tile.query_xyxy,device=DEVICE),v,image_hw=(23,29),
                origin_xy=tile.source_origin_xy,source_hw=tile.input_hw)
            qx0,qy0,qx1,qy1 = tile.query_xyxy
            expected = full_values[:,qy0:qy1,qx0:qx1].reshape_as(out.values)
            self.close(out.values,expected)
            self.assertTrue(torch.equal(out.sample_valid,
                full_valid[:,qy0:qy1,qx0:qx1].reshape_as(out.sample_valid)))
            maxerr = max(maxerr,float((out.values-expected).abs().max()))
        METRICS["full_vs_tile_max_error"] = maxerr

    def test_17_encoder_and_decoder_halo_reach_core(self):
        # Fixed convolutions are a boundary check, not an S2 model or training.
        raw = torch.rand(1,3,1,23,29,device=DEVICE)
        times = torch.tensor([[-3,1,2]],device=DEVICE)
        v = make_velocity_grid([-.5,0,.5],[-.5,0,.5],device=DEVICE)
        kernel = torch.tensor([[1,2,1],[2,4,2],[1,2,1]],device=DEVICE).float()[None,None]/16
        def encode(x):
            b,t,c,h,w=x.shape
            return F.conv2d(x.reshape(b*t,c,h,w),kernel,padding=1).reshape(b,t,1,h,w)
        def decode(evidence,hw):
            weight = evidence.sample_valid[:, :, None].to(evidence.values.dtype)
            denom = weight.sum((3,4,5)).clamp_min(1)
            value = (evidence.values*weight).sum((3,4,5))/denom
            image = value.transpose(1,2).reshape(1,1,*hw)
            return F.conv2d(F.conv2d(image,kernel,padding=1),kernel,padding=1)
        full = decode(sample_context(encode(raw),times,dense_queries((0,0,29,23),device=DEVICE),v),(23,29))
        assembled = torch.zeros_like(full)
        tiles = plan_tiles((23,29),(7,9),cpu(times[0]),cpu(v.reshape(-1,2)),encoder_radius=1,decoder_radius=2)
        for tile in tiles:
            x0,y0,x1,y1=tile.input_xyxy
            out=sample_context(encode(raw[...,y0:y1,x0:x1]),times,
                dense_queries(tile.query_xyxy,device=DEVICE),v,image_hw=(23,29),
                origin_xy=tile.source_origin_xy,source_hw=tile.input_hw)
            local=decode(out,tile.query_hw)
            cy,cx=tile.core_slice_in_query
            x0,y0,x1,y1=tile.core_xyxy
            assembled[...,y0:y1,x0:x1]=local[...,cy,cx]
        self.close(assembled,full)
        METRICS["fixed_local_pipeline_tile_max_error"]=float((assembled-full).abs().max())

    def test_18_invalid_geometry_is_rejected(self):
        f=torch.ones(1,1,1,4,5,device=DEVICE)
        times=torch.zeros(1,1,device=DEVICE)
        q=torch.tensor([[1.,1.]],device=DEVICE)
        v=make_velocity_grid([0],[0],device=DEVICE)
        for kw in [dict(query_chunk=0),dict(velocity_chunk=-1),dict(origin_xy=(-1,0)),
                   dict(image_hw=(4,5),origin_xy=(3,0),source_hw=(4,5)),
                   dict(origin_xy=(1,1)),dict(source_hw=(5,5))]:
            with self.assertRaises((ValueError,TypeError)):
                sample_context(f,times,q,v,**kw)
        with self.assertRaises(ValueError):
            sample_context(f,times,torch.tensor([[float('nan'),0]],device=DEVICE),v)
        with self.assertRaises(TypeError):
            sample_context(f.half(),times,q,v)
        clip,features=fixture_clip()
        with self.assertRaises(ValueError):
            sample_clip_features(features,replace(clip,center_index=7),q,v)

    def test_19_float64_inputs_and_fixed_geometry_contract(self):
        values=[0.,1.00000001,1.00000002]
        grid=make_velocity_grid(values,[0.],dtype=torch.float64,device=DEVICE)
        self.close(grid[0,:,0],values,atol=0,rtol=0)
        f=torch.arange(6,dtype=torch.float64,device=DEVICE).reshape(1,1,1,1,6).requires_grad_()
        zero=make_velocity_grid([0],[0],dtype=torch.float64,device=DEVICE)
        out=sample_context(f,[[0.]],[[2.00000001,0.]],zero)
        self.close(out.values.flatten(),[2.00000001],atol=1e-14,rtol=0)
        for coordinates in ['time','query','velocity']:
            t=torch.zeros(1,1,dtype=torch.float64,device=DEVICE,requires_grad=coordinates=='time')
            q=torch.tensor([[2.,0.]],dtype=torch.float64,device=DEVICE,requires_grad=coordinates=='query')
            v=zero.clone().requires_grad_(coordinates=='velocity')
            with self.assertRaises((ValueError,TypeError)):
                sample_context(f,t,q,v)


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.records=[]
    def startTest(self,test):
        self.started=time.perf_counter()
        super().startTest(test)
    def addSuccess(self,test):
        super().addSuccess(test)
        self.records.append({"name":test._testMethodName,"status":"PASS",
                             "seconds":time.perf_counter()-self.started})
    def addFailure(self,test,err):
        super().addFailure(test,err)
        self.records.append({"name":test._testMethodName,"status":"FAIL","error":self._exc_info_to_string(err,test)})
    def addError(self,test,err):
        super().addError(test,err)
        self.records.append({"name":test._testMethodName,"status":"FAIL","error":self._exc_info_to_string(err,test)})


def write_coordinate_svg(path):
    """Exact coordinate illustration, not a generated scientific result."""
    scale, ox, oy = 25, 45, 60
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="590" height="400" viewBox="0 0 590 400">',
           '<rect width="590" height="400" fill="white"/>',
           '<g font-family="sans-serif" font-size="14" fill="#172337">',
           '<text x="24" y="27">S1: q(t,v) = p + t v (native pixel coordinates)</text>']
    for y in range(11):
        for x in range(17):
            parts.append(f'<circle cx="{ox+x*scale}" cy="{oy+y*scale}" r="1.5" fill="#b6c1cc"/>')
    colors=['#2563eb','#cc3547','#16805f']
    for (vx,vy),color in zip([(1,.5),(0,0),(-.5,.5)],colors):
        coords=[(8+t*vx,5+t*vy) for t in list(range(-8,0))+list(range(1,9))]
        points=' '.join(f'{ox+x*scale},{oy+y*scale}' for x,y in coords)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x,y in coords:
            parts.append(f'<circle cx="{ox+x*scale}" cy="{oy+y*scale}" r="4" fill="{color}"/>')
    parts.extend([f'<circle cx="{ox+8*scale}" cy="{oy+5*scale}" r="8" fill="none" stroke="black" stroke-width="2"/>',
        '<text x="25" y="350">Circles: retained context samples; ring: center query p.</text>',
        '<text x="25" y="374">Time t=0 is excluded. Different colors are velocity hypotheses.</text>', '</g></svg>'])
    path.write_text('\n'.join(parts)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=PROJECT,
                        help='Root containing the frozen manifest-relative release paths')
    parser.add_argument('--config',type=Path,default=EXPERIMENT/'configs/s1_geometry.yaml')
    parser.add_argument('--device',default='cpu',help='Explicit device, e.g. cpu or cuda:1; no automatic GPU selection')
    parser.add_argument('--output',type=Path,default=EXPERIMENT/'reports/S1.json')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    global DEVICE,ATOL,RTOL
    DEVICE=args.device
    torch.set_num_threads(args.threads)
    config=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    ATOL=float(config['tolerance']['fp32_atol'])
    RTOL=float(config['tolerance']['fp32_rtol'])
    state={'stage':'S1','status':'RUNNING','gate_kind':'engineering','training_steps':0,
           'next_allowed_stage':None,'started_utc':datetime.now(timezone.utc).isoformat(),
           'environment':{'python':platform.python_version(),'torch':torch.__version__,
                          'device':DEVICE,'cuda_available':torch.cuda.is_available()},
           'scope':['sampling','validity','tiling','feature_gradients'],
           'configuration':config,'pending_requirements':[]}
    try:
        state['repository_base_commit']=subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()
    except (FileNotFoundError,subprocess.CalledProcessError):
        state['repository_base_commit']=None
    files=[*sorted((EXPERIMENT/'geometry').glob('*.py')),Path(__file__),
           EXPERIMENT/'checks/s1_runtime.py',args.config]
    state['code_and_config_sha256']={str(p.relative_to(EXPERIMENT)) if p.is_relative_to(EXPERIMENT) else str(p):
                                    hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    try:
        state['s0_prerequisite']=verify_s0(EXPERIMENT)
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(GeometryChecks)
        result=unittest.TextTestRunner(verbosity=2,resultclass=RecordedResult).run(suite)
        state['operator_checks']={'status':'PASS' if result.wasSuccessful() else 'FAIL',
                                  'count':result.testsRun,'checks':result.records,'metrics':METRICS}
        if not result.wasSuccessful():
            raise AssertionError('One or more S1 operator checks failed; no next stage permitted')
        state['real_training_batch']=check_real_batch(EXPERIMENT,args.project_root,DEVICE)
        if state['real_training_batch']['status']!='PASS':
            state['pending_requirements'].append('real_training_batch')
        gridcfg=config['velocity_grid']
        grid=make_velocity_grid(gridcfg['vx'],gridcfg['vy'],device=DEVICE)
        sampling=config['sampling']
        resource=config['resource']
        state['resource_measurement']=profile_sampler(DEVICE,grid,
            query_hw=tuple(resource['query_hw']),channels=resource['feature_channels'],
            query_chunk=sampling['query_chunk'],velocity_chunk=sampling['velocity_chunk'])
        if not gridcfg.get('formal_coverage_approved') or not gridcfg.get('coverage_evidence'):
            state['pending_requirements'].append('formal_grid_coverage_decision_from_training_data')
        if not str(DEVICE).startswith('cuda'):
            state['pending_requirements'].append('actual_cuda_peak_memory_and_time')
        else:
            budget=resource.get('max_peak_cuda_allocated_mib')
            if budget is None:
                state['pending_requirements'].append('declared_cuda_memory_budget')
            else:
                measured=state['resource_measurement']['cuda_memory']['peak_allocated_bytes']/1024**2
                if measured>float(budget):
                    raise RuntimeError(f'Peak allocated memory {measured:.2f} MiB exceeds {budget} MiB')
        state['status']='BLOCKED' if state['pending_requirements'] else 'PASS'
        state['next_allowed_stage']='S2' if state['status']=='PASS' else None
        args.output.parent.mkdir(parents=True,exist_ok=True)
        write_coordinate_svg(args.output.parent/'S1_sampling_coordinates.svg')
    except Exception as exc:
        state['status']='FAIL'
        state['error']=str(exc)
        state['traceback']=traceback.format_exc()
        print(state['traceback'],file=sys.stderr)
    state['finished_utc']=datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'stage':'S1','status':state['status'],
                      'operator_checks':state.get('operator_checks',{}).get('status'),
                      'pending_requirements':state['pending_requirements'],
                      'report':str(args.output)},ensure_ascii=False,indent=2))
    return {'PASS':0,'FAIL':1,'BLOCKED':2}[state['status']]


if __name__=='__main__':
    raise SystemExit(main())
