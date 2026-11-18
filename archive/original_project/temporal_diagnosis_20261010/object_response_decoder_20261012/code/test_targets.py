#!/usr/bin/env python3
"""Semantic checks for annotation-derived instance labels; CPU only."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import targets as targetmod


def ann(id=1,box=(20,20,8,8),tracking='1',numeric=101,category=3,ignore=0,crowd=0):
    return dict(id=id,bbox=list(box),tracking_id=tracking,track_id=numeric,category_id=category,ignore=ignore,iscrowd=crowd)


def sequence(frames=None):
    result=[[] for _ in range(160)]
    if frames:
        for t,aa in frames.items():result[t]=aa
    return SimpleNamespace(name='train_synthetic',annotations=result)


def build(seq,crop=(0,0)):
    return targetmod.build_window_targets(seq,dict(sequence=seq.name,start=0,crop_yx=crop))


class TargetTests(unittest.TestCase):
    def test_empty_window_negatives_active(self):
        a=build(sequence())
        self.assertEqual(a['observations'],[])
        self.assertTrue(a['negative_eligible'].all())
        self.assertEqual(int(a['background'].sum()),40*110*110)

    def test_missing_observation_never_becomes_absence(self):
        a=build(sequence({0:[ann()]}))
        self.assertEqual(a['negative_eligible'].nonzero()[0].tolist(),[0])
        self.assertEqual(len(a['observations']),1)

    def test_canonical_id_preserves_original_numeric_provenance(self):
        a=build(sequence({t:[ann(id=t+1,numeric=101 if t%2 else 120101)] for t in range(40)}))
        self.assertEqual(len(a['track_keys']),1)
        self.assertEqual({o['numeric_track_id'] for o in a['observations']},{101,120101})
        self.assertTrue(a['negative_eligible'].all())

    def test_category_namespace(self):
        a=build(sequence({0:[ann(),ann(id=2,box=(40,40,8,8),numeric=201,category=1)]}))
        self.assertEqual(len(a['track_keys']),2)

    def test_duplicate_identity_fails(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            build(sequence({0:[ann(),ann(id=2,box=(40,40,8,8),numeric=102)]}))

    def test_all_sizes_are_targets_and_removed_from_background(self):
        a=build(sequence({0:[ann(box=(20,20,20,24))]}))
        self.assertEqual(len(a['observations']),1)
        self.assertFalse(a['background'][0,18:46,18:42].any())
        self.assertTrue(a['valid'][0,20:44,20:40].all())

    def test_ignore_and_crowd_are_not_negative(self):
        for kwargs in ({'ignore':1},{'crowd':1}):
            a=build(sequence({0:[ann(**kwargs)]}))
            self.assertEqual(len(a['observations']),0)
            self.assertFalse(a['negative_eligible'][0])
            self.assertTrue(a['negative_eligible'][1:].all())
            self.assertFalse(a['background'][0,18:30,18:30].any())

    def test_partial_and_expanded_guard_intersecting_core(self):
        a=build(sequence({0:[ann(box=(120,20,3,4))]}))
        self.assertFalse(a['negative_eligible'][0])
        self.assertFalse(a['background'][0,20,118])
        self.assertEqual(len(a['observations']),0)

    def test_fully_outside_crop_does_not_disable_empty_crop(self):
        a=build(sequence({0:[ann(box=(150,20,8,8))]}))
        self.assertTrue(a['negative_eligible'].all())

    def test_zero_support_no_fallback_point_and_no_negative(self):
        a=build(sequence({0:[ann(box=(20.1,20.1,.1,.1))]}))
        self.assertEqual(a['observations'],[])
        self.assertEqual(a['stats']['empty_support_raw_quantization'],1)
        self.assertFalse(a['negative_eligible'][0])

    def test_exclusion_wins_over_positive(self):
        a=build(sequence({0:[ann(),ann(id=2,tracking='2',numeric=102,ignore=1)]}))
        self.assertEqual(a['observations'],[])
        self.assertEqual(a['stats']['empty_support_after_exclusion'],1)
        self.assertFalse(a['negative_eligible'][0])

    def test_two_positive_supports_stay_separate(self):
        a=build(sequence({0:[ann(),ann(id=2,box=(22,22,8,8),tracking='2',numeric=102)]}))
        self.assertEqual(len(a['observations']),2)
        first,second=a['observations']
        self.assertGreater(len(np.intersect1d(first['support_indices'],second['support_indices'])),0)
        self.assertFalse(a['background'][0].reshape(-1)[first['support_indices']].any())
        self.assertTrue(a['valid'][0].reshape(-1)[second['support_indices']].all())

    def test_crop_origin_and_float_bbox_unchanged(self):
        box=(60.25,70.75,8.5,9.25)
        a=build(sequence({0:[ann(box=box)]}),crop=(40,40))
        self.assertEqual(a['crop_yx'],(40,40))
        self.assertEqual(a['observations'][0]['bbox_original'],box)
        rect,q=targetmod.legacy.gaussian_region(box,(40,40))
        yy,xx=np.nonzero(q>0);expected=(yy+rect[1])*128+xx+rect[0]
        np.testing.assert_array_equal(a['observations'][0]['support_indices'],expected)

    def test_legacy_union_and_valid_exact_for_mixed_objects(self):
        seq=sequence({0:[ann(),ann(id=2,box=(23,23,8,8),tracking='2',numeric=102),ann(id=3,box=(40,40,20,20),tracking='3',numeric=103),ann(id=4,box=(5,20,8,8),tracking='4',numeric=104)],1:[ann(ignore=1)],2:[ann(box=(20.1,20.1,.1,.1))]})
        entry=dict(pairs=[dict(sequence=seq.name,crop_yx=[0,0],starts=[0,0]),dict(sequence=seq.name,crop_yx=[0,0],starts=[0,0])])
        bg=np.stack([targetmod.background_mask(seq,0,(0,0))]*4)
        union,valid,_=targetmod.legacy.build_targets(entry,{seq.name:seq},bg)
        actual=targetmod.build_entry_targets(entry,{seq.name:seq},background_masks=bg)
        for i,a in enumerate(actual):
            np.testing.assert_array_equal(a['valid'],valid[i])
            np.testing.assert_array_equal(targetmod.union_response(a),union[i])
        self.assertEqual(targetmod.targets_digest(actual),targetmod.targets_digest(actual))

    def test_corrupt_supplied_background_rejected(self):
        seq=sequence();bg=targetmod.background_mask(seq,0,(0,0));bg[0,20,20]=False
        with self.assertRaisesRegex(ValueError,'Background differs'):
            targetmod.build_window_targets(seq,dict(start=0,crop_yx=(0,0)),bg)

    def test_compact_cache_roundtrip(self):
        a=build(sequence({0:[ann(),ann(id=2,box=(40,40,8,8),tracking='2',numeric=102)]}))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'target.npz'
            targetmod.save_target(path,a);b=targetmod.load_target(path)
            self.assertEqual(a['digest'],b['digest'])
            np.testing.assert_array_equal(a['valid'],b['valid'])
            np.testing.assert_array_equal(a['observations'][0]['support_indices'],b['observations'][0]['support_indices'])
            with self.assertRaises(FileExistsError):targetmod.save_target(path,a)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TargetTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    print(json.dumps({'status':'PASS','semantic_tests':result.testsRun}))
