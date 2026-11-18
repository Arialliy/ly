"""Unconditional same-T-checkpoint diagnostics; no causal or official Fa claims."""
import hashlib
import json
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def mechanism_tables(objects, records, output, base, old, write_csv, write_json):
    sequences = ['DJI_0026_2', 'DJI_0067_2']
    rows = []
    for plan in ['O','S']:
        for group in ['pooled', *sequences]:
            for cohort in ['all','low_all_05','extreme_all_001','both_fixed_success']:
                selected = [r for r in objects if r['arrangement'] == plan
                            and (group == 'pooled' or r['sequence'] == group)
                            and (cohort == 'all' or r[cohort])]
                masks = {
                    'T_matched': lambda r: r['T_matched'],
                    'T_gain_vs_release': lambda r: r['T_matched'] and not r['release_matched'],
                    'T_gain_vs_S': lambda r: r['T_matched'] and not r['S_matched'],
                    'T_loss_vs_S': lambda r: not r['T_matched'] and r['S_matched'],
                    'T_gain_vs_current': lambda r: r['T_matched'] and not r['T_current_matched'],
                    'T_loss_vs_current': lambda r: not r['T_matched'] and r['T_current_matched'],
                    'T_release_gain_retained_current': lambda r: r['T_matched'] and not r['release_matched'] and r['T_current_matched'],
                    'T_release_gain_lost_current': lambda r: r['T_matched'] and not r['release_matched'] and not r['T_current_matched'],
                    'T_S_gain_retained_current': lambda r: r['T_matched'] and not r['S_matched'] and r['T_current_matched'],
                    'T_S_gain_lost_current': lambda r: r['T_matched'] and not r['S_matched'] and not r['T_current_matched'],
                }
                row = dict(arrangement=plan, group=group, cohort=cohort,
                           target_frames=len(selected), tracks=len({(r['sequence'],r['track_id']) for r in selected}),
                           clips=len({r['sequence'] for r in selected}))
                for name, predicate in masks.items():
                    subset = [r for r in selected if predicate(r)]
                    row[name] = len(subset)
                    row[name+'_tracks'] = len({(r['sequence'],r['track_id']) for r in subset})
                    row[name+'_clips'] = len({r['sequence'] for r in subset})
                assert row['T_release_gain_retained_current']+row['T_release_gain_lost_current']==row['T_gain_vs_release']
                assert row['T_S_gain_retained_current']+row['T_S_gain_lost_current']==row['T_gain_vs_S']
                rows.append(row)
    write_csv(output/'MECHANISM_SUMMARY.csv', rows)
    # Compare actual arrays, retaining the distinction between exact and tolerance agreement.
    comparisons, score_checks, hashes = [], [], {}
    reference_root = base.parent/'stage1_goal/predictions/official'
    for seq in sequences:
        for name in ['matched','component_area','frame_counts','coverage','strict_small','offsets']:
            a,b = records['T_bypass',seq][name], records['release',seq][name]
            exact = bool(np.array_equal(a,b))
            score_checks.append(dict(sequence=seq, field=name, exact=exact))
        for name in ['peak_x','peak_y']:
            exact = bool(np.array_equal(records['T_bypass',seq][name],records['release',seq][name],equal_nan=True))
            score_checks.append(dict(sequence=seq,field=name,exact=exact))
        bypass_areas=output/'T_bypass'/seq/'component_areas.npz'
        release_areas=old/'evaluation/release'/seq/'component_areas.npz'
        with np.load(bypass_areas,allow_pickle=False) as az, np.load(release_areas,allow_pickle=False) as bz:
            assert set(az.files)==set(bz.files)
            for name in az.files:
                score_checks.append(dict(sequence=seq,field='areas/'+name,exact=bool(np.array_equal(az[name],bz[name]))))
        for path in [bypass_areas,release_areas]: hashes[str(path)]=sha(path)
        for root in [base/'predictions/T_bypass',reference_root]:
            receipt = root/seq/'COMPLETE.json'
            hashes[str(receipt)] = sha(receipt)
        for label in [*(f'windows/window_{start:03d}.npz' for start in [0,24,36,60,72,96,108,120]), 'official_fused.npz']:
            actual = base/'predictions/T_bypass'/seq/label
            reference = reference_root/seq/label
            for path in [actual,reference]: hashes[str(path)] = sha(path)
            with np.load(actual,allow_pickle=False) as archive: a=archive['probabilities']
            with np.load(reference,allow_pickle=False) as archive: b=archive['probabilities']
            assert a.shape == b.shape and a.dtype == b.dtype == np.float32
            assert np.isfinite(a).all() and np.isfinite(b).all()
            maximum, different, threshold_differences = 0., 0, 0
            for af,bf in zip(a,b):
                maximum=max(maximum,float(np.abs(af-bf).max()))
                different+=int(np.count_nonzero(af!=bf))
                # Original helper's threshold is >= .5, verified in provenance.
                threshold_differences+=int(np.count_nonzero((af>=.5)!=(bf>=.5)))
            row=dict(sequence=seq,artifact=label,pixels=int(a.size),exact=different==0,
                     unequal_pixels=different,max_abs_error=maximum,absolute_tolerance=1e-4,
                     within_predeclared_tolerance=maximum<=1e-4,threshold05_disagreements=threshold_differences)
            comparisons.append(row)
    write_csv(output/'BYPASS_PROBABILITY_COMPARISON.csv', comparisons)
    passed=all(r['exact'] for r in score_checks) and all(r['within_predeclared_tolerance'] and r['threshold05_disagreements']==0 for r in comparisons)
    write_json(output/'BYPASS_VALIDATION.json',dict(status='PASS' if passed else 'FAIL',probability_absolute_tolerance=1e-4,
               max_abs_error=max(r['max_abs_error'] for r in comparisons),
               all_probabilities_bitwise_equal=all(r['exact'] for r in comparisons),
               threshold05_disagreements=sum(r['threshold05_disagreements'] for r in comparisons),
               all_matching_peak_area_background_arrays_exact=all(r['exact'] for r in score_checks),score_checks=score_checks,
               comparisons=comparisons,source_sha256=hashes,
               scope='Same frozen original state; published cache can differ by FP32 tiling arithmetic, bounded before evaluation. Zero threshold and exact diagnostic detection differences required. No tolerance retuning.'))
    assert passed, 'Bypass validation failed; retain BYPASS_VALIDATION.json and stop without tolerance adjustment'
