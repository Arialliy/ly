"""Add object records to unchanged v1 readouts; no model or loss modification."""
import numpy as np
import runtime as rt
import manifest_data

PREVIOUS = rt.ROOT / 'experiments/c_effective_update_v1'
legacy = rt.load_module('effective_update_v1_readouts', PREVIOUS / 'code/effect_data.py')


def measure(logits, risk, cohort, cache, windows, reference=None):
    result = legacy.measure(logits, risk, cohort)
    z = legacy._numpy(logits)
    qp = legacy._numpy(risk['q_positive'])
    qn = legacy._numpy(risk['q_negative'])
    weak = set(cohort['weak_target_indices'])
    weak_metadata = manifest_data.weak_metadata(int(cohort['index']))
    reliable_metadata = manifest_data.reliable_metadata(int(cohort['index']))
    targets = [o for o in cohort['target_objects'] if o['target_index'] in weak]
    _, peaks, xy = legacy._roi_values(z, [o['region'] for o in targets])
    previous_weak = {} if reference is None else {o['target_index']: o for o in reference['weak_objects']}
    result['weak_objects'] = []
    for obj, peak, position in zip(targets, peaks, xy):
        i = obj['target_index']
        ref_xy = position if reference is None else previous_weak[i]['peak_xy']
        result['weak_objects'].append({**obj, **weak_metadata[i],
            'q_positive': float(qp[i]), 'q_negative': float(qn[i]),
            # Preserve the original float32 subtraction before aggregation.
            'margin': float(qp[i] - qn[i]),
            'peak_logit': float(peak), 'peak_score': float(legacy._sigmoid(peak)),
            'peak_xy': position, 'peak_shift': float(np.linalg.norm(np.array(position) - ref_xy)),
            'negative_reference_indices': [int(j) for j in cache['negative_indices'][i] if j >= 0]})
    means, peaks, xy = legacy._roi_values(z, cohort['reliable_regions'])
    previous_reliable = None if reference is None else reference['reliable_objects']
    result['reliable_objects'] = []
    shifts = []
    for i, (obj, mean, peak, position) in enumerate(zip(cohort['reliable_objects'], means, peaks, xy)):
        ref_xy = position if reference is None else previous_reliable[i]['peak_xy']
        shift = float(np.linalg.norm(np.array(position) - ref_xy))
        shifts.append(shift)
        result['reliable_objects'].append({**obj, **reliable_metadata[i], 'q_positive': float(mean),
            'peak_logit': float(peak), 'peak_score': float(legacy._sigmoid(peak)),
            'peak_xy': position, 'peak_shift': shift})
    result['reliable_peak_shift'] = float(np.mean(shifts)) if shifts else None
    if reference is not None:
        result['reliable_peak_changed_count'] = int(np.count_nonzero(shifts))
    bg_means, bg_peaks, bg_xy = legacy._roi_values(z, cohort['background_regions'])
    result['background_objects'] = []
    for i, region, mean, peak, position in zip(cohort['background_indices'],
            cohort['background_regions'], bg_means, bg_peaks, bg_xy):
        wi, ti, *_ = region
        window = windows[wi]
        metadata = manifest_data.background_metadata(int(cohort['index']), region, i)
        result['background_objects'].append({**metadata, 'background_index': i, 'region': region,
            'sequence': window['sequence'], 'source_frame': window['source_frames'][ti],
            'window': wi, 'slot': ti, 'mean_logit': float(mean), 'peak_logit': float(peak),
            'peak_xy': position})
    result['detail_schema'] = 'weak/reliable objects preserve original cohort order; background ROI IDs stay fixed'
    return result


def assert_legacy_equal(current, previous):
    """Replayed endpoints must reproduce every old stored measurement exactly."""
    for key, expected in previous.items():
        assert key in current and current[key] == expected, (key, current.get(key), expected)
