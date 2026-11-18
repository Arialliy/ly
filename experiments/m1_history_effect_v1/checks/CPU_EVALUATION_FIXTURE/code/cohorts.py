"""Freeze existing target identities and release-selected hard-background ROIs.

No model or new prediction selects this diagnostic population or its references.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[1]
M1 = ROOT / 'experiments/m1_recurrent_v1'
Q = ROOT / 'archive/infrared_problem_screen_20261009/temporal_diagnosis_20261010'
LOW = Q / 'low_response_information_20261011'
OUT = E / 'cohorts'
PLANS = {'O': [0, 36, 72, 108, 120], 'S': [0, 24, 60, 96, 120]}
SEQS = ['DJI_0026_2', 'DJI_0067_2']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream: return list(csv.DictReader(stream))


def write_csv(path, rows):
    assert rows
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def key(row): return row['sequence'], int(row['frame']), int(row['annotation_id'])


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def remap(path):
    value = str(path)
    old = '/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009'
    if value.startswith(old + '/'):
        return Q.parent / value[len(old):].lstrip('/')
    return Path(value)


def groups(row):
    return ['strict_all'] + [name for name in ('low_all_05', 'extreme_all_001',
        'both_fixed_success', 'm1_low_common_recovered') if int(row[name])]


def verify():
    manifest = read_json(OUT / 'COMPLETE.json')
    assert manifest['status'] == 'COMPLETE'
    for name, expected in manifest['output_sha256'].items(): assert sha(OUT / name) == expected
    return manifest


def build():
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'COMPLETE.json').exists():
        print(json.dumps(verify()), flush=True); return
    if any(OUT.iterdir()): raise FileExistsError('Preserve incomplete cohort artifacts for inspection')
    source_paths = [Path(__file__), LOW / 'COHORTS.csv', LOW / 'analysis/LOW_RESPONSE_OBJECTS.csv',
        LOW / 'analysis/build_low_response_objects.py', LOW / 'logits/COMPLETE.json',
        M1 / 'evaluation/step_2048/OBJECT_COMPARISON.csv', M1 / 'evaluation/step_2048/SUMMARY_COMPLETE.json']
    m1_summary = read_json(source_paths[-1])
    assert m1_summary['status'] == 'COMPLETE'
    assert sha(source_paths[-2]) == m1_summary['output_sha256']['OBJECT_COMPARISON.csv']
    baseline = read_csv(LOW / 'COHORTS.csv')
    original = {key(row): row for row in baseline}; assert len(original) == 5200
    paired = read_csv(source_paths[-2]); assert len(paired) == 10400
    paired = {(key(row), row['arrangement']): row for row in paired}
    strict = []
    for row in baseline:
        row = dict(row)
        for plan in PLANS:
            old = paired[key(row), plan]
            assert old['Current_matched'] == old['History_matched']
            row['m1_Current_' + plan + '_matched'] = int(old['Current_matched'])
            row['m1_History_' + plan + '_matched'] = int(old['History_matched'])
        row['m1_low_common_recovered'] = int(int(row['low_all_05']) and any(
            row['m1_Current_' + plan + '_matched'] and row['m1_History_' + plan + '_matched'] for plan in PLANS))
        strict.append(row)
    assert sum(row['m1_low_common_recovered'] for row in strict) == 9
    for name, count in [('low_all_05', 4342), ('extreme_all_001', 4170), ('both_fixed_success', 646)]:
        assert sum(int(row[name]) for row in strict) == count
    strict_by_key = {key(row): row for row in strict}
    write_csv(OUT / 'STRICT_COHORTS.csv', strict)
    write_csv(OUT / 'COMMON_RECOVERED_9.csv', [row for row in strict if row['m1_low_common_recovered']])
    all_valid = []
    for sequence in SEQS:
        path = Q / 'top1_background_control_20261011/evaluation/release' / sequence / 'COMPLETE.json'
        source_paths.append(path)
        for item in read_json(path)['objects']:
            row = dict(sequence=sequence, **item)
            row['bbox'] = json.dumps(row['bbox'], separators=(',', ':'))
            s = strict_by_key.get(key(row))
            for name in ('low_all_05', 'extreme_all_001', 'both_fixed_success', 'm1_low_common_recovered'):
                row[name] = int(s[name]) if s else 0
            all_valid.append(row)
    assert len(all_valid) == 16297 and sum(int(row['strict_small']) for row in all_valid) == 5200
    write_csv(OUT / 'ALL_VALID_TARGETS.csv', all_valid)

    # Reuse precisely the old candidate geometry and annotation exclusion code.
    background = load_module('_history_existing_background_geometry', LOW / 'analysis/build_low_response_objects.py')
    scorer = load_module('_history_existing_gt_geometry', Q / 'stage1_goal/pilot/score_pilot.py')
    helper, helper_path = scorer.helper(); source_paths += [Q / 'stage1_goal/pilot/score_pilot.py', helper_path]
    details = {(key(row), row['arrangement']): row for row in read_csv(LOW / 'analysis/LOW_RESPONSE_OBJECTS.csv')}
    log_receipt = read_json(LOW / 'logits/COMPLETE.json')
    assert log_receipt['status'] == 'COMPLETE'
    log_hashes = {str(remap(path)): value for path, value in log_receipt['output_sha256'].items()}
    regions = []
    hard = []
    for sequence in SEQS:
        annotation = Q / 'stage1_goal/data' / sequence / 'annotations.json'; source_paths.append(annotation)
        data = read_json(annotation)
        imageframe = {image['id']: frame for frame, image in enumerate(data['images'])}
        allann = defaultdict(list)
        for ann in data['annotations']: allann[imageframe[ann['image_id']]].append(ann)
        excluded = background.exclusion_masks(allann, helper)
        seqrows = [row for row in strict if row['sequence'] == sequence]
        for plan in PLANS:
            log_path = LOW / 'logits' / sequence / f'fused_{plan}_logits.npy'
            assert sha(log_path) == log_hashes[str(log_path)]; source_paths.append(log_path)
            logits = np.load(log_path, mmap_mode='r', allow_pickle=False)
            assert logits.shape == (160, 512, 640)
            cached = {}
            for row in seqrows:
                old = details[key(row), plan]
                frame = int(row['frame'])
                x1, y1 = int(old['crop_x0']), int(old['crop_y0'])
                width, height = int(old['crop_w']), int(old['crop_h'])
                x2, y2 = x1 + width, y1 + height
                pair_id = f'{sequence}:{frame}:{row["annotation_id"]}:{plan}'
                common = {'sequence': sequence, 'frame': frame, 'source_frame': int(row['source_frame']),
                    'annotation_id': int(row['annotation_id']), 'track_id': int(row['track_id']), 'arrangement': plan,
                    'pair_id': pair_id, 'groups': json.dumps(groups(row), separators=(',', ':'))}
                regions.append(dict(common, id=pair_id + ':target', kind='target', x1=x1, y1=y1, x2=x2, y2=y2,
                    anchor_x=int(float(old['target_logit_peak_x'])), anchor_y=int(float(old['target_logit_peak_y'])),
                    reference_logit=float(old['target_box_max_logit'])))
                cache_key = (frame, width, height)
                if cache_key not in cached:
                    _, scores, valid = background.global_tiles(logits[frame], excluded[frame], width, height)
                    if not bool(valid.any()): raise RuntimeError('No legal hard-background reference: ' + pair_id)
                    iy, ix = divmod(int(np.argmax(np.where(valid, scores, -np.inf))), scores.shape[1])
                    bx, by = ix * width, iy * height
                    patch = logits[frame, by:by+height, bx:bx+width]
                    py, px = divmod(int(np.argmax(patch)), width)
                    cached[cache_key] = (bx, by, bx+width, by+height, bx+px, by+py, float(patch[py,px]), int(valid.sum()))
                bx, by, bx2, by2, px, py, value, candidates = cached[cache_key]
                assert value == float(old['global_bg_max_logit'])
                regions.append(dict(common, id=pair_id + ':background', kind='background', x1=bx, y1=by, x2=bx2, y2=by2,
                                    anchor_x=px, anchor_y=py, reference_logit=value))
                hard.append(dict(common, target_x1=x1, target_y1=y1, target_x2=x2, target_y2=y2,
                    background_x1=bx, background_y1=by, background_x2=bx2, background_y2=by2,
                    background_anchor_x=px, background_anchor_y=py, release_hard_background_logit=value,
                    legal_global_candidates=candidates))
    assert len(regions) == 20800 and len(hard) == 10400
    write_csv(OUT / 'REGIONS.csv', regions)
    write_csv(OUT / 'FIXED_HARD_BACKGROUND.csv', hard)
    outputs = {p.name: sha(p) for p in sorted(OUT.glob('*.csv'))}
    manifest = {'status': 'COMPLETE', 'strict_unique_objects': 5200, 'all_valid_unique_objects': 16297,
        'low_all_05': 4342, 'extreme_all_001': 4170, 'both_fixed_success': 646,
        'm1_low_common_recovered': 9, 'region_rows': len(regions), 'pairs_arrangement_objects': len(hard),
        'object_key': ['sequence', 'frame', 'annotation_id'], 'coordinates': 'integer pixels; half-open xyxy; anchors integer pixel indices',
        'background_rule': 'Original nonoverlapping target-size global grid anchored(0,0), discard incomplete tail blocks; '
                           'exclude every positive-area intersecting annotation including ignored labels plus2pixel buffer. '
                           'Select maximum release O/S fused raw-logit block, row-major ties; fixed release peak anchor within block.',
        'selection_source': 'published release raw logits, never M1 History/Current/Off outcomes',
        'scope': 'Fixed diagnostic reference only; no new detection/matching mask, no inference input label access. '
                 'Hardest among the original legal same-size global grid, not all possible ROIs or unannotated objects.',
        'source_sha256': {str(p): sha(p) for p in sorted(set(map(Path, source_paths)))}, 'output_sha256': outputs}
    save_json(OUT / 'COMPLETE.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('status', 'strict_unique_objects', 'm1_low_common_recovered', 'region_rows')}), flush=True)


def regions_for_window(sequence, start, rows=None):
    """Return 40 frame-region lists; references follow their fixed O/S arrangement."""
    if sequence not in SEQS or start not in sorted(set(PLANS['O'] + PLANS['S'])): raise ValueError((sequence, start))
    rows = read_csv(OUT / 'REGIONS.csv') if rows is None else rows
    result = [[] for _ in range(40)]
    for row in rows:
        frame = int(row['frame'])
        if row['sequence'] != sequence or not start <= frame < start+40 or start not in PLANS[row['arrangement']]: continue
        item = {k: row[k] for k in ('id', 'pair_id', 'kind', 'arrangement')}
        item.update({k: int(row[k]) for k in ('x1', 'y1', 'x2', 'y2', 'anchor_x', 'anchor_y')})
        item['groups'] = json.loads(row['groups']) if isinstance(row['groups'], str) else row['groups']
        result[frame-start].append(item)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    if args.build: build()
    else: print(json.dumps(verify()), flush=True)
