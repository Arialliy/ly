#!/usr/bin/env python3
"""Keep literal local block scores/positions when fewer than 20 are available."""
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_background_engine', BASE/'analysis/build_low_response_objects.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def main():
    rows = list(csv.DictReader((BASE/'analysis/LOW_RESPONSE_OBJECTS.csv').open()))
    scorer = engine.at('existing_scorer_for_archive', engine.P/'pilot/score_pilot.py')
    helper, _ = scorer.helper()
    values, index = [], []
    for seq in engine.SEQS:
        annpath = engine.P/'data'/seq/'annotations.json'
        if not annpath.exists():
            annpath = engine.P/'data_diagnostics'/seq/'annotations.json'
        ann = json.loads(annpath.read_text())
        frames = {im['id']: i for i, im in enumerate(ann['images'])}
        allann = defaultdict(list)
        for item in ann['annotations']:
            allann[frames[item['image_id']]].append(item)
        masks = engine.exclusion_masks(allann, helper)
        for arrangement in engine.PLANS:
            maps = np.load(BASE/'logits'/seq/f'fused_{arrangement}_logits.npy', mmap_mode='r')
            for row_number, row in enumerate(rows):
                if row['sequence'] != seq or row['arrangement'] != arrangement:
                    continue
                frame = int(row['frame'])
                for kind in ['target', 'reference']:
                    prefix = 'local' if kind == 'target' else 'reference_local'
                    if kind == 'reference' and row['reference_valid'] != '1':
                        continue
                    if int(row[prefix+'_valid_blocks']) >= 20:
                        continue
                    if kind == 'target':
                        x, y, w, h = [int(row[k]) for k in ['crop_x0', 'crop_y0', 'crop_w', 'crop_h']]
                        mask = masks[frame]
                    else:
                        x, y, w, h = [int(row[k]) for k in ['reference_x0', 'reference_y0', 'reference_w', 'reference_h']]
                        mask = masks[frame].copy()
                        mask[max(0,y-2):min(512,y+h+2), max(0,x-2):min(640,x+w+2)] = True
                    scores, blocks = engine.local_tiles(maps[frame], mask, w, h, x, y, engine.expanded_neighborhood(x,y,w,h))
                    assert len(blocks) == int(row[prefix+'_valid_blocks'])
                    qkey = prefix+('_bg_q95_logit' if kind == 'target' else '_q95')
                    if len(blocks):
                        assert float(np.quantile(scores, .95)) == float(row[qkey])
                    start = len(values)
                    # [score, grid shift x/y, block x/y, actual peak x/y].
                    values.extend(blocks)
                    index.append(dict(row_number=row_number, sequence=seq, arrangement=arrangement,
                        frame=frame, track_id=row['track_id'], annotation_id=row['annotation_id'],
                        kind=kind, crop_w=w, crop_h=h, offset=start, count=len(blocks)))
    np.savez_compressed(BASE/'analysis/INSUFFICIENT_LOCAL_BACKGROUND_BLOCKS.npz',
                        blocks=np.asarray(values, dtype=np.float64).reshape(-1,7),
                        columns=np.array(['max_logit','grid_shift_x','grid_shift_y','x0','y0','peak_x','peak_y']))
    engine.csvout(BASE/'analysis/INSUFFICIENT_LOCAL_BACKGROUND_INDEX.csv', index)
    engine.jsonout(BASE/'analysis/INSUFFICIENT_BACKGROUND_ARCHIVE.json', dict(status='PASS',
        records=len(index), actual_blocks=len(values),
        target_records=sum(r['kind']=='target' for r in index),
        reference_records=sum(r['kind']=='reference' for r in index),
        every_count_and_q95_matches=True, coordinate_convention='pixel centers for peaks; integer block top-left',
        purpose='Retain literal insufficient local samples; full score maps and deterministic sampling retain all other samples.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(records=len(index), blocks=len(values))))


if __name__ == '__main__':
    main()
