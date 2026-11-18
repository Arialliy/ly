"""Render predeclared examples from completed correspondence diagnostics.

Selection rule, fixed before reading sample outcomes:
  1. Sort all rows by sample_id; Python random.Random(46).sample selects 3.
  2. For all rows, then each dev clip in the order below, select the first
     competition['0.05'] row with KLT failure and the first with KLT success.
     KLT success means the existing lk_path_good_tol2 field (>=6/7 hits).
  3. Deduplicate sample IDs, retain every selection reason, and cap at 12.
     This exact rule selects at most 9 unique examples.

This is an illustration set, not an estimate of event prevalence. It does not
label score competition as intrinsic, irreversible, or physical ambiguity.
"""
from pathlib import Path
import hashlib
import json
import math
import random
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TRAIN = ['DJI_0030_9', 'DJI_0043_2', 'DJI_0057_1', 'DJI_0113_3',
         'DJI_0115_2', 'DJI_0217_2', 'DJI_0309_1', 'DJI_0323_2']
DEV = ['DJI_0026_2', 'DJI_0067_2']
ALLOWED = set(TRAIN + DEV)
FORBIDDEN = {'DJI_0051_3', 'DJI_0075_3', 'DJI_0139_2', 'DJI_0343_3'}
VIS = OUT / 'visuals'
RADIUS, SIDE, SCALE = 32, 65, 3
GT, BEST, ALT, KLT = '#ffe84a', '#13ddff', '#ff52dd', '#51ff71'
FG, MUTED, BG = '#ebeff4', '#b0bdcc', '#101720'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n')


def font(size):
    candidates = [
        OUT / 'assets/Go-Regular.ttf',
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
        ROOT / 'environment/site/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf',
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    # Pillow's default font is a supported fallback, with the receipt noting it.
    return ImageFont.load_default()


def text(draw, xy, value, size=16, color=FG):
    draw.text(xy, str(value), fill=color, font=font(size))


def wrapped(draw, xy, value, width=37, size=16, color=MUTED):
    x, y = xy
    for line in textwrap.wrap(value, width=width):
        text(draw, (x, y), line, size, color)
        y += size + 6
    return y


def selection(rows):
    rows = sorted(rows, key=lambda row: row['sample_id'])
    selected, reasons, checks = [], {}, []

    def add(row, reason):
        sid = row['sample_id']
        if sid not in reasons:
            selected.append(row)
            reasons[sid] = []
        reasons[sid].append(reason)

    for row in random.Random(46).sample(rows, min(3, len(rows))):
        add(row, 'seed46_uniform_sample_without_replacement')
    for scope in ['all'] + DEV:
        pool = [r for r in rows if scope == 'all' or r['sequence'] == scope]
        for success in [False, True]:
            match = next((r for r in pool if r['competition']['0.05']
                          and bool(r['lk_path_good_tol2']) == success), None)
            reason = scope + '_first_competition_klt_' + ('success' if success else 'failure')
            checks.append({'criterion': reason, 'sample_id': None if match is None else match['sample_id']})
            if match is not None:
                add(match, reason)
    return selected[:12], reasons, checks


def position(point, origin, scale):
    return ((float(point[0]) - origin[0] + .5) * scale,
            (float(point[1]) - origin[1] + .5) * scale)


def display_alternative(record):
    original = record.get('alternative_path')
    if original is None:
        return None, 'No original alternative was recorded.', False
    if original['positions'] != record['best_path']:
        return original, 'Original stored alternative differs from the best path; displayed unchanged.', False
    candidate = next((c for c in record['candidate_paths']
                      if c['positions'] != record['best_path']
                      and c['hits_tol1'] == 7 and c['gap_per_frame'] <= .05), None)
    if candidate is None:
        return None, ('Original alternative equals best. No different existing candidate '
                      'has 7/7 GT hits at +1 px and gap/frame <=0.05; no magenta marker shown.'), True
    return candidate, ('Original alternative equals best. First different existing candidate '
                       'with 7/7 GT hits at +1 px and gap/frame <=0.05, selected post hoc '
                       'using GT for display; not model selection or a GT-constrained oracle.'), True


def marks(panel, record, t, origin, scale):
    draw = ImageDraw.Draw(panel)
    x, y, w, h = record['gt_boxes'][t]
    # Bbox edges use the original continuous x/y convention.
    draw.rectangle([(x-origin[0])*scale, (y-origin[1])*scale,
                    (x+w-origin[0])*scale, (y+h-origin[1])*scale],
                   outline=GT, width=2)
    paths = [('best', record['best_path'][t], BEST)]
    alt = record.get('_display_candidate', record.get('alternative_path'))
    if alt:
        paths.append(('alternative', alt['positions'][t], ALT))
    paths.append(('klt', record['lk_path'][t], KLT))
    outside = []
    for kind, point, color in paths:
        px, py = position(point, origin, scale)
        if not (0 <= px < panel.width and 0 <= py < panel.height):
            outside.append(kind)
            continue
        if kind == 'best':
            draw.ellipse([px-5, py-5, px+5, py+5], outline=color, width=2)
        elif kind == 'alternative':
            draw.line([(px, py-8), (px+8, py), (px, py+8), (px-8, py), (px, py-8)],
                      fill=color, width=2)
        elif record['lk_valid'][t]:
            draw.line([(px-9, py), (px+9, py)], fill=color, width=2)
            draw.line([(px, py-9), (px, py+9)], fill=color, width=2)
        else:
            draw.line([(px-8, py-8), (px+8, py+8)], fill=color, width=2)
            draw.line([(px-8, py+8), (px+8, py-8)], fill=color, width=2)
    return outside


def heat_rgb(values):
    # Fixed NCC [-1, 1] scale, identical across all selected examples.
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140],
                      [94, 201, 98], [253, 231, 37]], dtype=float)
    f = np.clip((values + 1.) / 2., 0, 1)
    grid = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(f, grid, stops[:, c]) for c in range(3)], -1).astype(np.uint8)


def number(value):
    return 'n/a' if value is None else '{:.4f}'.format(value)


def render(record, reasons):
    seq = record['sequence']
    assert seq in ALLOWED and seq not in FORBIDDEN
    assert len(record['slots']) == 8
    assert len(record['gt_boxes']) == 8
    raw_path = ROOT / 'data/stage1' / seq / 'raw_frames.npy'
    raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
    assert raw.shape == (160, 512, 640) and raw.dtype == np.uint8
    anchor = record['anchor']
    x0, y0 = int(anchor[0])-RADIUS, int(anchor[1])-RADIUS
    assert 0 <= x0 and x0 + SIDE <= 640 and 0 <= y0 and y0 + SIDE <= 512
    heat_path = OUT / 'heatmaps' / ('%04d.npy' % record['sample_id'])
    heat = np.load(heat_path, allow_pickle=False)
    assert heat.shape == (49, 49) and np.isfinite(heat).all()
    candidate, display_reason, display_uses_gt = display_alternative(record)
    display_record = dict(record, _display_candidate=candidate)
    canvas = Image.new('RGB', (1320, 1250), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (24, 18), 'Observed correspondence candidates | sample %04d' % record['sample_id'], 26)
    text(draw, (24, 55), '%s | %s | track %s | frames %s-%s | current anchor (%s,%s)' % (
        seq, record['split'], record['track_id'], record['slots'][0], record['slots'][-1], anchor[0], anchor[1]), 18)
    text(draw, (24, 84), 'Native grayscale above / same pixels + marks below. Fixed crop: anchor +/-32 px; display: [0,255].', 17, MUTED)
    frame_status = []
    for t, frame in enumerate(record['slots']):
        tx, ty = 24 + (t % 4)*236, 126 + (t//4)*468
        tile = Image.new('RGB', (220, 452), '#18232f')
        td = ImageDraw.Draw(tile)
        text(td, (10, 6), 'frame %d | t=%d%s' % (frame, t, ' anchor' if t == 7 else ''), 17)
        crop = np.asarray(raw[frame, y0:y0+SIDE, x0:x0+SIDE])
        panel = Image.fromarray(crop).convert('RGB').resize((SIDE*SCALE, SIDE*SCALE), Image.NEAREST)
        tile.paste(panel, (12, 34))
        annotated = panel.copy()
        outside = marks(annotated, display_record, t, (x0, y0), SCALE)
        tile.paste(annotated, (12, 237))
        status = 'KLT valid' if record['lk_valid'][t] else 'KLT INVALID'
        if outside:
            status += '; outside: ' + ','.join(outside)
        text(td, (10, 434), status, 11, MUTED)
        canvas.paste(tile, (tx, ty))
        frame_status.append({'frame':frame, 't':t, 'klt_valid':bool(record['lk_valid'][t]),
                             'positions_outside_crop':outside})
    sx, sy = 982, 126
    text(draw, (sx, sy), 'Frame index 3: rich17 NCC', 19)
    text(draw, (sx, sy+27), 'Source frame: %d; radius 24 px' % record['slots'][3], 15, MUTED)
    heatpanel = Image.fromarray(heat_rgb(heat)).resize((294,294), Image.NEAREST)
    marks(heatpanel, display_record, 3, (anchor[0]-24, anchor[1]-24), 6)
    canvas.paste(heatpanel, (sx, sy+57))
    bar = np.tile(np.linspace(-1, 1, 294, dtype=np.float32), (15,1))
    canvas.paste(Image.fromarray(heat_rgb(bar)), (sx, sy+365))
    text(draw, (sx, sy+383), '-1', 14, MUTED)
    text(draw, (sx+142, sy+383), '0', 14, MUTED)
    text(draw, (sx+277, sy+383), '+1', 14, MUTED)
    yy = sy + 421
    for color, label in [(GT,'GT box (no dilation)'), (BEST,'Best path: circle'), (ALT,'Display candidate: diamond'), (KLT,'KLT: + valid / x invalid')]:
        draw.line([(sx,yy+8),(sx+18,yy+8)], fill=color, width=3)
        text(draw, (sx+27,yy), label, 16)
        yy += 28
    details = [
        'Competition proxy (delta=.05): %s' % record['competition']['0.05'],
        'Explicit candidate competition: %s' % record['explicit_candidate_competition'],
        'Best hits (+1 / +2 px): %s/7, %s/7' % (record['best_path_hits_tol1'], record['best_path_hits_tol2']),
        'KLT hits (+1 / +2 px): %s/7, %s/7' % (record['lk_history_hits_tol1'], record['lk_history_hits_tol2']),
        'KLT good (+2 px): %s' % record['lk_path_good_tol2'],
        'GT-compatible gap/frame: ' + number(record['gt_path_gap_per_frame']),
        'Wrong-path gap/frame: ' + number(record['wrong_path_gap_per_frame']),
        '|SCR| proxy: ' + number(record['absolute_scr']),
        'Release current-frame matched: ' + str(record.get('release_matched', 'not cached (train)')),
    ]
    yy += 12
    for line in details:
        yy = wrapped(draw, (sx, yy), line, width=35, size=14) + 4
    yy += 8
    text(draw, (sx, yy), 'Selection reasons:', 15)
    yy += 25
    for reason in reasons:
        yy = wrapped(draw, (sx, yy), reason.replace('_',' '), width=36, size=13) + 3
    if candidate is None:
        wrapped(draw, (sx, yy+4), 'No different explicit GT-compatible candidate displayed.', width=36, size=13, color=ALT)
    text(draw, (24, 1080), 'Interpretation: competing scores under this template/graph. The present target center is given by annotation.', 17)
    text(draw, (24, 1110), 'Yellow: original, undilated GT boxes. Hit metrics allow +1/+2 px. Markers show the position in each frame.', 16, MUTED)
    if display_uses_gt:
        text(draw, (24, 1140), 'Alternative candidate selected post hoc using GT for display; existing candidate, not model selection or GT-constrained oracle.', 16, ALT)
        text(draw, (24, 1170), 'Best and stored wrong path coincide here. Magenta shows a distinct existing GT-compatible candidate, if one is available.', 16, MUTED)
    else:
        text(draw, (24, 1140), 'Magenta: original recorded alternative, already distinct from best. No post-hoc GT selection was needed for this figure.', 16, MUTED)
        text(draw, (24, 1170), 'A different position alone does not imply a different physical target or intrinsically ambiguous observation.', 16, MUTED)
    text(draw, (24, 1200), 'No irreducible ambiguity claim. GT visibility/interpolation is unlabelled. Native [0,255] image display; no new inference.', 16, MUTED)
    stem = '%04d_%s_f%03d' % (record['sample_id'], seq, record['frame'])
    png = VIS / (stem + '.png')
    canvas.save(png)
    meta = {
        'sample_id':record['sample_id'], 'selection_reasons':reasons,
        'raw_source_relative_path':str(raw_path.relative_to(ROOT)),
        'crop_xyxy_exclusive':[x0,y0,x0+SIDE,y0+SIDE],
        'raw_shape':[160,512,640], 'display_grayscale_range':[0,255],
        'crop_resampling':'nearest', 'heatmap_source':str(heat_path.relative_to(OUT)),
        'heatmap_sha256':digest(heat_path), 'heatmap_frame_index':3,
        'heatmap_fixed_color_range':[-1,1], 'frame_status':frame_status,
        'original_alternative_path':record.get('alternative_path'),
        'display_candidate':candidate, 'display_candidate_selection_reason':display_reason,
        'display_candidate_selected_post_hoc_using_gt':display_uses_gt,
        'display_candidate_is_gt_constrained_oracle':False,
        'display_gt_boxes':'original, no dilation; evaluation tolerances are separate',
        'figure':png.name, 'sample':record,
        'interpretation':'A diagnostic score-competition illustration, not proof of intrinsic correspondence ambiguity.',
    }
    save_json(VIS/(stem+'.json'), meta)
    return meta


def main():
    results = json.loads((OUT/'RESULTS.json').read_text())
    assert results['status'] == 'COMPLETE'
    assert results['confirmation_loaded'] is False
    assert set(results['sampled_sequences']) <= ALLOWED
    rows = [json.loads(line) for line in (OUT/'SAMPLES.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows) == results['samples']
    assert len(set(r['sample_id'] for r in rows)) == len(rows)
    assert all(r['sequence'] in ALLOWED and r['sequence'] not in FORBIDDEN for r in rows)
    selected, reasons, checks = selection(rows)
    VIS.mkdir(exist_ok=False)
    save_json(VIS/'SELECTION.json', {
        'seed':46,'rng':'Python random.Random.sample','population':len(rows),
        'sort_key':'sample_id','random_count':min(3,len(rows)),
        'targeted_rule':'first competition[0.05]=true with lk_path_good_tol2=false/true, scope all then each dev',
        'targeted_scopes':['all']+DEV, 'deduplicated':True, 'limit':12,
        'selected_ids':[r['sample_id'] for r in selected], 'checks':checks,
        'reasons':reasons, 'prevalence_use':False,
        'source_hashes':{n:digest(OUT/n) for n in ['SAMPLES.jsonl','RESULTS.json','visualize.py']},
        'confirmation_loaded':False,
    })
    rendered = [render(row,reasons[row['sample_id']]) for row in selected]
    thumbs = Image.new('RGB', (1320, 450*math.ceil(len(rendered)/3)), BG)
    for i, m in enumerate(rendered):
        with Image.open(VIS/m['figure']) as figure:
            thumb=figure.resize((440,417),Image.LANCZOS)
            thumbs.paste(thumb, ((i%3)*440,(i//3)*450))
        text(ImageDraw.Draw(thumbs), ((i%3)*440+8,(i//3)*450+420),
             'sample %04d | %s' % (m['sample_id'],m['sample']['sequence']), 15)
    thumbs.save(VIS/'CONTACT_SHEET.png')
    lines=['# Correspondence diagnostic illustrations', '',
           'Selection was fixed before reading individual outcomes: seed 46 chooses three rows without replacement from sample_id order. Then first competition proxy (delta 0.05) with KLT failure / success is chosen globally and for each dev clip. Duplicates retain every reason. There are at most nine unique figures, below the cap of 12.', '',
           'Each figure contains untouched native grayscale display [0,255] above a copy with marks, for all eight frames. Crop coordinates are fixed at current anchor +/-32 px. Heatmaps show rich17 normalized correlation at index 3, with a fixed [-1,1] scale.', '',
           'Yellow: original, undilated GT box. Cyan circle: best path. Green plus: valid KLT; green x: invalid KLT. Magenta diamond: the original alternative when distinct. When the original alternative exactly equals best, display the first different already-generated candidate with 7/7 GT hits at +1 px and gap/frame <=0.05, or omit magenta if none exists. This choice is post hoc using GT for display, not model selection and not a GT-constrained oracle. All original and display paths/reasons are retained in the per-figure JSON. This visual set is not a prevalence estimate or proof of intrinsic ambiguity.', '',
           '|Sample|Sequence|Selection reason|Figure|', '|---|---|---|---|']
    for m in rendered:
        lines.append('|%s|%s|%s|[%s](%s)|' % (m['sample_id'],m['sample']['sequence'], '; '.join(m['selection_reasons']),m['figure'],m['figure']))
    (VIS/'README.md').write_text('\n'.join(lines)+'\n')
    save_json(VIS/'COMPLETE.json', {'status':'COMPLETE', 'count':len(rendered),
              'selected_ids':[m['sample_id'] for m in rendered], 'confirmation_loaded':False,
              'device':'cpu','inference':False,'training':False,
              'files_sha256':{p.name:digest(p) for p in sorted(VIS.iterdir()) if p.is_file()}})
    print(json.dumps({'status':'COMPLETE','count':len(rendered),
                      'selected_ids':[m['sample_id'] for m in rendered]}))


if __name__ == '__main__':
    main()
