#!/usr/bin/env python3
"""Acquire only the frozen selected clip; preserve raw bytes and all target labels."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import collections
import datetime
import gzip
import hashlib
import importlib.util
import json
import math
import urllib.parse
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
ROOT = STAGE.parent
LIMIT = 50_000_000
SEQUENCE = 'DJI_0028_2'
TRACK = 3800006
FIRST, LAST = 173, 332
INSPECT = [173, 213, 253, 293, 332]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    if (HERE / 'DOWNLOAD_PROVENANCE.json').exists():
        raise RuntimeError('Acquisition already recorded; refusing another network download.')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    selected_path = STAGE / 'SELECTED_CANDIDATE.json'
    selected = json.loads(selected_path.read_text())['candidate']
    assert (selected['sequence'], selected['track_id'], selected['start_frame'],
            selected['end_frame'], selected['frame_count']) == (SEQUENCE, TRACK, FIRST, LAST, 160)
    selected_rows = {row['frame']: row for row in selected['frames']}
    assert sorted(selected_rows) == list(range(FIRST, LAST + 1))
    manifest_path = ROOT / 'VT_TINY_MOT_PUBLIC_ZIP_MANIFEST.json'
    source_manifest = json.loads(manifest_path.read_text())
    prefix = f'VT-Tiny-MOT/train2017/{SEQUENCE}/01/'
    names = {prefix + f'{f:05d}.jpg' for f in range(FIRST, LAST + 1)}
    entries = sorted([e for e in source_manifest['entries'] if e['name'] in names],
                     key=lambda e: e['local_header_offset'])
    assert len(entries) == 160 and {e['name'] for e in entries} == names
    reduced = {k: v for k, v in source_manifest.items() if k != 'entries'}
    reduced.update(source_manifest_sha256=sha(manifest_path.read_bytes()),
                   source_manifest_path=str(manifest_path),
                   source_entries_count=source_manifest['entries_count'],
                   entries_count=len(entries), entries=entries,
                   selection={'sequence': SEQUENCE, 'track_id': TRACK, 'raw_start': FIRST,
                              'raw_end': LAST, 'frame_count': 160})
    write_json('SELECTED_ZIP_MANIFEST.json', reduced)

    # Reuse the supplied helper verbatim. Instrument reads to include the form body
    # in the same strict application-response-body budget as ZIP range responses.
    helper_path = ROOT / 'runtime/fetch_public_zip_subset.py'
    spec = importlib.util.spec_from_file_location('selected_clip_public_zip_helper', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    native_urlopen = helper.urllib.request.urlopen
    requests = []
    total_read = 0

    class CountedResponse:
        def __init__(self, response, request):
            self.response = response
            url = request if isinstance(request, str) else request.full_url
            u = urllib.parse.urlsplit(url)
            self.record = {'endpoint': u.scheme + '://' + u.netloc + u.path,
                           'request_range': None if isinstance(request, str) else request.get_header('Range'),
                           'status': response.status,
                           'content_range': response.headers.get('Content-Range'), 'body_bytes_read': 0}
            requests.append(self.record)

        def __getattr__(self, name):
            return getattr(self.response, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.response.close()

        def read(self, size=-1):
            nonlocal total_read
            if size < 0 or total_read + size > LIMIT:
                raise RuntimeError('Global response-body budget would be exceeded.')
            data = self.response.read(size)
            total_read += len(data)
            self.record['body_bytes_read'] += len(data)
            return data

    def counted_urlopen(request, **kwargs):
        return CountedResponse(native_urlopen(request, **kwargs), request)

    helper.urllib.request.urlopen = counted_urlopen
    archive = None
    try:
        archive = helper.PublicArchive(reduced, budget=LIMIT - 20_000)
        archive.extract(entries, HERE)
    finally:
        helper.urllib.request.urlopen = native_urlopen
        write_json('NETWORK_TRANSFER.json', {
            'response_body_bytes_read': total_read, 'response_body_budget_bytes': LIMIT,
            'budget_scope': 'All returned HTTP response bodies including public download form and retry reads; excludes transport headers.',
            'requests': requests, 'started_utc': started,
            'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})

    provenance = archive.provenance()
    provenance.update(response_body_bytes_including_form_and_retries=total_read,
                      response_body_budget_bytes=LIMIT,
                      form_response_bytes=sum(r['body_bytes_read'] for r in requests if not r['request_range']),
                      helper_path=str(helper_path), helper_sha256=sha(helper_path.read_bytes()),
                      acquisition_script_sha256=sha(Path(__file__).read_bytes()),
                      selected_candidate_sha256=sha(selected_path.read_bytes()),
                      reduced_manifest='SELECTED_ZIP_MANIFEST.json',
                      raw_bytes_preserved=True, sequence=SEQUENCE, target_original_track_id=TRACK,
                      raw_start_frame=FIRST, raw_end_frame=LAST, downloaded_frame_count=160)
    assert len(provenance['members']) == 160 and total_read <= LIMIT
    write_json('DOWNLOAD_PROVENANCE.json', provenance)

    annotation_path = ROOT / 'runtime/VT_TRAIN_01_ANNOTATIONS.json.gz'
    data = json.loads(gzip.decompress(annotation_path.read_bytes()))
    images = sorted([im for im in data['images'] if im['file_name'] in
                     {f'{SEQUENCE}/01/{f:05d}.jpg' for f in range(FIRST, LAST + 1)}],
                    key=lambda im: int(Path(im['file_name']).stem))
    assert len(images) == 160
    image_ids = {im['id'] for im in images}
    annotations = [a for a in data['annotations'] if a['image_id'] in image_ids]
    by_image = collections.defaultdict(list)
    for a in annotations:
        by_image[a['image_id']].append(a)
    video_ids = {im['video_id'] for im in images}
    subset = {'categories': data['categories'], 'images': images, 'annotations': annotations,
              'videos': [v for v in data['videos'] if v['id'] in video_ids],
              'tracks': [v for v in data['tracks'] if v['id'] in video_ids],
              'num_instances': sorted({a['track_id'] for a in annotations}),
              'num_track': len({a['track_id'] for a in annotations}),
              'subset_provenance': {
                  'source_file': str(annotation_path), 'source_sha256': sha(annotation_path.read_bytes()),
                  'retention': 'Every original annotation for every selected image; no target/category/ignore/size filtering.',
                  'sequence': SEQUENCE, 'raw_start': FIRST, 'raw_end': LAST,
                  'selected_original_track_id': TRACK,
                  'intended_use': 'Source annotation reference and future all-target camera-motion masking.'}}
    write_json('annotations.json', subset)
    member_records = {r['member']: r for r in provenance['members']}
    entry_records = {r['name']: r for r in entries}
    records = []
    frame_modes = collections.Counter()
    for im in images:
        frame = int(Path(im['file_name']).stem)
        member = 'VT-Tiny-MOT/train2017/' + im['file_name']
        path = HERE / member
        raw = path.read_bytes()
        rec, entry = member_records[member], entry_records[member]
        assert len(raw) == entry['uncompressed_bytes'] == rec['bytes']
        assert sha(raw) == rec['sha256']
        with Image.open(path) as image:
            image.load()
            assert image.format == 'JPEG' and image.mode == 'L' and image.size == (640, 512)
            frame_modes[image.mode] += 1
        target_annotations = [a for a in by_image[im['id']] if a['track_id'] == TRACK]
        assert len(target_annotations) == 1
        target = target_annotations[0]
        selected_row = selected_rows[frame]
        assert (im['id'], target['id'], target['bbox']) == (
            selected_row['image_id'], selected_row['annotation_id'], selected_row['bbox_xywh'])
        assert im['mot_frame_id'] == frame
        records.append({'raw_frame': frame, 'local_path': member, 'archive_member': member,
                        'sha256': sha(raw), 'bytes': len(raw), 'format': 'JPEG', 'native_mode': 'L',
                        'width': 640, 'height': 512, 'source_image_id': im['id'],
                        'source_video_id': im['video_id'], 'source_file_name': im['file_name'],
                        'source_mot_frame_id': im['mot_frame_id'], 'source_compact_frame_id': im['frame_id'],
                        'selected_target': {'original_track_id': TRACK, 'original_tracking_id': target['tracking_id'],
                                            'annotation_id': target['id'], 'category_id': target['category_id'],
                                            'bbox_xywh': target['bbox'],
                                            'center_xy': selected_row['center_xy']},
                        'all_source_annotation_ids': [a['id'] for a in by_image[im['id']]],
                        'all_target_annotation_count': len(by_image[im['id']]),
                        'zip_member': entry, 'crc_verified': rec['crc_verified']})
    write_json('FRAME_MANIFEST.json', {
        'candidate_id': selected['candidate_id'], 'sequence': SEQUENCE,
        'selected_original_track_id': TRACK, 'frame_count': 160,
        'raw_start_frame': FIRST, 'raw_end_frame': LAST,
        'source_url': provenance['source_url'], 'source_archive_bytes': provenance['archive_bytes'],
        'annotation_source_sha256': subset['subset_provenance']['source_sha256'],
        'rule_sha256': selected['rule_sha256'], 'raw_jpeg_total_bytes': sum(r['bytes'] for r in records),
        'all_target_annotations_file': 'annotations.json', 'frames': records})

    # All resizing and boxes are restricted to clearly named inspection artifacts.
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    width, row_h = 1584, 552
    sheet = Image.new('RGB', (width, row_h * len(INSPECT)), 'white')
    manifest_by_frame = {r['raw_frame']: r for r in records}
    visualization_records = []
    for index, frame in enumerate(INSPECT):
        record = manifest_by_frame[frame]
        with Image.open(HERE / record['local_path']) as raw_image:
            raw = raw_image.convert('RGB')
        x, y, w, h = record['selected_target']['bbox_xywh']
        boxed = raw.copy()
        ImageDraw.Draw(boxed).rectangle((x, y, x + w - 1, y + h - 1), outline=(255, 45, 30), width=2)
        cx, cy = record['selected_target']['center_xy']
        left = max(0, min(640 - 64, math.floor(cx - 32)))
        top = max(0, min(512 - 64, math.floor(cy - 32)))
        crop = raw.crop((left, top, left + 64, top + 64)).resize((256, 256), Image.Resampling.NEAREST)
        panel = Image.new('RGB', (width, row_h), 'white')
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), f'Raw frame {frame:05d} | native 640 x 512', fill='black', font=font)
        draw.text((656, 8), f'Target {TRACK} | bbox {x},{y},{w},{h}', fill='black', font=font)
        draw.text((1304, 8), '64 x 64 crop; 4x', fill='black', font=font)
        panel.paste(raw, (8, 36))
        panel.paste(boxed, (656, 36))
        panel.paste(crop, (1304, 36))
        draw.text((1304, 304), 'Raw crop; nearest', fill='black', font=font)
        draw.text((1304, 330), f'origin {left},{top}', fill='black', font=font)
        panel_name = f'INSPECTION_FRAME_{frame:05d}.png'
        panel.save(HERE / panel_name, optimize=False)
        sheet.paste(panel, (0, index * row_h))
        visualization_records.append({'raw_frame': frame, 'panel': panel_name,
                                      'crop_xywh': [left, top, 64, 64], 'crop_scale': 4,
                                      'crop_interpolation': 'Pillow Resampling.NEAREST'})
    sheet.save(HERE / 'CONTACT_SHEET.png', optimize=False)
    write_json('INSPECTION_VISUALIZATION.json', {
        'purpose': 'Deterministic annotation-guided visual inspection only; original JPEGs are untouched.',
        'fixed_raw_frames': INSPECT, 'contact_sheet': 'CONTACT_SHEET.png',
        'panels_left_to_right': ['native raw full frame', 'full frame with selected bbox in red',
                                'unboxed native 64 x 64 crop enlarged 4x by nearest neighbor'],
        'font': str(font_path) if font_path.exists() else 'Pillow default',
        'pillow_version': Image.__version__, 'frames': visualization_records})
    validation = {
        'status': 'PASS', 'exact_raw_frames': list(range(FIRST, LAST + 1)),
        'frame_count': len(records), 'native_mode_counts': dict(frame_modes), 'native_dimensions': [640, 512],
        'every_jpeg_fully_decoded': True, 'every_jpeg_sha256_matches_extracted_source_bytes': True,
        'source_image_and_selected_annotation_ids_match_fixed_candidate': True,
        'all_target_annotations_preserved': len(annotations),
        'unique_source_track_ids': len({a['track_id'] for a in annotations}),
        'raw_jpeg_total_bytes': sum(r['bytes'] for r in records),
        'ranged_zip_body_bytes': archive.transferred,
        'all_response_body_bytes_including_public_form': total_read,
        'response_body_budget_bytes': LIMIT,
        'no_model_outputs_read': True, 'no_gpu_or_inference': True,
        'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json('VALIDATION.json', validation)
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
