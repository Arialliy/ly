#!/usr/bin/env python3
"""Fetch queue entries 1:4, exactly as frozen before detector outputs.

Host Pillow decodes original JPEGs. NPY v1 is emitted directly as uint8 C-order;
no image resize, normalization, synthetic frames, GPU, or inference is used.
"""
import collections
import datetime
import gzip
import hashlib
import importlib.util
import json
import struct
import sys
import urllib.parse
from pathlib import Path
from PIL import Image

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
D = HERE.parent.parent
LIMIT = 100_000_000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def npy_header(shape):
    literal = repr({'descr': '|u1', 'fortran_order': False, 'shape': shape})
    n = len(literal.encode('latin1')) + 1
    padding = (-(10 + n)) % 64
    header = literal.encode('latin1') + b' ' * padding + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header


def main():
    queue_path = D / 'real_motion_stage2/INSPECTION_QUEUE.json'
    queue = json.loads(queue_path.read_text())
    candidates = queue['candidates'][1:4]
    expected = ['DJI_0028_4__4400005__00285__00444',
                'DJI_0032_5__4801536__00063__00222',
                'DJI_0047_3__4621500__00000__00159']
    assert [c['candidate_id'] for c in candidates] == expected
    assert all(c['frame_count'] == 160 for c in candidates)
    manifest_path = D / 'VT_TINY_MOT_PUBLIC_ZIP_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    annotation_path = D / 'runtime/VT_TRAIN_01_ANNOTATIONS.json.gz'
    annotation_bytes = annotation_path.read_bytes()
    data = json.loads(gzip.decompress(annotation_bytes))
    split_paths = {s: D / f'official_deeppro/datasets/RGB-T/{s}.txt' for s in ['train', 'test']}
    splits = {s: set(p.read_text().split()) for s, p in split_paths.items()}
    write(HERE / 'FIXED_SELECTION.json', {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'queue_source': str(queue_path), 'queue_sha256': sha(queue_path.read_bytes()),
        'queue_indices_zero_based': [1, 2, 3], 'candidate_ids': expected,
        'selection_rule': 'Exactly the next 3 entries after the first acquired candidate, preserving frozen queue order.',
        'selected_before_this_acquisition': True, 'model_outputs_read': False,
        'candidates': candidates,
        'scope': 'Additional development only. Exact name exclusion does not establish independent capture scenes.'})
    helper_path = D / 'runtime/fetch_public_zip_subset.py'
    spec = importlib.util.spec_from_file_location('fixed_public_zip_helper', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    native_urlopen = helper.urllib.request.urlopen
    requests, total_read = [], 0
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()

    class CountedResponse:
        def __init__(self, response, request):
            self.response = response
            url = request if isinstance(request, str) else request.full_url
            u = urllib.parse.urlsplit(url)
            self.record = {'endpoint': u.scheme + '://' + u.netloc + u.path,
                           'request_range': None if isinstance(request, str) else request.get_header('Range'),
                           'status': response.status, 'content_range': response.headers.get('Content-Range'),
                           'body_bytes_read': 0}
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
                raise RuntimeError('Global 100MB returned-body budget exceeded')
            result = self.response.read(size)
            total_read += len(result)
            self.record['body_bytes_read'] += len(result)
            return result

    def counted_urlopen(request, **kwargs):
        return CountedResponse(native_urlopen(request, **kwargs), request)

    helper.urllib.request.urlopen = counted_urlopen
    reports = []
    try:
        for candidate in candidates:
            folder = HERE / candidate['candidate_id']
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / 'VALIDATION.json').exists():
                raise RuntimeError('Previous validation found; refusing duplicate acquisition')
            sequence = candidate['sequence']
            assert sequence not in splits['train'] and sequence not in splits['test']
            first, last = candidate['start_frame'], candidate['end_frame']
            rows = {r['frame']: r for r in candidate['frames']}
            assert sorted(rows) == list(range(first, last + 1))
            names = {'VT-Tiny-MOT/train2017/' + r['file_name'] for r in rows.values()}
            entries = sorted([e for e in manifest['entries'] if e['name'] in names],
                             key=lambda e: e['local_header_offset'])
            assert len(entries) == 160 and {e['name'] for e in entries} == names
            reduced = {k: v for k, v in manifest.items() if k != 'entries'}
            reduced.update(entries_count=160, entries=entries,
                           source_manifest_path=str(manifest_path), source_manifest_sha256=sha(manifest_path.read_bytes()),
                           candidate_id=candidate['candidate_id'])
            write(folder / 'SELECTED_ZIP_MANIFEST.json', reduced)
            before = total_read
            archive = helper.PublicArchive(reduced, budget=LIMIT - total_read - 20_000)
            archive.extract(entries, folder)
            provenance = archive.provenance()
            provenance.update(candidate_id=candidate['candidate_id'], helper_path=str(helper_path),
                              helper_sha256=sha(helper_path.read_bytes()),
                              acquisition_script_sha256=sha(Path(__file__).read_bytes()),
                              response_body_bytes_including_form_and_retries=total_read - before,
                              global_response_body_bytes_so_far=total_read,
                              global_budget=LIMIT, raw_bytes_preserved=True)
            write(folder / 'DOWNLOAD_PROVENANCE.json', provenance)
            image_names = {r['file_name'] for r in rows.values()}
            images = sorted([im for im in data['images'] if im['file_name'] in image_names],
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
                      'subset_provenance': {'source_file': str(annotation_path),
                          'source_sha256': sha(annotation_bytes), 'retention': 'Every original annotation for each selected image; no target/category/ignore/size filtering.'}}
            write(folder / 'annotations.json', subset)
            member_records = {r['member']: r for r in provenance['members']}
            records = []
            npy_path = folder / 'raw_frames.npy'
            with npy_path.open('wb') as out:
                out.write(npy_header((160, 512, 640)))
                for im in images:
                    raw_frame = int(Path(im['file_name']).stem)
                    row = rows[raw_frame]
                    member = 'VT-Tiny-MOT/train2017/' + im['file_name']
                    path = folder / member
                    raw = path.read_bytes()
                    assert sha(raw) == member_records[member]['sha256']
                    assert len(raw) == member_records[member]['bytes']
                    with Image.open(path) as image:
                        image.load()
                        assert image.format == 'JPEG' and image.mode == 'L' and image.size == (640, 512)
                        pixels = image.tobytes()
                        assert len(pixels) == 512 * 640
                        out.write(pixels)
                    selected = [a for a in by_image[im['id']] if a['track_id'] == candidate['track_id']]
                    assert len(selected) == 1
                    selected = selected[0]
                    assert (im['id'], selected['id'], selected['bbox']) == (row['image_id'], row['annotation_id'], row['bbox_xywh'])
                    assert im['mot_frame_id'] == raw_frame
                    records.append({'raw_frame': raw_frame, 'local_path': member, 'archive_member': member,
                        'sha256': sha(raw), 'decoded_pixels_sha256': sha(pixels), 'bytes': len(raw),
                        'format': 'JPEG', 'native_mode': 'L', 'width': 640, 'height': 512,
                        'source_image_id': im['id'], 'source_video_id': im['video_id'],
                        'source_file_name': im['file_name'], 'source_mot_frame_id': im['mot_frame_id'],
                        'source_compact_frame_id': im['frame_id'],
                        'selected_target': {'original_track_id': candidate['track_id'],
                            'original_tracking_id': selected['tracking_id'], 'annotation_id': selected['id'],
                            'category_id': selected['category_id'], 'bbox_xywh': selected['bbox'], 'center_xy': row['center_xy']},
                        'all_source_annotation_ids': [a['id'] for a in by_image[im['id']]],
                        'all_target_annotation_count': len(by_image[im['id']]),
                        'crc_verified': member_records[member]['crc_verified']})
            frame_manifest = {'candidate_id': candidate['candidate_id'], 'sequence': sequence,
                'selected_original_track_id': candidate['track_id'], 'frame_count': 160,
                'raw_start_frame': first, 'raw_end_frame': last, 'source_url': provenance['source_url'],
                'source_archive_bytes': provenance['archive_bytes'], 'annotation_source_sha256': sha(annotation_bytes),
                'rule_sha256': candidate['rule_sha256'], 'raw_jpeg_total_bytes': sum(r['bytes'] for r in records),
                'all_target_annotations_file': 'annotations.json', 'frames': records}
            write(folder / 'FRAME_MANIFEST.json', frame_manifest)
            report = {'status': 'PASS', 'candidate_id': candidate['candidate_id'],
                'exact_raw_frames': list(range(first, last + 1)), 'frame_count': 160,
                'native_mode_counts': {'L': 160}, 'native_dimensions': [640, 512],
                'every_jpeg_fully_decoded': True, 'every_jpeg_sha256_matches_extracted_source_bytes': True,
                'all_member_crcs_verified': all(r['crc_verified'] for r in records),
                'selected_ids_boxes_match_frozen_candidate': True,
                'all_target_annotations_preserved': len(annotations),
                'unique_source_track_ids': len({a['track_id'] for a in annotations}),
                'raw_jpeg_total_bytes': sum(r['bytes'] for r in records),
                'clip_response_body_bytes': total_read - before, 'global_response_body_bytes_so_far': total_read,
                'source_split': 'VT-Tiny-MOT train2017/01', 'evaluation_role': 'extra development only',
                'deeppro_train_overlap': False, 'deeppro_test_overlap': False,
                'split_source_sha256': {s: sha(p.read_bytes()) for s, p in split_paths.items()},
                'independent_capture_scene_verified': False,
                'camera_motion_verified': False, 'continuous_visibility_verified': False,
                'no_model_outputs_read': True, 'no_gpu_or_inference': True,
                'raw_frames_npy': {'path': str(npy_path), 'sha256': sha(npy_path.read_bytes()),
                    'shape': [160, 512, 640], 'dtype': 'uint8', 'resized': False,
                    'normalized': False, 'decode_library': 'Pillow ' + Image.__version__},
                'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            write(folder / 'VALIDATION.json', report)
            reports.append(report)
            print(json.dumps({'READY': str(folder), 'candidate_id': candidate['candidate_id'],
                             'global_response_body_bytes_so_far': total_read}), flush=True)
    finally:
        helper.urllib.request.urlopen = native_urlopen
        write(HERE / 'NETWORK_TRANSFER.json', {'started_utc': started,
            'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'response_body_bytes_read': total_read, 'response_body_budget_bytes': LIMIT,
            'budget_scope': 'All returned HTTP response bodies including forms and retries; transport headers excluded.',
            'requests': requests})
    write(HERE / 'ACQUISITION_REPORT.json', {'status': 'PASS', 'candidate_ids': expected,
        'frames': 480, 'response_body_bytes_read': total_read, 'response_body_budget_bytes': LIMIT,
        'model_outputs_read': False, 'no_new_candidate_screen': True, 'clips': reports})


if __name__ == '__main__':
    main()
