"""Read-only inventory and annotation-only candidate audit; does not acquire images."""
from pathlib import Path
import collections, datetime, gzip, hashlib, json, os, re

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OFF = ROOT / 'official_deeppro'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prefix(seq):
    return seq.rsplit('_', 1)[0]

def get_split(split):
    return (OFF / 'datasets/RGB-T' / (split + '.txt')).read_text().split()

def main():
    train, test = get_split('train'), get_split('test')
    train_prefixes, test_prefixes = set(map(prefix, train)), set(map(prefix, test))
    developed = ['DJI_0024_2', 'DJI_0028_2', 'DJI_0028_4', 'DJI_0032_5', 'DJI_0047_3']
    excluded_prefixes = train_prefixes | set(map(prefix, developed))
    manifest = json.loads((ROOT / 'VT_TINY_MOT_PUBLIC_ZIP_MANIFEST.json').read_text())
    members = collections.defaultdict(dict)
    for e in manifest['entries']:
        s = e['name'].split('/')
        if len(s) == 5 and s[3] == '01' and s[4].endswith('.jpg'):
            members[(s[1], s[2])][s[4]] = e
    annotations_path = ROOT / 'runtime/VT_TRAIN_01_ANNOTATIONS.json.gz'
    data = json.load(gzip.open(annotations_path, 'rt'))
    images = {x['id']: x for x in data['images']}
    by_seq = collections.defaultdict(list)
    by_image = collections.defaultdict(list)
    category_names = {x['id']: x['name'] for x in data['categories']}
    for im in images.values():
        by_seq[im['file_name'].split('/')[0]].append(im)
    rect_count = 0
    for a in data['annotations']:
        by_image[a['image_id']].append(a)
        x, y, w, h = a['bbox']
        rect_count += a.get('segmentation') == [[x,y,x,y+h,x+w,y+h,x+w,y]]
    candidates = []
    stats_all = collections.Counter()
    for seq, ims in sorted(by_seq.items()):
        ims.sort(key=lambda x: int(Path(x['file_name']).stem))
        # First contiguous 160-frame span determined by numeric file names and archive presence.
        runs, run = [], []
        for im in ims:
            stem = int(Path(im['file_name']).stem)
            if Path(im['file_name']).name not in members[('train2017', seq)]:
                continue
            if run and stem != int(Path(run[-1]['file_name']).stem) + 1:
                runs.append(run); run = []
            run.append(im)
        if run: runs.append(run)
        eligible = [r for r in runs if len(r) >= 160]
        if not eligible: continue
        selected = eligible[0][:160]
        anns = [a for im in selected for a in by_image[im['id']]]
        small, small_clean = [], []
        for im in selected:
            aa = by_image[im['id']]
            for a in aa:
                x,y,w,h = a['bbox']
                if a.get('ignore',0) or a.get('iscrowd',0) or not (0 < w <= 16 and 0 < h <= 16): continue
                small.append(a)
                if x < 0 or y < 0 or x+w > im['width'] or y+h > im['height']: continue
                overlap = any(b['id'] != a['id'] and min(x+w,b['bbox'][0]+b['bbox'][2]) > max(x,b['bbox'][0]) and min(y+h,b['bbox'][1]+b['bbox'][3]) > max(y,b['bbox'][1]) for b in aa)
                if not overlap: small_clean.append(a)
        selected_members = [members['train2017',seq][Path(im['file_name']).name] for im in selected]
        c = {
            'sequence': seq, 'prefix': prefix(seq), 'archive_partition': 'train2017',
            'official_deeppro_train': seq in train, 'official_deeppro_test': seq in test,
            'checkpoint_train_prefix_overlap': prefix(seq) in train_prefixes,
            'developed_prefix_overlap': prefix(seq) in set(map(prefix,developed)),
            'confirmation_eligible_prefix_rule': prefix(seq) not in excluded_prefixes,
            'first_filename': Path(selected[0]['file_name']).name,
            'last_filename': Path(selected[-1]['file_name']).name,
            'frames': len(selected), 'sequence_annotated_frames': len(ims),
            'archive_sequence_ir_frames': len(members['train2017',seq]),
            'annotations': len(anns), 'small_not_ignore_crowd': len(small),
            'strict_small_inframe_nonoverlap': len(small_clean),
            'strict_small_tracks': len(set(a['track_id'] for a in small_clean)),
            'strict_small_categories': dict(collections.Counter(category_names[a['category_id']] for a in small_clean)),
            'compressed_image_bytes': sum(e['compressed_bytes'] for e in selected_members),
            'archive_member_prefix': 'VT-Tiny-MOT/train2017/' + seq + '/01/',
            'selected_members': [e['name'] for e in selected_members],
            'selected_image_ids': [im['id'] for im in selected],
            'annotation_frame_id_first_last': [selected[0]['frame_id'], selected[-1]['frame_id']],
            'annotation_mot_frame_id_first_last': [selected[0]['mot_frame_id'], selected[-1]['mot_frame_id']],
        }
        candidates.append(c)
    # Actual local raster sequence inventory. Skip hidden tool state and installed package trees.
    local, segmentation_dirs = [], []
    for sub in Path('/home/ly').iterdir():
        if not sub.is_dir() or sub.name.startswith('.'): continue
        for d, ds, fs in os.walk(sub):
            ds[:] = [x for x in ds if not x.startswith('.') and x not in ['node_modules','__pycache__','site-packages']]
            p = Path(d)
            if p.name == 'segmentations':
                segmentation_dirs.append({'path': d, 'direct_file_count': len(fs), 'child_dirs': len(ds)})
            if p.name == '01' and p.parent.name.startswith('DJI_'):
                ff = sorted(x for x in fs if x.lower().endswith(('.jpg','.png','.bmp','.jpeg')))
                if ff:
                    local.append({'sequence':p.parent.name, 'path':d, 'frame_count':len(ff), 'first':ff[0], 'last':ff[-1]})
    mask_members = [e['name'] for e in manifest['entries'] if any(v in e['name'].lower() for v in ['mask','segmentation'])]
    author_log = OFF / 'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/eval_ShootingRules_LocLen2_4.txt'
    author_frame_check = []
    for m in re.finditer(r'(DJI_\d+_\d+)results:\nTh_Seg = 0.000000e\+00:\s*PD:\[(\d+)/(\d+), [^\]]+\]\s*FA:\[(\d+), ([\deE.+-]+)\]',author_log.read_text()):
        seq,tp,gt,fp,fa = m.groups()
        source = members['test2017',seq]
        author_frame_check.append({'sequence':seq, 'archive_frame_count':len(source), 'first_filename':min(source), 'last_filename':max(source), 'approx_log_frames_from_rounded_fa': int(fp)/float(fa)/327680})
    train_annotations = [a for a in data['annotations'] if images[a['image_id']]['file_name'].split('/')[0] in train]
    train_small = [a for a in train_annotations if 0 < a['bbox'][2] <= 16 and 0 < a['bbox'][3] <= 16 and not a.get('ignore',0) and not a.get('iscrowd',0)]
    archive_train_names = {seq+'/01/'+name for (part,seq), mm in members.items() if part=='train2017' for name in mm}
    metadata_names = {im['file_name'] for im in images.values()}
    metadata_missing = sorted(archive_train_names-metadata_names)
    # All candidates are emitted; parent freezes the experiment before acquisition/inference.
    pool = {
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'selection_rule':'For each annotated sequence, earliest archive-present numeric-consecutive span of 160 frames; no prediction consulted; eligibility requires at least one strict small target-frame. Group rsplit("_",1)[0] conservatively, not verified physical scene.',
        'source_annotation_path':str(annotations_path), 'source_annotation_sha256':digest(annotations_path),
        'archive_manifest_path':str(ROOT/'VT_TINY_MOT_PUBLIC_ZIP_MANIFEST.json'),
        'archive_manifest_sha256':digest(ROOT/'VT_TINY_MOT_PUBLIC_ZIP_MANIFEST.json'),
        'official_train_candidates':[x for x in candidates if x['official_deeppro_train'] and x['strict_small_inframe_nonoverlap'] > 0],
        'new_prefix_candidates':[x for x in candidates if x['confirmation_eligible_prefix_rule'] and x['strict_small_inframe_nonoverlap'] > 0],
        'all_candidates':candidates,
        'excluded_confirmation_prefixes':sorted(excluded_prefixes),
    }
    (OUT/'DATA_CANDIDATE_POOL.json').write_text(json.dumps(pool,ensure_ascii=False,indent=2)+'\n')
    result = {
        'created_utc':pool['created_utc'], 'audit_mode':'read_only_no_training_no_image_download',
        'scan_scope':'All non-hidden immediate /home/ly directories recursively; exclude hidden dirs, node_modules, __pycache__, site-packages. /data does not exist; relevant Docker binds expose /home/ly.',
        'official_rgbt_masks_available':False, 'local_segmentation_dirs':segmentation_dirs,
        'official_train_sequences':train,'official_test_sequences':test,
        'official_train_prefixes':sorted(train_prefixes), 'official_test_prefixes':sorted(test_prefixes),
        'train_test_sequence_overlap':sorted(set(train)&set(test)),
        'train_test_prefix_overlap':sorted(train_prefixes & test_prefixes),
        'checkpoint_history_limit':'Published RGB-T training log is empty. Manifests are intended split, not cryptographic proof of checkpoint training exposure. Prefix distinctness is not proof of distinct geographic scene.',
        'local_vt_ir_inventory':local,
        'local_vt_ir_frame_count':sum(x['frame_count'] for x in local),
        'public_archive':{'id':manifest['file_id'],'bytes':manifest['archive_bytes'],'entries_count':manifest['entries_count'],'ir_frames_by_partition':{part:sum(len(v) for (p,s),v in members.items() if p==part) for part in ['train2017','test2017']},'sequences_by_partition':{part:len([1 for p,s in members if p==part]) for part in ['train2017','test2017']},'mask_or_segmentation_named_members':mask_members},
        'public_train_annotations':{'images':len(images),'annotations':len(data['annotations']),'videos':len(data['videos']),'exact_bbox_rectangle_polygons':rect_count,'official_train_annotations':len(train_annotations),'official_train_small_not_ignore_crowd':len(train_small),'official_train_small_categories':dict(collections.Counter(category_names[a['category_id']] for a in train_small))},
        'annotation_archive_alignment':{'duplicate_image_ids':len(data['images'])-len(images),'duplicate_image_filenames':len(images)-len(metadata_names),'metadata_images_missing_in_archive':sorted(metadata_names-archive_train_names),'archive_images_without_metadata_count':len(metadata_missing),'archive_images_without_metadata_by_sequence':dict(collections.Counter(n.split('/')[0] for n in metadata_missing)),'unlabelled_frames_must_not_be_treated_as_background':True},
        'official_train_excluded_from_160_frame_pool':[{ 'sequence':s,'annotated_frames':len(by_seq[s]),'reason':'less than 160 numeric-consecutive annotated frames'} for s in train if not any(c['sequence']==s for c in candidates)],
        'author_test_frame_count_crosscheck':author_frame_check,
        'candidate_counts':{'official_train_sequences':len(pool['official_train_candidates']),'official_train_unique_prefixes':len(set(x['prefix'] for x in pool['official_train_candidates'])),'new_sequences':len(pool['new_prefix_candidates']),'new_unique_prefixes':len(set(x['prefix'] for x in pool['new_prefix_candidates']))},
        'critical_files':{str(f.relative_to(ROOT)):{'bytes':f.stat().st_size,'sha256':digest(f)} for f in [OFF/'datasets/RGB-T/train.txt',OFF/'datasets/RGB-T/test.txt',OFF/'data_utils/TrainDataLoader.py',OFF/'data_utils/TestDataLoader.py',OFF/'ShootingRules.py',author_log,annotations_path,ROOT/'VT_TINY_MOT_PUBLIC_ZIP_MANIFEST.json']},
        'known_mask_protocol':'Paper IV-F describes rough Gaussian masks generated from author bounding boxes. Pinned train/test loaders binarize all positive label pixels; RGB-T test uses this label as centroid support. Gaussian parameters/generation script are not established.',
        'recommendation':'Proceed only as explicitly named common box/point-supervision protocol variant if original masks remain unavailable; identical labels, initialization, exposure, optimizer and evaluation across training controls. Reserve prefix-disjoint calibration and confirmation before prediction. Do not call weak-supervision metrics paper-native Pd/Fa.',
        'sources':['https://github.com/TinaLRJ/DeepPro','https://arxiv.org/html/2506.12766v5','https://github.com/XinyiYing/RGBT-Tiny','https://github.com/xuqingyu26/HGTMT','https://drive.google.com/file/d/17CgPc3huCf-KMr-I9zFVqmV0hzBVky9w/view'],
    }
    (OUT/'DATA_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'candidate_counts':result['candidate_counts'],'local_frames':result['local_vt_ir_frame_count'],'rectangle_polygons':rect_count,'total_annotations':len(data['annotations']),'pool':str(OUT/'DATA_CANDIDATE_POOL.json')},ensure_ascii=False))
    for group in ['official_train_candidates','new_prefix_candidates']:
        print(group)
        for x in pool[group]: print(x['sequence'],x['first_filename'],x['last_filename'],x['strict_small_inframe_nonoverlap'],x['strict_small_categories'],x['compressed_image_bytes'])

if __name__ == '__main__': main()
