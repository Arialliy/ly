#!/usr/bin/env python3
"""Independent CPU result audit on predeclared frames; no scorer imports."""
from __future__ import annotations

import ast
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
DATA = BASE.parent / "stage1_goal/data"
FIELDS = ["gt_count", "matched_count", "component_count", "unmatched_components",
          "positive_pixels", "positive_pixels_outside_boxes"]
HASHES = {}
CHECKS = []


def digest(path):
    path = Path(path)
    if str(path) not in HASHES:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b""):
                h.update(block)
        HASHES[str(path)] = h.hexdigest()
    return HASHES[str(path)]


def check(name, passed, **details):
    CHECKS.append(dict(name=name, passed=bool(passed), **details))
    if not passed:
        raise AssertionError(json.dumps(CHECKS[-1]))


def read_json(path):
    digest(path)
    return json.loads(Path(path).read_text())


def prepare_annotations(sequence):
    data = read_json(DATA / sequence / "annotations.json")
    images = data["images"]
    annotated = collections.defaultdict(list)
    for a in data["annotations"]:
        annotated[a["image_id"]].append(a)
    frames, metadata, offsets = [], [], [0]
    for t, im in enumerate(images):
        valid = []
        for a in sorted(annotated[im["id"]], key=lambda a: (a["track_id"], a["id"])):
            x, y, w, h = map(float, a["bbox"])
            if a.get("ignore", 0) or w <= 0 or h <= 0:
                continue
            if min(im["width"], x+w) <= max(0, x) or min(im["height"], y+h) <= max(0, y):
                continue
            valid.append(dict(a, bbox=[x, y, w, h]))
        frames.append(valid)
        for a in valid:
            x, y, w, h = a["bbox"]
            overlap = False
            for b in valid:
                if a["id"] == b["id"]:
                    continue
                xx, yy, ww, hh = b["bbox"]
                if min(x+w, xx+ww) > max(x, xx) and min(y+h, yy+hh) > max(y, yy):
                    overlap = True
            inside = x >= 0 and y >= 0 and x+w <= im["width"] and y+h <= im["height"]
            metadata.append(dict(frame=t, source_frame=int(Path(im["file_name"]).stem),
                                 image_id=im["id"], track_id=a["track_id"], annotation_id=a["id"],
                                 category_id=a["category_id"], bbox=a["bbox"],
                                 strict_small=int(max(w, h) <= 16 and inside and not overlap)))
        offsets.append(len(metadata))
    return frames, metadata, np.array(offsets)


def independent_components(probability, threshold):
    """Explicit 8-neighbor flood fill, independent of scipy's label routine."""
    height, width = probability.shape
    positive = probability >= threshold
    visited = np.zeros_like(positive)
    components = []
    for flat in np.flatnonzero(positive):
        sy, sx = divmod(int(flat), width)
        if visited[sy, sx]:
            continue
        visited[sy, sx] = True
        pending = [(sy, sx)]
        area, peak_index, peak_value = 0, int(flat), float(probability[sy, sx])
        while pending:
            y, x = pending.pop()
            area += 1
            p, index = float(probability[y, x]), y*width+x
            if p > peak_value or (p == peak_value and index < peak_index):
                peak_index, peak_value = index, p
            for yy in range(max(0, y-1), min(height, y+2)):
                for xx in range(max(0, x-1), min(width, x+2)):
                    if positive[yy, xx] and not visited[yy, xx]:
                        visited[yy, xx] = True
                        pending.append((yy, xx))
        py, px = divmod(peak_index, width)
        components.append(dict(x=px+.5, y=py+.5, score=peak_value, area=area))
    return components, positive


def independent_match(probability, gt, threshold):
    comps, positive = independent_components(probability, threshold)
    adjacency = []
    for ann in gt:
        x, y, w, h = ann["bbox"]
        selected = [i for i, c in enumerate(comps) if x <= c["x"] < x+w and y <= c["y"] < y+h]
        selected.sort(key=lambda i: (((comps[i]["x"]-x-w/2)/w)**2 + ((comps[i]["y"]-y-h/2)/h)**2,
                                    -comps[i]["score"], i))
        adjacency.append(selected)
    # Iterative DFS reproduces the frozen ordered augmenting-path protocol,
    # while implementing its state transitions independently of the scorer.
    owner = {}
    for start in range(len(gt)):
        seen = set()
        stack = [(start, iter(adjacency[start]), None)]
        while stack:
            target, choices, entered_via = stack[-1]
            candidate = next(choices, None)
            if candidate is None:
                stack.pop()
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate in owner:
                stack.append((owner[candidate], iter(adjacency[owner[candidate]]), candidate))
                continue
            owner[candidate] = target
            while len(stack) > 1:
                _, _, incoming = stack.pop()
                owner[incoming] = stack[-1][0]
            break
    assignments = {target: c for c, target in owner.items()}
    box_pixels = np.zeros(probability.shape, bool)
    yy, xx = np.arange(probability.shape[0])+.5, np.arange(probability.shape[1])+.5
    matched, maxima, peak_x, peak_y, target_area = [], [], [], [], []
    for i, ann in enumerate(gt):
        x, y, w, h = ann["bbox"]
        ys, xs = np.flatnonzero((yy >= y) & (yy < y+h)), np.flatnonzero((xx >= x) & (xx < x+w))
        region = np.ix_(ys, xs)
        box_pixels[region] = True
        maxima.append(float(probability[region].max()) if len(xs) and len(ys) else np.nan)
        match = assignments.get(i)
        matched.append(int(match is not None))
        peak_x.append(comps[match]["x"] if match is not None else np.nan)
        peak_y.append(comps[match]["y"] if match is not None else np.nan)
        target_area.append(comps[match]["area"] if match is not None else 0)
    counts = [len(gt), len(assignments), len(comps), len(comps)-len(assignments),
              int(positive.sum()), int((positive & ~box_pixels).sum())]
    areas = np.array([c["area"] for c in comps], dtype=np.int32)
    unmatched = np.array([int(i not in owner) for i in range(len(comps))], dtype=np.uint8)
    return dict(matched=np.array(matched), box_max=np.array(maxima, np.float32),
                peak_x=np.array(peak_x, np.float32), peak_y=np.array(peak_y, np.float32),
                component_area=np.array(target_area), frame_counts=np.array(counts),
                areas=areas, unmatched=unmatched)


def main():
    began=time.perf_counter()
    torch.set_num_threads(1)
    scope=read_json(BASE/'audit/AUDIT_SCOPE.json')
    freeze=read_json(BASE/'PROTOCOL_FREEZE.json')
    check('protocol_freeze_bound_to_predeclared_scope',digest(BASE/'PROTOCOL_FREEZE.json')==scope['frozen_protocol_sha256'])
    check('protocol_document_frozen',digest(BASE/'PROTOCOL.md')==freeze['protocol_sha256'])
    for p,h in freeze['code_sha256'].items():
        check('frozen_code_'+p,digest(BASE/p)==h)
    for p,h in freeze['preflight_sha256'].items():
        check('frozen_preflight_'+p,digest(BASE/p)==h)
    variant='gaussian_softiou_approx'
    check('single_predeclared_condition',scope['conditions']==[variant] and freeze['condition']==variant)
    run=read_json(BASE/'training'/variant/'run/RUN.json')
    checkpoint=BASE/'training'/variant/'run/final_model.pth'
    cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
    reference=Path(run['checkpoint_path'])
    numpy_core=np._core if hasattr(np,'_core') else np.core
    numeric_globals=[(numpy_core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(numeric_globals):
        release=torch.load(reference,map_location='cpu',weights_only=True)
    original=release['model_state_dict'];trained=cp['model_state_dict']
    check('trained_checkpoint_identity',run['status']=='complete' and
          run['steps']==run['final_step']==cp['step']==freeze['fixed_steps']==512 and
          run['condition']==run['variant']==cp['condition']==cp['variant']==variant and
          digest(checkpoint)==run['final_checkpoint_sha256'] and
          digest(reference)==run['checkpoint_sha256']==cp['initial_checkpoint_sha256'] and
          cp['plan_sha256']==run['plan_sha256'])
    check('original_architecture_state_identity',set(trained)==set(original) and
          all(trained[k].shape==original[k].shape and trained[k].dtype==original[k].dtype for k in original) and
          not any('gamma' in k or 'reference' in k or 'association' in k for k in trained) and
          run['parameter_count']==70913 and run['new_parameters']==0,
          state_tensor_count=len(trained),parameter_count=run['parameter_count'])
    parameter_count=sum(v.numel() for k,v in trained.items() if not k.endswith(('running_mean','running_var','num_batches_tracked')))
    check('independent_parameter_count',parameter_count==70913,parameters=parameter_count)
    check('trained_and_inference_source_identity',cp['model_source_sha256']==run['model_source_sha256']==digest(run['criterion_source_path']) and
          cp['label_source_sha256']==run['label_source_sha256']==digest(BASE/'code/gaussian_labels.py') and
          cp['train_source_sha256']==run['train_source_sha256']==digest(BASE/'code/train_supervision.py') and
          cp['label_config']==run['label_config']==freeze['label_config'] and
          run['bn_initial_sha256']==run['bn_final_sha256'])
    meta_by_seq={seq:prepare_annotations(seq) for seq in scope['sequences']}
    check('frozen_small_target_total',sum(sum(r['strict_small'] for r in vals[1]) for vals in meta_by_seq.values())==5200)
    audited_rows=[];tiling=[]
    starts_union=sorted(set(scope['plans']['O']+scope['plans']['S']))
    for seq in scope['sequences']:
        identity=variant+'/'+seq
        pred=BASE/'predictions'/variant/seq
        ev=BASE/'evaluation'/variant/seq
        receipt=read_json(pred/'COMPLETE.json')
        scores_receipt=read_json(ev/'COMPLETE.json')
        check(identity+'_inference_protocol',receipt['status']=='COMPLETE' and receipt['variant']==receipt['condition']==variant and
              receipt['sequence']==seq and receipt['checkpoint_sha256']==digest(checkpoint) and
              receipt['reference_checkpoint_sha256']==digest(reference) and
              receipt['model_source_sha256']==run['model_source_sha256']==digest(receipt['model_source']) and
              receipt['label_config']==run['label_config'] and
              receipt['label_source_sha256']==run['label_source_sha256'] and receipt['train_source_sha256']==run['train_source_sha256'] and
              receipt['raw_sha256']==digest(DATA/seq/'raw_frames.npy') and
              receipt['normalization']==[85.0799,47.4845] and receipt['precision']=='FP32' and
              receipt['tile_core']==128 and receipt['tile_halo']==16 and receipt['parameters']==70913 and
              receipt['receptive_radius_tyx']==[None,9,9] and receipt['model_labels_access'] is False and
              receipt['official_starts']==scope['plans']['O'] and receipt['diagnostic_starts']==starts_union and
              receipt['script_sha256']==digest(BASE/'code/infer_supervision.py'))
        witnesses=receipt['tiling_checks']
        check(identity+'_trained_tiling_witnesses',len(witnesses)==3 and
              [w['yx'] for w in witnesses]==[[0,0],[176,224],[352,448]] and
              all(w['threshold05_disagreements']==0 and math.isfinite(w['max_logit_error']) and
                  math.isfinite(w['max_probability_error']) for w in witnesses))
        tiling.extend([dict(sequence=seq,**w) for w in witnesses])
        gt,meta,offsets=meta_by_seq[seq]
        check(identity+'_scoring_metadata',scores_receipt['status']=='COMPLETE' and
              scores_receipt['condition']==variant and scores_receipt['sequence']==seq and
              scores_receipt['threshold']==scope['threshold']==.5 and scores_receipt['objects']==meta and
              scores_receipt['checkpoint_sha256']==digest(checkpoint))
        with np.load(ev/'scores.npz',allow_pickle=False) as ar:scores={k:ar[k] for k in ar.files}
        with np.load(ev/'component_areas.npz',allow_pickle=False) as ar:areas={k:ar[k] for k in ar.files}
        for name in ['scores.npz','component_areas.npz','FRAME_COUNTS.csv']:digest(ev/name)
        with (ev/'FRAME_COUNTS.csv').open() as stream:
            frame_csv={(r['arrangement'],int(r['frame'])):r for r in csv.DictReader(stream)}
        check(identity+'_score_array_protocol',float(scores['threshold'])==.5 and
              scores['plans'].tolist()==list(scope['plans']) and np.array_equal(scores['offsets'],offsets) and
              scores['strict_small'].tolist()==[bool(r['strict_small']) for r in meta])
        samples={}
        for start in starts_union:
            path=pred/'windows'/f'window_{start:03d}.npz'
            check(identity+f'_window_{start}_hash',digest(path)==receipt['window_sha256'][path.name])
            with np.load(path,allow_pickle=False) as ar:
                prob=ar['probabilities']
                check(identity+f'_window_{start}_valid',int(ar['start'])==start and int(ar['end'])==start+40 and
                      prob.shape==(40,512,640) and prob.dtype==np.float32 and np.isfinite(prob).all() and
                      np.all((prob>=0)&(prob<=1)))
                for frame in scope['frames_fixed_before_reading_scores']:
                    if start<=frame<start+40:samples[start,frame]=prob[frame-start].copy()
        del prob
        fused_path=pred/'official_fused.npz'
        check(identity+'_official_fused_hash',digest(fused_path)==receipt['fused_sha256'])
        with np.load(fused_path,allow_pickle=False) as ar:
            check(identity+'_official_fused_starts',ar['starts'].tolist()==scope['plans']['O'])
            full_fused=ar['probabilities']
            original_samples={frame:full_fused[frame].copy() for frame in scope['frames_fixed_before_reading_scores']}
        del full_fused
        for pi,(arrangement,starts) in enumerate(scope['plans'].items()):
            coverage=np.zeros(160,np.int16)
            for start in starts:coverage[start:start+40]+=1
            check(identity+'/'+arrangement+'_five_window_coverage',coverage.sum()==200 and np.all(coverage>0) and
                  np.array_equal(scores['coverage'][pi],coverage))
            for frame in scope['frames_fixed_before_reading_scores']:
                title=identity+f'/{arrangement}/{frame}'
                selected_starts=[s for s in starts if s<=frame<s+40]
                fused=np.maximum.reduce([samples[s,frame] for s in selected_starts])
                if arrangement=='O':check(title+'_fused_probability_exact',np.array_equal(fused,original_samples[frame]))
                independent=independent_match(fused,gt[frame],.5)
                lo,hi=offsets[frame:frame+2]
                for key in ['matched','box_max','peak_x','peak_y','component_area']:
                    check(title+'_'+key,np.array_equal(independent[key],scores[key][pi,lo:hi],equal_nan=True))
                check(title+'_frame_counts',np.array_equal(independent['frame_counts'],scores['frame_counts'][pi,frame]))
                row=frame_csv[arrangement,frame]
                check(title+'_frame_csv',all(int(row[k])==int(v) for k,v in zip(FIELDS,independent['frame_counts'])) and
                      float(row['threshold'])==.5 and int(row['coverage_windows'])==coverage[frame])
                al,ah=areas[arrangement+'_offsets'][frame:frame+2]
                check(title+'_component_areas_and_unmatched',np.array_equal(independent['areas'],areas[arrangement+'_areas'][al:ah]) and
                      np.array_equal(independent['unmatched'],areas[arrangement+'_unmatched'][al:ah]))
                ua=independent['areas'][independent['unmatched'].astype(bool)]
                check(title+'_background_area_csv',int(ua.sum())==int(row['unmatched_component_area']) and
                      int(ua.max(initial=0))==int(row['unmatched_component_max_area']) and
                      int(independent['areas'].max(initial=0))==int(row['component_max_area']))
                audited_rows.append(dict(condition=variant,sequence=seq,arrangement=arrangement,frame=frame,
                    covering_starts=selected_starts,**dict(zip(FIELDS,map(int,independent['frame_counts'])))))
    check('predeclared_total_recomputed_frames',len(audited_rows)==scope['expected_recomputed_frames']==20)
    digest(__file__)
    result=dict(status='PASS',scope=scope,checks=CHECKS,check_count=len(CHECKS),audited_frame_rows=audited_rows,
        tiling_witnesses=tiling,input_sha256=HASHES,elapsed_seconds=time.perf_counter()-began,
        limitations=['Only 20 of 640 new fused frame-arrangement cases independently rescored.',
            'Existing trained direct/tiled GPU witnesses were source-bound and checked, not rerun.',
            'Auditor verified labels but did not author or call scoring implementation; not an independent re-training.',
            'Diagnostic point-in-box matches and unmatched components are not paper Pd/Fa.'])
    (BASE/'INDEPENDENT_RESULT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    text=('# 独立结果抽样审计\n\n'+f'状态：**PASS**；{len(CHECKS)} 项检查，CPU {result["elapsed_seconds"]:.2f} 秒。\n\n'+
        '预先固定每段帧 [0,39,80,120,159]，对一个控制×两个片段×O/S共20个融合帧独立复算。'+
        '从原标注独立构建目标列表，独立8邻域遍历、行优先同值峰、半开框与有序增广匹配；未导入或调用评分器。\n\n'+
        '抽帧的框内最大分数、实际匹配、峰位置、组件面积、帧级前景/框外像素、未匹配组件及面积均与评分产物一致。'+
        '原70913参数结构、发布权重身份、最终512步权重、冻结源码/标签规则及O/S五窗覆盖吻合；16个窗口缓存hash吻合。'+
        '两个片段的6个既有训练后direct/tiled见证均在0.5阈值无判决差异，推理halo16。\n\n'+
        '范围：20/640个融合帧的抽样审计，不是全部预测的独立复算；未再跑GPU或模型实验。'+
        '审计员参与标签核验，独立于评分代码；本结果不替代来源外验证，也不把诊断U当作官方Fa。\n\n'+
        '完整检查与输入SHA见 [INDEPENDENT_RESULT_AUDIT.json](INDEPENDENT_RESULT_AUDIT.json)。\n')
    (BASE/'INDEPENDENT_RESULT_AUDIT.md').write_text(text)
    print(json.dumps({'status':'PASS','checks':len(CHECKS),'audited_frames':len(audited_rows),'seconds':result['elapsed_seconds']}))


if __name__=='__main__':main()


