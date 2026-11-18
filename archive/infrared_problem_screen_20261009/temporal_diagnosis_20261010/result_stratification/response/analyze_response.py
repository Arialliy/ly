#!/usr/bin/env python3
"""CPU/stdlib-only descriptive reanalysis of frozen saved predictions and diagnostics."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def boolean(x):
    return str(x).lower() in ('true', '1')


def number(x):
    if x in (None, ''):
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(rows):
    bytrack = defaultdict(list)
    for r in rows:
        bytrack[(r['sequence'], r['track_id'])].append(r)
    counts = Counter({k: sum(r['low_box_response'] for r in rr) for k, rr in bytrack.items()})
    low = sum(counts.values())
    n = len(rows)
    return {
        'target_frames': n,
        'sequences': len(set(r['sequence'] for r in rows)),
        'tracks': len(bytrack),
        'matched_frames': sum(r['matched'] for r in rows),
        'point_match_fraction': sum(r['matched'] for r in rows) / n,
        'response_frames': n - low,
        'response_fraction': (n - low) / n,
        'low_box_response_frames': low,
        'low_box_response_fraction': low / n,
        'tracks_with_low_response': sum(v > 0 for v in counts.values()),
        'sequences_with_low_response': len(set(k[0] for k, v in counts.items() if v > 0)),
        'top_track_share_of_low_frames': max(counts.values()) / low if low else 0,
        'top_low_track': ':'.join(map(str, counts.most_common(1)[0][0])) if low else '',
        'track_macro_response_fraction': statistics.mean(sum(not r['low_box_response'] for r in rr)/len(rr) for rr in bytrack.values()),
        'track_median_response_fraction': median(sum(not r['low_box_response'] for r in rr)/len(rr) for rr in bytrack.values()),
        'box_max_median': median(r['box_max'] for r in rows),
        'max_side_median': median(r['max_side_px'] for r in rows),
        'signed_contrast_median': median(r['signed_mean_contrast'] for r in rows),
        'absolute_standardized_mean_contrast_median': median(r['absolute_standardized_mean_contrast'] for r in rows),
        'local_ring_std_median': median(r['image_ring_std'] for r in rows),
        'motion_valid_frames': sum(r['motion_valid'] for r in rows),
        'adjusted_speed_valid_median': median(r['adjusted_speed_px_per_frame'] for r in rows if r['motion_valid']),
        'response_without_pointmatch_frames': sum(not r['low_box_response'] and not r['matched'] for r in rows),
        'pointmatch_without_response_frames': sum(r['low_box_response'] and r['matched'] for r in rows),
    }


def run_records(rows):
    result = []
    active = []
    previous_frame = None
    for r in sorted(rows, key=lambda x: x['source_frame']):
        frame = r['source_frame']
        if (not r['low_box_response']) or (previous_frame is not None and frame != previous_frame + 1):
            if active:
                result.append(active)
                active = []
        if r['low_box_response']:
            active.append(frame)
        previous_frame = frame
    if active:
        result.append(active)
    return [{'start_source_frame': a[0], 'end_source_frame': a[-1], 'frames': len(a)} for a in result]


def make_report(summary, focus):
    tables = summary['tables']
    def chosen(name, population='all_tracks', subset='clean_inframe_no_gt_overlap', threshold=.5):
        return [r for r in tables[name] if r['population']==population and r['subset']==subset and r['threshold']==threshold]
    def table(rows, fields):
        header = '| ' + ' | '.join(label for key,label in fields) + ' |\n'
        header += '| ' + ' | '.join('---' for _ in fields) + ' |\n'
        def fmt(value):
            return ('%.2f'%value) if isinstance(value,float) else str(value)
        return header + ''.join('| ' + ' | '.join(fmt(r.get(key,'')) for key,label in fields) + ' |\n' for r in rows)
    fields = [('target_frames','目标帧'),('tracks','IDs'),('response_frames','框响应帧'),('matched_frames','点匹配帧'),('low_box_response_frames','低框响应帧')]
    report = '# 原始输出：按结果分层的探索性诊断\n\n'
    report += ('本报告只重算既有预测、原始框标注和图像/运动诊断。主分析是原始 T40/stride36/last-anchored、重叠取最大值的输出；阈值 0.5，清洁条件为框完整在图内且不与其他 GT 框重叠。低框响应定义为框内最大预测概率 < 阈值，与点匹配失败分开。所有计数的单位是目标帧，目标帧不是独立样本；没有逐帧显著性检验。分析规范是在看过主结果后固定，属于探索性分析。\n\n')
    report += ('**优先观察的能力缺口：在两个道路片段中，确有标注位置移动、框较小、相对环形背景均值更亮的目标持续或反复低响应；主要落在行人，但正的原始均值差常伴随较大的局部背景标准差。** 当前数据支持这种条件组合下的响应缺口，尚不能把原因归给运动融合、类别或训练缺样。静止/近静止船类提供了另外的失败条件，但两段 DJI_0028 可能来自同一拍摄场景，而且原任务针对运动目标，不能把这项边界现象写成违背原方法承诺。\n\n')
    report += '## 样本和基本结果\n\n'
    report += table([dict(r,rate=100*r['response_fraction']) for r in chosen('sequence')],[('sequence','片段')]+fields+[('rate','框响应率 %')])+'\n'
    report += '4 个片段共 640 图像帧、4,973 标注目标帧、84 个原始 track ID；39 个 ID 至少有 40 个标注帧。清洁主集保留 3,553 目标帧、80 个 ID；长轨迹清洁子集保留 2,990 目标帧、37 个 ID。ID 数不是独立物体数，有缺帧和潜在 ID 碎片；未擅自合并。\n\n'
    report += table([dict(r,rate=100*r['response_fraction']) for r in tables['overall'] if r['threshold']==.5],[('population','轨迹集'),('subset','过滤')]+fields+[('rate','框响应率 %')])+'\n'
    report += ('主集在 0.1/0.5/0.9 的框响应帧分别为 1,685/1,605/1,516，分母均为 3,553；低响应不是仅靠把阈值由 0.5 降到 0.1 就能消除。0.5 下有 57 帧框内有响应而点匹配失败，因此不能把点匹配曲线直接称为“无响应”。额外排除所有可用自然窗口任一关联歧义及原始融合关联歧义后，删去 662 帧，其中 599 帧原本有框响应；过滤会选择性改变目标组成，34.80% 与主集 45.17% 不能解释成模型性能下降。\n\n')
    report += '## 优先条件：可用运动残差 >0.5、最大框边 <=16、原始均值差 >10\n\n'
    report += ('以下是父任务要求补查的已注册分箱边界的组合，属于探索性追加切片，并非事先注册的主要假设。运动是 t→t+5 的相机补偿后图像位置残差，分配给起始帧，不是物理速度或人工确认运动；原始均值差为图像框均值减局部环均值，“>10”不保证目标相对于局部波动的标准化对比度强。\n\n')
    fr = [r for r in focus['tables']['sequence'] if r['population']=='all_tracks' and r['subset']=='clean_inframe_no_gt_overlap' and r['threshold']==.5]
    report += table(fr,[('sequence','片段')]+fields)+'\n'
    fr = [r for r in focus['tables']['category'] if r['population']=='all_tracks' and r['subset']=='clean_inframe_no_gt_overlap' and r['threshold']==.5]
    report += table(fr,[('category','类别')]+fields)+'\n'
    report += ('主集共 315 条符合条件，125 条低响应；两个道路片段分别为 60/140、65/113 低响应，船类为 0/62。行人贡献 112/203 低响应，分布于 6 个有低响应的 ID、2 个道路片段；最大 ID 4621500 贡献 53/112（47.3%），不能由单一长轨迹解释全部，但也不能忽视该轨迹的权重。\n\n')
    fr = [r for r in focus['tables']['sequence__track_id__category'] if r['population']=='all_tracks' and r['subset']=='clean_inframe_no_gt_overlap' and r['threshold']==.5 and r['category']=='pedestrian']
    report += table(fr,[('sequence','片段'),('track_id','ID'),('target_frames','符合条件帧'),('low_box_response_frames','低框响应帧'),('absolute_standardized_mean_contrast_median','中位 |z|'),('local_ring_std_median','中位局部环 std')])+'\n'
    fr = [r for r in focus['tables']['sequence__category__absolute_standardized_contrast_bin'] if r['population']=='all_tracks' and r['subset']=='clean_inframe_no_gt_overlap' and r['threshold']==.5 and r['category']=='pedestrian']
    report += table(fr,[('sequence','片段'),('absolute_standardized_contrast_bin','|z| 分箱')]+fields)+'\n'
    report += ('行人 112 条低响应中，只有 11 条来自 |z|>=1（总共 28 条此类符合条件帧）；其余 101 条来自 |z|<1。因此“正对比度、小、在动也漏”是结果描述，进一步称“清晰目标也普遍不响应”证据不足。道路片段和船类片段的背景标准差、类别、速度、尺度均不同，不能由两种场景的平均率识别单一原因。\n\n')
    report += '## 分箱和混杂\n\n'
    for name, label in [('scale_bin','最大框边'),('signed_contrast_bin','有符号原始均值差'),('absolute_standardized_contrast_bin','绝对标准化均值差'),('motion_bin','有效图像残差 px/frame')]:
        report += table([dict(r,rate=100*r['response_fraction']) for r in chosen(name)],[(name,label)]+fields+[('rate','框响应率 %')])+'\n'
    report += ('尺度结果不支持“越小越失败”：<=8 的总体框响应 475/650（73.08%），(8,12] 为 292/1,033（28.27%），但前者包含船类高响应轨迹。两个船类片段 <=8 均为 100%，两个道路片段 <=8 分别 46/88、60/193；必须按片段和类别交叉看。\n\n')
    report += table(chosen('sequence__category'),[('sequence','片段'),('category','类别')]+fields)+'\n'
    report += ('船类只出现在两个 DJI_0028 片段，公交只在 DJI_0032_5。骑行者的低框响应在两个道路片段都较多（54/92、217/336），但尺度/背景与汽车不同。行人总体低响应 254/472 中 143 条来自 DJI_0047_3 唯一行人 ID；不能用总体行人率概括所有片段。\n\n')
    report += table(chosen('sequence__absolute_standardized_contrast_bin'),[('sequence','片段'),('absolute_standardized_contrast_bin','|z| 分箱')]+fields)+'\n'
    report += ('|z|<0.5 的低响应 1,333/1,984，|z|>=1 的低响应 215/897；这个总体差异在船类片段最大。道路片段的分箱不单调，例如 DJI_0032_5 的 |z|<0.5 响应 500/734，而 [0.5,1) 为 230/382；不能把背景对比度当唯一因果解释。完整 sequence×category×contrast 和 sequence×category×motion 表已另存。\n\n')
    report += table(chosen('sequence__motion_bin'),[('sequence','片段'),('motion_bin','有效残差分箱')]+fields)+'\n'
    report += ('<=0.15 的 906 条中有 878 条来自船类两段；低响应 869/906，最大单 ID 只占这些低响应的 15.65%，所以并非仅一条大轨迹。但是近静止与低对比度、船类、场景一起变化。<=0.15 且 |z|>=1 的 87 条有 67 条低响应，全部来自 DJI_0028_4 的 3 个 ID；这说明仅靠低对比度无法概括所有失败，也不足以证明运动是原因。无有效补偿值的 626 条单独保留为 invalid，不拿原始位移替换。\n\n')
    report += '## 真正连续的低框响应\n\n'
    rr = [r for r in summary['low_response_run_summary'] if r['population']=='all_tracks' and r['subset']=='clean_inframe_no_gt_overlap' and r['threshold']==.5 and r['longest_consecutive_low_box_response']>=40]
    report += table(rr,[('sequence','片段'),('track_id','ID'),('category','类别'),('target_frames','清洁目标帧'),('low_box_response_frames','低响应帧'),('longest_consecutive_low_box_response','最长连续低响应')])+'\n'
    report += ('这些连续段按 source_frame 连续性计算；漏标帧、被清洁规则排除的帧都会切断连续段。六条船轨迹在两个可能同场景片段中重复；道路片段有一条骑行者 40 帧和一条行人 142 帧。这里不是“点没匹配”的连续段，也不表示整个连续段始终满足上面的移动/小框/正差值组合。所有轨迹、三阈值、三个清洁条件与长轨迹子集的逐段起止和长度分布均已保存。\n\n')
    report += '## 能据此说到哪里\n\n'
    report += ('结果已经把可重复的失败条件缩到：两个道路片段的移动、小框、正原始差值目标，尤其是背景波动下标准化对比度不强的行人；以及同拍摄场景的近静止船类。现有数据还不能分离类别、外观、局部背景、轨迹运动和上下文的贡献。没有在这里建议或修改网络、训练数据、损失、阈值。官方训练标注中是否存在这些类别以及真实 Gaussian mask/检查点曝光须与父任务的数据覆盖审计共同解释，不能从漏检倒推“训练集中没有行人”。\n\n')
    report += ('限制：框标注不等于像素目标掩码，持续标注不证明每帧可见；负框减环均值差不等于热极性；四个片段预先筛选、两个船片段可能同场景；所有轨迹数均是原始标注 ID，缺帧/ID 碎片使独立样本数更小；原始响应是融合后的框内最大概率，可能受覆盖/饱和/形状影响，因此点匹配与响应始终分别列出。\n\n')
    report += '## 重现\n\n```bash\npython3 analyze_response.py\n```\n\n仅 Python 标准库、CPU，无安装和推理。输出 JOINED_TRACK_FRAME.csv、SUMMARIES.json、SUMMARY_*.csv、FOCUS_*.csv、FOCUS_SUMMARIES.json、LOW_RESPONSE_RUNS.csv、LOW_RESPONSE_RUN_SUMMARY.csv、ANNOTATION_TRACK_AUDIT.csv 和本报告。SUMMARIES.json 记录所有输入 SHA256。图像标准化代理为 abs(image_box_mean-image_ring_mean)/max(image_ring_std,1)，1 的标准差下限沿用已保存 motion/PROTOCOL.json；预测 box_mean 与 image_box_mean 字段明确分开。\n'
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    q, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=True)
    spec_path = q/'result_stratification/ANALYSIS_SPEC.json'
    spec = json.loads(spec_path.read_text())
    input_paths = [spec_path]
    sequences = ['DJI_0028_2', 'DJI_0028_4', 'DJI_0032_5', 'DJI_0047_3']
    joined = []
    annotation_audit = []
    for sequence in sequences:
        annpath = (q.parent/'real_motion_stage2/selected_raw/annotations.json' if sequence == sequences[0]
                   else next((q/'additional_clips').glob(sequence+'*/annotations.json')))
        motiondir = q/'motion' if sequence == sequences[0] else q/'motion'/sequence
        path = q/'analysis'/sequence/'per_track_frame.csv'
        contrastpath = motiondir/'TRACK_FRAME_DIAGNOSTICS.csv'
        motionpath = motiondir/'LAG5_TRACK_MOTION.csv'
        contextpath = q/'window_offset_experiment/analysis'/sequence/'per_context_all_tracks.csv'
        input_paths += [annpath, path, contrastpath, motionpath, contextpath]
        annotation = json.loads(annpath.read_text())
        categories = {x['id']: x['name'] for x in annotation['categories']}
        images = {x['id']: x for x in annotation['images']}
        ann = {str(x['id']): x for x in annotation['annotations']}
        track_frames = defaultdict(set)
        for a in annotation['annotations']:
            track_frames[str(a['track_id'])].add(images[a['image_id']]['frame_id'])
        for track, frames in track_frames.items():
            ordered = sorted(frames)
            missing = sum(b-a-1 for a,b in zip(ordered,ordered[1:]))
            annotation_audit.append({'sequence': sequence, 'track_id': track,
                'annotation_frames':len(frames), 'at_least_40_annotation_frames': len(frames)>=40,
                'first_source_frame':ordered[0], 'last_source_frame':ordered[-1],
                'missing_frames_between_first_and_last':missing,
                'annotations_in_track':sum(str(a['track_id'])==track for a in annotation['annotations'])})
        contrast = {(r['track_id'], r['raw_frame']): r for r in read_csv(contrastpath)}
        motion = {(r['track_id'], r['raw_frame_from']): r for r in read_csv(motionpath)}
        ambiguous = defaultdict(bool)
        contexts = Counter()
        for r in read_csv(contextpath):
            key = (r['track_id'], r['source_frame'], float(r['threshold']))
            ambiguous[key] |= boolean(r['association_ambiguous'])
            contexts[key] += 1
        rows = read_csv(path)
        seen = set()
        for r in rows:
            key = (r['track_id'],r['source_frame'],float(r['threshold']))
            assert key not in seen, key
            seen.add(key)
            a = ann[r['annotation_id']]
            im = images[a['image_id']]
            assert int(r['source_frame']) == im['frame_id']
            assert int(r['track_id']) == a['track_id']
            assert [float(r[k]) for k in ('bbox_x','bbox_y','bbox_w','bbox_h')] == a['bbox']
            c = contrast[(r['track_id'],r['source_frame'])]
            assert c['annotation_id'] == r['annotation_id']
            m = motion.get((r['track_id'],r['source_frame']))
            assert contexts[key] > 0
            threshold = float(r['threshold'])
            assert threshold in spec['thresholds']
            maxside = max(float(r['bbox_w']),float(r['bbox_h']))
            cv = boolean(c['contrast_valid'])
            signed = number(c.get('mean_difference')) if cv else None
            std = number(c.get('ring_std')) if cv else None
            absolute = abs(signed)/max(std,1) if cv and signed is not None and std is not None else None
            raw_adjusted = number(m['adjusted_speed_px_per_frame']) if m else None
            valid = bool(m and boolean(m['camera_adjustment_valid']) and raw_adjusted is not None)
            clean = not boolean(r['partly_outside']) and int(r['label_overlap_count']) == 0
            anyamb = ambiguous[key] or boolean(r['association_ambiguous'])
            j = dict(r)
            j.update(sequence=sequence, source_frame=int(r['source_frame']), local_frame=int(r['local_frame']),
                threshold=threshold, matched=int(r['matched']), box_max=float(r['box_max']),
                category=categories[a['category_id']], category_id=a['category_id'],
                track_annotation_frames=len(track_frames[r['track_id']]),
                at_least_40_annotation_frames=len(track_frames[r['track_id']])>=40,
                clean_inframe_no_gt_overlap=clean,
                ambiguous_any_natural_context_or_fused=anyamb,
                clean_plus_no_context_ambiguity=clean and not anyamb,
                available_natural_contexts=contexts[key],
                max_side_px=maxside,
                scale_bin=('<=8' if maxside<=8 else '(8,12]' if maxside<=12 else '(12,16]' if maxside<=16 else '>16'),
                contrast_valid=cv, image_box_mean=number(c.get('box_mean')), image_ring_mean=number(c.get('ring_mean')),
                image_ring_std=std, signed_mean_contrast=signed,
                absolute_standardized_mean_contrast=absolute,
                signed_contrast_bin=('invalid' if signed is None else '<-10' if signed < -10 else '[-10,10]' if signed <=10 else '>10'),
                absolute_standardized_contrast_bin=('invalid' if absolute is None else '<0.5' if absolute<.5 else '[0.5,1)' if absolute<1 else '>=1'),
                motion_join_present=m is not None,
                camera_adjustment_valid=boolean(m['camera_adjustment_valid']) if m else False,
                global_registration_valid=boolean(m['global_registration_valid']) if m else False,
                source_center_inside_inlier_hull=boolean(m['source_center_inside_inlier_hull']) if m else False,
                nearest_inlier_distance_px=number(m['nearest_inlier_distance_px']) if m else None,
                motion_raw_frame_to=int(m['raw_frame_to']) if m else None,
                raw_speed_px_per_frame=number(m['raw_speed_px_per_frame']) if m else None,
                camera_displacement_at_center_px_per_frame=number(m['camera_displacement_at_center_px_per_frame']) if m else None,
                adjusted_speed_raw_csv_value=raw_adjusted,
                adjusted_speed_px_per_frame=raw_adjusted if valid else None,
                motion_valid=valid,
                motion_bin=('invalid' if not valid else '<=0.15' if raw_adjusted<=.15 else '(0.15,0.5]' if raw_adjusted<=.5 else '(0.5,1]' if raw_adjusted<=1 else '>1'),
                low_box_response=float(r['box_max'])<threshold)
            joined.append(j)
        assert len(rows) == len(ann)*len(spec['thresholds'])
    write_csv(out/'JOINED_TRACK_FRAME.csv',joined)
    write_csv(out/'ANNOTATION_TRACK_AUDIT.csv',annotation_audit)
    dimensions = [(),('sequence',),('sequence','track_id','category'),('category',),('scale_bin',),
        ('signed_contrast_bin',),('absolute_standardized_contrast_bin',),('motion_bin',),
        ('sequence','category'),('sequence','scale_bin'),('sequence','signed_contrast_bin'),
        ('sequence','absolute_standardized_contrast_bin'),('sequence','motion_bin'),
        ('category','scale_bin'),('category','signed_contrast_bin'),('category','absolute_standardized_contrast_bin'),
        ('category','motion_bin'),('sequence','category','scale_bin'),
        ('sequence','category','signed_contrast_bin'),('sequence','category','absolute_standardized_contrast_bin'),
        ('sequence','category','motion_bin'),('scale_bin','absolute_standardized_contrast_bin'),
        ('motion_bin','absolute_standardized_contrast_bin'),
        ('sequence','category','motion_bin','absolute_standardized_contrast_bin')]
    subsets = {'all_annotations':lambda r:True,
        'clean_inframe_no_gt_overlap':lambda r:r['clean_inframe_no_gt_overlap'],
        'clean_plus_no_context_ambiguity':lambda r:r['clean_plus_no_context_ambiguity']}
    tables = defaultdict(list)
    run_summary = []
    all_runs = []
    for population in ('all_tracks','tracks_ge40_annotation_frames'):
        for subset, select in subsets.items():
            for threshold in spec['thresholds']:
                rr = [r for r in joined if r['threshold']==threshold and select(r)
                      and (population=='all_tracks' or r['at_least_40_annotation_frames'])]
                metadata = {'population':population,'subset':subset,'threshold':threshold}
                for dims in dimensions:
                    groups = defaultdict(list)
                    for r in rr:
                        groups[tuple(r[k] for k in dims)].append(r)
                    name = '__'.join(dims) if dims else 'overall'
                    for vals, group in sorted(groups.items()):
                        tables[name].append(dict(metadata,**dict(zip(dims,vals)),**summarize(group)))
                tracks = defaultdict(list)
                for r in rr:
                    tracks[(r['sequence'],r['track_id'],r['category'])].append(r)
                for (sequence,track,category), group in sorted(tracks.items()):
                    runs = run_records(group)
                    base = dict(metadata,sequence=sequence,track_id=track,category=category)
                    run_summary.append(dict(base,target_frames=len(group),
                        low_box_response_frames=sum(r['low_box_response'] for r in group),
                        longest_consecutive_low_box_response=max([r['frames'] for r in runs],default=0),
                        low_response_run_count=len(runs),
                        median_low_response_run_length=median(r['frames'] for r in runs),
                        run_length_1=sum(r['frames']==1 for r in runs),
                        run_length_2_to_4=sum(2<=r['frames']<=4 for r in runs),
                        run_length_5_to_19=sum(5<=r['frames']<=19 for r in runs),
                        run_length_20_to_39=sum(20<=r['frames']<=39 for r in runs),
                        run_length_ge40=sum(r['frames']>=40 for r in runs)))
                    all_runs.extend(dict(base,**r) for r in runs)
    for name,rows in tables.items():
        write_csv(out/('SUMMARY_'+name+'.csv'),rows)
    write_csv(out/'LOW_RESPONSE_RUN_SUMMARY.csv',run_summary)
    write_csv(out/'LOW_RESPONSE_RUNS.csv',all_runs)
    focus_tables = defaultdict(list)
    focus_rows = [r for r in joined if r['motion_valid'] and r['adjusted_speed_px_per_frame']>.5
        and r['max_side_px']<=16 and r['signed_mean_contrast'] is not None and r['signed_mean_contrast']>10]
    write_csv(out/'FOCUS_JOINED_TRACK_FRAME.csv', focus_rows)
    for population in ('all_tracks','tracks_ge40_annotation_frames'):
        for subset, select in subsets.items():
            for threshold in spec['thresholds']:
                rr = [r for r in focus_rows if r['threshold']==threshold and select(r)
                    and (population=='all_tracks' or r['at_least_40_annotation_frames'])]
                for dims in [(),('sequence',),('category',),('sequence','track_id','category'),
                    ('absolute_standardized_contrast_bin',),('sequence','category','absolute_standardized_contrast_bin')]:
                    groups = defaultdict(list)
                    for r in rr:
                        groups[tuple(r[k] for k in dims)].append(r)
                    name = '__'.join(dims) if dims else 'overall'
                    for vals, group in sorted(groups.items()):
                        focus_tables[name].append(dict(population=population,subset=subset,threshold=threshold,
                            **dict(zip(dims,vals)),**summarize(group)))
    focus = {'criterion':'valid adjusted image residual >0.5 px/frame AND maximum box side <=16 px AND image box-minus-ring mean >10',
        'status':'Exploratory supplemental conjunction of fixed bin boundaries requested after primary stratification; not a preregistered hypothesis',
        'tables':dict(focus_tables)}
    for name, rows in focus_tables.items():
        write_csv(out/('FOCUS_'+name+'.csv'),rows)
    (out/'FOCUS_SUMMARIES.json').write_text(json.dumps(focus,ensure_ascii=False,indent=2)+'\n')
    summary = {'status':'complete','spec':spec,'inputs':[
        {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in input_paths],
        'joined_rows':len(joined),'unique_target_frames':len(joined)//len(spec['thresholds']),
        'annotation_track_audit':annotation_audit,
        'standardization':'abs(image_box_mean - image_ring_mean) / max(image_ring_std,1); same std floor as saved motion protocol',
        'original_response':'boxmax from saved original max-fused T40/stride36/last-anchored outputs',
        'ambiguity':'OR across all 11 natural contexts available for this target-frame and original fused output, per threshold',
        'run_rule':'Strict consecutive source frames; missing annotation frames and excluded frames break runs; low means box_max < threshold',
        'tables':dict(tables),'low_response_run_summary':run_summary}
    (out/'SUMMARIES.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'REPORT.md').write_text(make_report(summary,focus))
    print(json.dumps({'joined_rows':summary['joined_rows'],'target_frames':summary['unique_target_frames'],
        'tracks':len(annotation_audit),'tables':len(tables),'run_rows':len(all_runs)},indent=2))


if __name__ == '__main__':
    main()
