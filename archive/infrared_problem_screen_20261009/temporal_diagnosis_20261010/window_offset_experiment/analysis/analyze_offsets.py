#!/usr/bin/env python3
"""Analyze frozen natural contiguous-window intervention, preserving primary definition."""
import collections
import csv
import json
from pathlib import Path
import sys

import numpy as np


def run(D):
    Q=D/'temporal_diagnosis_20261010';E=Q/'window_offset_experiment';outroot=E/'analysis'
    sys.path.insert(0,str(Q/'analysis'))
    import analyze as base
    protocol=json.loads((E/'PROTOCOL.json').read_text())
    candidates=json.loads((D/'real_motion_stage2/INSPECTION_QUEUE.json').read_text())['candidates'][:4]
    byseq={c['sequence']:c for c in candidates}
    assert list(byseq)==protocol['all_sequences']
    all_summary=[]
    for seq in ['DJI_0047_3']+[s for s in protocol['all_sequences'] if s!='DJI_0047_3']:
        c=byseq[seq];out=outroot/seq;out.mkdir(parents=True,exist_ok=True)
        data=D/'real_motion_stage2/selected_raw' if seq=='DJI_0028_2' else Q/'additional_clips'/c['candidate_id']
        anns=json.loads((data/'annotations.json').read_text());images=sorted(anns['images'],key=lambda im:im['frame_id'])
        annotation_by_image=collections.defaultdict(list)
        for ann in anns['annotations']:annotation_by_image[ann['image_id']].append(ann)
        gt=[base.prepare_gt(annotation_by_image[im['id']],im['height'],im['width'])[0] for im in images]
        with (Q/'analysis'/seq/'per_track_frame.csv').open() as f:
            baseline={(int(r['local_frame']),float(r['threshold']),int(r['track_id'])):r for r in csv.DictReader(f)}
        files=sorted((Q/'runs'/seq/'windows').glob('window*.npz'))+sorted((E/'runs'/seq/'windows').glob('window*.npz'))
        assert len(files)==11
        context_rows=[];seen_starts=[];inputs={}
        for path in files:
            with np.load(path,allow_pickle=False) as saved:
                p=saved['probabilities'];start=int(saved['start']);end=int(saved['end'])
            assert end-start==40 and p.dtype==np.float32 and p.shape==(40,512,640)
            seen_starts.append(start);inputs[str(path)]=base.digest(path)
            source='original' if start in protocol['reuse_original_starts'] else 'new'
            for fi in range(start,end):
                for th in [.1,.5,.9]:
                    rows,_,_=base.match_frame(p[fi-start],gt[fi],th)
                    for r in rows:
                        context_rows.append(dict(sequence=seq,source_frame=images[fi]['frame_id'],local_frame=fi,
                            window_start=start,window_end=end,slot=fi-start,window_source=source,threshold=th,**r))
        assert sorted(seen_starts)==list(range(0,121,12))
        base.save_csv(out/'per_context_all_tracks.csv',context_rows)
        groups=collections.defaultdict(list)
        for row in context_rows:groups[row['local_frame'],row['threshold'],row['track_id']].append(row)
        frame_rows=[]
        for key,rr in sorted(groups.items()):
            fi,th,tid=key;rr.sort(key=lambda r:r['window_start']);orig=baseline[key]
            original_windows=[r for r in rr if r['window_source']=='original']
            new_windows=[r for r in rr if r['window_source']=='new']
            values=[r['box_max'] for r in rr if r['box_max'] is not None]
            any_original_match=any(r['matched'] for r in original_windows)
            any_new_match=any(r['matched'] for r in new_windows)
            any_ambig=any(r['association_ambiguous'] for r in rr)
            overlap=rr[0]['label_overlap_count']>0;partial=bool(rr[0]['partly_outside'])
            point_states=[r['matched'] for r in rr]
            original_match=int(orig['matched'])
            frame_rows.append({'sequence':seq,'local_frame':fi,'source_frame':images[fi]['frame_id'],
                'track_id':tid,'threshold':th,'selected_track':int(tid==c['track_id']),
                'bbox_w':rr[0]['bbox_w'],'bbox_h':rr[0]['bbox_h'],
                'current_frame_max_box_side':max(rr[0]['bbox_w'],rr[0]['bbox_h']),
                'context_count':len(rr),'new_context_count':len(new_windows),
                'starts':';'.join(str(r['window_start']) for r in rr),'slots':';'.join(str(r['slot']) for r in rr),
                'point_match_count':sum(point_states),'point_match_any':int(any(point_states)),
                'point_match_all':int(all(point_states)),
                'point_disagreement':int(min(point_states)!=max(point_states)),
                'boxmax_min':min(values) if values else None,'boxmax_max':max(values) if values else None,
                'boxmax_range':max(values)-min(values) if values else None,
                'boxmax_threshold_crossing':int(min(values)<th<=max(values)) if values else None,
                'boxmax_positive_contexts':sum(v>=th for v in values),
                'any_matching_ambiguity':int(any_ambig),'label_overlap':int(overlap),
                'label_identical_boxes':rr[0]['label_identical_box_count'],'partly_outside':int(partial),
                'protocol_eligible':int(len(rr)>=2 and not overlap and not any_ambig),
                'conservative_eligible':int(len(rr)>=2 and not overlap and not any_ambig and not partial),
                'original_fused_point_match':original_match,
                'original_fused_boxmax':float(orig['box_max']) if orig['box_max'] else None,
                'any_original_window_point_match':int(any_original_match),
                'any_new_window_point_match':int(any_new_match),
                'oracle_any_context_recovers_original_unmatched':int(not original_match and any(point_states)),
                'oracle_new_context_recovers_original_unmatched':int(not original_match and any_new_match),
                'oracle_recovery_requires_new_context':int(not original_match and not any_original_match and any_new_match)})
        base.save_csv(out/'same_gt_frame_context_summary.csv',frame_rows)

        def summarize(rr):
            many=[r for r in rr if r['context_count']>=2]
            original_unmatched=[r for r in rr if not r['original_fused_point_match']]
            return {'gt_frames':len(rr),'gt_frames_multiple_contexts':len(many),
                'gt_frames_with_new_context':sum(r['new_context_count']>0 for r in rr),
                'point_disagreement_gt_frames':sum(r['point_disagreement'] for r in many),
                'boxmax_crossing_gt_frames':sum(bool(r['boxmax_threshold_crossing']) for r in many),
                'multiple_context_boxmax_range_median':float(np.median([r['boxmax_range'] for r in many])) if many else None,
                'original_fused_unmatched_gt_frames':len(original_unmatched),
                'oracle_any_context_recovers_original_unmatched':sum(r['oracle_any_context_recovers_original_unmatched'] for r in original_unmatched),
                'oracle_recovery_requires_new_context':sum(r['oracle_recovery_requires_new_context'] for r in original_unmatched),
                'original_unmatched_no_context_matches':sum(not r['point_match_any'] for r in original_unmatched),
                'all_contexts_low_boxmax':sum(r['boxmax_max'] is not None and r['boxmax_max']<r['threshold'] for r in rr),
                'partly_outside_gt_frames':sum(r['partly_outside'] for r in rr),
                'label_overlap_gt_frames':sum(r['label_overlap'] for r in rr),
                'matching_ambiguous_gt_frames':sum(r['any_matching_ambiguity'] for r in rr)}

        stats=[];track_summary=[]
        for th in [.1,.5,.9]:
            rr=[r for r in frame_rows if r['threshold']==th]
            selected=[r for r in rr if r['selected_track']]
            record={'threshold':th,'all_gt':summarize(rr),
                'protocol_subset':summarize([r for r in rr if r['protocol_eligible']]),
                'conservative_fully_in_frame_subset':summarize([r for r in rr if r['conservative_eligible']]),
                'selected_track':summarize(selected),
                'selected_protocol_subset':summarize([r for r in selected if r['protocol_eligible']]),
                'selected_conservative_subset':summarize([r for r in selected if r['conservative_eligible']]),
                'current_frame_max_side_le16_all':summarize([r for r in rr if r['current_frame_max_box_side']<=16]),
                'current_frame_max_side_le16_conservative':summarize([r for r in rr if r['current_frame_max_box_side']<=16 and r['conservative_eligible']]),
                'current_frame_max_side_le12_all':summarize([r for r in rr if r['current_frame_max_box_side']<=12]),
                'current_frame_max_side_le12_conservative':summarize([r for r in rr if r['current_frame_max_box_side']<=12 and r['conservative_eligible']])}
            if seq=='DJI_0047_3':
                low=[r for r in selected if 18<=r['source_frame']<=159]
                record['previously_observed_source_18_to_159']=summarize(low)
                record['previously_observed_18_to_159_multiple_contexts']=summarize([r for r in low if r['context_count']>=2])
                record['previously_observed_18_to_159_has_new_context']=summarize([r for r in low if r['new_context_count']>0])
                record['recovered_source_frames']=[r['source_frame'] for r in low if r['oracle_any_context_recovers_original_unmatched']]
                record['still_single_context_source_frames']=[r['source_frame'] for r in low if r['context_count']==1]
            stats.append(record)
            for tid in sorted({r['track_id'] for r in rr}):
                group=[r for r in rr if r['track_id']==tid]
                track_summary.append(dict(track_id=tid,threshold=th,selected_track=tid==c['track_id'],
                    all_frames=summarize(group),conservative_subset=summarize([r for r in group if r['conservative_eligible']])))
        result={'sequence':seq,'selected_track_id':c['track_id'],'stats':stats,
            'window_starts':sorted(seen_starts),'primary_definition':'same GT and frame; >=2 real contexts, no GT overlap, no matching ambiguity in any context',
            'conservative_additional_filter':'also excludes partly out-of-frame GT, without changing frozen primary counts',
            'protocol_sha256':base.digest(E/'PROTOCOL.json'),'code_sha256':base.digest(Path(__file__)),
            'primary_matching_code_sha256':base.digest(Q/'analysis/analyze.py'),'window_input_sha256':inputs,
            'limits':['Extra windows and GT-guided any-context matching are an oracle diagnosis, not method performance.',
                      'Surrounding content, time slots and boundary padding vary together; no specific cause is isolated.',
                      'Original max-fused components can merge; rescue requiring new context is separately counted.',
                      'Only source frames with >=2 windows received multiple contexts; last12 frames have just one context.',
                      'Continuous visibility remains unverified; bbox labels and representative point diagnostics are limited.',
                      'No single-frame baseline, model modification, retraining or threshold selection was introduced.']}
        base.save_json(out/'SUMMARY.json',result);base.save_json(out/'TRACK_SUMMARY.json',track_summary)
        base.save_json(out/'ARTIFACT_SHA256.json',{p.name:base.digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
        print(json.dumps({'sequence':seq,'stats':stats}),flush=True)
        all_summary.append(result)
    all_summary.sort(key=lambda r:protocol['all_sequences'].index(r['sequence']))
    base.save_json(outroot/'ALL_CLIPS_SUMMARY.json',all_summary)
    lines=['# 自然滑窗起点对照：CPU诊断结果','',
        '固定官方RGB-T DeepPro-Plus权重、seed46、原始真实帧顺序。原5窗口加预先固定6窗口，共11个起点0,12,…,120。没有改阈值或模型。', '',
        '## 全轨迹，同一GT同一帧的窗口敏感性（阈值0.5）','',
        '计数单位是GT×帧，不是窗口对。保守子集要求至少2个窗口、GT无重叠、任一窗口匹配均无歧义，并且GT完整位于画面内。', '',
        '| 片段 | 所有多窗口GT帧 | 点匹配分歧 | 框max跨阈值 | 保守GT帧 | 保守点分歧 | 保守max跨阈值 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for result in all_summary:
        r=next(s for s in result['stats'] if s['threshold']==.5);a=r['all_gt'];b=r['conservative_fully_in_frame_subset']
        lines.append(f"| {result['sequence']} | {a['gt_frames_multiple_contexts']} | {a['point_disagreement_gt_frames']} | {a['boxmax_crossing_gt_frames']} | {b['gt_frames']} | {b['point_disagreement_gt_frames']} | {b['boxmax_crossing_gt_frames']} |")
    lines+=['','## 尺度分层（阈值0.5，保守子集）','',
        '按每个当前帧原标注框的max(width,height)分层，≤12是≤16的嵌套子集。不是按整条轨迹最大尺度分层，也不改变主协议。', '',
        '| 片段 | ≤16 GT帧 | ≤16点分歧 | ≤16框max跨阈值 | ≤12 GT帧 | ≤12点分歧 | ≤12框max跨阈值 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for result in all_summary:
        r=next(s for s in result['stats'] if s['threshold']==.5)
        a=r['current_frame_max_side_le16_conservative'];b=r['current_frame_max_side_le12_conservative']
        lines.append(f"| {result['sequence']} | {a['gt_frames']} | {a['point_disagreement_gt_frames']} | {a['boxmax_crossing_gt_frames']} | {b['gt_frames']} | {b['point_disagreement_gt_frames']} | {b['boxmax_crossing_gt_frames']} |")
    lines+=['','## 预选轨迹的窗口诊断（阈值0.5）','',
        '任意窗口匹配只表示GT辅助的可达响应，不能称部署算法改善。原max融合可能改变组件形状，因此同时列出必须依靠新窗口才恢复的数量。','',
        '| 片段 / 轨迹 | 原融合未匹配 | 任意窗口恢复 | 必须依靠新增窗口 | 仍所有窗口未匹配 | 所有窗口框max低于0.5 |',
        '|---|---:|---:|---:|---:|---:|']
    for result in all_summary:
        a=next(s for s in result['stats'] if s['threshold']==.5)['selected_track']
        lines.append(f"| {result['sequence']} / {result['selected_track_id']} | {a['original_fused_unmatched_gt_frames']} | {a['oracle_any_context_recovers_original_unmatched']} | {a['oracle_recovery_requires_new_context']} | {a['original_unmatched_no_context_matches']} | {a['all_contexts_low_boxmax']} |")
    r=next(s for x in all_summary if x['sequence']=='DJI_0047_3' for s in x['stats'] if s['threshold']==.5)
    a=r['previously_observed_source_18_to_159'];b=r['previously_observed_18_to_159_multiple_contexts'];c=r['previously_observed_18_to_159_has_new_context']
    lines+=['','## DJI_0047_3原帧18–159的142帧连续低响应','',
        f"这142帧中，{b['gt_frames']}帧拥有至少2个窗口，{c['gt_frames']}帧实际得到新增窗口。任意窗口恢复{a['oracle_any_context_recovers_original_unmatched']}帧，其中{a['oracle_recovery_requires_new_context']}帧必须依靠新增窗口；恢复帧为{r['recovered_source_frames']}。",'',
        f"仍只有一个窗口的原始帧为{r['still_single_context_source_frames']}，不能称其经过多个上下文检验。",'',
        '## 解释范围','',
        '- 若同一GT同一帧输出随窗口变化，说明窗口上下文/位置会影响响应；不能区分内容变化、绝对slot或时间padding。',
        '- 持续低响应若未被新窗口救回，只排除本次有限起点集合下的恢复，不能证明所有窗口策略无效。',
        '- 帧数高度相关，不作独立样本显著性检验。真实目标连续可见性和暗亮极性/局部对比度仍需核对。',
        '- bbox点响应诊断不等于官方Pd/Fa/AP。全部轨迹、三阈值、标注歧义、越界情况和逐窗口原始记录均保留。',
        '- 单窗口预测数增加，GT辅助任意窗口上包络不可写成新方法收益。','',
        '## 文件','',
        '- `ALL_CLIPS_SUMMARY.json`：全部片段、三阈值、完整与保守子集。',
        '- 每片 `per_context_all_tracks.csv`：每窗口每GT原始匹配和响应。',
        '- 每片 `same_gt_frame_context_summary.csv`：同帧汇总、slot、上下文数、原融合与窗口oracle恢复。',
        '- 每片 `TRACK_SUMMARY.json`、`ARTIFACT_SHA256.json`：全部轨迹及复查哈希。','']
    (outroot/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':run(Path(sys.argv[1]))
