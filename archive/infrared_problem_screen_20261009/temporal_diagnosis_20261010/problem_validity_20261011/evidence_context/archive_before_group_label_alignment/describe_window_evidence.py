#!/usr/bin/env python3
"""Describe every geometric-small system opportunity using original labels only."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT=Path(__file__).resolve().parent
V=ROOT.parent
P=V.parent/'stage1_goal'
CASES={'original_window_match_lost_by_fusion','extra_window_only_opportunity'}
OFFICIAL={0,36,72,108,120}
NEW={'DJI_0026_2','DJI_0067_2'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row['sequence'],int(row['frame']),int(row['annotation_id'])


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path,rows):
    assert rows
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def usable(ann,im):
    x,y,w,h=map(float,ann['bbox'])
    return (not ann.get('ignore',0) and all(math.isfinite(v) for v in [x,y,w,h]) and w>0 and h>0
            and min(im['width'],x+w)>max(0,x) and min(im['height'],y+h)>max(0,y))


def stats(values,prefix):
    return {prefix+'_min':min(values) if values else '',
            prefix+'_max':max(values) if values else '',
            prefix+'_mean':statistics.mean(values) if values else '',
            prefix+'_median':statistics.median(values) if values else ''}


def p90(values):
    if not values:return ''
    ordered=sorted(values);j=(len(values)-1)*0.9
    lo=math.floor(j);hi=math.ceil(j)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(j-lo)


def main():
    outputs=['WINDOW_EVIDENCE.csv','OBJECT_CONTEXT.csv','SEQUENCE_SUMMARY.csv','SUMMARY.json','README.md']
    for name in outputs:
        if (ROOT/name).exists():raise FileExistsError(ROOT/name)
    input_hashes={n:sha(V/n) for n in ['OBJECTS.csv','WINDOWS.csv']}
    all_objects=read_csv(V/'OBJECTS.csv')
    objects=[r for r in all_objects if r['small_geometry']=='1' and r['system_case'] in CASES]
    assert len(objects)==157
    object_lookup={key(r):r for r in objects}
    assert len(object_lookup)==len(objects)
    window_lookup=collections.defaultdict(list)
    for row in read_csv(V/'WINDOWS.csv'):
        if key(row) in object_lookup:window_lookup[key(row)].append(row)
    data={};source_hashes={}
    for seq in sorted({r['sequence'] for r in objects}):
        path=P/('data' if seq in NEW else 'data_diagnostics')/seq/'annotations.json'
        raw=json.loads(path.read_text());images=raw['images']
        assert len(images)==160
        source=[int(Path(im['file_name']).stem) for im in images]
        assert source==list(range(source[0],source[0]+160))
        loc={im['id']:i for i,im in enumerate(images)}
        anns=[[] for _ in images];bytrack=[{} for _ in images];byid={}
        for ann in raw['annotations']:
            t=loc[ann['image_id']]
            assert ann['track_id'] not in bytrack[t]
            anns[t].append(ann);bytrack[t][ann['track_id']]=ann;byid[ann['id']]=ann
        data[seq]=dict(images=images,source=source,anns=anns,bytrack=bytrack,byid=byid,path=path,
                       rawcounts=[len(aa) for aa in anns],
                       validcounts=[sum(usable(a,im) for a in aa) for aa,im in zip(anns,images)])
        source_hashes[seq]={'path':str(path),'sha256':sha(path),'resolved_path':str(path.resolve())}
    evidence=[];object_rows=[]
    for obj in sorted(objects,key=key):
        seq,t,annid=key(obj);track=int(obj['track_id']);d=data[seq]
        ann=d['byid'][annid]
        assert ann['track_id']==track and ann['image_id']==d['images'][t]['id']
        assert d['source'][t]==int(obj['source_frame'])
        assert list(map(float,ann['bbox']))==[float(obj[k]) for k in ['bbox_x','bbox_y','bbox_w','bbox_h']]
        windows=sorted(window_lookup[key(obj)],key=lambda r:int(r['window_start']))
        expected=[s for s in range(0,121,12) if s<=t<s+40]
        assert [int(r['window_start']) for r in windows]==expected
        assert len(windows)==int(obj['window_count'])
        assert all(int(w['track_id'])==track and int(w['source_frame'])==d['source'][t] and
                   w['system_case']==obj['system_case'] for w in windows)
        official=[w for w in windows if int(w['original5'])]
        success=[w for w in windows if int(w['matched'])]
        assert success and official and obj['fused_matched']=='0'
        orig_all_boundary=all(not 8<=int(w['slot'])<=31 for w in official)
        common=dict(group=obj['group'],sequence=seq,frame=t,source_frame=d['source'][t],
                    annotation_id=annid,track_id=track,category=obj['category'],category_id=int(obj['category_id']),
                    system_case=obj['system_case'],fused_matched=0,
                    original5_all_current_slots_boundary=int(orig_all_boundary),
                    any_success_current_slot_internal=int(any(8<=int(w['slot'])<=31 for w in success)))
        selected_evidence=[]
        for w in windows:
            start=int(w['window_start']);end=start+40;slot=t-start
            assert int(w['slot'])==slot and bool(int(w['original5']))==(start in OFFICIAL)
            present=[j for j in range(start,end) if track in d['bytrack'][j]]
            missing=[j for j in range(start,end) if track not in d['bytrack'][j]]
            valid=[j for j in present if usable(d['bytrack'][j][track],d['images'][j])]
            centers={j:(float(d['bytrack'][j][track]['bbox'][0])+float(d['bytrack'][j][track]['bbox'][2])/2,
                        float(d['bytrack'][j][track]['bbox'][1])+float(d['bytrack'][j][track]['bbox'][3])/2) for j in valid}
            xs=[v[0] for v in centers.values()];ys=[v[1] for v in centers.values()]
            steps=[math.dist(centers[j],centers[j-1]) for j in valid if j-1 in centers]
            pairs=[(a,b) for a,b in zip(valid,valid[1:]) if b-a>1]
            points=list(centers.values())
            diameter=max((math.dist(a,b) for i,a in enumerate(points) for b in points[i+1:]),default=0.)
            row=dict(common,window_start=start,window_end_exclusive=end,
                     window_source_first=d['source'][start],window_source_last_inclusive=d['source'][end-1],
                     current_slot=slot,current_slot_internal_8_31=int(8<=slot<=31),
                     original5=int(w['original5']),actual_match=int(w['matched']),
                     box_max=float(w['box_max']),peak_x=w['peak_x'],peak_y=w['peak_y'],
                     track_annotation_frames=len(present),track_missing_annotation_frames=len(missing),
                     track_annotation_source_frames=';'.join(str(d['source'][j]) for j in present),
                     track_missing_source_frames=';'.join(str(d['source'][j]) for j in missing),
                     track_usable_box_frames=len(valid),track_unusable_box_frames=len(present)-len(valid),
                     track_ignored_annotation_frames=sum(bool(d['bytrack'][j][track].get('ignore',0)) for j in present),
                     track_center_x_min=min(xs) if xs else '',track_center_x_max=max(xs) if xs else '',
                     track_center_y_min=min(ys) if ys else '',track_center_y_max=max(ys) if ys else '',
                     track_center_x_range=max(xs)-min(xs) if xs else '',
                     track_center_y_range=max(ys)-min(ys) if ys else '',
                     track_center_max_pair_distance=diameter if points else '',
                     track_center_net_displacement=math.dist(points[0],points[-1]) if points else '',
                     consecutive_step_count=len(steps),consecutive_step_zero_count=sum(x==0 for x in steps),
                     consecutive_step_nonzero_count=sum(x>0 for x in steps),
                     consecutive_step_sum=sum(steps),consecutive_step_p90_linear=p90(steps),
                     nonconsecutive_valid_label_gaps=len(pairs),
                     **stats(steps,'consecutive_step'),
                     window_raw_annotation_rows=sum(d['rawcounts'][start:end]),
                     window_usable_annotation_rows=sum(d['validcounts'][start:end]),
                     **stats(d['rawcounts'][start:end],'window_raw_target_count'),
                     **stats(d['validcounts'][start:end],'window_usable_target_count'))
            assert row['track_annotation_frames']+row['track_missing_annotation_frames']==40
            assert row['track_usable_box_frames']+row['track_unusable_box_frames']==row['track_annotation_frames']
            evidence.append(row);selected_evidence.append(row)
        object_rows.append(dict(common,available_windows=len(windows),official_windows=len(official),
                                matched_windows=len(success),
                                matched_official_windows=sum(int(w['original5']) for w in success),
                                matched_extra_windows=sum(not int(w['original5']) for w in success),
                                success_window_starts=';'.join(w['window_start'] for w in success),
                                success_current_slots=';'.join(w['slot'] for w in success),
                                any_window_missing_track_annotations=int(any(r['track_missing_annotation_frames'] for r in selected_evidence)),
                                all_windows_full40_track_annotations=int(all(r['track_annotation_frames']==40 for r in selected_evidence)),
                                success_window_annotation_frames_min=min(r['track_annotation_frames'] for r in selected_evidence if r['actual_match']),
                                success_window_annotation_frames_max=max(r['track_annotation_frames'] for r in selected_evidence if r['actual_match']),
                                all_window_center_diameter_min=min(r['track_center_max_pair_distance'] for r in selected_evidence),
                                all_window_center_diameter_max=max(r['track_center_max_pair_distance'] for r in selected_evidence)))
    sequence_rows=[]
    for seq in sorted(data):
        for case in sorted(CASES):
            oo=[r for r in object_rows if r['sequence']==seq and r['system_case']==case]
            ww=[r for r in evidence if r['sequence']==seq and r['system_case']==case]
            sequence_rows.append(dict(sequence=seq,group=('annotation_selected_development' if seq in NEW else 'prior_motion_selected_development'),
                                      system_case=case,object_frames=len(oo),tracks=len({r['track_id'] for r in oo}),
                                      window_object_rows=len(ww),matched_window_rows=sum(r['actual_match'] for r in ww),
                                      objects_any_success_internal=sum(r['any_success_current_slot_internal'] for r in oo),
                                      objects_original5_all_boundary=sum(r['original5_all_current_slots_boundary'] for r in oo),
                                      objects_all_windows_full40_annotations=sum(r['all_windows_full40_track_annotations'] for r in oo),
                                      objects_any_window_missing_track_annotations=sum(r['any_window_missing_track_annotations'] for r in oo)))
    assert sum(r['object_frames'] for r in sequence_rows)==157
    for name,rows in [('WINDOW_EVIDENCE.csv',evidence),('OBJECT_CONTEXT.csv',object_rows),('SEQUENCE_SUMMARY.csv',sequence_rows)]:
        write_csv(ROOT/name,rows)
    assert input_hashes=={n:sha(V/n) for n in input_hashes}
    summary=dict(status='COMPLETE',object_frames=len(objects),window_object_rows=len(evidence),
                 sequences=len(data),tracks=len({(r['sequence'],r['track_id']) for r in objects}),
                 case_counts=dict(collections.Counter(r['system_case'] for r in objects)),
                 group_counts=dict(collections.Counter(r['group'] for r in objects)),
                 input_hashes=input_hashes,annotation_sources=source_hashes,script_sha256=sha(Path(__file__)),
                 objects_original5_all_boundary=sum(r['original5_all_current_slots_boundary'] for r in object_rows),
                 objects_any_success_internal=sum(r['any_success_current_slot_internal'] for r in object_rows),
                 objects_all_windows_full40_annotations=sum(r['all_windows_full40_track_annotations'] for r in object_rows),
                 objects_any_window_missing_track_annotations=sum(r['any_window_missing_track_annotations'] for r in object_rows),
                 output_hashes={n:sha(ROOT/n) for n in outputs[:3]},
                 interpretation='Complete descriptive opportunity subset; label presence is not visibility, image-coordinate movement is not physical motion, natural window content and slot co-vary; no causal inference.')
    (ROOT/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    text='''# 窗口内容与可用标注证据：完整机会集合

本表覆盖 `OBJECTS.csv` 中全部 `small_geometry=1` 且最终融合失配、至少一个诊断窗实际匹配的对象帧，不按响应高低再挑案例。两类为 `original_window_match_lost_by_fusion` 和 `extra_window_only_opportunity`。窗口来自固定 11 起点网格中覆盖当前帧的所有真实窗口，并非每个对象都有 11 个覆盖窗口。

- `WINDOW_EVIDENCE.csv`：逐对象/窗口实际匹配、源帧范围、当前槽位、是否原五窗，以及标注存在、图像坐标运动、整窗目标标注密度。
- `OBJECT_CONTEXT.csv`：逐机会对象的成功槽位、原五窗槽位是否全在边界区间、缺标情况。
- `SEQUENCE_SUMMARY.csv`：逐序列/系统案例完整分布，零样本行保留。
- `SUMMARY.json`：输入、原标注、脚本和输出 SHA256，以及集合数量。

`window_source_last_inclusive` 是包含的最后源帧，`window_end_exclusive` 是片段内不包含的结束索引；当前槽位取 0..39。内部槽位固定定义为 8..31，其余称边界槽位，仅对应原先时间感受野诊断划分，不定义窗口质量。

`track_annotation_frames` 统计该 track 的原标注是否存在，包括后来被诊断规则排除的记录；缺标等于 40 减存在数。运动只对与原评价资格一致的框计算：不 ignore、有限坐标、正宽高且与图像有交集；部分越界框仍可计入，和原 prepare_gt 一致。没有跨缺标间隔计算单帧步长。所有运动单位为原始图像像素；中心为 x+w/2、y+h/2，范围取窗内有效中心极值，最大两点距离为实际中心点对最大欧氏距离，净位移为首末有效中心距离，连续步长只计算相邻源帧。

全目标数量同时给原标注条数和按上述资格保留的条数，包含所有类别/大小；它们是标注密度，不是经人工确认的可见目标数。`track_annotation_source_frames`/`track_missing_source_frames` 允许逐帧追溯标注存在情况。90百分位使用排序位置 (N-1)*0.9 的线性插值。

**解释边界**：标注存在不等于目标可见，缺标也不自动等于目标消失。图像坐标运动混合了目标运动、相机运动和标注波动，不等于物理运动。这批对象按最终检测失败与某窗口成功定义，是机会条件集合；不能用它估计全部目标的运动条件发生率。窗口内容、槽位、全目标标注密度共同变化，不能由本表认定有害上下文、遮挡、光照或运动因果，也不能把 8..31 内成功直接证明为位置效应。原五窗中实际匹配与额外窗口中的成功仍须区分，后者尚不是同预算系统收益。
'''
    text+='\n集合核验：%d 个对象帧、%d 个对象/窗口行、%d 段、%d 个序列内轨迹。原五窗融合损失 %d 个对象帧；额外窗口机会 %d 个。\n' % (len(objects),len(evidence),len(data),summary['tracks'],summary['case_counts']['original_window_match_lost_by_fusion'],summary['case_counts']['extra_window_only_opportunity'])
    (ROOT/'README.md').write_text(text)
    print(json.dumps({k:summary[k] for k in ['status','object_frames','window_object_rows','sequences','tracks','case_counts','objects_all_windows_full40_annotations','objects_any_window_missing_track_annotations']}))


if __name__=='__main__':main()
