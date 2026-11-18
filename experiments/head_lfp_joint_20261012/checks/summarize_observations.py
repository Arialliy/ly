#!/usr/bin/env python3
"""Summarize existing fixed training observations; no inference or sampling."""
import csv, hashlib, json, statistics
from pathlib import Path
E=Path(__file__).resolve().parents[1]
OUT=E/'observations'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open() as f: return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    sources={}; totals=[]; margins=[]; all_summary=[]
    release=None
    for condition,step in [('release',0)]+[(c,s) for c in ['H','HL'] for s in [128,512,1024,2048]]:
        d=OUT/condition/f'step_{step:04d}'
        receipt=json.loads((d/'COMPLETE.json').read_text())
        assert receipt['status']=='COMPLETE' and receipt['training_updates']==0
        for f,expected in receipt['csv_sha256'].items(): assert sha(d/f)==expected
        sources[str((d/'COMPLETE.json').relative_to(E))]=sha(d/'COMPLETE.json')
        objects=read(d/'OBJECTS.csv')
        core=[r for r in objects if r['core_strict_small']=='1']
        if condition=='release': release={r['key']:r for r in core}
        assert {r['key'] for r in core}==set(release)
        summaries=read(d/'SUMMARY.csv'); all_summary+=summaries
        pooled=dict(next(r for r in summaries if r['sequence']=='ALL'))
        low=[r for r in core if float(r['release_box_max'])<.5]
        recovered=[r for r in low if r['matched']=='1']
        pooled.update(low_recovery_tracks=len({(r['sequence'],r['track_id']) for r in recovered}),
            low_recovery_sources=len({r['sequence'] for r in recovered}),
            low_recovered_unique_target_frames=len({(r['sequence'],r['track_id'],r['frame']) for r in recovered}))
        totals.append(pooled)
        for label,rows in [('all_core',core),('release_low_window',low),('release_success',[r for r in core if r['release_matched']=='1'])]:
            valid=[r for r in rows if r['margin_vs_fixed_background_logit']!='']
            values=[float(r['margin_vs_fixed_background_logit']) for r in valid]
            delta=[float(r['margin_vs_fixed_background_logit'])-float(release[r['key']]['margin_vs_fixed_background_logit']) for r in valid]
            margins.append(dict(condition=condition,step=step,cohort=label,window_target_instances=len(rows),
                tracks=len({(r['sequence'],r['track_id']) for r in rows}),sources=len({r['sequence'] for r in rows}),
                valid_background_instances=len(valid),median_margin=statistics.median(values) if values else '',
                median_paired_margin_change=statistics.median(delta) if delta else '',
                margin_positive=sum(v>0 for v in values),margin_improved=sum(v>0 for v in delta),margin_worsened=sum(v<0 for v in delta)))
    write(OUT/'ALL_STEPS_SUMMARY.csv',totals)
    write(OUT/'ALL_STEPS_SEQUENCE_SUMMARY.csv',all_summary)
    write(OUT/'FIXED_BACKGROUND_MARGINS.csv',margins)
    report=['# 固定训练观察', '',
        '读取预登记的16个训练裁剪窗口，不增加训练或推理。窗口覆盖8个训练来源；其中严格小目标来自7个来源、16条轨迹，共1170个窗口—目标帧实例，对应743个唯一目标帧；窗口重叠实例不能当成独立样本。', '',
        '| 配置 | 步数 | 匹配 | 相对发布新增／丢失 | 141个原单窗低响应实例恢复 | 恢复轨迹／来源 | U | 框外前景 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in totals:
        report.append('| {condition} | {step} | {matched} | {gain_vs_release}／{loss_vs_release} | {release_low_recovered} | {low_recovery_tracks}／{low_recovery_sources} | {unmatched_components} | {outside_pixels} |'.format(**r))
    report += ['', '这里的低响应是这组训练裁剪中发布模型的框响应低于0.5，不是开发集合中的4342个全窗低响应目标帧。U只覆盖这组裁剪，不能套用完整开发集的4015预算。', '',
        '固定背景位置是发布模型在每个窗口帧有效背景掩码中的最高响应位置，排除全部标注框及既定保护边界。随后保持位置不变。`FIXED_BACKGROUND_MARGINS.csv` 汇总原始目标logit与该位置logit差；只表示相对这个冻结参照的变化，不代表已战胜当前所有困难背景，更不证明泛化或因果机制。', '',
        '来源数量表示观察覆盖。低响应恢复的来源／轨迹数单列；不能将8个观察来源写成8个恢复来源。学习过程节点用于解释固定2048步终点，不用于挑选检查点。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    outputs={p.name:sha(p) for p in [OUT/'ALL_STEPS_SUMMARY.csv',OUT/'ALL_STEPS_SEQUENCE_SUMMARY.csv',OUT/'FIXED_BACKGROUND_MARGINS.csv',OUT/'REPORT.md']}
    (OUT/'SUMMARY_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',source_sha256=sources,output_sha256=outputs,script_sha256=sha(Path(__file__)),new_training_or_inference=False),indent=2)+'\n')
    print(json.dumps(totals,ensure_ascii=False))
if __name__=='__main__':main()
