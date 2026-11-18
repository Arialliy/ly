#!/usr/bin/env python3
"""Paired H / HL detection summaries, with fixed 2048-step main endpoint.

All tables derive from scored full-frame probabilities. No threshold search,
checkpoint selection, new matching rule or favorable-object selection is used.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

import evaluate_joint as scoring

BASE=scoring.BASE
CONDITIONS=scoring.CONDITIONS
PLANS=scoring.PLANS
SEQUENCES=scoring.SEQUENCES
STEPS=scoring.STEPS
digest=scoring.digest
write_csv=scoring.write_csv
write_json=scoring.write_json


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def comparability():
    runs={}
    source={}
    for condition in ['H','HL']:
        folder=BASE/'training'/condition/'run'
        run=json.loads((folder/'RUN.json').read_text())
        assert run['status']=='complete' and run['final_step']==2048
        assert run['seed']==46 and run['head_lr']==1e-5 and run['adapter_lr']==1e-4
        assert run['trainable_parameters']==(273 if condition=='H' else 373)
        assert run['bn_initial_sha256']==run['bn_final_sha256']
        assert run['original_nonhead_initial_sha256']==run['original_nonhead_final_sha256']
        assert run['frozen_original_nonhead_unchanged']
        assert run['train_log_sha256']==digest(folder/'TRAIN.jsonl')
        assert run['parameter_diff_sha256']==digest(folder/'PARAMETER_DIFF.json')
        assert run['final_checkpoint_sha256']==digest(folder/'final_model.pth')
        assert [int(row['step']) for row in run['snapshots']]==[0,128,512,1024,2048]
        for snapshot in run['snapshots']:
            assert snapshot['sha256']==digest(folder/snapshot['path'])
        logs=[json.loads(line) for line in (folder/'TRAIN.jsonl').read_text().splitlines()]
        assert len(logs)==2048 and [row['step'] for row in logs]==list(range(1,2049))
        runs[condition]=run
        source[str(folder/'RUN.json')]=digest(folder/'RUN.json')
    keys=['seed','steps','head_lr','adapter_lr','weight_decay','optimizer','betas','eps','precision',
          'bn_initial_sha256','original_nonhead_initial_sha256','original_initial_state_sha256',
          'initial_checkpoint_sha256','plan_sha256','selection_sha256','old_log_sha256','exposure_rule',
          'label_config','effective_supervision_config','frozen_plan_config','training_sequences',
          'loss_reduction','protocol','cumulative_input_sha256','cumulative_label_sha256']
    for key in keys:
        assert runs['H'][key]==runs['HL'][key], ('Training mismatch',key)
    return {'status':'COMPLETE','matched_fields':keys,'source_sha256':source,
            'scope':'Original head learns in both; only HL adds the identical-initialized spatial LFP. Non-head original tensors and every BN state remain frozen.'},runs


def background(records, areas, condition, pi, plan, sequences):
    counts=sum((records[condition,s]['frame_counts'][pi].sum(0) for s in sequences),np.zeros(6,np.int64))
    area=np.concatenate([areas[condition,s][plan+'_areas'] for s in sequences])
    unmatched=np.concatenate([areas[condition,s][plan+'_unmatched'].astype(bool) for s in sequences])
    result=dict(zip(scoring.FIELDS,map(int,counts)))
    for name, selected in [('component',area),('unmatched_component',area[unmatched])]:
        result[name+'_area_sum']=int(selected.sum())
        result[name+'_area_max']=int(selected.max()) if len(selected) else 0
        result[name+'_area_mean']=float(selected.mean()) if len(selected) else 0.
        for label,q in [('q50',.5),('q95',.95),('q99',.99)]:
            result[name+'_area_'+label]=float(np.quantile(selected,q)) if len(selected) else 0.
    return result


def observed_counts(selected, predicate):
    chosen=[row for row in selected if predicate(row)]
    return len(chosen),len({(r['sequence'],r['track_id']) for r in chosen}),len({r['sequence'] for r in chosen})


def all_gt_tables(records,metadata,step,output):
    objects=[]
    for sequence in SEQUENCES:
        for idx,meta in enumerate(metadata[sequence]):
            for pi,plan in enumerate(PLANS):
                row=dict(step=step,sequence=sequence,track_id=meta['track_id'],frame=meta['frame'],
                         source_frame=meta['source_frame'],annotation_id=meta['annotation_id'],
                         arrangement=plan,strict_small=int(meta['strict_small']),category_id=meta['category_id'],
                         bbox=json.dumps(meta['bbox'],separators=(',',':')),threshold=.5)
                for condition in CONDITIONS:
                    for key in ['matched','box_max','peak_x','peak_y','component_area']:
                        value=records[condition,sequence][key][pi,idx].item()
                        row[condition+'_'+key]=value if np.isfinite(value) else ''
                    for reference in CONDITIONS:
                        before=bool(records[reference,sequence]['matched'][pi,idx])
                        after=bool(records[condition,sequence]['matched'][pi,idx])
                        row[condition+'_gain_vs_'+reference]=int(after and not before)
                        row[condition+'_loss_vs_'+reference]=int(before and not after)
                objects.append(row)
    assert len(objects)==2*16297 and sum(r['strict_small'] for r in objects)==10400
    summaries=[]
    for condition in CONDITIONS:
        for plan in PLANS:
            for group,sequences in [('pooled',SEQUENCES)]+[(s,[s]) for s in SEQUENCES]:
                for scope in ['all_valid_gt','strict_small','other_valid_gt']:
                    selected=[r for r in objects if r['arrangement']==plan and r['sequence'] in sequences
                              and (scope=='all_valid_gt' or bool(r['strict_small'])==(scope=='strict_small'))]
                    row=dict(step=step,condition=condition,arrangement=plan,group=group,scope=scope,
                             target_frames=len(selected),tracks=len({(r['sequence'],r['track_id']) for r in selected}),
                             clips=len({r['sequence'] for r in selected}),M=sum(r[condition+'_matched'] for r in selected))
                    for ref in CONDITIONS:
                        row[ref+'_M']=sum(r[ref+'_matched'] for r in selected)
                        row['G_vs_'+ref]=sum(r[condition+'_gain_vs_'+ref] for r in selected)
                        row['L_vs_'+ref]=sum(r[condition+'_loss_vs_'+ref] for r in selected)
                        assert row['M']==row[ref+'_M']+row['G_vs_'+ref]-row['L_vs_'+ref]
                    summaries.append(row)
    write_csv(output/'ALL_GT_OBJECT_COMPARISON.csv',objects)
    write_csv(output/'ALL_GT_SUMMARY.csv',summaries)
    changed=[r for r in objects if r['HL_gain_vs_H'] or r['HL_loss_vs_H']]
    write_csv(output/'HL_VS_H_CHANGED_OBJECTS.csv',changed,fields=list(objects[0]))
    return summaries


def summarize(step, preflight=False):
    output=BASE/'evaluation_preflight' if preflight else scoring.evaluation_root(step)
    if (output/'SUMMARY_COMPLETE.json').exists():
        raise FileExistsError(output/'SUMMARY_COMPLETE.json')
    output.mkdir(parents=True,exist_ok=True)
    conditions=['release'] if preflight else CONDITIONS
    collector,data=scoring.collector(step,conditions)
    records,areas,metadata,hashes,frames,objects=data
    for row in objects:
        row['step']=step
    for row in frames:
        row['step']=step
    summaries=[]
    distribution=[]
    protection=[]
    cohort_keys=['low_all_05','extreme_all_001','both_fixed_success','extra_only','new_window_dependent',
                 'high_unmatched','candidate_window_mixed','release_O_only','release_S_only']
    for condition in conditions:
        for pi,plan in enumerate(PLANS):
            for group,sequences in [('pooled',SEQUENCES)]+[(s,[s]) for s in SEQUENCES]:
                selected=[r for r in objects if r['sequence'] in sequences and r['arrangement']==plan]
                costs=background(records,areas,condition,pi,plan,sequences)
                row=dict(step=step,is_main=int(step==2048 and not preflight),condition=condition,
                         arrangement=plan,group=group,threshold=.5,small_gt=len(selected),
                         small_tracks=len({(r['sequence'],r['track_id']) for r in selected}),clips=len(sequences),
                         M=sum(r[condition+'_matched'] for r in selected),**costs)
                for reference in conditions:
                    ref_cost=background(records,areas,reference,pi,plan,sequences)
                    row[reference+'_M_same']=sum(r[reference+'_matched'] for r in selected)
                    row['G_vs_'+reference+'_same']=sum(r[condition+'_matched'] and not r[reference+'_matched'] for r in selected)
                    row['L_vs_'+reference+'_same']=sum(not r[condition+'_matched'] and r[reference+'_matched'] for r in selected)
                    row['net_vs_'+reference+'_same']=row['G_vs_'+reference+'_same']-row['L_vs_'+reference+'_same']
                    assert row['M']==row[reference+'_M_same']+row['net_vs_'+reference+'_same']
                    for key in ['unmatched_components','positive_pixels_outside_boxes','unmatched_component_area_sum']:
                        row[reference+'_'+key]=ref_cost[key]
                        row['delta_'+key+'_vs_'+reference]=costs[key]-ref_cost[key]
                    row['background_costs_not_increased_vs_'+reference]=int(all(costs[k]<=ref_cost[k]
                        for k in ['unmatched_components','positive_pixels_outside_boxes','unmatched_component_area_sum']))
                row['U_budget']=4015 if group=='pooled' else ''
                row['passes_pooled_U_budget']=int(row['unmatched_components']<=4015) if group=='pooled' else ''
                for cohort in cohort_keys:
                    sub=[r for r in selected if r[cohort]]
                    row[cohort+'_gt']=len(sub)
                    recovered,tracks,clips=observed_counts(sub,lambda r:r[condition+'_matched'])
                    row[cohort+'_matched']=recovered
                    row[cohort+'_matched_tracks']=tracks
                    row[cohort+'_matched_clips']=clips
                summaries.append(row)
                if group=='pooled':
                    outcomes={
                        'gain_vs_release':lambda r:r[condition+'_matched'] and not r['release_matched'],
                        'loss_vs_release':lambda r:not r[condition+'_matched'] and r['release_matched'],
                        'low_recovered':lambda r:r['low_all_05'] and r[condition+'_matched'],
                        'extreme_recovered':lambda r:r['extreme_all_001'] and r[condition+'_matched'],
                    }
                    for label,predicate in outcomes.items():
                        count=collections.Counter((r['sequence'],r['track_id']) for r in selected if predicate(r))
                        ranked=sorted(count.items(),key=lambda pair:(-pair[1],pair[0]))
                        distribution.append(dict(step=step,condition=condition,arrangement=plan,outcome=label,
                            frames=sum(count.values()),tracks=len(count),clips=len({k[0] for k in count}),
                            top1_frames=ranked[0][1] if ranked else 0,top5_frames=sum(v for _,v in ranked[:5]),
                            ranked_tracks=json.dumps([dict(sequence=k[0],track_id=k[1],frames=v) for k,v in ranked])))
        first={(r['sequence'],r['frame'],r['annotation_id']):r for r in objects if r['arrangement']=='O'}
        second={(r['sequence'],r['frame'],r['annotation_id']):r for r in objects if r['arrangement']=='S'}
        for group,sequences in [('pooled',SEQUENCES)]+[(s,[s]) for s in SEQUENCES]:
            for cohort in ['all']+cohort_keys:
                selected=[key for key,r in first.items() if r['sequence'] in sequences and (cohort=='all' or r[cohort])]
                left=np.array([first[key][condition+'_matched'] for key in selected],bool)
                right=np.array([second[key][condition+'_matched'] for key in selected],bool)
                protection.append(dict(step=step,condition=condition,group=group,cohort=cohort,target_frames=len(selected),
                    tracks=len({(key[0],first[key]['track_id']) for key in selected}),clips=len({key[0] for key in selected}),
                    O_matched=int(left.sum()),S_matched=int(right.sum()),both_matched=int((left&right).sum()),
                    O_only=int((left&~right).sum()),S_only=int((right&~left).sum()),neither_matched=int((~left&~right).sum()),
                    lost_dual_success=int((~(left&right)).sum()) if cohort=='both_fixed_success' else '',
                    lost_both_arrangements=int((~left&~right).sum()) if cohort=='both_fixed_success' else ''))
    assert [(r['M'],r['unmatched_components']) for r in summaries if r['condition']=='release' and r['group']=='pooled']==[(700,4015),(720,3967)]
    transitions=collector.cohort_transitions(conditions,objects)
    for row in transitions:
        row['step']=step
    write_csv(output/'SUMMARY.csv',[r for r in summaries if r['group']=='pooled'])
    write_csv(output/'SEQUENCE_SUMMARY.csv',[r for r in summaries if r['group']!='pooled'])
    write_csv(output/'OBJECT_COMPARISON.csv',objects)
    write_csv(output/'FRAME_COUNTS.csv',frames)
    write_csv(output/'COHORT_TRANSITIONS.csv',transitions)
    write_csv(output/'RECOVERY_CONCENTRATION.csv',distribution)
    write_csv(output/'PROTECTED_AND_WINDOW_RETENTION.csv',protection)
    trajectory_path=scoring.PREVIOUS/'evaluation_code/trajectory_tables.py'
    trajectory=scoring.load_module(trajectory_path,'joint_frozen_trajectory_tables')
    trajectory.trajectory_tables(objects,conditions,conditions,output,scoring.Q)
    if not preflight:
        training_check,runs=comparability()
        write_json(output/'TRAINING_COMPARABILITY.json',training_check)
        hashes.update(training_check['source_sha256'])
        all_gt_tables(records,metadata,step,output)
        resources=[]
        for condition in ['H','HL']:
            for sequence in SEQUENCES:
                path=BASE/'predictions'/condition/f'step_{step:04d}'/sequence/'COMPLETE.json'
                inference=json.loads(path.read_text())
                for plan in PLANS:
                    resources.append(dict(step=step,condition=condition,sequence=sequence,arrangement=plan,
                        trainable_parameters=runs[condition]['trainable_parameters'],
                        full_2048_training_seconds=runs[condition]['duration_seconds'],
                        training_peak_allocated_bytes=runs[condition]['max_cuda_memory_allocated_bytes'],
                        training_peak_reserved_bytes=runs[condition]['max_cuda_memory_reserved_bytes'],
                        five_window_model_plus_cpu_copy_seconds=inference[plan+'_model_seconds'],
                        union_eight_window_elapsed_seconds=inference['elapsed_seconds'],
                        inference_peak_allocated_bytes=inference['peak_cuda_allocated_bytes'],
                        inference_peak_reserved_bytes=inference['peak_cuda_reserved_bytes']))
        write_csv(output/'RESOURCES.csv',resources)
    hashes.update({str(Path(__file__).resolve()):digest(__file__),str(Path(scoring.__file__).resolve()):digest(scoring.__file__),
                   str(scoring.PREVIOUS/'evaluation_code/evaluate.py'):digest(scoring.PREVIOUS/'evaluation_code/evaluate.py'),
                   str(trajectory_path):digest(trajectory_path)})
    outputs={p.name:digest(p) for p in sorted(output.glob('*.csv'))}
    write_json(output/'SUMMARY_COMPLETE.json',dict(status='COMPLETE',step=step,main_step=2048,is_main=step==2048 and not preflight,
        preflight=preflight,conditions=conditions,threshold=.5,plans=PLANS,strict_small_target_frames=5200,
        strict_small_tracks=130,development_clips=2,all_valid_target_frames=16297,cohort_counts={'low_all_05':4342,'extreme_all_001':4170,'both_fixed_success':646},
        checkpoint_policy='2048 main;512 and1024 descriptive process checkpoints, no selection',
        scope='Same one-to-one component-peak-in-box diagnostic; M/U are not official Pd/Fa. H versus HL is the LFP increment. Frozen non-head original state, all BN state and matching protocol unchanged.',
        source_sha256=hashes,output_sha256=outputs))
    print(json.dumps({'status':'COMPLETE','step':step,'preflight':preflight,'rows':[
        {key:r[key] for key in ['condition','arrangement','M','unmatched_components','low_all_05_matched','G_vs_release_same','L_vs_release_same']}
        for r in summaries if r['group']=='pooled']}),flush=True)


def combine():
    output=BASE/'evaluation'
    tables=['SUMMARY.csv','SEQUENCE_SUMMARY.csv','ALL_GT_SUMMARY.csv','COHORT_TRANSITIONS.csv',
            'PROTECTED_AND_WINDOW_RETENTION.csv','CONTINUOUS_MISS_SUMMARY.csv','RESOURCES.csv','RECOVERY_CONCENTRATION.csv']
    input_hashes={}
    for step in STEPS:
        folder=scoring.evaluation_root(step)
        complete=json.loads((folder/'SUMMARY_COMPLETE.json').read_text())
        assert complete['status']=='COMPLETE' and complete['step']==step
        assert complete['is_main']==(step==2048)
        for filename,expected in complete['output_sha256'].items():
            assert digest(folder/filename)==expected
        input_hashes[str(folder/'SUMMARY_COMPLETE.json')]=digest(folder/'SUMMARY_COMPLETE.json')
    for filename in tables:
        rows=[]
        for step in STEPS:
            for row in read_csv(scoring.evaluation_root(step)/filename):
                row['step']=step
                row['is_main']=int(step==2048)
                rows.append(row)
        write_csv(output/('ALL_STEPS_'+filename),rows)
    main=read_csv(scoring.evaluation_root(2048)/'SUMMARY.csv')
    retention=read_csv(scoring.evaluation_root(2048)/'PROTECTED_AND_WINDOW_RETENTION.csv')
    trajectories=read_csv(scoring.evaluation_root(2048)/'CONTINUOUS_MISS_SUMMARY.csv')
    all_gt=read_csv(scoring.evaluation_root(2048)/'ALL_GT_SUMMARY.csv')
    report=[
        '# H 与 H＋LFP：原预测头协同适配结果', '',
        '主结论固定读取 2048 步；512、1024 步全部保留为学习过程，不挑选最佳节点。阈值固定 0.5，O／S 均为预登记五窗安排。', '',
        '本表 M 是 5,200 个严格小目标帧的实际组件峰值点匹配数，U 是全体有效标注匹配后的未匹配组件数；两者均不是论文 Pd/Fa。', '',
        '| 配置 | 安排 | M | 相对发布新增／丢失 | 低响应恢复（帧／轨迹／片段） | 极低响应恢复 | U | 框外前景像素 | 未匹配组件面积 | U≤4015 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in main:
        report.append('| {condition} | {arrangement} | {M} | {G_vs_release_same}／{L_vs_release_same} | {low_all_05_matched}／{low_all_05_matched_tracks}／{low_all_05_matched_clips} | {extreme_all_001_matched} | {unmatched_components} | {positive_pixels_outside_boxes} | {unmatched_component_area_sum} | {passed} |'.format(**row,passed='通过' if row['passes_pooled_U_budget']=='1' else '未通过'))
    report.extend(['','## LFP 相对仅适配预测头的增量','','| 安排 | 新增／丢失匹配 | 匹配净变化 | ΔU | Δ框外前景 | Δ未匹配组件面积 |','|---|---:|---:|---:|---:|---:|'])
    for row in main:
        if row['condition']=='HL':
            report.append('| {arrangement} | {G_vs_H_same}／{L_vs_H_same} | {net_vs_H_same} | {delta_unmatched_components_vs_H} | {delta_positive_pixels_outside_boxes_vs_H} | {delta_unmatched_component_area_sum_vs_H} |'.format(**row))
    report.extend(['','逐对象差异保存在 `evaluation/step_2048/HL_VS_H_CHANGED_OBJECTS.csv`；包含严格小目标及其余有效标注。净增不能替代新增与丢失的集合核对。',
                   '','## 保护集合及连续漏检','','| 配置 | 646 集合仍双侧成功 | 不再双侧成功 | 两侧均漏检 | O 最长漏检缩短／变长轨迹 | S 最长漏检缩短／变长轨迹 |','|---|---:|---:|---:|---:|---:|'])
    for condition in CONDITIONS:
        protected=next(r for r in retention if r['condition']==condition and r['cohort']=='both_fixed_success' and r['group']=='pooled')
        runs={plan:next(r for r in trajectories if r['condition']==condition and r['arrangement']==plan and r['cohort']=='all' and r['group']=='pooled') for plan in PLANS}
        report.append(f"| {condition} | {protected['both_matched']} | {protected['lost_dual_success']} | {protected['lost_both_arrangements']} | {runs['O']['tracks_shorter_vs_release']}／{runs['O']['tracks_longer_vs_release']} | {runs['S']['tracks_shorter_vs_release']}／{runs['S']['tracks_longer_vs_release']} |")
    report.extend(['','连续性按原局部帧号与源帧号均相邻定义；缺标、帧号中断、筛选空缺打断，不外推到未观察部分。每安排包含 130 条轨迹、两个开发片段。',
                   '','## 其余有效标注的保持','','| 配置 | 安排 | 其余目标帧数 | 匹配 | 相对发布新增／丢失 |','|---|---|---:|---:|---:|'])
    for row in all_gt:
        if row['scope']=='other_valid_gt' and row['group']=='pooled':
            report.append('| {condition} | {arrangement} | {target_frames} | {M} | {G_vs_release}／{L_vs_release} |'.format(**row))
    report.extend(['','## 结果边界与后续判定','',
        '分别检查 H 对发布权重、HL 对发布权重、HL 对 H。预算通过并不自动等于背景代价不增加：S 仍需与自身发布参照 U=3967 及其框外前景、组件面积比较。',
        '','有恢复但背景或保持未达标，可以支持有针对性的后续研发；当前不能称预算内有效方法。只改变临界对象、只有背景减少或分数统一上移，均不能自动解释为持续漏检已解决。',
        '','训练观察对象的响应／匹配应与本表联合阅读后再区分学习失败、泛化缺口与背景代价；本报告不单凭开发总数把原因归为冻结预测头或 LFP 机制。',
        '','全部过程节点：`evaluation/ALL_STEPS_SUMMARY.csv`。逐片段、逐轨迹、全量逐对象、背景组件面积、资源表均保存在对应 step 目录。',
    ])
    (BASE/'RESULTS.md').write_text('\n'.join(report)+'\n')
    write_json(output/'ALL_STEPS_COMPLETE.json',dict(status='COMPLETE',steps=STEPS,main_step=2048,
        source_sha256=input_hashes,summary_script_sha256=digest(__file__),report_sha256=digest(BASE/'RESULTS.md'),
        scope='Fixed endpoint report, no model-success or causal-mechanism claim automatically inferred from totals.'))
    print(json.dumps({'status':'COMPLETE','report':str(BASE/'RESULTS.md')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--step',type=int,choices=STEPS)
    mode.add_argument('--all',action='store_true')
    mode.add_argument('--baseline-check',action='store_true')
    args=parser.parse_args()
    if args.all:
        combine()
    elif args.baseline_check:
        summarize(0,preflight=True)
    else:
        summarize(args.step)


if __name__=='__main__':
    main()
