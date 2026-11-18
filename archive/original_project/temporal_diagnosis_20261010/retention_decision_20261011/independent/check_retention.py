#!/usr/bin/env python3
"""Read only existing combination tables; no predictions are regenerated."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
SRC = OUT.parent.parent / 'combination_feasibility_20261011'
U_CAP = 4015
METRICS = ['unmatched_components', 'positive_pixels_outside_boxes', 'positive_pixels', 'component_count']
INTS = ['small_gt', 'small_matched', 'gained', 'lost', 'net', 'original_sym_union_retained', 'symmetric_gains_retained', 'beyond_original_sym_union', 'gt_count', 'matched_count'] + METRICS

def read(name):
    rows = list(csv.DictReader((SRC/name).open()))
    for r in rows:
        for k in INTS:
            r[k] = int(r[k])
        for k in ['U_budget_feasible', 'both_sequence_U_within_original']:
            r[k] = r[k] == 'True'
        r['id'] = r['choice_0026'] + '/' + r['choice_0067']
        assert r['U_budget_feasible'] == (r['unmatched_components'] <= U_CAP)
        assert r['small_matched'] == 700 + r['gained'] - r['lost']
        assert r['component_count'] == r['matched_count'] + r['unmatched_components']
    return rows

def write(name, rows):
    if not rows:
        (OUT/name).write_text('')
        return
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

source_hashes = {n: digest(SRC/n) for n in ['SHARED_COMBINATIONS.csv', 'PER_CLIP_COMBINATIONS.csv']}
shared, per = read('SHARED_COMBINATIONS.csv'), read('PER_CLIP_COMBINATIONS.csv')
assert len(shared) == 10 and len(per) == 100
assert len({r['id'] for r in per}) == 100
for s in shared:
    p = next(p for p in per if p['id'] == s['id'])
    assert all(s[k] == p[k] for k in INTS)
sf = [r for r in shared if r['U_budget_feasible']]
pf = [r for r in per if r['U_budget_feasible']]
queries = {}
for scope, rows in [('shared', sf), ('per_clip_hindsight', pf)]:
    high = [r for r in rows if r['small_matched'] >= 720]
    zero = [r for r in rows if r['lost'] == 0]
    queries[scope] = {
        'feasible_count': len(rows),
        'M_ge_720': {'count': len(high), 'minimum_lost': min((r['lost'] for r in high), default=None), 'all_rows': high},
        'zero_loss': {'count': len(zero), 'maximum_M': max((r['small_matched'] for r in zero), default=None), 'all_rows': zero},
    }
write('M_GE_720_ALL.csv', [r for rows in [sf,pf] for r in rows if r['small_matched'] >= 720])
write('ZERO_LOSS_ALL.csv', [r for rows in [sf,pf] for r in rows if r['lost'] == 0])

all_improvements, metric_minima, near_rows, exact_rows = [], [], [], []
reference_queries = {}
for s in sf:
    dom = [p for p in pf if p['small_matched'] >= s['small_matched'] and p['lost'] <= s['lost']]
    exact = [p for p in pf if p['small_matched'] == s['small_matched'] and p['lost'] == s['lost']]
    near = [p for p in pf if abs(p['small_matched']-s['small_matched']) <= 1 and abs(p['lost']-s['lost']) <= 1]
    reference_queries[s['id']] = {'reference': s, 'dominance_candidates': len(dom), 'exact_candidates': len(exact), 'near_candidates': len(near), 'metric_minima': {}}
    for p in dom:
        all_improvements.append({'reference_id':s['id'], **p})
    for metric in METRICS:
        v = min(p[metric] for p in dom)
        opts = [p for p in dom if p[metric] == v]
        reference_queries[s['id']]['metric_minima'][metric] = {'value': v, 'delta': v-s[metric], 'rows':opts}
        for p in opts:
            metric_minima.append({'reference_id':s['id'], 'minimized_metric':metric, 'metric_delta':v-s[metric], **p})
    for condition, candidates, dest in [('exact',exact,exact_rows), ('near_delta_1',near,near_rows)]:
        for p in candidates:
            dest.append({'reference_id':s['id'], 'comparison':condition,
                         'delta_M':p['small_matched']-s['small_matched'], 'delta_lost':p['lost']-s['lost'],
                         **{'delta_'+m:p[m]-s[m] for m in METRICS}, **p})
write('NO_WORSE_MATCH_LOSS_ALL.csv',all_improvements)
write('NO_WORSE_MATCH_LOSS_METRIC_MINIMA.csv',metric_minima)
write('EXACT_MATCH_LOSS_ALL.csv',exact_rows)
write('NEAR_MATCH_LOSS_ALL.csv',near_rows)

groups = defaultdict(list)
for p in pf:
    groups[p['small_matched'],p['lost']].append(p)
range_rows = []
for (m,l), rs in sorted(groups.items()):
    for metric in METRICS:
        lo,hi = min(r[metric] for r in rs),max(r[metric] for r in rs)
        range_rows.append({'M':m, 'lost':l,'number_of_pairs':len(rs), 'metric':metric, 'min':lo,'max':hi,'range':hi-lo,
            'min_ids':'|'.join(r['id'] for r in rs if r[metric]==lo), 'max_ids':'|'.join(r['id'] for r in rs if r[metric]==hi)})
write('EXACT_M_LOST_BACKGROUND_RANGES.csv',range_rows)

summary = {'source_hashes':source_hashes, 'source_unchanged':all(digest(SRC/n)==v for n,v in source_hashes.items()),
           'fixed_threshold':0.5, 'U_cap':U_CAP, 'queries':queries, 'shared_reference_queries':reference_queries,
           'maximum_exact_M_lost_group_background_ranges':{m:max((r for r in range_rows if r['metric']==m), key=lambda r:r['range']) for m in METRICS},
           'near_definition':{'abs_delta_M_max':1, 'abs_delta_lost_max':1, 'descriptive_only':True},
           'cautions':['All per-clip selection is GT-assisted hindsight on only two development clips.',
                       'Metrics minima refer to different actual rows unless all identifiers coincide; do not combine them.',
                       'No statistical significance or generalization claim. U and outside-box pixels are diagnostic, not official Fa.']}
(OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

def line(r):
    return f"| {r['id']} | {r['small_matched']} | {r['gained']} | {r['lost']} | {r['unmatched_components']} | {r['positive_pixels_outside_boxes']} | {r['positive_pixels']} | {r['component_count']} |"
header = '| 0026/0067 方案 | M | 新增 | 新丢失 | U | 框外前景 | 总前景 | 总组件 |\n|---|---:|---:|---:|---:|---:|---:|---:|'
parts=['# 现有穷举表的保持性能独立复核','仅读取既有 10 行共用安排和 100 行逐片组合；未生成任何新预测。阈值固定 0.5，合计 U ≤ 4015。所有选择依赖开发集标注，仅诊断候选池空间。',
       '## 匹配不少于 720 时的全部预算可行方案',header]
for r in [x for x in pf if x['small_matched']>=720]: parts.append(line(r))
parts += ['',f"共用安排最少丢失 {queries['shared']['M_ge_720']['minimum_lost']}；逐片选择最少丢失 {queries['per_clip_hindsight']['M_ge_720']['minimum_lost']}。",'','## 零新增丢失的全部预算可行方案',header]
parts += [line(r) for r in pf if r['lost']==0]
parts += ['',f"共用最多匹配 {queries['shared']['zero_loss']['maximum_M']}；逐片最多匹配 {queries['per_clip_hindsight']['zero_loss']['maximum_M']}。",'','## 每个预算可行共用参照的背景取舍','下列每个最小值均附真实方案；不同方案的最低值不能拼成一个方案。']
for s in sf:
    q=reference_queries[s['id']]
    parts += ['',f"### 参照 {s['id']}：M={s['small_matched']}、丢失={s['lost']}",header,line(s),f"满足 M 不降且丢失不增的逐片组合共 {q['dominance_candidates']} 个。"]
    selected={p['id']:p for v in q['metric_minima'].values() for p in v['rows']}
    parts += [header] + [line(p) for p in selected.values()]
    for metric,v in q['metric_minima'].items():
        parts += [f"- {metric} 最小值 {v['value']}（相对参照 {v['delta']:+d}）："+', '.join(p['id'] for p in v['rows'])+'。']
    parts += [f"精确同 M、同丢失有 {q['exact_candidates']} 个；M 与丢失各差不超过 1 的描述性近邻有 {q['near_candidates']} 个。完整背景差异见 CSV。"]
parts += ['','## 结论',
          '最佳逐片组合相对最佳共用安排只多匹配 1、少丢失 2，U 减少 32（0.81%），但框外前景增加 88（0.23%）、总前景增加 81。不能解释为背景整体改善。',
          '零新增丢失条件下仍有事后恢复 7 个目标帧的空间，不能写成“保持空间为零”。该组合比原安排 U 少 66，但框外前景多 106，仍为取舍。',
          '全池预算可行组合按完全相同 M、lost 分组，组内最大跨度为 U 14、框外前景 320、总前景 397、组件数 24；各指标端点不一定是同一对组合。精确同于最佳共用安排 M720/lost54 的另一方案，除总组件少 10 外，其 U、框外及总前景均更高。',
          '这些结果支持停止为当前候选池继续追加自适应选窗实验，将 7 个零丢失恢复案例保留为有限上界参照。此为投入决策，不是对其他模型或其他窗口池的一般否定。',
          '', '## 使用边界','“相近”在本复核中仅表示 M 和新增丢失各差不超过 1；这是用户问题的描述性读取，不是新的成功标准，也不代表统计显著。只有两个片段，不能由目标帧数量推出独立样本意义上的稳定改善。背景 U 和框外前景像素均沿用既有诊断定义，并非官方 Fa。','']
formatted=[]
for part in parts:
    if (part.startswith('#') or part.startswith('| 0026')) and formatted and formatted[-1] != '':
        formatted.append('')
    formatted.append(part)
    if part.startswith('#'):
        formatted.append('')
(OUT/'REPORT.md').write_text('\n'.join(formatted))
print(json.dumps({'feasible_shared':len(sf),'feasible_per_clip':len(pf),'high_rows':[{'id':r['id'],**{k:r[k] for k in ['small_matched','gained','lost']+METRICS}} for r in pf if r['small_matched']>=720], 'zero_rows':[{'id':r['id'],**{k:r[k] for k in ['small_matched','gained','lost']+METRICS}} for r in pf if r['lost']==0]},indent=2))
