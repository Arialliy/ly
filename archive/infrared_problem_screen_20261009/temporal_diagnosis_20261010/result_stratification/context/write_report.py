#!/usr/bin/env python3
"""Render measured context results; standard library only."""
import csv
import hashlib
import json
from pathlib import Path

P=Path(__file__).resolve().parent
S=json.loads((P/'SUMMARY.json').read_text())

def one(table,**kw):
    rr=[r for r in S[table] if all(r[k]==v for k,v in kw.items())]
    assert len(rr)==1,(table,kw,len(rr))
    return rr[0]

def get(table='sensitivity',**kw):
    d=dict(threshold=.5,subset='strict',sequence='ALL',scale='<=16',stratum_kind='all',stratum='all');d.update(kw)
    return one(table,**d)

def ratio(a,b):return f'{a}/{b} ({100*a/b:.1f}%)' if b else '0/0'

lines=['# 固定窗口选择：同一 GT 当前帧的条件增益、损失与未解决缺口','',
'输入为四个既定160帧片段、全部4973个有效 GT×帧、84个标注 track ID。每片使用11个已保存的真实连续40帧窗口，起点0,12,…,120；没有新增推理、阈值搜索或训练。统计固定阈值0.1/0.5/0.9，主文0.5。', '',
'结果先说明：同一帧的窗口上下文与slot变化会改变一部分点匹配和框内响应，但固定居中规则没有一致净收益。严格≤16子集仍有超过一半 GT帧在所有本次可用窗口里低于0.5；这部分缺口无法靠本次窗口起点集合解释为单纯选择失误。', '',
'## 计数和选择规则','',
'单位为同一 GT×当前帧，主比较只保留至少两个可用窗口（每片local frame12–147，共544个图像帧）。clean要求原框完整在画面内且不与其他GT框重叠；strict进一步要求任一可用窗口都没有匹配歧义。所有规则在同一子集、同一可用窗口集合比较，strict随阈值改变，因为匹配歧义与阈值有关。', '',
'EARLIEST=最小窗口起点；LATEST=最大起点；CENTER=slot距19.5最近，平局取更小起点。每帧所有GT共享同一选择；选择只使用帧号与窗口几何，不使用GT、响应分数或成败。下面“增益”是A失败/B成功，“损失”是A成功/B失败，均完整保留。boxmax使用原框内像素中心的最大概率；点匹配使用既有8邻域组件峰值及一对一匹配。', '',
'## 清洁子集与严格子集都报告','',
'| 子集/当前帧尺度 | GT帧 | 标注轨迹ID | 任一窗口点分歧 | 框max跨0.5 | EARLIEST→CENTER点增/损 | 框响应增/损 |',
'|---|---:|---:|---:|---:|---:|---:|']
for sub in ['clean','strict']:
    for sc in ['all','<=16','<=12']:
        r=get(subset=sub,scale=sc);p=get('pairwise',subset=sub,scale=sc,a='earliest',b='center',metric='point');b=get('pairwise',subset=sub,scale=sc,a='earliest',b='center',metric='response')
        lines.append(f"| {sub}/{sc} | {r['gt_frames']} | {r['tracks']} | {r['point_disagreement']} | {r['boxmax_crossing']} | {p['gains']}/{p['losses']} | {b['gains']}/{b['losses']} |")
lines+=['','≤12是≤16的嵌套子集。全部多窗口GT、不排除标注问题的结果也保存在CSV，避免只呈现清洁样本。','',
'## 固定规则的双向变化','',
'| strict尺度 | A→B | 点A成功→B成功 | 点增/损/净 | 框响应增/损/净 | 点变化轨迹/符合轨迹 |',
'|---|---|---:|---:|---:|---:|']
for sc in ['all','<=16','<=12']:
    for a,b in [('earliest','latest'),('earliest','center'),('latest','center')]:
        p=get('pairwise',scale=sc,a=a,b=b,metric='point');r=get('pairwise',scale=sc,a=a,b=b,metric='response')
        lines.append(f"| {sc} | {a}→{b} | {p['a_positive']}→{p['b_positive']} | {p['gains']}/{p['losses']}/{p['net']:+d} | {r['gains']}/{r['losses']}/{r['net']:+d} | {p['changed_tracks']}/{p['tracks']} |")
lines+=['','全尺度strict中，EARLIEST→CENTER点匹配79增/81损，净-2；≤16净+20，≤12净+26。尺度条件改变结论，且CENTER相对LATEST在≤16反而净-11，因此这些数据不支持把“居中/固定窗口选择”当作已经成立的新方法。','',
'## 四片段与类别条件','',
'| 片段，strict≤16 | GT帧/轨迹ID | 点分歧/有分歧轨迹 | 框跨阈值/有跨越轨迹 | EARLIEST→CENTER点增/损 | 框增/损 | 所有窗口点失败 / 所有窗口框低 |',
'|---|---:|---:|---:|---:|---:|---:|']
for seq in S['metadata']:
    r=get(sequence=seq);p=get('pairwise',sequence=seq,a='earliest',b='center',metric='point');b=get('pairwise',sequence=seq,a='earliest',b='center',metric='response')
    lines.append(f"| {seq} | {r['gt_frames']}/{r['tracks']} | {r['point_disagreement']}/{r['point_sensitive_tracks']} | {r['boxmax_crossing']}/{r['response_sensitive_tracks']} | {p['gains']}/{p['losses']} | {b['gains']}/{b['losses']} | {r['all_contexts_point_negative']}/{r['all_contexts_boxmax_low']} |")
lines+=['','≤16框响应跨越出现在DJI_0028_4、DJI_0032_5、DJI_0047_3三片段，DJI_0028_2为0/214。点分歧四片段都存在，但0028_2的5帧只来自一条轨迹，且没有框max跨越，更接近组件峰值/形态变化。不同序列不能直接视为独立拍摄场景；0028两个片段尤其不能据名称推断场景独立。','',
'| 片段，strict≤16 | EARLIEST→LATEST点增/损 | 框增/损 | EARLIEST→CENTER点增/损 | 框增/损 |',
'|---|---:|---:|---:|---:|']
for seq in S['metadata']:
    a=get('pairwise',sequence=seq,a='earliest',b='latest',metric='point');b=get('pairwise',sequence=seq,a='earliest',b='latest',metric='response')
    c=get('pairwise',sequence=seq,a='earliest',b='center',metric='point');d=get('pairwise',sequence=seq,a='earliest',b='center',metric='response')
    lines.append(f"| {seq} | {a['gains']}/{a['losses']} | {b['gains']}/{b['losses']} | {c['gains']}/{c['losses']} | {d['gains']}/{d['losses']} |")
lines+=['','EARLIEST→CENTER的≤16点净+20中，0047贡献+14，0028_4为+5，0032为+1，0028_2为0；框净+35中0047贡献+29。更晚窗口相对更早窗口的收益分散一些，但仍有损失，0028_2点净-1。','',
'## 明显响应翻转与0.5阈值附近变化','',
'固定使用strict@0.5、≤16的同一1417 GT帧队列，再读取既有连续boxmax数值：若同帧不同上下文的最小值<0.1、最大值≥0.9，记为明显翻转。0.1/0.9来自既定阈值集合，没有新选阈值；原0.5指标不变。','',
'| 片段 | 固定GT帧 | min<0.1且max≥0.9 | 受影响track ID | 0.5框跨越总数 |',
'|---|---:|---:|---:|---:|']
for seq in ['ALL']+list(S['metadata']):
    r=get(sequence=seq)
    lines.append(f"| {seq} | {r['gt_frames']} | {r['strong_response_flip']} | {r['strong_response_flip_tracks']} | {r['boxmax_crossing']} |")
lines+=['','97/137=70.8%的0.5框跨越满足上述明显翻转条件，不能全部解释成0.5附近的小幅波动。其余40帧未满足双端条件，但这不等于它们都在0.5附近；未再定义或搜索“附近”阈值。逐帧min/max及对应annotation ID可见`strong_flips_strict_half_le16.csv`。','',
'| 片段/原标注类别，strict≤16 | GT帧/轨迹ID | 点分歧 | 框跨0.5 | 所有窗口框低 |',
'|---|---:|---:|---:|---:|']
for r in S['sensitivity']:
    if r['threshold']==.5 and r['subset']=='strict' and r['scale']=='<=16' and r['stratum_kind']=='category' and r['sequence']!='ALL':
        lines.append(f"| {r['sequence']}/{r['stratum']} | {r['gt_frames']}/{r['tracks']} | {r['point_disagreement']} | {r['boxmax_crossing']} | {r['all_contexts_boxmax_low']} |")
lines+=['','类别取自源标注，未按外观重标。小汽车、行人、骑行者之间的比较受到片段、尺寸与标注身份变化的混杂；例如0032≤16汽车只有9个GT帧/7个ID，不能作为稳定类别规律。','',
'## 轨迹覆盖与集中度','',
'| strict尺度 | 点分歧轨迹/符合轨迹 | 点变化总数/Top1/Top3/Top5 | 框跨越轨迹/符合轨迹 | 框变化总数/Top1/Top3/Top5 |',
'|---|---:|---:|---:|---:|']
for sc in ['all','<=16','<=12']:
    a=one('concentration',threshold=.5,subset='strict',sequence='ALL',scale=sc,metric='point_disagreement');b=one('concentration',threshold=.5,subset='strict',sequence='ALL',scale=sc,metric='boxmax_crossing')
    lines.append(f"| {sc} | {a['sensitive_tracks']}/{a['eligible_tracks']} | {a['gt_frame_changes']}/{a['top1_changes']}/{a['top3_changes']}/{a['top5_changes']} | {b['sensitive_tracks']}/{b['eligible_tracks']} | {b['gt_frame_changes']}/{b['top1_changes']}/{b['top3_changes']}/{b['top5_changes']} |")
r=get(stratum_kind='selected_track',stratum='0');t=get(stratum_kind='long_track',stratum='1')
lines+=['',f"不是只看四条预选轨迹：strict≤16的非预选轨迹共有{r['gt_frames']}GT帧/{r['tracks']}个ID，包含{r['point_disagreement']}次点分歧和{r['boxmax_crossing']}次框跨越；预选四轨只贡献12与10次。至少40标注帧的长轨子集包含{t['gt_frames']}GT帧/{t['tracks']}个ID，仍有{t['point_disagreement']}次点分歧与{t['boxmax_crossing']}次框跨越。",'',
'≤16框跨越Top3为0047/4500503（37）、0032/4801517（20）、0028_4/4400003（11），合计68/137=49.6%；Top5占63.5%。≤12的Top5更集中，占84/108=77.8%。应优先把这些轨迹作为局部解释案例，同时保留其余轨迹的反例。','',
'源ID复核发现84个处理后track ID对应53个“序列+tracking_id”标签：0028_4为9→8、0032_5为57→32、0047_3为10→5。strict≤16合并为27个源标签组，其中点敏感20组、框敏感17组；框跨越Top3源标签组占84/137=61.3%。这只是按原始标签重分组，不宣称53个独立物体，也不能凭84→53确认ID碎片；同一序列的source tracking_id可跨类别复用，可能来自类别内独立编号。0047的source id 0等映射到多个类别，不能据此断言同一物体改变类别。`source_tracking_id_summary.csv`保留类别和完整ID映射。','',
'## 边缘slot与内部slot的联合上下文对照','',
'只比较同一GT当前帧同时具有边缘和内部窗口的情况：边缘slot0..3或36..39，内部4..35。每一侧固定选距19.5最近的窗口，平局取较小起点。窗口内容与slot一起改变，因此不能归因为padding或slot本身。','',
'| strict尺度/边缘定义 | 配对GT帧 | 边缘→内部点增/损 | 框增/损 |',
'|---|---:|---:|---:|']
for sc in ['all','<=16','<=12']:
    for bb in ['boundary','left_boundary','right_boundary']:
        a=get('boundary',scale=sc,boundary=bb,metric='point');b=get('boundary',scale=sc,boundary=bb,metric='response')
        lines.append(f"| {sc}/{bb} | {a['gt_frames']} | {a['gains']}/{a['losses']} | {b['gains']}/{b['losses']} |")
lines+=['','≤16联合边缘→内部仅19点增/11损、21框增/12损；不是所有边缘失败都能由内部窗口消除。逐片段和左右边缘全部明细见CSV。','',
'## 所有已测窗口都未解决的缺口','',
'| strict尺度 | 所有窗口点失败 | 所有窗口框max<0.5 | 任一窗口点分歧 | 框跨越 |',
'|---|---:|---:|---:|---:|']
for sc in ['all','<=16','<=12']:
    r=get(scale=sc);n=r['gt_frames']
    lines.append(f"| {sc} | {ratio(r['all_contexts_point_negative'],n)} | {ratio(r['all_contexts_boxmax_low'],n)} | {ratio(r['point_disagreement'],n)} | {ratio(r['boxmax_crossing'],n)} |")
lines+=['','这是有限自然窗口集合的诊断，不能称所有可能上下文都无法恢复，也没有使用“任意窗口最好结果”作为性能。所有窗口低框响应说明本轮选窗口本身无法补齐主缺口；还需结合原始热图、对比度/尺度、可见性与训练分布解释。','',
'## 全GT的组件负担（CPU读取既存NPZ）','',
'每帧仅取指定规则的一张概率图，以8邻域在固定阈值标记组件。匹配数取既有全GT一对一匹配，未匹配数=组件数−匹配数；没有过滤大目标、越界框或重叠框来缩小分母。下表限定同样544个多窗口图像帧；含4230个GT帧。这些组件不是已经核实的假阳性，不能称官方Fa。','',
'| 片段 | 规则 | GT帧 | 点匹配 | 组件 | 未匹配组件 | 每图未匹配 |',
'|---|---|---:|---:|---:|---:|---:|']
for r in S['components']:
    if float(r['threshold'])==.5 and r['subset']=='multi_context_frames':
        lines.append(f"| {r['sequence']} | {r['rule']} | {r['gt_count']} | {r['matched_count']} | {r['component_count']} | {r['unmatched_components']} | {float(r['unmatched_components_per_frame']):.3f} |")
lines+=['','CENTER相对EARLIEST在全GT上点匹配1438→1450，未匹配组件12478→11220；但0032匹配833→825、0028_2为180→176，收益不一致。LATEST的未匹配组件更少（10523），同时匹配也更少（1383）。这说明规则改变存在检测数量与响应形态的权衡，不能仅挑总恢复帧数包装收益。','',
'## 固定阈值敏感性','',
'| 阈值，strict≤16 | GT帧 | EARLIEST→CENTER点增/损 | 框增/损 | 全窗口框低 |',
'|---|---:|---:|---:|---:|']
for th in [.1,.5,.9]:
    r=get(threshold=th);a=get('pairwise',threshold=th,a='earliest',b='center',metric='point');b=get('pairwise',threshold=th,a='earliest',b='center',metric='response')
    lines.append(f"| {th} | {r['gt_frames']} | {a['gains']}/{a['losses']} | {b['gains']}/{b['losses']} | {r['all_contexts_boxmax_low']} |")
lines+=['','各阈值strict分母不同，不能把上表当作同一队列的阈值收益曲线。clean和all_multi的阈值完整表已保存。0.5是既定主阈值，没有根据这些结果重新选阈值。','',
'## 从测量到可检验的解释','',
'1. **已经测到的能力缺口：** 同一当前帧对周围窗口和位置不够稳定；小目标三片段出现实际框响应跨越，四片段出现组件点匹配分歧。该结论覆盖全部轨迹，但变化集中度、标签编号不等于物体身份和片段差异限制泛化。',
'2. **原因尚未拆开：** 同一帧像素保持不变，变化来源于时序上下文/位置这一联合干预；不能据此单独断言padding错误、长程运动依赖或边缘slot机制。组件峰值可能移出框，解释了部分有框响应而点失败的情况。',
'3. **先核对训练曝光：** 官方TrainDataLoader.py已对序列构造多个结束位置的窗口[max(0,i−40),i)，并按样本长度加权随机采样。当前不能断言训练没有见过不同起点或位置，也不能直接把“增加不同起点覆盖”当作修正。下一步应核对这种已有采样对已测问题条件的实际曝光；任何数据改法都必须明确超出现有多起点随机采样的部分，再检验增益、损失和组件负担。此处不直接提出已成立的解决方案。',
'4. **更大的待解释缺口：** 778/1417个strict≤16 GT帧在全部已测窗口都低于0.5，远多于137个跨阈值帧。优先结合其他分层结果确定低响应对应哪些尺度、局部对比度、类别/场景和连续可见性条件，再决定训练样本覆盖或现有表征/监督的修正；当前结果没有识别出唯一模型内部原因。', '',
'本分析为查看主结果后的探索性分层；不做相邻帧独立显著性检验。只有四个预筛片段，未验证每帧可见性，且原标注存在重叠、编号与类别映射不唯一。clean/strict不能替代人工真值核验。','',
'## 复现与文件','',
'- `analyze_context.py`：统计全部固定规则、源标签、组件数（`--components`）；普通模式使用已有组件计数。',
'- `same_gt_frame_choices.csv`：逐GT帧三个规则、边缘/内部规则、响应与过滤标记。',
'- `pairwise_gain_loss.csv`：按序列、类别、尺度、长轨/预选轨和清洁度的增益/损失；含原max融合补充对照。',
'- `sensitivity_and_remaining_gap.csv`、`all_track_summary.csv`、`track_concentration.csv`：覆盖、集中度及仍全窗口失败的数量。',
'- `source_tracking_id_summary.csv`：原source tracking id标签分组，保留跨类别编号复用的核查信息。',
'- `boundary_interior_pairs.csv`：边缘/内部联合上下文配对。',
'- `chosen_per_frame_components.csv`、`component_summary.csv`：全GT组件负担，三阈值、每片全部160帧和多窗口136帧均保留。',
'- `SUMMARY.json`、`KEY_FINDINGS.json`：完整机器表与精简主结果；输入CSV/JSON、44个NPZ有SHA256记录。',
'- `write_report.py`：按结果自动生成本文。', '',
'CPU复现（热镜像，无GPU、无安装）：', '',
'```bash',
'docker run --rm --cpus 4 --memory 4g --user 1004:1004 --entrypoint python -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=1 -v /home/ly:/home/ly deeppro-official-eval:20261009 '+str(P/'analyze_context.py')+' --components',
'python3 '+str(P/'write_report.py'),
'```','']
(P/'REPORT.md').write_text('\n'.join(lines))
keys=dict(strict_le16=get(),strict_le12=get(scale='<=12'),strict_all=get(scale='all'),
    clean_le16=get(subset='clean'),
    strict_le16_pairs=[get('pairwise',a=a,b=b,metric=m) for a,b in [('earliest','latest'),('earliest','center'),('latest','center')] for m in ['point','response']],
    strict_le16_by_sequence=[get(sequence=s) for s in S['metadata']],
    strict_le16_nonselected=get(stratum_kind='selected_track',stratum='0'),
    strict_le16_longtracks=get(stratum_kind='long_track',stratum='1'),
    components_at_half=[r for r in S['components'] if float(r['threshold'])==.5 and r['subset']=='multi_context_frames'],
    limits=S['methods']['limitations'])
(P/'KEY_FINDINGS.json').write_text(json.dumps(keys,indent=2)+'\n')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(P.iterdir()) if p.is_file() and p.name!='ARTIFACT_SHA256.json'}
(P/'ARTIFACT_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(P/'REPORT.md')
