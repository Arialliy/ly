#!/usr/bin/env python3
"""Render the fixed-pilot report from completed tables, with no model execution."""
import csv,json,hashlib,datetime
from pathlib import Path
C=Path(__file__).resolve().parent
E=C/'evaluation'; P=C/'evaluation_path'
def rows(name,base=E):
    return list(csv.DictReader((base/name).open()))
def table(headers, records):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in records])
def n(r,k): return int(r[k])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
summary=rows('SUMMARY.csv'); lookup={(r['condition'],r['arrangement']):r for r in summary}
assert all(n(lookup[v,a],'unmatched_components')>4015 for v in ['E0','E1','E2'] for a in ['O','S'])
primary=['release','E0','E1','E2']
main=table(['配置','安排','M','新增/丢失¹','U','框外前景A','未匹配组件面积','低响应恢复²'],[(r['condition'],r['arrangement'],r['M'],r['G_vs_release_same']+'/'+r['L_vs_release_same'],r['unmatched_components'],r['positive_pixels_outside_boxes'],r['unmatched_component_area_sum'],r['low_all_05_matched']+'帧/'+r['low_all_05_matched_tracks']+'轨迹') for r in summary if r['condition'] in primary])
stable=table(['配置','双成功','仅O成功','仅S成功','双失败'],[(r['condition'],r['both_matched'],r['O_only'],r['S_only'],r['neither_matched']) for r in rows('WINDOW_DEPENDENCE.csv') if r['group']=='pooled' and r['cohort']=='both_fixed_success' and r['condition'] in primary])
compar=[]
for v,ref in [('E1','E0'),('E2','E0'),('E2','E1')]:
    for a in ['O','S']:
        r=lookup[v,a];b=lookup[ref,a]
        compar.append((v+'−'+ref,a,r['G_vs_'+ref+'_same'],r['L_vs_'+ref+'_same'],n(r,'M')-n(b,'M'),n(r,'low_all_05_matched')-n(b,'low_all_05_matched'),n(r,'unmatched_components')-n(b,'unmatched_components'),n(r,'positive_pixels_outside_boxes')-n(b,'positive_pixels_outside_boxes')))
compare=table(['比较','安排','实际新增','实际丢失','M净变化','低响应恢复净变化','U变化','A变化'],compar)
mechanism=table(['条件','安排','M','对E2新增/丢失','U','A','低响应恢复'],[(r['condition'],r['arrangement'],r['M'],r['G_vs_E2_same']+'/'+r['L_vs_E2_same'],r['unmatched_components'],r['positive_pixels_outside_boxes'],r['low_all_05_matched']) for r in summary if r['condition'] in ['E2','E2_mass','E2_current']])
seq=table(['配置','安排','片段','M','新增/丢失¹','U','低响应恢复帧/轨迹'],[(r['condition'],r['arrangement'],r['group'],r['M'],r['G_vs_release_same']+'/'+r['L_vs_release_same'],r['unmatched_components'],r['low_all_05_matched']+'/'+r['low_all_05_matched_tracks']) for r in rows('SEQUENCE_SUMMARY.csv') if r['condition'] in ['E0','E1','E2']])
trajectory=table(['配置','安排','各轨迹最长漏检段均值','最长段最大值','对发布缩短/变长轨迹','对E0缩短/变长轨迹'],[(r['condition'],r['arrangement'],f"{float(r['longest_miss_track_mean']):.3f}",r['longest_miss_max'],r['tracks_shorter_vs_release']+'/'+r['tracks_longer_vs_release'],r['tracks_shorter_vs_E0']+'/'+r['tracks_longer_vs_E0']) for r in rows('CONTINUOUS_MISS_SUMMARY.csv') if r['group']=='pooled' and r['cohort']=='all' and r['condition'] in primary])
source_records=[]
for r in rows('PROVENANCE_SUMMARY.csv'):
    if r['group']!='pooled' or r['cohort'] not in ['low_all_05','both_fixed_success']:continue
    for prefix in ['forward','mutual']:
        source_records.append((r['arrangement'],r['cohort'],prefix,f"{float(r[prefix+'_nonnull_mass_object_equal_mean']):.6f}",*[f"{100*float(r[prefix+'_'+cat+'_mass_fraction']):.3f}%" for cat in ['same_track_region','other_target_region','outside_all_annotated_boxes','uncertain']]))
source=table(['安排','冻结集合','权重','非空权重总量均值','同ID粗框','其他目标粗框','全部标注框外','不确定'],source_records)
costs=rows('COMPUTE_COST.csv'); costrows=[]
for v in ['E0','E1','E2']:
    run=json.loads((C/'training'/v/'run/RUN.json').read_text()); rr=[r for r in costs if r['condition']==v]
    costrows.append((v,run['parameter_count'],f"{run['duration_seconds']:.2f}",f"{run['max_cuda_memory_allocated_bytes']/2**30:.3f}",*[f"{sum(float(r['model_plus_cpu_copy_seconds']) for r in rr if r['arrangement']==a):.3f}" for a in ['O','S']],f"{max(int(r['peak_cuda_allocated_bytes']) for r in rr)/2**30:.3f}"))
cost=table(['配置','参数','训练秒','训练峰值GiB','两片O五窗秒','两片S五窗秒','推理峰值GiB'],costrows)
path_status=(P/'SUMMARY_COMPLETE.json').is_file() and (P/'TABLE_VALIDATION.json').is_file()
if path_status:
    pp={(r['condition'],r['arrangement']):r for r in rows('SUMMARY.csv',P)}; vals=[]
    for step in [0,128,256,384,512]:
        for a in ['O','S']:
            cells=[]
            for v in ['E0','E1','E2']:
                r=pp['release' if step==0 else v if step==512 else f'{v}_step{step}',a]
                cells.append('/'.join(r[k] for k in ['M','unmatched_components','low_all_05_matched','L_vs_release_same']))
            vals.append((step,a,*cells))
    path=table(['步数','安排','E0：M/U/低恢复/丢失¹','E1：同口径','E2：同口径'],vals)
    path_note='所有非零固定检查点均超过U预算。该结论只覆盖128/256/384/512已观测节点，不能推断第1步起就失控，也不排除节点之间存在未观测状态。未挑选中间检查点替代终点。'
    assert all(n(r,'unmatched_components')>4015 for r in pp.values() if r['condition'].startswith('E'))
else:
    path='固定中间检查点评估尚在完成；主终点结果与停止决定已成立。'
    path_note='不依据中间结果变更终点。'
validation=json.loads((E/'TABLE_VALIDATION.json').read_text()); independent=json.loads((C/'RESULTS_AUDIT.json').read_text())
text=f'''# TPro前可拒绝局部时序聚合：固定首轮结果

2026-10-12。**当前版本未通过，停止扩展。** E1未带来预算内恢复；E2与同监督原结构E0的实际匹配几乎相同，且两个机制诊断几乎不改变匹配。不能将双向核验确定为论文主线。

**研究目标保持：恢复真实场景中可重复的持续漏检，同时约束背景成本和原成功退化。** 本轮负结果限定于当前结构、监督、半径2、固定seed46与512步协议，不证明所有局部时序聚合无效，也不将自定义适配引起的背景扩张归为官方时序缺陷。

## 协议与执行

正式训练仅E0/E1/E2各一次，全部从同一发布权重开始，E1/E2新增状态相同。共同监督保留近似Gaussian派生二值标签；正窗沿用masked官方Soft-IoU，仅空标签窗使用既有top-1%背景负项。全部原参数可训练，BN运行缓存固定。训练来源、数据顺序与512步曝光一致。唯一新增模块在layer1后、原TPro前，描述子8维、t±1、5×5局部范围；E1单向，E2前后向独立归一化后相乘，保留空选项，不强制非空归一化。

依据用户最新指令，**seed固定46，不追加其他seed**。[当前有效协议](EFFECTIVE_PROTOCOL.md)覆盖原启动协议的后续多seed条款；[启动冻结清单](PROTOCOL_FREEZE.json)保持不变供审计。官方源码只定义了默认46的种子函数，未找到调用，worker还使用当前时间；**作者发布权重实际训练seed未确认**。本轮显式固定46，详见[源码种子核对](OFFICIAL_SEED_NOTE.md)。本实验不是作者完整训练配方复现。

评估固定两个开发片段、5200严格小目标帧/130轨迹，O/S均五窗，阈值0.5、原概率max融合和组件峰值点匹配不变。U≤4015为冻结预算；U不是论文Fa，M不是论文Pd。4342低响应、4170极低响应与646保护集合成员不变。40帧原生512×640采用core128/halo20分块；不宣称完成原生整图直接前向。3个固定真实裁剪的直接/分块对照在每项推理上检查。

## 主终点：恢复伴随超预算背景

{main}

¹ 新增/丢失均相对发布权重的同一安排。² 低响应集合固定4342帧；恢复均覆盖两个开发片段，不能视为新来源验证。极低响应4170子集中，E0 O/S恢复246/268，E1为235/239，E2为244/271。全量[逐对象对照](evaluation/OBJECT_COMPARISON.csv)含10400行，不删失败对象。

{compare}

E1对E0的U减少，但A增加，且匹配和低响应恢复下降。E2对E0仅新增/丢失3/4(O)、3/1(S)，低响应恢复净变化−2/+3；其343/375个恢复不能全部归因于结构。E2对E1多15/47个匹配，同时多1330/1334个未匹配组件，仍没有预算内收益。

E0相对先前Gaussian近似控制减少背景组件，但匹配也下降：旧控制O/S为M1568/1656、U19048/19446。本轮只说明空窗负监督发生了作用并改变检测取舍，没有建立可靠背景控制；旧固定终点失败保持不变。

## 原成功保持与来源分布

{stable}

E2有43个保护对象帧不再保持双侧成功，其中22个仅一侧失败、21个双侧失败；不能把43全部写成双侧漏检。

{seq}

E2低响应恢复中，321/343(O)、343/375(S)来自0067，涉及该来源60/61条轨迹；0026分别22/32帧、3条轨迹。增益并非单轨迹，但仍集中于0067；只有两个开发来源，不能声称稳定泛化。

## 持续漏检：E2没有缩短E0的各轨迹最长漏检段

{trajectory}

完整观察范围均为5200帧/130轨迹/2片段；缺标、local或source帧号中断、原筛选空缺均打断连续段。[全部1547个发布漏检段](evaluation/RELEASE_MISS_SEGMENTS.csv)保留，包括没有任何恢复的段。E2相对E0，O的130条轨迹最长漏检段全部相同；S有1条变长、没有缩短。相对发布权重的改善主要已经出现在共同训练E0，不能作为双向核验的独立贡献。

## 两项固定权重机制检查

{mechanism}

**mass：** 保持每query/reference的E2非空权重总量，改按前向概率分配。O匹配集合完全不变；S只丢1帧。没有显示双向所选位置的明显独立价值。保持权重和不等于保持特征修正范数，诊断不构成强因果证明。

**current：** 仅新增分支参考特征换成当前帧。O只丢1帧，S匹配集合完全相同，原低响应恢复也相同。当前结果未显示该分支的显式邻帧输入带来有意义的实际收益；前端和TPro仍处理时间，不能称整个模型单帧；训练后替换也有分布变化。

最后128次训练均值：E1/E2空选项概率0.004192/0.004265，非空权重总量0.995808/0.117058，修正/特征范数0.074376/0.004410。E2较小修正与检测接近E0、mass/current结果近似一致，**更符合新增路径作用很弱的描述，没有支持对应核验有效的证据链**。小非空权重主要涉及乘积收缩，不能直接称为学会了拒绝错误参考。

### 粗框来源描述

{source}

查询预先按几何冻结：4342低响应＋646保护帧，共4988对象帧；先在对象内平均各窗口/参考，再做对象等权统计。该表是E2同一特征下forward与mutual权重的描述，不是E1训练模型与E2之间的对应准确率。低响应同ID粗框权重占比只从约82.43%到83.43%；近中心5×5搜索本来就常落入同一粗框。粗框不能提供像素对应真值，框外不自动等于真实背景，权重归属也不证明恢复的因果来源。详见[来源逐对象表](evaluation/PROVENANCE_OBJECTS.csv)。

## 固定训练路径

{path}

{path_note}

## 计算成本与核查

{cost}

五窗耗时为两个片段各自五窗耗时之和；同RTX5090型号、相同分块、包含输出CPU拷贝，不含写盘压缩。并行执行时测量，非独立性能基准。E2扣除了稀疏来源采集计时，但仍可能含查询准备开销，峰值显存含遥测。新增1281参数不等于计算成本可忽略：本轮E1/E2五窗耗时约为E0的3倍。

- 实现测试137项、共同损失13项、整模型/资源24项、独立运行器源码31项通过。可见[checks](checks/)。
- [训练执行审计](TRAINING_AUDIT.md)216项通过：1536次更新的实际输入/标签哈希相同，15个快照正确，BN缓存未变；385空窗、16个全空batch均获得负监督梯度，三组正式容器退出0。
- [主表一致性核查](evaluation/TABLE_VALIDATION.json)通过；[独立结果审计](RESULTS_AUDIT.md)1886项通过，独立复算5条件×2片段×O/S×5固定帧，共100组。
- [执行收尾核查](FINAL_EXECUTION_AUDIT.json)262项通过：28推理和28评分完成，84个分块对照的0.5阈值判决差异均为0，最大logit误差7.439e−5；无本轮活跃GPU任务。训练路径表核查112741项通过。
- 主结果、固定节点路径和两项机制检查均不进行选阈值或选优节点。原源码及历史结果只读；本轮无新环境安装、无BC-TPro重跑。

## 去留决定

**停止本候选的扩展，不进入新来源确认，不追加模块、其他seed、训练步数或阈值试验来挽救本轮。** 三组共同训练仍超预算，E1未显示有利的完整取舍，E2未显示相对E0的实际恢复/持续性收益；mass和current诊断也没有支持核心机制必要性。

研究问题保留，当前没有获得成功方法。下一项结构若要启动，必须有新的具体证据和可区分于本轮的机制假设；不自动转入第二版。本轮不另外命名新模块或新增训练。

交付索引：[有效协议](EFFECTIVE_PROTOCOL.md) · [单一模块](code/module.py) · [主结果CSV](evaluation/SUMMARY.csv) · [逐轨迹表](evaluation/TRACK_SUMMARY.csv) · [固定训练路径](evaluation_path/SUMMARY.csv) · [文献机制定位](LITERATURE_ALIGNMENT.md)。
'''
(C/'REPORT.md').write_text(text)
inputs=[E/n for n in ['SUMMARY.csv','SEQUENCE_SUMMARY.csv','WINDOW_DEPENDENCE.csv','CONTINUOUS_MISS_SUMMARY.csv','PROVENANCE_SUMMARY.csv','COMPUTE_COST.csv','TABLE_VALIDATION.json']]+[C/'TRAINING_AUDIT.json',C/'RESULTS_AUDIT.json',C/'SEED_AMENDMENT.json']
if path_status:inputs += [P/'SUMMARY.csv',P/'TABLE_VALIDATION.json']
(C/'REPORT_SOURCES.json').write_text(json.dumps({'rendered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'path_complete':path_status,'source_sha256':{str(f):sha(f) for f in inputs},'renderer_sha256':sha(Path(__file__)),'report_sha256':sha(C/'REPORT.md')},indent=2)+'\n')
print(json.dumps({'report':str(C/'REPORT.md'),'path_complete':path_status}))
