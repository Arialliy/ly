from pathlib import Path
import csv,json,hashlib,datetime
N=Path(__file__).resolve().parents[1];E=N/'evaluation'
read=lambda name:list(csv.DictReader((E/(name+'.csv')).open()))
summary=read('SUMMARY');traj=[r for r in read('CONTINUOUS_MISS_SUMMARY') if r['group']=='pooled' and r['cohort']=='all']
protection=[r for r in read('WINDOW_DEPENDENCE') if r['group']=='pooled' and r['cohort']=='both_fixed_success']
trans=[r for r in read('COHORT_TRANSITIONS') if r['group']=='pooled' and r['condition']=='R2' and r['reference'] in ('R0','R1') and r['cohort'] in ('all','low_all_05','extreme_all_001')]
assert all(int(r['unmatched_components'])>4015 for r in summary if r['condition']!='release')
assert json.loads((E/'TABLE_VALIDATION.json').read_text())['status']=='PASS'
assert json.loads((N/'checks/TRAINING_AUDIT.json').read_text())['status']=='PASS'
lines=['## 最终结果与去留','',
'**三组均未通过：停止当前目标级时序响应解码器原型，不进入确认实验，不选择中间检查点，不追加训练或模块。** 研究目标保留；本轮不证明所有目标查询、轨迹监督或直接响应学习无效。',
'', '以下O/S分别为原五窗／最佳共用五窗，阈值均为0.5。M为小目标帧诊断匹配，U为全部GT参与对应后仍未匹配的预测组件。', '',
'| 配置 | 安排 | M | 新增／丢失原成功 | 低响应集合匹配 | 极低响应集合匹配 | U（预算4015） |',
'|---|---|---:|---:|---:|---:|---:|']
for r in summary:
 lines.append('| {condition} | {arrangement} | {M} | {G_vs_release_same}／{L_vs_release_same} | {low_all_05_matched} | {extreme_all_001_matched} | {unmatched_components} |'.format(**r))
lines+=['','R2确实在冻结极低响应集合中形成613／576个匹配，涉及78／77条标注轨迹ID及两个片段；不能写成没有任何低响应匹配。但这些计数同时伴随严重背景扩张，不能单独证明恢复了可靠的目标证据。大量预测区域覆盖背景时，落框匹配也不足以建立预算内的检测改善。本轮不追加诊断来挽救候选。','',
'| 配置 | 安排 | 框外前景像素 | 未匹配组件面积和 | 646保护集合单侧保持 |','|---|---|---:|---:|---:|']
for r in summary:
 lines.append('| {condition} | {arrangement} | {positive_pixels_outside_boxes} | {unmatched_component_area_sum} | {both_fixed_success_matched} |'.format(**r))
lines+=['','R2的U约为共同预算的71倍，框外前景像素与未匹配组件面积也大幅恶化；不是仅因组件划分导致的单一计数问题。646个保护对象帧中，R0／R1／R2仍双侧成功82／60／87个；R2有446个双侧均失败、113个只在一侧成功。因而“失去双侧保持559个”不能写成559个全部双侧漏检。','',
'| 配置 | 安排 | 最长漏检段缩短／变长的轨迹数（对发布权重） | 逐轨迹最长漏检段均值 |','|---|---|---:|---:|']
for r in traj:
 lines.append(f"| {r['condition']} | {r['arrangement']} | {r['tracks_shorter_vs_release']}／{r['tracks_longer_vs_release']} | {float(r['longest_miss_track_mean']):.3f} |")
lines+=['','连续性统计使用130条评价标注轨迹ID，缺标、来源帧号中断与筛选空缺均打断连续段。R2相对发布权重确实缩短50／54条轨迹的最长漏检段，也延长22／23条。这些局部变化如实保留，但发生于严重超预算且原成功大量丢失的工作状态，不能据此宣布持续检出问题解决。','',
'### 三个控制分别说明什么','',
'- R0相对发布权重：独立直接读出已有极低响应匹配，但没有背景受控、保持受限的可用收益。',
'- R1相对R0：O/S少309／285个匹配。新增跨帧交互没有形成实用优势；不能把时序结构本身当作已验证贡献。',
'- R2相对R1：M净增409／368；相对R0净增100／83，低响应匹配净增84／78。这些条件内增量不能抹掉，也不能升级为预算内的机制成功。',
'- R2相对R0的实际集合变化为新增497／482、丢失397／399，并非保留全部R0成功后再增加100／83个。相对R1新增588／567、丢失179／199。',
'- R2相对R0的最长漏检段缩短36／40条，也变长26／24条。相对R1缩短54／52条、变长8／9条。二者都仍不满足检测约束。',
'- R2低响应匹配主要来自0067片段：O的587/635、S的563/601；0026分别48／38。虽涉及两个片段，也不能声称跨来源稳定有效。','',
'### 执行与归档','',
'三组均固定seed46，从相同发布原网络与新增参数初始化训练2048步，耗时932.20／950.18／957.76秒。12个检查点的原参数及BN逐张量不变；完整输入和标签曝光流三组一致。六次最终全图推理、六组O/S评分已完成，逐对象明细10400行，表格独立复算27366项通过。实现／训练完整性通过与检测候选失败分开记录。',
'',
'GPU1由用户补充授权后用于R2训练。GPU1/2在训练结束后有其他任务占用，原卡推理启动在进入Docker前被阻止；所有六次实际推理最终用GPU3完成。阻止记录及资源调度保留，未中止其他任务、未改变模型或预算。',
'',
'全部交付见 evaluation/SUMMARY.csv、OBJECT_COMPARISON.csv、COHORT_TRANSITIONS.csv、CONTINUOUS_MISS_SUMMARY.csv、CONTINUOUS_MISS_TRACKS.csv、WINDOW_DEPENDENCE.csv；具体保护、恢复和丢失集合均可追溯。当前候选状态：CLOSED_FAILED_BACKGROUND_AND_RETENTION_NO_EXTENSION。',
'']
report=N/'REPORT.md';text=report.read_text();old='状态：正式训练运行中，尚无本轮开发检测结论。研究对象仍是背景受控、原成功退化受限的真实持续漏检恢复；训练目标下降不是方法通过的判据。'
assert old in text
text=text.replace(old,'**首轮已完成，R0／R1／R2均未通过背景与保持约束，当前原型停止扩展。** 研究目标保留；条件内低响应匹配与连续性变化保留，不认定为可用方法。',1)
pos=text.index('## 本轮改变及可解释范围')
text=text[:pos]+'\n'.join(lines)+'\n'+text[pos:];report.write_text(text)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
decision={'status':'CLOSED_FAILED_BACKGROUND_AND_RETENTION_NO_EXTENSION','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'question_retained':True,'successful_method_established':False,'core_mechanism_established':False,
 'main_endpoint':2048,'seed':46,'threshold':.5,'U_budget':4015,'all_variants_over_budget':True,
 'confirmatory_run_authorized_by_this_outcome':False,'automatic_new_modules':False,'automatic_more_training':False,
 'old_candidates_unchanged':True,'scope':'Two reused development clips, diagnostic matching; not official Pd/Fa or independent test',
 'main_rows':summary,'protected_rows':protection,'continuous_miss_rows':traj,'R2_comparisons':trans,
 'supported':'R2 conditional matching/count and miss-run gains exist in a severely over-budget state; no usable constrained improvement',
 'not_supported':['query-decoder prototype solves repeated misses under budget','original successes are retained','all object-level temporal learning is disproved','unique failure root cause determined'],
 'source_sha256':{str(p):sha(p) for p in [E/'SUMMARY.csv',E/'COHORT_TRANSITIONS.csv',E/'CONTINUOUS_MISS_SUMMARY.csv',E/'WINDOW_DEPENDENCE.csv',E/'TABLE_VALIDATION.json',N/'checks/TRAINING_AUDIT.json']}}
(N/'DECISION.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False)+'\n')
(N/'STATUS.json').write_text(json.dumps({'status':decision['status'],'execution_complete':True,'formal_training_complete':3,'final_inferences_complete':6,'scoring_complete':6,'main_checkpoint_step':2048,'table_validation':'PASS','training_integrity':'PASS','active_own_gpu_jobs':[],'report':'REPORT.md','decision':'DECISION.json','old_candidates_remain_closed':True},indent=2)+'\n')
print('closeout written')
