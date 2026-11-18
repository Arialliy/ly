import json,csv,hashlib
from pathlib import Path
from datetime import datetime,timezone
L=Path(__file__).resolve().parent
summary=json.loads((L/'SUMMARY.json').read_text());run=json.loads((L/'run/RUN.json').read_text());protocol=json.loads((L/'PROTOCOL.json').read_text())
rows=[r for r in summary['metrics'] if r['group']=='pooled' and r['reference']=='same_arrangement']
assert len(rows)==2
lines=[]
for r in rows:
    lines.append(f"| {r['arrangement']} | {r['reference_small_matched']} | {r['M']} | {r['G']} | {r['L']} | {r['unmatched_components']} | {r['positive_pixels_outside_boxes']} | {r['reference_U_same_arrangement']} | {r['reference_A_same_arrangement']} |")
all_failed=all(not r['passes_pooled_U_budget'] for r in rows)
conclusion=('λ=0 在两个固定安排下都超过原冻结 U≤4015 预算。额外正项不是本次共同背景失控的必要条件；当前弱框底损失/全参数适配设置仍不能作为稳定训练基础。这不证明额外项完全无影响，也没有定位共同设置中的唯一原因。' if all_failed else '至少一个安排的 λ=0 通过 U≤4015；应结合完整新增/丢失和框外前景判断稳定性，不能单凭正响应升高或U通过认定方法改善。')
text='''# 同预算 λ=0 底损失控制

本控制已完成；它补齐旧弱框训练的缺失前提，不是官方 Gaussian 监督复现，也不是本轮新方法成绩。原权重的三表诊断保持独立。

从旧 `pilot/frozen_v1` 复制 trainer 与 loss 到新目录；trainer 字节不变，实际训练损失唯一修改为 `total = pos + neg`。仍计算 extra 用于原日志，但不加入优化目标。原 TRAIN_PLAN、seed46、512更新、fresh Adam、LR1e-4、weight decay1e-4、4×40×128×128输入、FP32、全参数更新及BN running buffers固定均保留，取固定最终步权重。

**元数据边界：** 原计划与 trainer 的历史 CONFIG.extra_lambda=1、RUN.strategy=paired_mean 保留原样以复用冻结计划；它们不是本控制的有效损失系数。有效系数为0，由 [PROTOCOL.json](PROTOCOL.json)、[精确一行差异](losses.py.diff)、新 loss 源码SHA和训练日志总损失检查共同确定。权重与训练记录位于独立 `run/`。

检测阈值固定0.5；O/S各自仍为原40帧×5窗、完整160帧覆盖、先sigmoid再实际probability max，再运行原组件峰—框一对一点匹配。没有阈值调优、放宽框或更改峰规则。旧推理脚本缓存11窗，实际评分只采用两固定安排各自的5窗，额外缓存不进入融合。

| 安排 | 原权重 M | λ0 M | 新增G | 丢失L | λ0 U | λ0 框外前景A | 原权重 U | 原权重A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(lines)+'\n\n'+conclusion+f'''

G/L均相对相同安排的原发布权重，原安排700与共用对称安排720是两个分开的控制；[METRICS.csv](METRICS.csv)还另列S相对原O的差异。分母为5200严格小目标帧；背景匹配仍用全部有效标注。U和A是诊断量，不是官方Fa。

训练耗时 {run['duration_seconds']:.2f} 秒，评分耗时 {summary['elapsed_seconds']:.2f} 秒。训练日志记录项均有限、总loss等于positive+negative，512步完成；BN buffer前后SHA一致，旧源码/计划/选择文件SHA均保持一致。环境规格与既有ledger哈希一致，复用已经通过kernel witness和doc-follow的Docker镜像，未重建环境、未停止其它作业。

可复算产物：[逐对象实际匹配](OBJECTS.csv)、[完整汇总与校验](SUMMARY.json)、[训练运行记录](run/RUN.json)、[训练日志](run/TRAIN.jsonl)、[完整启动参数](LAUNCH.json)。所有λ0结果只解释当前弱框训练底座，不能替代下一项数据覆盖/监督干预的公平对照。
'''
(L/'REPORT.md').write_text(text)
record=dict(status='COMPLETE',effective_strategy='lambda0_base_weak_box_MIL',effective_extra_lambda=0,legacy_trainer_strategy=run['strategy'],legacy_trainer_extra_lambda=run['config']['extra_lambda'],fixed_threshold=.5,final_checkpoint_sha256=run['final_checkpoint_sha256'],training_checks=summary['training_checks'],all_arrangements_exceed_U_budget=all_failed,generated_at_utc=datetime.now(timezone.utc).isoformat())
(L/'CONTROL_METADATA.json').write_text(json.dumps(record,indent=2)+'\n')
files={str(p.relative_to(L)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(L.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json' and '__pycache__' not in p.parts}
(L/'ARTIFACT_SHA256.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps({'pooled':rows,'artifacts':len(files),'all_failed':all_failed}))
