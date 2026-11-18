import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
Q=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
P=Q/'stage1_goal'; D=Q/'low_response_information_20261011/training_readiness'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=list((P/'runs').glob('*/RUN.json'))
runs=[]
for f in files:
    d=json.loads(f.read_text())
    runs.append(dict(path=str(f),sha256=sha(f),**{k:d.get(k) for k in ['strategy','status','steps','final_step','lr','seed','duration_seconds','plan_sha256','checkpoint_sha256']},extra_lambda=d.get('config',{}).get('extra_lambda')))
sources=[P/'pilot/train_pilot.py',P/'pilot/losses.py',P/'pilot/frozen_v1/train_pilot.py',P/'pilot/frozen_v1/losses.py',P/'pilot/TRAIN_PLAN.json',P/'TRAINING_FAILURE_DIAGNOSIS.md',P/'PILOT_RESULTS.md',P/'CVPR2027_RESEARCH_PLAN (1).md']
record=dict(audited_at_utc=datetime.now(timezone.utc).isoformat(),scope='Read-only inspection of stage1_goal runs, pilot source, frozen plan, training audit and supplied updated plan; project-wide filename/text search found no lambda-zero run artifact.',runs=runs,lambda_zero_completed=False,lambda_zero_launched_this_review=False,stable_training_base_demonstrated=False,official_gaussian_supervision_available=False,source_sha256={str(f):sha(f) for f in sources},technical_note='pilot/losses.py computes total=pos+neg+extra, hardcoded lambda=1; merely changing CONFIG.extra_lambda does not alter the loss. train_pilot.py compares frozen plan config exactly and has no extra-lambda CLI option.')
(D/'AUDIT.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
(D/'REPORT.md').write_text('''# 训练基础只读审查

当前未发现已完成的 λ=0 底损失控制，稳定训练基础也尚未建立。本轮未启动训练；原权重的分数/峰/轨迹诊断不受此项阻塞。

直接读取了 `stage1_goal/runs/*/RUN.json`：paired_mean、paired_max、ohem、random_pair_max 均完成 512 步，seed=46、学习率 1e-4、extra_lambda=1；另有 2 步 smoke。项目范围内文件名/文本检索没有找到独立 λ=0 权重、日志或评估产物。不存在的证据限于当前项目缓存范围。

`pilot/losses.py` 的实际表达式是 `total = pos + neg + extra`，`pilot/train_pilot.py` 没有 λ 参数，且逐项核对冻结计划。仅将 CONFIG 的 extra_lambda 改成 0 不会取消额外正项。后续若补控制，应在新目录复制并明确修改实际总损失为 `pos + neg`，同时记录源码差异；旧冻结文件保持不变。

可复用的材料是 `pilot/TRAIN_PLAN.json`（相同 512 步数据曝光及裁剪）、原发布权重、数据指纹、现有推理与点匹配脚本。控制需保持 seed46、fresh Adam、学习率1e-4、weight decay1e-4、每步4×40×128×128、FP32、全参数更新、BN running buffers 固定，以及固定最终步取权重；只移除额外正项。

这个 λ=0 仍是弱框 MIL 的底损失控制，不是原作者 Gaussian/SoftIoU 训练复现。原作者真实 Gaussian 标签与精确生成参数仍未在现有材料中取得。四个 λ=1 组共同背景失控，不能把其公共失败归因于额外正项，也不能将其视为稳定底座直接开展“普通样本 vs 针对性样本”。

若后续选定数据覆盖干预，必须先判定 λ=0 能否保持合理背景成本；若也失控，应先恢复稳定监督/训练基础。实际训练是否进入下一步由本轮三张主表的干预决策决定，本审查只报告前提与可复用材料。

所有读取源码与运行记录的 SHA256、完整运行清单见 [AUDIT.json](AUDIT.json)。
''')
print(json.dumps({'runs_checked':len(runs),'lambda_zero_completed':False,'training_launched':False}))
