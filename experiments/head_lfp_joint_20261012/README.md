# H / H＋LFP 联合适配实验

本轮运行及输出均在`/home/ly/DeepProPlus_TLFP_Code`内。两组训练与全部固定节点评价已经完成，勿重复运行训练。

| 入口 | 内容 |
|---|---|
| [结果图](figures/final_results.png)／[PDF](figures/final_results.pdf) | 匹配、背景、低响应恢复与保持，独立核对通过 |
| [主结果表](RESULTS.md)与[本轮判定](DECISION.md) | 固定2048步；H的恢复与HL增量分别判断 |
| [训练观察汇总](observations/REPORT.md) | 固定训练窗口上的匹配、背景与相对分数过程 |
| [固定协议](PROTOCOL.md) | 两组参数范围、监督、2048步预算、观察与评价规则 |
| [用户原方案](USER_PLAN.md) | 本轮执行依据 |
| [协议与源码冻结](PROTOCOL_FREEZE.json) | 正式训练前的SHA256 |
| [工作流实时状态](runtime/WORKFLOW.json) | 按项运行、退出码及当前工作 |
| [环境与容器命令](runtime/ENVIRONMENT.md) | 包内数据挂载、GPU检查及独立执行核验 |
| [训练与检查点](training/) | H/HL的0、128、512、1024、2048节点 |
| [固定训练观察](observations/) | 16个训练来源裁剪窗口，独立于正式开发统计 |
| [完整画面预测](predictions/) | 各组各正式节点两个片段的窗口预测 |
| [正式评价](evaluation/) | 三个节点的O/S结果和对象集合差异 |
| [实现与执行核验](checks/) | 实际模型CPU、GPU、BN和逐名训练审计 |

仅适配原预测头为H；预测头与已验证空间LFP共同学习为HL。二者从同一发布权重出发，seed46。2048步是主终点，512/1024全部记录为学习过程，不挑选最好节点。

训练计划是原512条裁剪顺序重复四轮，不是2048条独立新采样。完整公开数据的收录不改变8训练／2开发／4确认边界。确认集本轮未推理。

`code/workflow.py`是本次单次工作流，已有状态即拒绝重复。具体任务由`code/launch.py`写独立回执；原项目只读归档挂载到容器兼容路径，实际宿主机读取的文件来自统一包。固定公开镜像与`environment/site`复用，不安装包或改动原代码。

正式表格在`RESULTS.md`，范围解释和决策在`DECISION.md`；M/U继续是诊断匹配／未匹配组件，不能代替官方Pd/Fa。
