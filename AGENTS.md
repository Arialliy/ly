# 项目与执行约定

项目根目录为 `/home/ly/DeepProPlus_TLFP_Code`。项目代码、数据、检查点、训练日志、预测和报告存放在该目录内。自主模型实现在 `experiments/irstd_independent_v1/`，新实验写入相应运行目录；历史数据、发布权重、参考复现和已冻结的运行记录按原路径只读使用。

## 运行身份

- S4 固定训练种子 46，共同比较 M0、Msep、Mraw、Mint。实际协议、依赖摘要、输入清单和恢复边界以正式运行目录内的绑定文件为准。[子运行绑定](experiments/irstd_independent_v1/reports/s4_runs/s4_24k_seed46_q256v81_20261118_02/execution_binding.json)
- 三个证据模型保留父运行各自前 600 次更新的权重、优化器和抽样游标。子运行继续执行原定日程。查看 S4 状态或恢复前先核对运行 ID、控制器状态、已有作业和已提交的检查点；恢复操作使用绑定该运行的代码与配置。[父运行](experiments/irstd_independent_v1/reports/s4_runs/s4_24k_seed46_20261118_01/) · [子运行](experiments/irstd_independent_v1/reports/s4_runs/s4_24k_seed46_q256v81_20261118_02/)
- 评价沿用 S0 的原生坐标、完整中心清单和逐帧目标匹配。历史测试仅在共同终点及开发阈值锁定后进入正式评价。[数据契约](experiments/irstd_independent_v1/reports/S0_data_contract.md) · [研究协议](1/S4_research_protocol.md)

## 本地环境

- S4 正式训练使用本机 GPU 1、2、3。每次启动先按物理 UUID 核对占用进程、显存和利用率，等待空闲卡并保留其他作业。使用已经启动的正式控制器及其独占锁。[本地调度器](experiments/irstd_independent_v1/checks/run_s4_local.py)
- Docker 镜像固定为 `sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92`，复用 `environment/site/`。正式单 GPU 容器设置 4 个 CPU、20 GiB 内存，使用非 root 用户并禁网。项目只读挂载，本次运行目录可写。精度和确定性按正式协议执行。监控使用本地状态文件。[本地依赖](environment/site/) · [执行锁](experiments/irstd_independent_v1/reports/S4_protocol_lock.json)
