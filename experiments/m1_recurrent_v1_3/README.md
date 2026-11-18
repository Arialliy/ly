# M1-v1.3 源帧证据库

当前状态以 [运行收据](runtime/WORKFLOW.json) 为准；`RESULTS.md` 仅在全部正式评价、作用链汇总和最终快照核验完成后生成。

本轮直接接入用户提供的原样 evidence_bank 核心。原 M1 53 个可训练张量、PDA/TSFM、两投影及预测链保留；未融合源帧投影 X 按 FIFO4 保存，每个源帧独立重新对齐、融合、均值读取。Bank-Current 用当前 X 替代每个历史槽，保持相同150次算子对调用。无 M2、无 v1.2 门、无新增参数。

- [固定实验协议](PROTOCOL.md) / [训练前哈希冻结](PROTOCOL_FREEZE.json)
- [接入代码](code/revision.py) / [用户原样核心](reference/M1_v13_evidence_bank_reference/evidence_bank.py)
- [来源边界](PROVENANCE.md)
- [原包29项测试](checks/REFERENCE_TESTS.json)
- [真实CUDA状态/梯度/trace一致性](checks/GPU_SEMANTICS.json)
- [完整分辨率和真实训练成本](checks/GPU_RESOURCES_V2.json)
- [Bank-Current训练](training/BankCurrent/RUN.json) / [Bank-History训练](training/BankHistory/RUN.json)
- [同权重作用链采集](code/trace.py) / [冻结组件匹配评价](code/evaluate.py)
- [完成后结果](RESULTS.md) / [可核验结果数据](RESULTS.json)

每组从旧 History2048 完整恢复权重、Adam 与 RNG，seed46、同曝光各512步，最终2560为主。过程快照2048/2176/2304/2560只用于续跑与解释，不择优。512步实测预算约4—5小时，两GPU并行；完整窗口推理约3.35—3.75秒、峰值allocated8.01GiB（预检条件），不等于部署综合吞吐。

首次资源预检 `GPU_RESOURCES.json` 保留失败记录：测试重复使用同一个内存中Adam字典，CPU step共享造成第二组起点错误。修正仅限测试隔离，重跑 `GPU_RESOURCES_V2.json` 通过；磁盘原检查点和正式双进程训练均未受影响。

缓存、状态差异及梯度测试通过只说明实现符合约定。是否推进持续漏检恢复，以独立训练控制和同History权重干预的实际R/L、连续漏检、原成功保持和背景成本共同判定。
