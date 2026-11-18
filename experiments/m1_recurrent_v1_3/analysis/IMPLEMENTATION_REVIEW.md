# M1-v1.3 实现及定位边界复核

检查时间：2026-10-14T13:11:16.805480+00:00。这里只读复核冻结源码、已有预检收据和正在运行的训练工件；未执行模型、GPU工作或新增测试。没有开发集恢复结论。

冻结协议 SHA256：`ffd7e09f92708bed1899a50bd268e72c17e646915ad17b90030b17b8ce8b3acd`。已逐文件重算全部 12 个 code/reference SHA256，变更列表为 `[]`。工作流 PID 1040399 在检查时存在：True，收据状态 `RUNNING`。

| 训练臂 | RUN状态 | 已落盘日志步数 | 全局末步 | 末条plan_index | RUN与冻结source一致 |
| --- | --- | ---: | ---: | ---: | --- |
| BankCurrent | RUNNING | 8 | 2056 | 7 | True |
| BankHistory | RUNNING | 10 | 2058 | 9 | True |

这只是运行时快照，日志步数会继续增加。RUN在指定步数才更新进度字段，可能落后逐步JSONL；不能把这种落后本身解释为失败。

实现与已登记预检相符：

- `revision.py` 复用原始包的 `scan_features/read_then_write`，不重写来源bank核心。每个forward仅有局部BankState，容量4；写入对齐融合前的投影X，先读后写，TSFM candidate/readout均不回写。每帧配对数min(t,4)，40帧合计150。Current每个slot替换成当前X，仍执行同数量PDA/TSFM。
- 原53个可训练张量及模块对象保留，没有v1.2 gate和新增参数。`GPU_RESOURCES_V2.json`报告两臂原模型权重及Adam各张量exact迁移，109350可训练参数、180263总参数；它还记录丢弃模型上的预检训练，不能当作正式512步结果。
- `GPU_SEMANTICS.json`报告原核心与安装adapter exact、原始source缓存未被后续读取修改、t0等于当前X、chunk7输出exact、窗口隔离exact。输入/参数梯度分块比较有显式容差；未宣称DCNv2反向逐bit确定性。
- 同一预检报告trace三模式与标准forward的logit absmax均为0、150条slot记录及source-order readout均值一致。该parity来自训练来源小裁剪检查；完整512×640可行性由独立资源预检覆盖，二者不能合并声称已逐像素验证完整开发集trace。
- `infer.py`正式输出只含BankCurrentCurrent、BankHistoryHistory、BankHistoryCurrent，终点2560。BankCurrent权重下额外history计算仅用于trace诊断。Off走共享原H和原head；旧release缓存跨runtime尾差仍按既有2e−5/threshold-diff0约束记录。

解释边界：

- bank的来源年龄4指投影特征索引；X来自已有时空前端，不能称完整模型仅看4帧或Current为单帧模型。前端分块后重建完整F/H，PDA/TSFM在完整空间图执行；TSFM含全局池化。
- `previous_input`和兼容别名`before_pda`是来源X均值，`after_pda`为分别对齐后的均值，`after_tsfm`为实际candidate平均readout。它们不是沿单一递归状态逐步传递的三个位置；均值也可能抵消逐源差异。source-age sidecar只保留每源全图标量，不证明目标定位正确。
- ROI标签只在模型计算之后用于固定区域统计；目标、困难背景、anchor及cohort均复用旧冻结定义。主ROI汇总仍按重叠窗口→可用O/S→目标帧等权，t0单列。逐source_age的补充统计按window/frame/slot出现次数等权，不能混作独立对象证据。
- 同History权重history/current干预、独立训练Current/History比较分别回答不同问题。非零状态差、Jacobian、背景下降或正margin都不能替代目标增强及实际恢复；需同时看有符号目标/背景logit、逐对象G/L、连续漏检、轨迹覆盖、U/面积和保护对象。

证据链接及SHA256：

- [PROTOCOL_FREEZE.json](../PROTOCOL_FREEZE.json)：`ffd7e09f92708bed1899a50bd268e72c17e646915ad17b90030b17b8ce8b3acd`
- [checks/GPU_SEMANTICS.json](../checks/GPU_SEMANTICS.json)：`f5d9455222883c1566064629afa9602043e1c6bf2f819ac62fc6fc000751ea0e`
- [checks/GPU_RESOURCES_V2.json](../checks/GPU_RESOURCES_V2.json)：`3df3a6da6e709e63b86eceab3a9b2a116122e31a3028b38be4278478419ec42a`
- [checks/REFERENCE_TESTS.json](../checks/REFERENCE_TESTS.json)：`08bf906eac2715f505a83e9c8f7779ec164e723bb3faf8ccd4d763a7707f4ab4`
- [analysis/aggregate_trace.py](../analysis/aggregate_trace.py)：`775594afe9ed9f299e5ac582f4c6c771847ce76574185558b57d7ecac444c91e`
