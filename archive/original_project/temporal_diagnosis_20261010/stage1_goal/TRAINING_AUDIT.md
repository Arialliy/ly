# 四组正式训练独立完成性审计

日期：2026-10-10（Asia/Taipei）。独立审计者 `/root/stage1_independent_audit`。完整状态、命令、哈希、分布和张量统计见 [TRAINING_AUDIT.json](TRAINING_AUDIT.json)。

**结论：PASS_TRAINING_COMPLETION_ONLY。四组真实容器均已退出且退出码0，均完成512步；共享初始化、数据曝光、训练配置和BN控制一致，最终权重真实更新且数值有限。这个结论只涉及训练完成性，不代表检测效果或第一阶段目标通过。**

## 真实执行及产物

以 `runtime/TRAIN_LAUNCHES.json` 中记录的容器名和完整ID查询 Docker inspect，并读取实际 Docker logs；初查四组尚在运行，最终复查四组均 `exited / Running=false / ExitCode=0 / OOMKilled=false`。未以运行中的 ExitCode字段或单独RUN.json认定完成。实际Cmd均指向 `pilot/frozen_v1/train_pilot.py`；官方源码目录以只读方式挂载。

|策略|实际退出UTC|更新步数|参数张量改变|对初始参数L2差|训练耗时秒|
|---|---|---:|---:|---:|---:|
|paired_mean|2026-10-09T20:13:59.716805789Z|512|64/64|0.913549|229.90|
|paired_max|2026-10-09T20:14:05.084644075Z|512|64/64|0.980496|232.21|
|ohem|2026-10-09T20:14:07.420648802Z|512|64/64|0.954919|233.63|
|random_pair_max|2026-10-09T20:14:05.009059595Z|512|64/64|0.938998|231.71|

四组日志均有33条记录：第1、17、…、497及512步；实际容器stdout中的JSON记录与磁盘 `TRAIN.jsonl` 逐条一致。所有记录的loss及梯度范数有限，梯度范数均正。每组RUN状态complete、final_step512、checkpoint内部step512、容器退出0相互印证。最终checkpoint和日志SHA256均与RUN记录匹配。

## 共享条件与输入曝光

四组实际使用相同发布权重、新Adam、seed46、FP32、512步、lr1e-4及weight_decay1e-4，模型与损失源码、数据选择和预生成plan哈希相同：

- 初始checkpoint：`4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc`。
- 冻结训练源码：`ca37be9d9ecb68ca1e779c2ba7177d1f87e498d57ef272d99470f46e97b702af`。
- 冻结loss源码：`bbb51878fc2b442626707ff117474e2df4cad992a715c569aae56c9784f325b6`，包含random与OHEM不同的新增自检。
- 训练plan：`6f86d929ad1275bbb120456dd23a25c7ea9571517f60b4c64a9d5ce6413c975c`；selection：`1f5e617764c0fe27a63440cc5b26d0e27a05db7dbc8c997b0bc5fd74a4b39c92`。

独立读取实际环境中Adam的函数签名，并结合已执行源码中明确传入的参数，确认共享betas=(.9,.999)、eps=1e-8、amsgrad=False、maximize=False、decoupled_weight_decay=False；foreach/fused为None。**最终文件未保存optimizer状态，因此这些超参的证据是执行源码哈希及实际库默认值，不是从保存的optimizer对象反序列化得到。**

未调用原计划生成器；从8段原始annotations逐步重建strict目标、core包含关系、extra_specs及slots，全部512步一致。所有原图和标注哈希与plan指纹匹配；plan sidecar及selection哈希正确，随机配对均使用2N项各一次且不重配原同对象帧两路。

- 1024对、2048窗，每窗40帧；每组共81920次帧裁剪曝光。两对来自不同序列，全部是真实40帧窗，crop和anchor合法。
- 内部标签512对均满足两个slot在8..31。边界标签512对中444对至少一端在外，68对双端仍在内部。
- 目标附近crop768对、随机crop256对。内部delta4/8/12分别205/166/141；边界delta12/24/36分别234/167/111。
- 额外对象帧总数1786。N<2共113步，其中N=0为19步、N=1为94步；四组RUN累计计数均与独立重算一致。

输入曝光序列 `(sequence,start,crop_y,crop_x)` 的规范JSON哈希为 `17149b1abc656b0c70bcbb9b5ba6c25303b6301147b057e3ed8492a33e527f99`。四组共享相同plan和数据，冻结build_batch不含随机增广；33个记录步上的extra对象数、底损失对象数及背景像素数逐项相同。未保存每步batch张量，因此输入相同由这些证据共同支持，未声称逐batch字节比较。

## 最终权重、BN及官方文件

在无GPU暴露、无网络、只读挂载的CPU容器里以 `torch.load(..., map_location="cpu", weights_only=True)` 读取初始和四份最终checkpoint；没有模型forward、optimizer更新或确认集推理。

模型共100个state张量，其中64个参数张量、70913个参数元素、36个BN缓冲。四组全部state张量有限，key、shape和dtype均与初始一致；每组64个参数张量都有变化，24个BN affine张量全部更新。四份最终权重两两比较也都存在真实参数差异。

36个BN running mean/var/counter缓冲在四组最终checkpoint中逐张量精确等于官方初始值；独立哈希 `51832be8d623f18e372232ddefdf4214ad67b3e45f795fbaa2ea9daa7fd0ecab` 同时匹配四组RUN的前后记录。已执行源码每步断言BN缓冲不变且容器正常退出，与最终直接比较一致。

官方仓库最终 `git status --porcelain` 为空，发布checkpoint SHA256仍为上述 `4fd4aa…abcc`。没有修改官方文件或正式训练产物。

## 审计边界

本审计通过训练完成性，不检查校准阈值、确认集匹配、误报约束或有利上下文保护。训练日志不是512条逐步记录；512次更新的证据来自冻结固定循环、末步日志、checkpoint内部step以及真实容器退出状态。没有从loss下降或权重变化推导检测改善。

审计自身首次尝试在完全只读容器里构造Adam以读取默认值，触发PyTorch临时目录需求而退出；未影响训练进程或文件。随后改为只读函数签名检查，完整CPU权重审计退出码0。正式四个训练容器均无此失败。
