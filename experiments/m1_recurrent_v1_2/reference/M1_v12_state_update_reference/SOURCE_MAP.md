# 源码依据、借鉴边界与未验证项

核对日期：2026-10-14。只核对公开论文及网页可读取源码；未取得本地作者仓库commit，不编造哈希。本包没有分发作者代码、权重或数据。

## 1. 直接母方法：RFR（作者仓库列为IEEE TGRS 2025）

- 仓库与出版说明：https://github.com/XinyiYing/RFR
- 代码：https://raw.githubusercontent.com/XinyiYing/RFR/main/codes/model/RFR_framework.py
- 论文：https://arxiv.org/html/2409.12448v3

明确的源事实：`forward_train`/`forward_test`对历史特征先执行`PDA`，再执行`TSFM(aligned,current)`；原TSFM包含相关性调制、当前/历史拼接、空间与频率处理及ReLU。原公开递归更新不是`aligned + TSFM(aligned,current)`。保留本地已验证的PDA、TSFM，不复制新的检测头。

## 2. 修改思路：MatAnyone（CVPR 2025）

- 正式论文页面：https://openaccess.thecvf.com/content/CVPR2025/html/Yang_MatAnyone_Stable_Video_Matting_with_Consistent_Memory_Propagation_CVPR_2025_paper.html
- 作者稿：https://arxiv.org/html/2501.14677v2
- 实际融合：https://raw.githubusercontent.com/pq-yang/MatAnyone/main/matanyone/model/matanyone.py
- 预测器：https://raw.githubusercontent.com/pq-yang/MatAnyone/main/matanyone/model/big_modules.py
- 训练目标：https://raw.githubusercontent.com/pq-yang/MatAnyone/main/matanyone/model/losses.py

`read_memory`使用预测概率在新读出与上一帧值之间插值；`UncertPred`使用前后特征、上一帧mask、memory差异，经1x1和3x3卷积输出logit。训练代码通过真实相邻mask/alpha的变化构造对应监督。

关键限制：这是指定目标的视频抠图，完整推理要求首帧目标mask。其memory已包含对象信息；输入条件、任务和监督都不等于未知红外弱目标发现。不能把MatAnyone的预训练权重或“变化/不确定性”含义直接移到本项目。

## 3. 我们独立实现了什么

- 从M1已计算的对齐历史A、TSFM候选R、当前特征X，构造`[A,X,R-A]`。
- 紧凑卷积预测保留系数g，不接收任何标签、轨迹、首帧mask或测试检测阈值。
- 更新为`(1-g)*R + g*A`；保留与更新竞争，不再无条件相加。
- frame/pixel两模式使用同一个预测器及相同参数；frame只把同一概率图空间平均后广播。
- 不添加原MatAnyone的变化监督，不声称训练了其UncertPred，不声称g是校准的不确定性或目标概率。
- 没有新位置标签或额外辅助损失；本地原M1的检测目标保持。

因此，本包是“借鉴有界插值形式的任务适配原型”，不是MatAnyone或RFR完整复现，不声称门控或插值算子新颖。

## 4. DeepPro-Plus接口依据

https://raw.githubusercontent.com/TinaLRJ/DeepPro/main/networks/models/DeepPro-Plus.py

唯一TPro及原预测头保持。公开接口只是参照，实际本地M1读取及回注位置不改。

## 5. 验证范围

本包只测试自写的状态更新核心，不运行作者PDA/DCN/TSFM、作者完整方法、用户M1或真实检测器。没有访问`/home/ly/DeepProPlus_TLFP_Code`。

如在本地复制作者代码，应遵守其许可证，并记录实际commit/SHA；不要通过下载浮动main覆盖已冻结实现。
