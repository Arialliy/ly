# 从近期论文思路和官方代码建立可核对的参考实现

日期：2026-10-12。用途：源码学习、接口验证及候选设计；不是已验证的检测方法。

## 本轮边界

用户报告的R0/R1/R2均未通过；R2 O/S匹配824/786，未匹配区域284982/285156，646保护集合仅87个仍双侧成功。该结果保持归档，本包不修复或重启R2，不改其阈值或输出后处理。

本轮已补齐参考代码和验证。`upstream/`保存固定commit的少量作者源码，供本地只读核查；本包没有作者预训练权重、训练数据或正式检测器训练脚本。自写参考不能叫作者原方法复现，不能加载官方NS-FPN权重后假定等价。官方整模型在用户数据上的成功不是借鉴思想的先决条件。

## 已核对的来源与接口

|来源|文件/接口|可借鉴的内容|限制|
|---|---|---|---|
|NS-FPN，CVPR 2026|`model/NS_FPN.py`: `ConvDWT`, `ConvIDWT`, `wav_Enhance`; `model/diff_cross_attns.py`|低频引导高频调制、特征净化先于融合|原模型是MSHNet+多尺度FPN；DeepPro-Plus没有相同FPN接口。SFS依赖自定义扩展，不在本包引入。|
|TDCNet，AAAI 2026|`model/TDCNet/TDCR.py`: `TDC`, `RepConv3D`; `background_alignment.py`; `model/gim/matcher.py`|不同时间差分核与可重参数化时空卷积；先处理背景坐标一致性|官方流程有背景配准，主模型为框检测，不直接替换当前密集预测头；训练/部署核需等价测试。|
|SPIRE，作者仓库标注ECCV 2026|`model/spire_net.py`, `utils/loss.py`, `train.py`|监督表示、原始响应回归与定位输出的配套设计|其网络返回直接热图，损失文件包含MSE类；不是本项目R2的query存在概率乘空间响应再取max；当前核查未完整取得训练工具函数，因此不推断哪一个loss类是运行默认值。|
|DeepPro-Plus，TPAMI 2026|`networks/models/DeepPro-Plus.py`|唯一全分辨率TPro及原预测链|公开代码不替代用户已冻结本地快照。|

### 主要来源

- NS-FPN论文：https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html
- NS-FPN仓库：https://github.com/mengduann/NS-FPN
- NS-FPN算子：https://raw.githubusercontent.com/mengduann/NS-FPN/main/model/NS_FPN.py
- NS-FPN主网络：https://raw.githubusercontent.com/mengduann/NS-FPN/main/model/MSHNet_NSFPN.py
- TDCNet论文：https://ojs.aaai.org/index.php/AAAI/article/view/37385
- TDCNet仓库：https://github.com/IVPLabs/TDCNet
- TDC算子：https://raw.githubusercontent.com/IVPLabs/TDCNet/main/model/TDCNet/TDCR.py
- TDCNet配准：https://raw.githubusercontent.com/IVPLabs/TDCNet/main/background_alignment.py
- SPIRE仓库：https://github.com/NIRIXIANG/SPIRE-IRSTD
- SPIRE网络：https://raw.githubusercontent.com/NIRIXIANG/SPIRE-IRSTD/main/model/spire_net.py
- SPIRE损失：https://raw.githubusercontent.com/NIRIXIANG/SPIRE-IRSTD/main/utils/loss.py
- DeepPro-Plus：https://raw.githubusercontent.com/TinaLRJ/DeepPro/main/networks/models/DeepPro-Plus.py

NS-FPN本轮已下载并固定commit `b857bef068ba48f1258b62de6bf082f73dbafde4`，算子文件SHA256为`4c293334e05c98256d91b535242316ac4e3c725e02cd4ba7ba81aa86f310409f`，见`upstream/PROVENANCE.json`。仓库根目录未见LICENSE/COPYING文件，未据此推定再分发许可。其余上述链接仍是原检索记录，本轮没有补跑这些作者整模型或checkpoint。

## 文件

- `haar_reference.py`：自写Haar分解/重建和LFP思路参考；明确保留`[N,C,3,h,w]`子带轴。
- `framewise_adapter.py`：自写`[B,C,T,H,W]`逐帧适配器，原特征加零初始化的修正路径；不是官方公式。
- `load_official_lfp.py`：在用户环境中读取可信的本地官方`NS_FPN.py`，只装载LFP相关类，不引入SFS CUDA依赖。仅抽取定义，不改写作者类；AST执行不是安全沙箱。
- `check_upstream_lfp.py`：对本地作者实现检查变换往返与形状；若不通过停止，不自动打补丁。
- `test_adapter.py`, `test_haar_reference.py`：本包的合成张量测试。
- `check_deeppro_integration.py`：冻结本地Plus的TRAIN裁块初始化、梯度及分块检查。
- `test_upstream_loader.py`：作者类AST不变及导入隔离检查。
- `run_reference_checks.py`、`runtime/ENVIRONMENT.md`：独立依赖层和固定Docker启动方式。
- `VALIDATION.json`：已做与未做的验证范围。
- **`REFERENCE_REPORT.md`：本轮实际结果、失败记录及解释边界。**

## 安装与测试

本包自写实现只需PyTorch。测试还需pytest；本地检查作者LFP还需pytorch_wavelets。依赖版本需按用户工程锁定，不建议覆盖已有环境。

```bash
python -m pytest -q test_haar_reference.py test_adapter.py test_upstream_loader.py
# 可选：在安装pytorch_wavelets且已有可信官方仓库后运行
python check_upstream_lfp.py --source /path/NS-FPN/model/NS_FPN.py
```

## 代码层面的重要发现

当前可见的官方`ConvDWT`将高频子带的C与3两个轴转置后展开；`ConvIDWT`随后直接reshape为`[B,C,3,h,w]`。对C>1，这两种展开顺序通常不互逆。本轮已运行未修改作者类的CPU检查：C=1偶数尺寸通过；C=3/32偶数尺寸分别出现约3.453/4.789的往返最大误差；17×19输入返回16×18。14项中4通过、10失败，执行错误为0；详见`reference_checks/UPSTREAM_CPU.launch.result.json`。这是裸变换与形状的检查，不是作者整模型检测结果；不据此否定论文成绩，也未修改作者类。

自写`haar_reference.py`采用显式子带轴，奇数尺寸用右/下复制填充并在重建后裁回，另对Gaussian sigma平方加数值下限。这些均需作为相对原实现的差异登记；本包不是checkpoint兼容替代品。高频并不等于噪声，真实小弱目标也可能在高频中。

## 已验证的接入点（仅有限实现检查）

只选`layer1`之后、原维度置换与唯一TPro之前。保持原TPro、输出头和返回契约；本轮未运行新的sigmoid/五窗融合/组件匹配流程。共增加100个参数。56项CPU测试通过；固定TRAIN的40×97×101裁块通过GPU整网检查。在相同冻结状态下，两路原输出初始化误差为0；非零门控与非零注意力权重下，halo20、绝对偶数起点的分块logit最大误差为9.54e-6。原70913参数及全部BN状态不变。

```python
from framewise_adapter import FramewiseFeatureAdapter
from haar_reference import SpatialLFPReference

# 在加载/核对原发布模型后增加；不要用strict=False掩盖原权重缺失。
self.feature_refiner = FramewiseFeatureAdapter(
    SpatialLFPReference(channels=32), channels=32, frame_chunk=8
)

# 在原forward中仅增加这一行；先检查本地真实张量。
seq_feats = self.layer1(self.conv_in(seq_imgs))
seq_feats = self.feature_refiner(seq_feats)
# 之后保留本地原维度变换、TPro、conv_out1、conv_out2与返回契约。
```

本适配器是`F + tanh(scale)*(P(F)-F)`，只保证scale为0且算子输出有限时的初始等价；不保证训练后检出保持，也不保证抑噪一定降低组件数量。不要在训练中用`torch.no_grad()`包住新增模块之后的冻结网络；参数冻结与切断梯度不是一回事。逐帧处理后恢复原连续BCTHW排列；帧块1/8/17与40的对照已通过。空间分块需要绝对偶数相位及halo20；当前仅验证指定TRAIN裁块及合成输入，不外推为任意尺寸、整数据集的等价保证。

## 研究解释与下一步

此空间LFP只能作为一个代码可核查的借鉴控制，不能被称作新的时序贡献。原研究对象仍是背景约束下的持续漏检恢复，不是视觉上平滑背景。

后续自己的候选可以检验：与逐帧低频条件化相比，利用合法窗口内低频的时间结构，是否更能选择性保留目标相关高频。这里只给出候选问题，不在本包实现新时间门控，也不预设高频稳定性等于目标。

在正式训练前固定：原结构同监督控制、空间LFP控制、自己的时序扩展之间的唯一差异；监督、可训练范围、数据曝光、检查点与推理预算按同一协议，不更换输出语义，不自动重用R2终点。不是要求借鉴控制必须先优于发布权重，而是要求对照可解释。

只有实际恢复、最长漏检、原成功保持与背景成本形成有意义的改善，并优于空间控制，才能主张时序扩展有价值。仅U下降或仅匹配增加都不够。沿用既有0.5和U≤4015；不把其他方法的原生Fa当成当前U，不暗加NMS或目标数量上限。

不得把这份代码包本身当成第一个正向检测结果。也不据代码完整性推断候选一定有效。

## 当前状态

参考实现已交付，有限接口检查通过；官方裸变换往返异常单独保留为失败。尚未完成正式检测训练、全画面恢复评价、跨来源验证或时序扩展。旧`DECISION.json`、`CLOSEOUT.md`及`ARTIFACT_HASHES.json`仍属于历史E0/E1/E2结题，未改写；本轮状态只看`VALIDATION.json`和`REFERENCE_REPORT.md`。
