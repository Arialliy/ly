# 从近期论文思路和官方代码建立可核对的参考实现

日期：2026-10-12。用途：源码学习、接口验证及候选设计；不是已验证的检测方法。

## 本轮边界

用户报告的R0/R1/R2均未通过；R2 O/S匹配824/786，未匹配区域284982/285156，646保护集合仅87个仍双侧成功。该结果保持归档，本包不修复或重启R2，不改其阈值或输出后处理。

本包不包含作者代码、预训练权重、训练数据或检测器训练脚本。自写参考不能叫作者原方法复现，不能加载官方NS-FPN权重后假定等价。官方整模型在用户数据上的成功不是借鉴思想的先决条件。

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

以上是检索时访问的main页面，不是冻结commit。本环境网络不能直接下载仓库；作者整模型和checkpoint没有在这里运行。实际复用前应在本地记录git commit、源文件SHA256、依赖版本和许可证。不要因为可以访问源代码就推定允许任意再分发。

## 文件

- `haar_reference.py`：自写Haar分解/重建和LFP思路参考；明确保留`[N,C,3,h,w]`子带轴。
- `framewise_adapter.py`：自写`[B,C,T,H,W]`逐帧适配器，原特征加零初始化的修正路径；不是官方公式。
- `load_official_lfp.py`：在用户环境中读取可信的本地官方`NS_FPN.py`，只装载LFP相关类，不引入SFS CUDA依赖。仅抽取定义，不改写作者类；AST执行不是安全沙箱。
- `check_upstream_lfp.py`：对本地作者实现检查变换往返与形状；若不通过停止，不自动打补丁。
- `test_adapter.py`, `test_haar_reference.py`：本包的合成张量测试。
- `VALIDATION.json`：已做与未做的验证范围。

## 安装与测试

本包自写实现只需PyTorch。测试还需pytest；本地检查作者LFP还需pytorch_wavelets。依赖版本需按用户工程锁定，不建议覆盖已有环境。

```bash
python -m pytest -q
# 可选：在安装pytorch_wavelets且已有可信官方仓库后运行
python check_upstream_lfp.py --source /path/NS-FPN/model/NS_FPN.py
```

## 代码层面的重要发现

当前可见的官方`ConvDWT`将高频子带的C与3两个轴转置后展开；`ConvIDWT`随后直接reshape为`[B,C,3,h,w]`。对C>1，这两种展开顺序通常不互逆。这是静态张量次序分析，不是已运行作者整模型后的结论；也不能由此否定作者训练权重的性能。本包提供独立的往返检查，保留结果而不悄悄修复原代码。

自写`haar_reference.py`采用显式子带轴，奇数尺寸用右/下复制填充并在重建后裁回，另对Gaussian sigma平方加数值下限。这些均需作为相对原实现的差异登记；本包不是checkpoint兼容替代品。高频并不等于噪声，真实小弱目标也可能在高频中。

## 候选接入点（尚未在本地DeepPro工程运行）

只选`layer1`之后、原维度置换与唯一TPro之前。保持原TPro、输出头、sigmoid、五窗融合、阈值和组件匹配。

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

本适配器是`F + tanh(scale)*(P(F)-F)`，只保证scale为0且算子输出有限时的初始等价；不保证训练后检出保持，也不保证抑噪一定降低组件数量。不要在训练中用`torch.no_grad()`包住新增模块之后的冻结网络；参数冻结与切断梯度不是一回事。帧分块不改变时间长度，但空间分块仍需独立考虑小波相位、对齐及足够halo。

## 研究解释与下一步

此空间LFP只能作为一个代码可核查的借鉴控制，不能被称作新的时序贡献。原研究对象仍是背景约束下的持续漏检恢复，不是视觉上平滑背景。

后续自己的候选可以检验：与逐帧低频条件化相比，利用合法窗口内低频的时间结构，是否更能选择性保留目标相关高频。这里只给出候选问题，不在本包实现新时间门控，也不预设高频稳定性等于目标。

在正式训练前固定：原结构同监督控制、空间LFP控制、自己的时序扩展之间的唯一差异；监督、可训练范围、数据曝光、检查点与推理预算按同一协议，不更换输出语义，不自动重用R2终点。不是要求借鉴控制必须先优于发布权重，而是要求对照可解释。

只有实际恢复、最长漏检、原成功保持与背景成本形成有意义的改善，并优于空间控制，才能主张时序扩展有价值。仅U下降或仅匹配增加都不够。沿用既有0.5和U≤4015；不把其他方法的原生Fa当成当前U，不暗加NMS或目标数量上限。

不得把这份代码包本身当成第一个正向检测结果。也不据代码完整性推断候选一定有效。
