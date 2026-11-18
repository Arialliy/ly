ConFIG 来源和本地封装核验通过，范围限于来源、接口与新增 CPU 合成检查。

[ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/94e85561a342de88b559b72c9b29f638-Abstract-Conference.html) 指向[作者项目页](https://tum-pbs.github.io/ConFIG/)，项目页指向 [tum-pbs/ConFIG](https://github.com/tum-pbs/ConFIG)。固定 commit `94862437f451f175673bce9c85f3e14bd9182c21`，包版本 0.1.8，MIT 许可。15 个原始文件已按该 commit 的 Git blob SHA1 验证，并单独登记 SHA256；源码及许可保留原样。原 README 的 BibTeX 写 2024，发表年份以 ICLR 2025 proceedings 为准。

封装直接调用作者 `ConFIG_update`，保持 FP32、`EqualWeight`、`ProjectionLength`、`use_least_square=True`、`losses=None`。作者函数先归一化各行梯度，用最小二乘求方向，再以各梯度在此方向的投影和定长度。`EqualWeight` 返回长度为损失项数的全 1 向量，使用默认浮点 dtype；本地不把官方求解改为 double。double 只用于可靠性判断与范数匹配标量运算。依赖声明为 torch>=1.9.0，工具文件另导入 numpy；使用既有不可变 Docker 环境，未安装或升级包。

坐标由一份固定可训练参数清单确定。每项未使用、其他项使用的参数补零；三项均未使用的参数保持 `None`。官方工具默认 `skip`，不能直接用于各项使用范围不同的情形。本地三项按 fit/keep/track 排列。正常候选额外匹配普通分组和的 L2 范数，这是本地控制条件，不是作者默认。原标量 full_F 的 autograd 梯度由调用方独立保留并核对舍入差。

有限性输入错误直接中止；全未使用、零和、某项全零、归一化矩阵奇异值比≤1e-6、候选投影余弦≤1e-7、等投影相对偏差>1e-4、范数匹配相对误差>1e-6，以及求解异常、零或非有限候选，均有明确处理并写入诊断。允许回退的情况返回原分组和；零和仍进入 Adam，不跳步；全未使用保留 `None`。

18 项 CPU 合成检查全部通过：正常方向与未修改作者函数一致、固定坐标对齐、已有 Adam 动量和权重衰减下零梯度与 `None` 的差异、退化回退、非法输入。记录见 `CONFIG_OPERATOR_CHECK.json` 与 `CONFIG_OPERATOR_TESTS.xml`。仅有作者原 docstring 的 8 条无效转义弃用警告。

论文主要验证 PINN，也包含 CelebA 多任务实验；没有提供本任务红外检出的验证。[作者 troubleshooting](https://tum-pbs.github.io/ConFIG/start/troubleshooting/) 及论文附录说明梯度长度和 Adam 动量会影响实际效果。因此方向相容或输入梯度同范数，不保证 Adam 实际更新同范数、每项下降或漏检恢复。尚未执行真实检查点 ConFIG 单步、GPU 算子检查或训练；是否启用仍由固定 16 对批次的实际更新证据决定。
