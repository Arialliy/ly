# 原权重简单控制：CPU 缓存重组与评分

只处理 `DJI_0026_2`、`DJI_0067_2`；不训练、不重新前向、不触碰四段确认材料。规则和阈值选择归根目录 `PROTOCOL.md`；这里不挑阈值、不挑控制赢家。

`official5_probmean` 固定五窗起点 `[0,36,72,108,120]`，概率以起点升序 float64 累加、除以覆盖数，输出转 float32。`symmetric5_probmax` 固定起点 `[0,24,60,96,120]`，逐像素 float32 概率最大值。两个方案均 5×40 帧输入、完整覆盖同一 160 帧；缓存的 11 窗是研究诊断成本，不能称实际只运行了 5 窗分析。

`temporal_offsets=[0,39]` 记录的是当前帧在输入窗内的合法槽位；以当前帧为原点的参考帧相对时间可在 `[-39,39]` 内，随槽位改变。两控制与原系统共享完整 160 帧离线访问范围；不保证逐帧在线延迟相同。

每个控制/序列目录内：

- `fused.npz`：probabilities `[160,512,640]`、coverage `[160]`、starts `[5]`。
- `fused_matches.npz`：与原 `score_pilot.py` 同名字段；matched `[23,G]`、frame_counts `[23,160,6]`、box_max `[G]`、strict_small `[G]`、offsets `[161]`；其余定位字段同原定义。
- `fused_curves.csv`：每阈值的全目标/小目标匹配、组件、未匹配组件、正像素和框外正像素。
- `fused_COMPLETE.json`：阈值、逐对象 metadata、五窗预算、融合规则、源窗口/权重/源码/原图/评分脚本/输出 SHA256 等。

每段均先精确重构原官方五窗最大值，并以相同包装器在全部 160 帧重新评分 0.5：matched、center_error、peak_x、peak_y、component_area、frame_counts、box_max、strict_small、offsets 均须与旧官方产物完全相等（NaN 同位相等）。该结果在 `<sequence>_BASELINE_ADAPTER_CHECK.json`。两种新融合另逐帧用独立数组公式验证与重组输出完全相等，禁止伪造原 scorer 的官方融合元数据。

真实运行用既有 `deeppro-official-eval:20261009` 镜像，仅 CPU；每段一容器、2 CPU、8 GiB 内存限制，段内两控制顺序执行。`LAUNCHES.json` 保存真实容器 ID、镜像 ID、命令及代码哈希；`RUNTIME_STATUS.json` 保存最近实际状态（运行中 exit_code=0 不表示已完成），结束以 `Running=false`、`Status=exited`、`ExitCode=0` 为准。日志是对应 `<sequence>_container.log`。

评分复用原 `analysis/analyze.py::match_frame`，没有改变组件连通、峰值、框判定或匹配规则。所有结果仍是点匹配和未匹配组件诊断，不是论文 Pd/Fa。概率阈值网格全部扫描，不假定组件数关于阈值单调。
