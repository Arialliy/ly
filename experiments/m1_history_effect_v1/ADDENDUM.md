作用链统计的样本资格补记（2026-10-14）。本补记不修改已经冻结引用的 `RESULTS.json`，其 SHA256 仍为 `11f87da4beb57cee26dcc9d77a89a9034a03e7a873f9f500001747d7ea63ca0e`。

`t1_39 / pooled_unique / target` 统计只纳入至少有一个局部时间 t=1…39 观测的目标帧。仅出现在首帧 t=0 的观测独立保留在 t0 层，不补零，也不把不同时间位置的目标构成变化解释成受控记忆衰减。

| 冻结集合 | 全集合目标帧 | t1–39有资格目标帧 | 仅无t1–39资格的目标帧 |
| --- | ---: | ---: | ---: |
| strict_all | 5200 | 5162 | 38 |
| low_all_05 | 4342 | 4307 | 35 |
| extreme_all_001 | 4170 | 4135 | 35 |
| both_fixed_success | 646 | 643 | 3 |
| m1_low_common_recovered | 9 | 9 | 0 |

因此，低响应目标的 PDA 后相对差 `0.4317908874933576` 与 TSFM 后相对差 `0.014014997714347326`，描述的是 **4307/4342 个具有 t1–39 资格的目标帧**；它们不是整个4342集合的无条件均值。表内 TSFM 相对差指标在各自有资格目标帧中均无缺失，但其他指标须继续按各行的 `values_present / values_missing` 判断。

最终检测评分未使用该资格子集过滤：strict5200、low4342、extreme4170、protect646及全部16297有效GT仍完整进入原O/S匹配与连续性诊断。资格限制仅适用于作用链相应时间层的统计。完整定义和逐指标计数见 [TRACE_SUMMARY.json](analysis/trace_summary/TRACE_SUMMARY.json) 与 [STAGE_SUMMARY.csv](analysis/trace_summary/STAGE_SUMMARY.csv)。
