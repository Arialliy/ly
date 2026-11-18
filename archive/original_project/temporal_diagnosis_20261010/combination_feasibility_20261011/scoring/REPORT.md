# 五窗组合的实际融合评分

已对两段开发片段的10种合法整段五窗组合逐一重做逐像素max和原组件峰匹配；另评分全部11窗参照。只读取既有概率，没有训练或新的模型前向。

候选11窗中11选5共462个子集，恰好10个覆盖全部160帧。本表每行对两段使用相同的相对起点；小目标分母5200，全有效目标16297。阈值始终为0.5，U是对全部有效标注匹配后剩余的组件数。

| 组合 | 起点 | 小目标匹配 | 相对O新增 | 相对O丢失 | U | 框外前景像素 |
|---|---|---:|---:|---:|---:|---:|
| C00 | 0,12,48,84,120 | 690 | 62 | 72 | 4114 | 38505 |
| C01 | 0,24,48,84,120 | 705 | 68 | 63 | 4100 | 39026 |
| C02 | 0,24,60,84,120 | 699 | 68 | 69 | 3939 | 38521 |
| C03=S | 0,24,60,96,120 | 720 | 74 | 54 | 3967 | 38641 |
| C04 | 0,36,48,84,120 | 686 | 42 | 56 | 4102 | 39249 |
| C05 | 0,36,60,84,120 | 687 | 42 | 55 | 4003 | 39170 |
| C06=I2 | 0,36,60,96,120 | 708 | 48 | 40 | 4031 | 39290 |
| C07 | 0,36,72,84,120 | 685 | 26 | 41 | 3979 | 38170 |
| C08=I1 | 0,36,72,96,120 | 713 | 32 | 19 | 4026 | 38548 |
| C09=O | 0,36,72,108,120 | 700 | 0 | 0 | 4015 | 38368 |
| ALL11 | 0,12,...,120 | 792 | 101 | 9 | 5139 | 48589 |

共用安排的事后最大值为S的720，比O净增20，但仍丢失54个原成功。它满足冻结的U<=4015，而框外前景比O增加273；后者仅为诊断，不改作事后约束。

按片段独立事后选择，0026的最大值为C08的301，0067的最大值为C03的420，合计721、+73/-52、U3935。0026选C08、0067保持O则为707、+7/-0、U3949。这些选择用了标注进行事后评价，只衡量候选池的可行空间，不能作为可部署选择器或独立测试成绩。全部100对组合及冻结丢失容限的选择见上级汇总。

ALL11得到792，但每片输入窗口帧数由200增至440，U增至5139，比冻结预算高1124；框外前景增加10221。增加窗口没有保证峰匹配单调，仍丢失9个原成功。ALL11不属于五窗方案，也没有枚举所有更大预算的实际融合，故不把它称作更大预算下最优解。

## 复核

- O、S、I1、I2在两片共8组的matched、box_max、peak_x/y、component_area、frame_counts、coverage与旧结果逐元素精确相同；O/S还独立与原评价NPZ对照。
- 全22份缓存概率均通过float32、40×512×640和全部有限值检查。
- 每种组合均完整覆盖160帧；五窗覆盖次数和200、全部11窗和440。
- [verify_scores.py](verify_scores.py)独立重枚举几何组合，重算CSV/NPZ/82417条组件记录的计数关系，验证全部5200小目标上的匹配位等价于eligible_components>0，验证源文件哈希运行后未变化，全部通过，见[VALIDATION.json](VALIDATION.json)。
- [audit_root_aggregate.py](audit_root_aggregate.py)独立复算上级10种共享、100种按片段选择、16种冻结取舍选择、Pareto集合及选定对象/轨迹表，全部通过，见[ROOT_AGGREGATE_AUDIT.json](ROOT_AGGREGATE_AUDIT.json)。
- 三个CPU容器均退出0，无OOM；实际评分约21.1秒，见[RUNTIME.json](RUNTIME.json)。

## 文件与数组

[METRICS.csv](METRICS.csv)保存逐片及pooled的33行指标、组件面积均值/分位数；[COMPONENT_AREA_DISTRIBUTIONS.csv](COMPONENT_AREA_DISTRIBUTIONS.csv)保留all/assigned/unmatched面积的完整离散直方图。[COMPONENTS.csv](COMPONENTS.csv)保存组件逐行记录；[FRAME_COUNTS.csv](FRAME_COUNTS.csv)保存逐帧计数；[TRACK_METRICS.csv](TRACK_METRICS.csv)保存逐轨迹增损。[OBJECTS.csv](OBJECTS.csv)定义NPZ对象索引。

两个`DJI_*_scores.npz`的plans为C00...C09,ALL11。matched/box_max/peak_x/peak_y/component_area/eligible_components形状为(11,G)；frame_counts为(11,160,6)，字段顺序gt_count,matched_count,component_count,unmatched_components,positive_pixels,positive_pixels_outside_boxes；small为(G,)，offsets为(161,)，另存frame/source_frame/annotation_id/track_id。所有结果都是诊断峰入框匹配，不是论文Pd/Fa；本轮没有检验内容与位置的因果机制。
