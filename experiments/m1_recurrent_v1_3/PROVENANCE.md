# M1-v1.3 公开来源与本地复用记录

核验日期：2026-10-14。本文只登记来源、代码语义和适配边界；本次来源核验没有运行 StreamVGGT、安装其依赖、下载权重或启动训练。用户原包与冻结实验代码均未修改。

## StreamVGGT：保存与读取分离的实现参照

固定作者仓库 [wzzheng/StreamVGGT](https://github.com/wzzheng/StreamVGGT) 的提交 [05682bcb05683c4c71522e9cb618fc8ca349063c](https://github.com/wzzheng/StreamVGGT/commit/05682bcb05683c4c71522e9cb618fc8ca349063c)。该提交日期为 2025-10-22，提交说明为 `Update results.png`，是本次访问时的 main；不把它称为 ICLR 投稿或最终发表代码快照。

作者仓库 About 标注 ICLR 2026；[ICLR 官方 2026 Downloads](https://iclr.cc/Downloads/2026) 收录 *Streaming Visual Geometry Transformer*，链接到其[会议条目](https://iclr.cc/virtual/2026/poster/10011495)。作者仓库沿用带“4D”的旧标题，所核验的 [arXiv v2](https://arxiv.org/html/2507.11539v2) 标题为 *Streaming Visual Geometry Transformer*，版本日期 2026-03-31。OpenReview 页面本次返回浏览器验证页，未将其内容用作接收决定依据。

论文 §3.2 的参照内容是因果注意力和历史 K/V 缓存；§3.3 涉及 VGGT 教师蒸馏。其任务与证据属于三维几何、深度和相机姿态，不能据此认定本项目的红外漏检恢复已经有效。这里未移植作者的 Transformer、RoPE、编码器、几何头或蒸馏训练。

原始文件保存于 [upstream/streamvggt_reference](upstream/streamvggt_reference/)，完整 URL、字节数、SHA-256 和 Git blob SHA-1 见 [MANIFEST.json](upstream/streamvggt_reference/MANIFEST.json)。五个文件均与该提交的官方 Git tree blob 完全相符，无本地修改。`aggregator.py` 与 `block.py` 因 GitHub TLS 间歇失败经公开 jsDelivr 缓存获取，仍逐字节匹配官方 blob；取得方式分别记录在 manifest，未将 CDN 当作新的作者来源。

| 文件 | SHA-256 |
| --- | --- |
| `src/streamvggt/layers/attention.py` | `9e93c86005709195cc54da2cc417c1edcdcd7b32d3ad0b2427c27b8085df917b` |
| `src/streamvggt/layers/block.py` | `d10b39e357042d627133bfeb8cf12e6561e20c68e19c33c85a41eb5acfc22585` |
| `src/streamvggt/models/aggregator.py` | `3cf86125bdeb0d54a48c3c39d336bdfd37b0bc3c7388c18e20d390f700b0da20` |
| `README.md` | `176d4190b3e77d30ca25ecbf85d0a7522e3fed4b781a32cc26884b4c81f63144` |
| `LICENSE.txt` | `e0eb0aae5e2658b11ad65864bca446e7ed1d59a461ff02395b9e637bf41052e4` |

作者根目录 [LICENSE.txt](upstream/streamvggt_reference/LICENSE.txt) 为 CC BY-NC-SA 4.0，原文及源码内已有声明完整保留。本地这组文件仅供阅读，不被实验 runtime 导入。

## RFR：复用已核验的真实 PDA/TSFM 与 native DCN

继续使用作者 [XinyiYing/RFR](https://github.com/XinyiYing/RFR/tree/fd71ccdf17f056f80e0a5ea8726b87c82c27fb69) 的固定提交 `fd71ccdf17f056f80e0a5ea8726b87c82c27fb69`，无需重新取得或复制整库。原始源、兼容副本、构建证据和许可边界已保存在 [v1 upstream/PROVENANCE.json](../m1_recurrent_v1/upstream/PROVENANCE.json) 与 [COMPATIBILITY.md](../m1_recurrent_v1/upstream/COMPATIBILITY.md)。本次重新计算下列文件哈希并与既有记录逐项一致。

| 本地既有文件（相对 v1/upstream） | SHA-256 |
| --- | --- |
| `RFR/codes/model/RFR_framework.py` | `a569e5cd0599a8cf25efb67314ff8249ba7456f0363b8755cdee729b40028bda` |
| `RFR/codes/model/utils.py` | `540a4b05409d8fad9221b581fcc5ec67fc7a4aa89baedb5df522e5f176a671a9` |
| `compat/RFR_framework.py` | `60de85347a49c80202f68e1f5eb8f48624feb5dd1f13fed79653bbc053a888a8` |
| `compat/utils.py` | `540a4b05409d8fad9221b581fcc5ec67fc7a4aa89baedb5df522e5f176a671a9` |
| `compat/dcn/DCN2.cpython-310-x86_64-linux-gnu.so` | `93aba2d218912d868c9e9073ec73fff1b4088a845cdd99109aead3f554ecc7ca` |

既有兼容变更限于隔离导入和 PyTorch C++/CUDA API 适配；PDA/TSFM 类 AST 与原作一致，使用真实 native DCN。构建环境仍为既有 PyTorch 2.2.0a0/CUDA 12.3；本次未重编译。上游 RFR checkout 没有根 LICENSE，不能补称 MIT 或其他全库许可；嵌入源码声明保留，沿用既有记录。

## 用户原包与项目适配

核心来自原样解包的 [reference/M1_v13_evidence_bank_reference/evidence_bank.py](reference/M1_v13_evidence_bank_reference/evidence_bank.py)，SHA-256 为 `035925964a651df6e02a9d48a3748f8e774aa9522a461f69762ca75340de443c`。原包 [SOURCE_MAP.md](reference/M1_v13_evidence_bank_reference/SOURCE_MAP.md) 保持不变，其 SHA-256 为 `7f192e9dc6bef20540e3d6aedf8632c723a5c5cc8bda2c8de8ead5f2f4242269`。本次更精确的逐项归属见实验根目录 [SOURCE_MAP.md](SOURCE_MAP.md)。

K=4 FIFO、保存投影后的源特征 X、读后写入、每个源帧分别执行 PDA/TSFM、等权均值，以及算子次数匹配的 current_only 对照，属于本项目登记的适配；不是 StreamVGGT 原模块移植，也不能仅因采用缓存或 FIFO 宣称方法贡献。当前实验仍需用实际匹配、连续漏检与背景代价验证这一适配。
