"""Build an offline Chinese index for the recorded, data-derived figures."""
import html
import json
from pathlib import Path

V = Path(__file__).resolve().parents[1]
FIGURES = [
    ('main_results', '检出与背景组件总数', '五种配置在两套固定安排下的实际结果。坐标从零起，虚线为 U=4,015 的冻结诊断预算。M 是严格小目标帧匹配数，U 是全部有效标注匹配后剩余的组件数，均不是官方 Pd/Fa。T-current 与 T-bypass 是同一 T 权重的推理控制。'),
    ('background_deltas', '背景成本变化', '相对同安排发布权重的差值；负值表示该项成本下降。三个面板单位和范围分别标明。O/S 的原基数：U 4,015/3,967，框外像素 38,368/38,641，未匹配面积 45,232/45,136。差值较小，不代表持续漏检已恢复，也不代表全部尺度检出均被保持。'),
    ('training_diagnostics', '训练过程中的分支变化', '34个已保存日志样点，横坐标是已完成更新数。日志取自该步更新前，最后一点对应511次更新后的状态；不等同于512步终点权重。曲线仅连接真实样点、没有平滑。这是参数与分支变化的遥测，不是检出恢复曲线。'),
]

def card(name, title, caption):
    return f'<section class="card"><h2>{html.escape(title)}</h2><a href="{name}.png"><img loading="lazy" src="{name}.png" alt="{html.escape(title)}"></a><p>{html.escape(caption)}</p><p class="links"><a href="{name}.png">原尺寸 PNG</a> · <a href="{name}.pdf">PDF</a></p></section>'

def main():
    records=json.loads((V/'QUALITATIVE_RENDER.json').read_text())['figures']
    intro=('前一轮S/T-LFP结果的可视化：2个开发片段、5,200个严格小目标帧、130条轨迹；阈值固定为 ≥0.5。'
           'S-LFP／T-LFP均为 seed 46、512步终点，只有一个seed，没有误差条或显著性主张。'
           'O是原五窗安排，S是对称五窗安排；S-LFP是空间模型名称。')
    conclusion=('严格小目标匹配集合保持不变：O为700、S为720；冻结的4,342个全窗低响应目标帧恢复为0。'
                '这些图保留S/T-LFP未通过的结果。非严格小目标仍有检出丢失，见完整报告。')
    qualitative=('以下整帧只展示O安排。预先固定两个片段的第0、80、159帧，没有按T-LFP输出挑选案例。'
                 '每组从左到右为原红外图、发布权重、T-LFP；绿色标注框表示实际匹配，橙色标注框表示漏检。'
                 '彩色框是严格小目标的真实标注框，不是模型预测框；绿色加号为匹配峰。浅蓝轮廓是所有预测前景的0.5等值线，可能对应其他尺度目标，不能直接称为误检。')
    zoom_intro=('两组局部图按各片段第80帧中“冻结低响应集合的最小轨迹ID”选择。'
                '均使用96×96原图裁剪，原图固定0–255显示，概率图共享0–1色标，没有逐图增强。'
                '极低响应在此色标下接近黑色；这不证明原输入不存在信息。粗框不代表精确目标形状或中心。'
                '0026的目标靠近原画面右边界，保留原位置；图周白边只是绘图区留白，不是补造的图像像素。')
    body=[f'<header><div class="eyebrow">DEEPRO-PLUS · S-LFP · T-LFP</div><h1>实验结果与真实画面对照</h1><p>{intro}</p><p class="conclusion">{conclusion}</p><nav><a href="../results/tlfp_pilot_20261012/REPORT.md">完整报告</a><a href="../results/tlfp_pilot_20261012/evaluation/SUMMARY.csv">汇总 CSV</a><a href="captions.md">详细图注</a><a href="../README.md">目录说明</a></nav></header>']
    md=['# 实验可视化\n', '[浏览器查看全部图片](index.html) · [完整实验报告](../results/tlfp_pilot_20261012/REPORT.md)\n',intro+'\n',conclusion+'\n','## 定量图\n']
    latest='../experiments/head_lfp_joint_20261012'
    if (V/latest/'figures/final_results.png').exists():
        latest_caption='H/HL均为seed46、2048步固定终点。S安排两组M均为745，U分别4400/4292，仍超过4015；LFP相对H未增加匹配总数。下方S/T-LFP历史图使用512步，二者不可混为同一实验。'
        body.insert(0, '<header><h1>最新：H与H＋LFP协同适配</h1><p>'+latest_caption+'</p><nav><a href="'+latest+'/RESULTS.md">主结果</a><a href="'+latest+'/DECISION.md">结论与边界</a><a href="'+latest+'/figures/CAPTION.md">图注</a></nav></header>'+card(latest+'/figures/final_results','2048步：匹配、背景、恢复与保持',latest_caption))
        md[0:0]=['# 最新：H与H＋LFP\n',latest_caption+'\n','[结果]('+latest+'/RESULTS.md) · [判定]('+latest+'/DECISION.md) · [PNG]('+latest+'/figures/final_results.png) · [PDF]('+latest+'/figures/final_results.pdf)\n']
    for name,title,caption in FIGURES:
        body.append(card(name,title,caption));md.append(f'- [{title} PNG]({name}.png) · [PDF]({name}.pdf)\n')
    md.append('\n详细定义、数据来源与训练日志口径见[captions.md](captions.md)。\n\n## 低响应目标局部图\n\n'+zoom_intro+'\n')
    body.append('<div class="explanation"><h2>低响应目标局部图</h2><p>'+zoom_intro+'</p></div>')
    for r in records:
        if not r['name'].startswith('zoom_'):continue
        title=f"{r['sequence']} · 第 {r['frame']} 帧 · 轨迹 {r['track_id']}"
        caption=f"框内最大概率：发布权重 {r['release_box_max']:.3e}，T-LFP {r['T_box_max']:.3e}。两个预测均未恢复该目标；此例不能代表全体目标的物理可见性。"
        body.append(card(r['name'],title,caption));md.append(f"\n![{title}]({r['name']}.png)\n\n{caption} [PDF]({r['name']}.pdf)\n")
    body.append('<div class="explanation"><h2>固定帧的完整画面对照</h2><p>'+qualitative+'</p></div>')
    md.append('\n## 完整画面对照\n\n'+qualitative+'\n')
    for r in records:
        if not r['name'].startswith('frame_'):continue
        title=f"{r['sequence']} · 第 {r['frame']} 帧"
        caption=f"本帧严格小目标 {r['strict_small_gt']} 个；发布权重匹配 {r['release_matched']} 个，T-LFP 匹配 {r['T_matched']} 个。实际匹配直接读取冻结明细表，没有按图上轮廓重新判定。"
        body.append(card(r['name'],title,caption));md.append(f"- [{title} PNG]({r['name']}.png) · [PDF]({r['name']}.pdf)\n")
    body.append('<footer><p>全部图片来自已保存的真实数据与预测，未执行新训练或模型推理。绘图脚本、选样规则和SHA来源记录与图片一同保存。PNG为300dpi；统计图PDF保留矢量图元，真实画面PDF内图像仍为栅格、文字与框为矢量。</p><a href="README.md">重绘方法与来源</a></footer>')
    style='''body{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#142534;background:#f3f5f7;margin:0;line-height:1.75}main{max-width:1280px;margin:auto;padding:32px 24px}h1{font-size:32px;line-height:1.3}h2{font-size:21px}header,.explanation,footer{padding:12px 8px 24px}.eyebrow{color:#426579;font-size:13px;letter-spacing:.12em}p{max-width:1150px}.conclusion{border-left:4px solid #a55719;padding:12px 18px;background:#fff4e9}nav{display:flex;gap:20px;flex-wrap:wrap}a{color:#006696;text-decoration:underline;text-underline-offset:3px}.card{padding:20px 24px;margin:18px 0 30px;background:#fff;border:1px solid #dbe3e8;border-radius:10px}.card img{display:block;width:100%;height:auto}.card h2{margin-top:0}.links{font-size:14px}footer{font-size:14px;color:#526474}@media(max-width:640px){main{padding:16px 10px}.card{padding:12px}h1{font-size:26px}}'''
    (V/'index.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeepPro-Plus 本轮结果可视化</title><style>'+style+'</style></head><body><main>'+''.join(body)+'</main></body></html>')
    md.append('''
## 来源与重绘

- [固定选样规则](QUALITATIVE_SELECTION.json)、[原始输入SHA](QUALITATIVE_EXTRACTION.json)、[逐图元数据](QUALITATIVE_RENDER.json)。
- [定量数据来源](QUANTITATIVE_SOURCES.json)、[渲染自查](QUANTITATIVE_RENDER_CHECK.json)、[独立定量图审核](QUANTITATIVE_REVIEW.json)。
- `data/` 保留6帧选定原图、发布／T概率和对应标注明细。仅这些样本被收录，不等于全量训练数据。
- PNG为300dpi；统计图PDF为矢量图元，真实画面PDF的图像为栅格，文字和标注框为矢量。

在具备NumPy、Matplotlib的环境中运行下列命令即可用包内数据重绘，不使用GPU，也不需要原实验数据目录：

```bash
python3 visualizations/scripts/build_quantitative.py
python3 visualizations/scripts/plot_qualitative.py
python3 visualizations/scripts/build_gallery.py
```

`scripts/extract_qualitative.py`是首次提取样本的来源记录；重绘无需再次运行它。当前绘图复用已验证Docker环境，执行命令见[QUALITATIVE_EXECUTION.json](QUALITATIVE_EXECUTION.json)。图片仅辅助查看已完成实验，评价结论仍以全量结果表为准。
''')
    (V/'README.md').write_text('\n'.join(md))
    print('Wrote offline index.html and README.md')

if __name__=='__main__':main()
