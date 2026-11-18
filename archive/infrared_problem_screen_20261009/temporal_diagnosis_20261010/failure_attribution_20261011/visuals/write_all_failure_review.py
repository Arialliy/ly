#!/usr/bin/env python3
"""Persist manual observations made after individually viewing 16 + 4 panels."""
import csv
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

D=Path(__file__).resolve().parent;C=D.parent/'conditions'
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# This mapping is the reviewer's direct image observation, not inferred from outcome counts.
notes={
805208:('可区分弱亮斑；类别细节不足','框内有小而模糊的浅亮团，位于图像右边界附近。','上方约 y=305..311 有横向较亮带；框触及原图右侧。','框附近原/新及示例窗图均近黑；与弱响应层一致。','轨迹中心仅在少数相邻 x 位置变化；不能称物理静止。','未见足以确认明显错框的证据；亮斑身份待确认'),
805083:('弱可判读','框内仅弱的弥散浅亮结构，单帧中难以辨认船体轮廓。','弱水平纹理穿过邻域，没有清楚孤立边界。','框附近输出近黑；响应阈值层来自全窗数值表。','标注中心横向跨越多位置；只证明图像坐标轨迹变化。','无法仅靠该单帧排除背景纹理或核实目标身份'),
805444:('弱可判读','框中部有模糊浅亮团，能看出局部变化但轮廓不清。','周围较平滑，伴随弱水平灰度带。','原/新与展示窗在框附近近黑。','该轨迹横向延伸；固定中心和标注中心灰度曲线不同，不能从中推断遮挡。','未见明确错框证据；目标边界待核实'),
805940:('可区分弱亮斑；类别细节不足','框内有小亮斑，6×7 像素不足以辨认细节。','上方横向亮带和原图右缘邻近；右上方另有高响应区域。','当前框内低响应，而上方邻近结构有明显响应；两者空间上分开。','轨迹图仅短时标注，灰度曲线晚段才有橙色采样；不能称真实目标刚出现。','未见明确错框证据；小斑身份与边界待核实'),
871417:('局部结构较清楚','框内暗顶亮底的紧凑车状结构可辨；右上方还有另一车状结构。','上下存在桥侧长边缘及邻近车辆。','高响应团位于框右缘并向右延伸，首失败窗也可见跨框响应；图中没有精确峰标记。','标注轨迹沿斜向道路方向延伸；固定像素强波动不等于目标自身强弱变化。','当前框大体包围车状结构，未见明显指向纯背景证据'),
867655:('局部结构较清楚','框内有暗顶亮底的车状轮廓。','邻近车辆、纵向细结构和上下桥侧边缘同时存在。','响应主要在框右下方并越出框边；原/新及唯一覆盖窗外观相同。','当前帧靠近片段开头；轨迹在图像坐标中移动，非物理运动判定。','当前框有车状结构，未见明显纯背景错框证据'),
875413:('局部结构可见但受画面边界限制','左侧画面边缘有深浅车状局部，当前框贴原图左边界，完整轮廓不在图中。','原图左边界及邻近道路/桥侧结构。','原融合在框右边外侧有小响应，展示失败窗与对称输出在当前框附近弱；高层来自其他可用窗数值。','标注轨迹图延伸到画面左缘；不能据此把离画视为模型失误原因。','需核对边界目标框与实际可见部分，未确认错框'),
869314:('难与邻近长边缘分离','狭窄框内有弱暗竖向结构，但与桥侧长暗带交叠，行人形态不能可靠辨认。','框横跨桥侧深色长边缘；右侧有细竖杆及邻近高响应。','当前框右边邻近有高响应团，框内大部分暗；不从彩色团直接声称精确峰位置。','轨迹中心沿长边缘缓慢改变；无法区分目标、相机或标注变化。','待确认：不能据细暗纹理认定有目标或纯背景'),
874661:('局部结构清楚；类别细节不足','框内孤立暗斑与较平滑的灰背景有明确对比，不能简单称当前帧不可判读。','近框没有显著强边缘穿过目标。','框内原/新及示例窗均低响应，和可区分的暗斑同时存在。','标注中心只在数像素范围变化；不等于真实船体静止。','框与暗斑对应，未见明显纯背景错框证据'),
874662:('局部结构清楚；类别细节不足','框内有清楚的斜向暗轮廓，较周围灰背景明显。','下方远于框的长暗边缘存在，未穿过当前框。','该暗轮廓没有明显输出响应。','轨迹中心在数像素范围内变化，灰度曲线不能判断物理运动或可见性。','框与暗轮廓对应，未见明显纯背景错框证据'),
874664:('局部结构可区分但很薄','8×4 框内有扁平小暗斑，细节有限。','上方有长斜暗边缘，当前框与其分离；目标靠近原图右缘但未触边。','框内低响应，未见局部高分团。','标注中心在数像素范围内变化。','局部暗斑存在，类别与标注精确边界仍待确认'),
874665:('弱可判读','12×3 的薄框内仅很弱的水平暗纹，难可靠分离目标与背景纹理。','背景较平滑但存在弱横向纹理。','原/新及展示窗框内低响应。','标注中心及原始灰度变化都很小；不能由此宣布物理静止或不可见。','待确认：弱纹理不足以核实或否定标注'),
867462:('局部结构可见但受画面边界限制','原图左缘有暗顶亮底的部分车状结构，当前框覆盖该局部。','左侧图像边界、上下桥侧/道路边缘。','当前框内输出低，邻近下方有很小的孤立响应。','标注轨迹穿过图像；当前接近片段首帧，不能把后续轨迹反推当前完整可见性。','需核实边界部分目标，未确认明显错框'),
877979:('难与邻近长边缘分离','框内有浅亮小结构及暗上部，紧邻桥侧纹理，不能从这几像素识别骑行者形态。','桥侧长边缘穿过/贴近框，上下邻近车状亮暗结构。','原/新高响应在框右邻及右下方；首个失败窗框内弱，高响应层由全部可用窗表确定。','标注轨迹沿桥侧延伸，灰度曲线变化不构成遮挡证明。','待确认：需相邻原图核实对象边界与类别'),
875774:('局部弱结构可区分；类别细节不足','框内为弱亮竖向结构，顶部接近暗带，细节不足以辨认骑行者。','桥侧长边缘紧贴框；右侧另有更亮的目标状结构。','当前框内低响应，右側邻近亮结构有高响应团，两者分开。','橙色轨迹灰度采样中存在缺段，不能把连线或缺段解释为连续可见或遮挡。','未见明确纯背景错框证据；身份与邻近边缘关系待确认'),
876372:('局部结构可见但受边界及短标注限制','框贴原图左缘，框内下部有亮小块、上部较暗，难可靠识别人形。','左画面边界和桥侧竖向/斜向结构邻近。','框内输出低。','本轨迹图仅一个点，不能计算运动、证明静止或解释为短暂遮挡。','待确认：单点标注加边界局部不足以核实标注正确性'),
}
stable_notes={
805372:'同一船轨 400008 的帧 90：原图仍是低对比浅亮团，但原/新及展示窗都匹配；可与全失败帧 98 定性并置，不能声称两当前帧或其参考内容相同。',
805300:'同一船轨 400008 的帧 82：当前浅亮团轮廓同样模糊，框内原/新均有高响应并匹配；低单帧可判读性本身不足以解释所有响应失败。',
870580:'0067 轨 7702017 帧 35：小暗竖向/浅亮底部结构贴桥侧纹理，轮廓并不清楚，仍在框边附近形成匹配响应；不能把桥侧难判读一概视为不可检测。',
873182:'0067 轨 7701908 帧 64：3 像素宽窄框贴桥侧长边缘，局部外观难辨，输出在框附近有强响应并匹配；只作为成功外观参照，不作因果对照。',
}
samples=read(C/'ALL_VISUAL_SAMPLE_INDEX.csv');records=[];viewed=[]
for r in samples:
    aid=int(r['annotation_id']);assert aid in notes
    filename=f"{r['sequence']}_track{r['track_id']}_frame{int(r['frame']):03d}_ann{aid}.png";path=D/filename
    clarity,current,edge,response,temporal,mislabel=notes[aid]
    records.append(dict(sample_id=r['sample_id'],sequence=r['sequence'],frame=r['frame'],source_frame=r['source_frame'],track_id=r['track_id'],annotation_id=aid,
        annotated_category=r['category'],response_layer=r['response_layer'],diagnostic_box_max=r['diagnostic_box_max'],image_viewed=str(path),image_sha256=sha(path),
        raw_array_source=r['raw_frames_path'],raw_array_frame_index=r['raw_array_frame_index'],original_source_file_name=r['source_file_name'],
        local_readability=clarity,current_observation=current,nearby_edges_or_boundary=edge,score_map_observation=response,
        temporal_plot_observation=temporal,obvious_background_misannotation_assessment=mislabel,
        true_visibility='待确认',true_visible_duration='未定量确认；不能以标注存在时长代替',occlusion='待确认',camera_motion='未定量确认',physical_motion='待确认',precise_peak_or_component_cause='须数值组件/峰审计；本视觉表不确认',
        review_scope='逐个实际查看固定渲染详情PNG；没有查看额外相邻原图'))
    viewed.append(dict(role='all_failure',path=str(path),sha256=sha(path)))
with (D/'ALL_FAILURE_REVIEW.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
stables=[]
for r in read(D/'STABLE_VISUAL_SELECTION.csv'):
    aid=int(r['annotation_id']);fn=f"{r['sequence']}_track{r['track_id']}_frame{int(r['frame']):03d}_ann{aid}.png";path=D/fn
    stables.append(dict(**r,image_viewed=str(path),image_sha256=sha(path),observation=stable_notes[aid]))
    viewed.append(dict(role='response_defined_stable_reference',path=str(path),sha256=sha(path)))
with (D/'STABLE_REFERENCE_REVIEW.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(stables[0]));w.writeheader();w.writerows(stables)
record=dict(status='COMPLETE',reviewer='/root/failure_conditions',recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            review_method='Each listed PNG actually opened with view_image; 16 frozen all-failure samples plus 4 existing response-defined stable references.',
            full_failure_images_viewed=16,stable_reference_images_viewed=4,other_original_images_viewed=0,
            source_selection_sha256={str(p):sha(p) for p in (C/'SAMPLE_SELECTION.json',C/'SUPPLEMENTAL_SAMPLE_SELECTION.json',C/'ALL_VISUAL_SAMPLE_INDEX.csv',D/'STABLE_VISUAL_SELECTION.csv')},
            images_viewed=viewed,limits='Exploratory, purposive samples; no 4391-population estimates; image structure is not a ground-truth visibility, occlusion, class identity or physical-motion label.',
            script_sha256=sha(Path(__file__)))
(D/'ALL_FAILURE_REVIEW_RECORD.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(all_failure_reviewed=len(records),stable_reviewed=len(stables),images_logged=len(viewed)),ensure_ascii=False))
