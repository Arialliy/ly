import csv,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
D=Path(__file__).resolve().parent
notes={
'DJI_0026_2_both_success_1_track520005':('可区分弱亮团','5帧框内都有紧凑浅亮团，位于较亮水平结构下方；上下文中位置明显横移。','类别轮廓不可辨；图像坐标变化不能独立区分目标与相机运动。'),
'DJI_0026_2_both_success_2_track400007':('弱可判读','5帧框内仅浅淡横向亮纹/亮团，在较平滑灰背景上持续存在；框附近没有清楚孤立物体轮廓。','只有中间帧属成功层；邻帧层明确保留。不能用此图证明船体身份。'),
'DJI_0026_2_extra_only_1_track400008':('弱可判读','5帧框内皆有模糊浅亮团，周围横向纹理相近；没有肉眼清楚的新物体出现或消失。','外观相近不等于输入证据相同，弱团身份无法独立确定。'),
'DJI_0026_2_extra_only_2_track400007':('弱可判读','5帧框内皆有浅亮横向纹理，82帧上部局部较亮；单帧轮廓不足以识别类别。','并不能由82帧额外窗匹配反推当前图像具备唯一可识别信号。'),
'DJI_0026_2_extreme_low_1_track520008':('可区分弱亮团','152—156帧框内有连续的小浅亮团；156帧较紧凑明亮，位置沿画面横向变化。','同段包含两端不同旧分层，只有153—155帧极低；弱亮团不保证可学习或类别正确。'),
'DJI_0026_2_extreme_low_2_track400006':('可区分局部亮团但近边界','42—46帧右画面边界旁持续有框内紧凑亮斑，上方更亮水平带与其分开；亮斑并未肉眼消失。','黑区域是画外padding，非暗背景；类别/精确边缘不可确定。'),
'DJI_0026_2_low_not_extreme_1_track520008':('可区分弱亮团','150—154帧都有框内浅亮团，150帧更紧凑，151帧更弱；上下文位置横移。','与另一个520008案例重复3帧，应去重后计对象，不能当独立来源。'),
'DJI_0026_2_low_not_extreme_2_track400008':('弱可判读','98—102帧浅亮团始终混于弱水平灰纹，99帧与相邻帧肉眼外观相近。','与extra-only案例重复4帧，不得重复计数估计发生率。'),
'DJI_0067_2_both_success_1_track7700519':('局部车状结构清楚','19—23帧目标有暗顶、亮底和较亮右侧的紧凑车状轮廓，在道路上下长边缘之间横移；邻近多车和窄目标。','23帧已不属两安排成功；外观清楚不能单独解释响应变化。'),
'DJI_0067_2_both_success_2_track7700522':('局部车状结构可区分','2个有效连续帧中都有暗顶亮底车状局部，邻近车辆及桥侧纹理清楚。','筛选后连续段只有2帧，不跨空缺拼接5帧，不足以判断长期持续性。'),
'DJI_0067_2_extra_only_1_track7701625':('弱竖向结构与桥侧混合','116—120帧框内弱竖直深浅结构可见，和桥侧长暗带混合；右侧多个同样窄的已标注目标及下方车辆。','无法由这5帧可靠区分行人本体与栏杆/阴影，需谨慎解释附近高峰。'),
'DJI_0067_2_extra_only_2_track7702010':('局部竖向结构可区分','单有效帧框内有暗左亮右的竖向小结构，其下紧邻另一个已标注车状目标。','只有1帧，无法核查峰是否跟随；类别与目标边界不足以明确。'),
'DJI_0067_2_extreme_low_1_track7700544':('局部车状结构清楚','121—124帧框内持续有暗顶、浅亮下部的紧凑车状物，位于道路上；可与周围车辆区分。','4帧都极低但当前结构可区分，不能因此直接归因于模型架构或训练覆盖。'),
'DJI_0067_2_extreme_low_2_track7822007':('局部弱竖向结构近边界','148—150帧左边界附近有框内小竖向浅亮块，左侧长竖杆、右下车辆和桥侧纹理同时存在。','画外黑区不是可用背景；仅3帧，类别形态不可辨。'),
'DJI_0067_2_high_unmatched_1_track7701568':('弱竖向结构与桥侧混合','20—24帧框内窄竖向弱结构贴桥侧长暗带；右边有另一个已标注窄目标，车流在下方。','前3帧失配后2帧匹配，粗框中心不等于精确目标中心；仅原图不能判定位错。'),
'DJI_0067_2_high_unmatched_2_track7702012':('竖向亮块可区分但邻近密集','125—129帧框内有竖向亮块，左右紧邻多个已标注窄目标、下方车辆；边界与长暗带相接。','相邻目标和车辆可能吸引邻域峰，必须看精确峰坐标；不能用框外高峰直接判偏移。'),
'DJI_0067_2_low_not_extreme_1_track7701917':('弱可判读','130—134帧窄框内暗上浅下的细竖向纹理与桥侧暗带混合，左边另有窄目标和竖杆。','没有足够轮廓认定精确行人本体，不从标注存续推断真实可见性。'),
'DJI_0067_2_low_not_extreme_2_track7702013':('小竖向亮块可区分但密集','120—123帧框内小亮块持续存在，暗背景上局部可分；下方车、左侧密集窄目标、右侧另一窄目标。','类别轮廓不足且相邻响应可能竞争；4帧不能证明长期跟随。'),
}
cases=list(csv.DictReader((D/'SELECTED_CASES.csv').open()));records=[]
for c in cases:
    assert c['case_id'] in notes
    p=D/(c['case_id']+'_raw.png');readability,obs,limit=notes[c['case_id']]
    records.append(dict(**c,raw_image=str(p),raw_image_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),raw_readability=readability,raw_observation=obs,interpretation_limit=limit,viewed_with='functions.tools.view_image',peak_attribution='pending exact new-logit peak overlay'))
with (D/'RAW_REVIEW.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(D/'RAW_REVIEW_RECORD.json').write_text(json.dumps(dict(reviewed_at_utc=datetime.now(timezone.utc).isoformat(),reviewer='/root/low_visual_review',cases_actually_viewed=len(records),case_frames=79,unique_object_frames=72,unique_tracks=15,protocol='All 18 final raw PNGs opened with view_image after initial zoom-cropping issue corrected; no local contrast stretch; first row fixed scene coordinates, second row target-centered zoom.',notes='Manual observations are not ground-truth visibility/class/occlusion or physical motion labels; not population estimates; peak attribution deferred to exact overlay.',source_selection_sha256=hashlib.sha256((D/'SELECTED_CASES.csv').read_bytes()).hexdigest()),indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'raw_cases_reviewed':len(records)}))
