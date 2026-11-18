import csv,json,hashlib
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
D=Path(__file__).resolve().parent;R=D.parent
cases=list(csv.DictReader((D/'SELECTED_CASES.csv').open()));frames=list(csv.DictReader((D/'SELECTED_FRAMES.csv').open()))
rawreview={r['case_id']:r for r in csv.DictReader((D/'RAW_REVIEW.csv').open())}
details={(r['sequence'],r['annotation_id'],r['arrangement']):r for r in csv.DictReader((R/'analysis/LOW_RESPONSE_OBJECTS.csv').open())}
# These labels were recorded after individually opening all 18 exact peak panels,
# together with all 18 raw continuous panels. They are not inferred from the bbox test.
manual={
'DJI_0026_2_both_success_1_track520005':('TTTTT','TTTTT','邻域峰在小浅亮团上，5帧均维持对应关系；上下文中的横向位移可见，但不能独立判物理运动。'),
'DJI_0026_2_both_success_2_track400007':('TTTTT','TTTTT','峰位于模糊亮团或其左缘，响应强度随帧变而弱亮纹持续；小图像位移不足以独立证明跟随。'),
'DJI_0026_2_extra_only_1_track400008':('UUTTT','TTTTT','O在97帧落于左侧弱纹理，98帧落于框下角，弱图像不足判归属；99—101帧与团状结构较一致。S五帧与弱亮团较一致。'),
'DJI_0026_2_extra_only_2_track400007':('UUTBB','TTTBB','O的80—81帧在亮团左界附近且接近背景分数，无法可靠归属；82帧共定位。83—84帧两安排峰远离当前亮团、转向周围纹理；S的80—82帧共定位。'),
'DJI_0026_2_extreme_low_1_track520008':('TBTBT','TBTBT','152/154/156帧峰与亮团共定位，153/155帧峰落于分离的周围纹理，未形成连续稳定目标峰。'),
'DJI_0026_2_extreme_low_2_track400006':('BBBBB','BBBUB','O五帧峰均在框上方/左上方背景亮带或纹理，S除45帧外也偏向上方背景；45帧弱峰在框边角但不足确定为目标本体。'),
'DJI_0026_2_low_not_extreme_1_track520008':('TBTBT','TBTBT','150/152/154帧峰与弱亮团共定位，151/153帧峰转到左侧背景；相同轨迹未稳定保持目标归属。'),
'DJI_0026_2_low_not_extreme_2_track400008':('UTTTU','TTTTT','O的98/102帧弱峰处于亮团边缘或邻近纹理，不能可靠归属；99—101共定位。S五帧均在弱团/左缘附近。'),
'DJI_0067_2_both_success_1_track7700519':('TTTTT','TTTTT','5帧原始logit峰均位于车状轮廓右侧亮边并随其移动；23帧真实点匹配仍失败，说明原始logit峰与概率组件实际峰不能互换。'),
'DJI_0067_2_both_success_2_track7700522':('TB','TB','18帧邻域峰位于当前车右边；19帧最大峰转到右方另一辆已标注车，当前目标仍实际匹配。只有两帧，不判长期持续性。'),
'DJI_0067_2_extra_only_1_track7701625':('BBBBB','BBBBB','O峰在下方其他已标注车辆；S前三帧在车辆、末两帧在右侧其它窄目标。S118帧当前目标仍实际匹配；最高邻域峰属于别人不等于自身无有效峰。'),
'DJI_0067_2_extra_only_2_track7702010':('U','U','最高峰在当前窄框下方0.5像素、接触下方另一目标；单帧弱形态不足区分当前目标底部与邻车贡献，不能直接判定位偏移。'),
'DJI_0067_2_extreme_low_1_track7700544':('BBBB','BBBB','当前车状轮廓持续清楚；O最高峰在左侧其它车辆，S在右侧其它车辆。邻域高响应不是当前目标可恢复峰的直接证据。'),
'DJI_0067_2_extreme_low_2_track7822007':('BBB','BBB','两安排峰分别落在左边界竖杆/上方灰背景/下方道路结构，与框内小竖向亮结构分开；位置在3帧间跳变。'),
'DJI_0067_2_high_unmatched_1_track7701568':('UUUBB','UUUBB','20—22帧峰在框右0.5—1.5像素，弱竖纹与桥侧混合，无法可靠分离目标本体和邻近结构；23—24帧邻域最高峰转到其它已标注目标而本目标已匹配。'),
'DJI_0067_2_high_unmatched_2_track7702012':('BBBBB','BBBBB','两安排邻域最高峰均落在附近已标注车状目标；当前窄目标附近也有较弱局部响应，但只凭本图不能判其归属。'),
'DJI_0067_2_low_not_extreme_1_track7701917':('BBBBB','BBBBB','两安排邻域最高峰均在下方其他已标注车辆。当前框右侧另有非最高响应团，不能把单个邻域argmax当作附近所有峰的穷尽或认定完全缺少响应。'),
'DJI_0067_2_low_not_extreme_2_track7702013':('BBBB','BBBB','两安排最高峰均对应下方已标注车辆，当前小亮块本身在原图仍可区分；不能将该高峰当成当前目标位置错误。'),
}
names={'T':'疑似目标相关','B':'背景或其他目标相关','U':'无法判断'}
caseframe=[]
for c in cases:
    rr=[r for r in frames if r['case_id']==c['case_id']];o,s,note=manual[c['case_id']];assert len(o)==len(s)==len(rr)
    for arr,labels in [('O',o),('S',s)]:
        for r,label in zip(rr,labels):
            d=details[r['sequence'],r['annotation_id'],arr]
            keys=['neighborhood_peak_x','neighborhood_peak_y','neighborhood_peak_logit','neighborhood_peak_probability','neighborhood_peak_inside_target','neighborhood_peak_other_gt_ids','neighborhood_peak_distance_to_box','target_box_max_logit','target_box_max_probability','matched_at_05']
            caseframe.append(dict(case_id=c['case_id'],sequence=r['sequence'],track_id=r['track_id'],frame=r['frame'],source_frame=r['source_frame'],annotation_id=r['annotation_id'],arrangement=arr,selected_group=c['group'],actual_frame_group=r['visual_group'],review_label_code=label,review_label=names[label],raw_readability=rawreview[c['case_id']]['raw_readability'],raw_observation=rawreview[c['case_id']]['raw_observation'],peak_observation=note,**{k:d[k] for k in keys},peak_image=str(D/(c['case_id']+'_peaks.png')),raw_image=rawreview[c['case_id']]['raw_image']))
unique={}
for r in caseframe:
    key=(r['sequence'],r['annotation_id'],r['arrangement'])
    if key in unique:
        assert unique[key]['review_label']==r['review_label'],key
        unique[key]['case_ids']+=';'+r['case_id']
        unique[key]['selected_groups']+=';'+r['selected_group']
        unique[key]['peak_images']+=';'+r['peak_image']
    else:
        unique[key]={k:v for k,v in r.items() if k not in ['case_id','selected_group','peak_image']}
        unique[key].update(case_ids=r['case_id'],selected_groups=r['selected_group'],peak_images=r['peak_image'])
def write(name,rr):
    with (D/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
write('PEAK_REVIEW_CASE_FRAMES.csv',caseframe)
unique_rows=sorted(unique.values(),key=lambda r:(r['sequence'],int(r['track_id']),int(r['frame']),r['arrangement']))
write('PEAK_REVIEW.csv',unique_rows)
counts={arr:dict(Counter(r['review_label'] for r in unique_rows if r['arrangement']==arr)) for arr in ['O','S']}
manifest=dict(reviewed_at_utc=datetime.now(timezone.utc).isoformat(),reviewer='/root/low_visual_review',status='complete',raw_panels_actually_opened=18,peak_panels_actually_opened=18,n_cases=18,n_unique_tracks=15,n_unique_object_frames=72,n_unique_object_arrangement_rows=len(unique_rows),n_case_arrangement_rows=len(caseframe),counts=counts,interpretation='Labels describe attribution of the single predeclared neighborhood raw-logit argmax, not all possible nearby peaks. Suspected target-related is an exploratory visual judgment, not validated segmentation, visibility or class truth. Neighbor frames can cross original strata; row-level stratum retained; repeated case frames deduplicated consistently. No population rates should be estimated from these 18 cases.',selection_sha256=hashlib.sha256((D/'SELECTED_CASES.csv').read_bytes()).hexdigest(),detail_csv_sha256=hashlib.sha256((R/'analysis/LOW_RESPONSE_OBJECTS.csv').read_bytes()).hexdigest(),images={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*_peaks.png'))})
(D/'PEAK_REVIEW_RECORD.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'unique_rows':len(unique_rows),'counts':counts},ensure_ascii=False))
