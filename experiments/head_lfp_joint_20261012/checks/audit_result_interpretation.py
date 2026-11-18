"""Independently recount saved observations and endpoint detections; no inference."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np

E=Path(__file__).resolve().parents[1];B=E.parents[1]
Q=B/'archive/original_project/temporal_diagnosis_20261010'
D=E/'evaluation/step_2048'
CONDITIONS=('release','H','HL');PLANS=('O','S');SEQS=('DJI_0026_2','DJI_0067_2')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))
def check(x,why):
    if not x:raise AssertionError(why)
def flag(row,key):return bool(int(row[key]))
def key(row):return (row['sequence'],row['frame'],row['annotation_id'],row.get('arrangement',''))
def identities(rows):return {(r['sequence'],r['track_id']) for r in rows}


def receipt_check(folder,filename='SUMMARY_COMPLETE.json',field='output_sha256'):
    receipt=json.loads((folder/filename).read_text())
    check(receipt['status']=='COMPLETE',str(folder)+' status')
    for name,digest in receipt[field].items():check(sha(folder/name)==digest,str(folder/name)+' hash')
    return receipt


def observations():
    O=E/'observations';receipt_check(O)
    totals={(r['condition'],int(r['step'])):r for r in read(O/'ALL_STEPS_SUMMARY.csv')}
    margin_table={(r['condition'],int(r['step']),r['cohort']):r for r in read(O/'FIXED_BACKGROUND_MARGINS.csv')}
    release_objects={r['key']:r for r in read(O/'release/step_0000/OBJECTS.csv')}
    release_frames={r['key']:r for r in read(O/'release/step_0000/FRAMES.csv')}
    outputs={};matched_sets={}
    for condition,step in [('release',0)]+[(c,s) for c in ('H','HL') for s in (128,512,1024,2048)]:
        folder=O/condition/f'step_{step:04d}'
        receipt=receipt_check(folder,'COMPLETE.json','csv_sha256')
        check(receipt['training_updates']==0 and receipt['model_state_unchanged'],'Observation changed model')
        objects=read(folder/'OBJECTS.csv');frames=read(folder/'FRAMES.csv')
        check({r['key'] for r in objects}==set(release_objects),'Observation object sampling changed')
        check({r['key'] for r in frames}==set(release_frames),'Observation frame sampling changed')
        frame_map={r['key']:r for r in frames}
        for row in frames:
            check(row['fixed_bg_index']==release_frames[row['key']]['fixed_bg_index'],'Background position moved')
        for row in objects:
            base=release_objects[row['key']]
            check(row['release_matched']==base['matched'] and row['release_box_max']==base['box_max'],'Release join changed')
            check(flag(row,'gain_vs_release')==(flag(row,'matched') and not flag(base,'matched')),'Observation gain flag')
            check(flag(row,'loss_vs_release')==(flag(base,'matched') and not flag(row,'matched')),'Observation loss flag')
            frame=frame_map[row['key'].rsplit(':',1)[0]]
            if row['margin_vs_fixed_background_logit']!='':
                check(abs(float(row['margin_vs_fixed_background_logit'])-(float(row['box_max_logit'])-float(frame['fixed_bg_logit'])))<1e-7,'Margin rawlogit arithmetic')
        core=[r for r in objects if flag(r,'core_strict_small')]
        low=[r for r in core if float(r['release_box_max'])<.5]
        recovered=[r for r in low if flag(r,'matched')]
        current=totals[condition,step]
        counts={'core_strict_small_window_target_frames':len(core),'unique_core_strict_target_frames':len({(r['sequence'],r['track_id'],r['frame']) for r in core}),
                'tracks':len(identities(core)),'matched':sum(flag(r,'matched') for r in core),
                'gain_vs_release':sum(flag(r,'gain_vs_release') for r in core),'loss_vs_release':sum(flag(r,'loss_vs_release') for r in core),
                'release_low_window_objects':len(low),'release_low_recovered':len(recovered),
                'low_recovery_tracks':len(identities(recovered)),'low_recovery_sources':len({r['sequence'] for r in recovered}),
                'low_recovered_unique_target_frames':len({(r['sequence'],r['track_id'],r['frame']) for r in recovered}),
                'unmatched_components':sum(int(r['unmatched_components']) for r in frames),
                'outside_pixels':sum(int(r['positive_pixels_outside_boxes']) for r in frames),
                'unmatched_component_area':sum(int(r['unmatched_component_area']) for r in frames)}
        for name,value in counts.items():check(int(current[name])==value,f'Observation aggregate {condition}/{step}/{name}')
        for cohort,chosen in [('all_core',core),('release_low_window',low),('release_success',[r for r in core if flag(r,'release_matched')])]:
            valid=[r for r in chosen if r['margin_vs_fixed_background_logit']!='']
            vals=[float(r['margin_vs_fixed_background_logit']) for r in valid]
            deltas=[v-float(release_objects[r['key']]['margin_vs_fixed_background_logit']) for r,v in zip(valid,vals)]
            saved=margin_table[condition,step,cohort]
            for name,value in {'window_target_instances':len(chosen),'tracks':len(identities(chosen)),'sources':len({r['sequence'] for r in chosen}),
                               'valid_background_instances':len(valid),'margin_positive':sum(v>0 for v in vals),
                               'margin_improved':sum(v>0 for v in deltas),'margin_worsened':sum(v<0 for v in deltas)}.items():
                check(int(saved[name])==value,f'Observation margins {name}')
            if vals:
                check(abs(float(saved['median_margin'])-statistics.median(vals))<1e-8,'Median margin')
                check(abs(float(saved['median_paired_margin_change'])-statistics.median(deltas))<1e-8,'Median pairedmargin')
        if step==2048:
            outputs[condition]={**counts,'low_recovered_instances':[r['key'] for r in recovered],
                                'low_cohort_median_paired_margin_change':float(margin_table[condition,step,'release_low_window']['median_paired_margin_change'])}
            matched_sets[condition]={r['key'] for r in core if flag(r,'matched')}
    check(len(release_frames)==640,'16*40 observation frames')
    core=[r for r in release_objects.values() if flag(r,'core_strict_small')]
    low=[r for r in core if float(r['release_box_max'])<.5]
    scope={'windows':len({(r['sequence'],r['window_start'],r['crop_y'],r['crop_x']) for r in release_frames.values()}),
           'window_sources':len({r['sequence'] for r in release_frames.values()}),'strict_small_instances':len(core),
           'strict_small_sources':len({r['sequence'] for r in core}),'strict_small_tracks':len(identities(core)),
           'strict_small_unique_target_frames':len({(r['sequence'],r['track_id'],r['frame']) for r in core}),
           'low_instances':len(low),'low_tracks':len(identities(low)),'low_sources':len({r['sequence'] for r in low})}
    check(scope=={'windows':16,'window_sources':8,'strict_small_instances':1170,'strict_small_sources':7,'strict_small_tracks':16,
                  'strict_small_unique_target_frames':743,'low_instances':141,'low_tracks':5,'low_sources':1},'Frozen observation scope')
    return {'status':'PASS','scope':scope,'endpoint':outputs,
            'HL_vs_H_matched_set_gains':sorted(matched_sets['HL']-matched_sets['H']),
            'HL_vs_H_matched_set_losses':sorted(matched_sets['H']-matched_sets['HL']),
            'scope_wording':'Corrected report distinguishes8 window sources/7 strict-small sources/1 low-response source. Instances can overlap; fixed background is not current maximum.'}


def lengths(rows,condition=None):
    runs=[];count=0;previous=None
    for row in rows:
        adjacent=previous is not None and int(row['frame'])==int(previous['frame'])+1 and int(row['source_frame'])==int(previous['source_frame'])+1
        if previous is not None and not adjacent:
            if count:runs.append(count)
            count=0
        misses=condition is None or not flag(row,condition+'_matched')
        if misses:count+=1
        elif count:runs.append(count);count=0
        previous=row
    if count:runs.append(count)
    return runs


def endpoint():
    receipt=receipt_check(D)
    check(receipt['threshold']==.5 and receipt['step']==2048 and receipt['is_main'],'Endpoint protocol')
    small=read(D/'OBJECT_COMPARISON.csv');allgt=read(D/'ALL_GT_OBJECT_COMPARISON.csv')
    check(len(small)==10400 and len(allgt)==32594,'Object table population')
    check(len({key(r) for r in small})==len(small),'Small duplicatekeys')
    small_map={key(r):r for r in small};gt_map={key(r):r for r in allgt}
    frozen_small={key(r):r for r in read(B/'results/tlfp_pilot_20261012/evaluation/OBJECT_COMPARISON.csv')}
    check(set(frozen_small)==set(small_map),'Frozen small population changed')
    for k,row in small_map.items():
        for name in ('low_all_05','extreme_all_001','both_fixed_success','release_matched'):
            check(row[name]==frozen_small[k][name],'Frozen small cohorts changed')
    background={};raw_checked=0
    for condition in CONDITIONS:
        for sequence in SEQS:
            folder=Q/'top1_background_control_20261011/evaluation/release'/sequence if condition=='release' else D/condition/sequence
            meta=json.loads((folder/'COMPLETE.json').read_text())
            with np.load(folder/'scores.npz') as archive:scores={k:archive[k] for k in archive.files}
            with np.load(folder/'component_areas.npz') as archive:areas={k:archive[k] for k in archive.files}
            check(scores['plans'].tolist()==list(PLANS) and float(scores['threshold'])==.5,'Raw score protocol')
            for pi,plan in enumerate(PLANS):
                for i,obj in enumerate(meta['objects']):
                    k=(sequence,str(obj['frame']),str(obj['annotation_id']),plan);row=gt_map[k]
                    check(flag(row,condition+'_matched')==bool(scores['matched'][pi,i]),'Raw match vs allGT table')
                    check(flag(row,'strict_small')==bool(obj['strict_small']),'Strict flag vsmetadata')
                    if k in small_map:check(row[condition+'_matched']==small_map[k][condition+'_matched'],'Small/allGT inconsistency')
                    raw_checked+=1
                counts=scores['frame_counts'][pi].sum(axis=0)
                unmatched=areas[plan+'_areas'][areas[plan+'_unmatched'].astype(bool)]
                background[condition,plan,sequence]={'unmatched_components':int(counts[3]),'positive_pixels_outside_boxes':int(counts[5]),
                                                    'unmatched_component_area_sum':int(unmatched.sum())}
    summary={(r['condition'],r['arrangement']):r for r in read(D/'SUMMARY.csv')}
    details={};recovery_details={}
    for condition in CONDITIONS:
        for plan in PLANS:
            rows=[r for r in small if r['arrangement']==plan]
            current=summary[condition,plan];match=[r for r in rows if flag(r,condition+'_matched')]
            low=[r for r in match if flag(r,'low_all_05')];extreme=[r for r in match if flag(r,'extreme_all_001')]
            derived={'M':len(match),'low_all_05_matched':len(low),'low_all_05_matched_tracks':len(identities(low)),
                     'low_all_05_matched_clips':len({r['sequence'] for r in low}),'extreme_all_001_matched':len(extreme)}
            for ref in CONDITIONS:
                derived['G_vs_'+ref+'_same']=sum(flag(r,condition+'_matched') and not flag(r,ref+'_matched') for r in rows)
                derived['L_vs_'+ref+'_same']=sum(not flag(r,condition+'_matched') and flag(r,ref+'_matched') for r in rows)
            for name in ('unmatched_components','positive_pixels_outside_boxes','unmatched_component_area_sum'):
                derived[name]=sum(background[condition,plan,s][name] for s in SEQS)
            for name,value in derived.items():check(int(current[name])==value,f'Endpoint summary {condition}/{plan}/{name}')
            derived['passes_U_budget']=derived['unmatched_components']<=4015
            details[condition+'_'+plan]=derived
            if low:
                frequencies=defaultdict(int)
                for r in low:frequencies[r['sequence'],r['track_id']]+=1
                scores_=[float(r['release_box_max']) for r in low]
                recovery_details[condition+'_'+plan]={
                    'same_arrangement_release_box_max_min':min(scores_),
                    'same_arrangement_release_box_max_median':statistics.median(scores_),
                    'same_arrangement_release_box_max_max':max(scores_),
                    'frames_per_track':[{'sequence':s,'track_id':t,'recovered_frames':n} for (s,t),n in sorted(frequencies.items())],
                    'interpretation':'Scores are release probabilities in the same fixed arrangement. These values do not prove selective discrimination or exclude general score rescaling.'}
    changed=[r for r in allgt if flag(r,'HL_matched')!=flag(r,'H_matched')]
    changed_csv=read(D/'HL_VS_H_CHANGED_OBJECTS.csv')
    check({key(r) for r in changed}=={key(r) for r in changed_csv},'HLvsH exact allGT changedset')
    allgt_summary=read(D/'ALL_GT_SUMMARY.csv')
    retention={}
    for saved in allgt_summary:
        rows=[r for r in allgt if r['arrangement']==saved['arrangement'] and (saved['group']=='pooled' or r['sequence']==saved['group'])
              and (saved['scope']=='all_valid_gt' or flag(r,'strict_small')==(saved['scope']=='strict_small'))]
        cond=saved['condition']
        check(len(rows)==int(saved['target_frames']),'AllGT group population')
        check(sum(flag(r,cond+'_matched') for r in rows)==int(saved['M']),'AllGT matches')
        for ref in CONDITIONS:
            check(sum(flag(r,cond+'_matched') and not flag(r,ref+'_matched') for r in rows)==int(saved['G_vs_'+ref]),'AllGT gains')
            check(sum(not flag(r,cond+'_matched') and flag(r,ref+'_matched') for r in rows)==int(saved['L_vs_'+ref]),'AllGT losses')
        if saved['group']=='pooled':retention[cond+'_'+saved['arrangement']+'_'+saved['scope']]={k:int(saved[k]) for k in ('target_frames','M','G_vs_release','L_vs_release','G_vs_H','L_vs_H')}
    protection={}
    for saved in read(D/'PROTECTED_AND_WINDOW_RETENTION.csv'):
        if saved['cohort']!='both_fixed_success' or saved['group']!='pooled':continue
        cond=saved['condition'];left={k[:3]:r for k,r in small_map.items() if k[3]=='O' and flag(r,'both_fixed_success')}
        right={k[:3]:r for k,r in small_map.items() if k[3]=='S' and flag(r,'both_fixed_success')}
        check(len(left)==len(right)==646 and set(left)==set(right),'Protected population')
        both=sum(flag(r,cond+'_matched') and flag(right[k],cond+'_matched') for k,r in left.items())
        neither=sum(not flag(r,cond+'_matched') and not flag(right[k],cond+'_matched') for k,r in left.items())
        check(both==int(saved['both_matched']) and 646-both==int(saved['lost_dual_success']) and neither==int(saved['lost_both_arrangements']),'Protection counts')
        protection[cond]={'both_matched':both,'lost_dual_success':646-both,'neither_matched':neither}
    partitions=defaultdict(list)
    for row in small:partitions[row['arrangement'],row['sequence'],row['track_id']].append(row)
    track_deltas={};checked_tracks=0;changed_lengths=[]
    for saved in read(D/'CONTINUOUS_MISS_TRACKS.csv'):
        rows=sorted(partitions[saved['arrangement'],saved['sequence'],saved['track_id']],key=lambda r:(int(r['frame']),int(r['source_frame'])))
        rows=[r for r in rows if saved['cohort']=='all' or flag(r,saved['cohort'])]
        runs={cond:lengths(rows,cond) for cond in CONDITIONS};cond=saved['condition'];observation=lengths(rows)
        derived={'target_frames':len(rows),'missed_frames':sum(runs[cond]),'miss_run_count':len(runs[cond]),
                 'longest_miss_run':max(runs[cond],default=0),'observation_segments':len(observation),'observation_longest':max(observation)}
        for name,value in derived.items():check(int(saved[name])==value,'Track '+name)
        check(json.loads(saved['miss_run_lengths'])==runs[cond],'Track run lengths')
        for ref in CONDITIONS:
            delta=max(runs[cond],default=0)-max(runs[ref],default=0)
            check(int(saved['longest_miss_delta_vs_'+ref])==delta,'Track reference delta')
            if saved['cohort']=='all':track_deltas[cond,saved['arrangement'],saved['sequence'],saved['track_id'],ref]=delta
            if saved['cohort']=='all' and ref=='release' and delta:
                changed_lengths.append({'condition':cond,'arrangement':saved['arrangement'],
                                        'sequence':saved['sequence'],'track_id':saved['track_id'],
                                        'release_longest':max(runs['release'],default=0),
                                        'candidate_longest':max(runs[cond],default=0),'delta':delta})
        checked_tracks+=1
    continuity={}
    for cond in CONDITIONS:
        for plan in PLANS:
            ds=[v for (c,p,s,t,ref),v in track_deltas.items() if c==cond and p==plan and ref=='release']
            check(len(ds)==130,'Trackpopulation')
            continuity[cond+'_'+plan]={'shorter':sum(v<0 for v in ds),'same':sum(v==0 for v in ds),'longer':sum(v>0 for v in ds)}
            saved=next(r for r in read(D/'CONTINUOUS_MISS_SUMMARY.csv') if r['condition']==cond and r['arrangement']==plan and r['cohort']=='all' and r['group']=='pooled')
            for key_,value in continuity[cond+'_'+plan].items():check(int(saved['tracks_'+key_+'_vs_release'])==value,'Continuous summary')
    return {'status':'PASS','raw_matched_entries_checked':raw_checked,'allGT_rows':len(allgt),'strict_small_rows':len(small),
            'main_rows':details,'HL_vs_H_changed_objects':[{k:r[k] for k in ('sequence','track_id','frame','annotation_id','arrangement','strict_small','H_matched','HL_matched')} for r in changed],
            'allGT_retention':retention,'protection':protection,'continuity':continuity,'trajectory_rows_recomputed':checked_tracks,
            'low_recovery_score_and_track_distribution':recovery_details,'changed_longest_miss_lengths':changed_lengths}


def main():
    started=time.perf_counter();out={'status':'RUNNING','audit_script_sha256':sha(__file__),'no_new_training_inference_or_probes':True}
    try:
        out['observations']=observations();out['endpoint']=endpoint()
        text=(E/'RESULTS.md').read_text()
        for label,row in out['endpoint']['main_rows'].items():
            cond,plan=label.rsplit('_',1)
            expected=f"| {cond} | {plan} | {row['M']} | {row['G_vs_release_same']}／{row['L_vs_release_same']} | {row['low_all_05_matched']}／{row['low_all_05_matched_tracks']}／{row['low_all_05_matched_clips']} | {row['extreme_all_001_matched']} | {row['unmatched_components']} | {row['positive_pixels_outside_boxes']} | {row['unmatched_component_area_sum']} |"
            check(expected in text,'RESULTS primary table mismatch: '+label)
        out['report_sha256']=sha(E/'RESULTS.md');out['observation_report_sha256']=sha(E/'observations/REPORT.md')
        rows=out['endpoint']['main_rows']
        out['supported_conclusions']=[
            f"H恢复全窗低响应O/S={rows['H_O']['low_all_05_matched']}/{rows['H_S']['low_all_05_matched']}帧；HL={rows['HL_O']['low_all_05_matched']}/{rows['HL_S']['low_all_05_matched']}。恢复实际存在，但两组两安排均超过4015预算。",
            f"HL相对H严格小目标O新增/丢失={rows['HL_O']['G_vs_H_same']}/{rows['HL_O']['L_vs_H_same']}，S={rows['HL_S']['G_vs_H_same']}/{rows['HL_S']['L_vs_H_same']}。S总数相等不能写作匹配集合相同。",
            'HL相对H降低U、框外前景和未匹配组件面积；现有数据支持背景代价取舍，未支持LFP增加目标恢复能力。',
            '训练观察终点H和HL实际匹配集合相同；5个低响应恢复实例分布3条轨迹且只来自1个训练来源。不能将8个观察来源写成8个恢复来源。',
            '开发恢复每条低响应轨迹仅1至3帧，来源明显集中，但并非全部仅为0.5附近框响应：部分原同安排框最大概率低至0.004。',
            '连续漏检确有局部缩短，包括O下21到13帧、S下24到15帧；S也有19到22帧的恶化。不能写作所有持续漏检改善。',
            '4170个极低响应子集恢复为0；646保护集合仍双侧成功644，两侧均漏检0。其余大小目标存在新增丢失，不能宣称全体成功保持。'
        ]
        out['interpretation_limits']=[
            'Training observation instances overlap;1170 instances are743 unique target frames.8 window sources are not8 recovery sources.',
            'Training low-response141 instances are not the4342 development all-window-low cohort.',
            'Margin is against a release-fixed background location; no conclusion about all current hard backgrounds or uniform score-shift exclusion follows.',
            'M/U are frozen development diagnostics, not official Pd/Fa or new-source generalization.',
            'Only HL-vs-H set differences can support incremental LFP detection claims. Comparing HL against release includes head adaptation.',
            'Implementation/gradient execution success does not prove LFP mechanism, temporal causality, or recovery under background constraints.',
            'A few recovering frames do not alone establish robust shortening of persistent misses; inspect per-track lengths and newly lengthened trajectories.'
        ]
        out['status']='PASS'
    except BaseException as error:
        out.update(status='FAILED',error=repr(error));raise
    finally:
        out['seconds']=time.perf_counter()-started
        (E/'checks/RESULT_INTERPRETATION_AUDIT.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({'status':out['status'],'output':str(E/'checks/RESULT_INTERPRETATION_AUDIT.json')}))


if __name__=='__main__':main()
