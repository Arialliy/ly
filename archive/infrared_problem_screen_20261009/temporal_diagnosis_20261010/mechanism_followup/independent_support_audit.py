#!/usr/bin/env python3
"""Independent saved-result audit of the common projection-support experiment."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE/'support_probe'

def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()

def key(row):
    return row['sequence'],int(row['local_frame']),int(row['track_id']),int(row['annotation_id'])

def n(row,col):
    return float(row[col])

def main():
    rows=[]
    original=[]
    input_hashes={}
    shard_counts=[]
    metadata=[]
    for shard in range(4):
        directory=OUT/('shard_%02d_of_04'%shard)
        meta=json.loads((directory/'metadata.json').read_text())
        metadata.append(meta)
        assert meta['status']=='COMPLETE'
        selected=read(directory/'selected_pairs.csv')
        result=read(directory/'results.csv')
        prev_selected=read(BASE/('shard_%02d_of_04'%shard)/'selected_pairs.csv')
        assert selected==prev_selected
        assert meta['completed']==len(result)==len(selected)==meta['selected_total']
        assert [key(r) for r in selected]==[key(r) for r in result]
        for s,r in zip(selected,result):
            assert all(r[k]==v for k,v in s.items())
        for file,field in [('results.csv','result_sha256'),('selected_pairs.csv','selected_pairs_sha256')]:
            actual=sha(directory/file)
            assert actual==meta[field]
            input_hashes[str(directory/file)]=actual
        assert meta['frozen_dependency_sha256']==sha(BASE/'run_factorial_frozen.py')
        command_path=Path(meta['command'][0])
        assert meta['wrapper_sha256']==sha(command_path)
        rows.extend(result)
        original.extend(read(BASE/('shard_%02d_of_04'%shard)/'results.csv'))
        shard_counts.append(len(result))
    rows.sort(key=lambda r:int(r['pair_index']))
    original={key(r):r for r in original}
    assert len(rows)==len({key(r) for r in rows})==1417
    assert [int(r['pair_index']) for r in rows]==list(range(1417))
    assert set(key(r) for r in rows)==set(original)
    disagreements=0
    max_lambda_endpoint_error=0.
    for r in rows:
        a,b=int(r['a_start']),int(r['b_start'])
        shared=set(range(a,a+40))&set(range(b,b+40))
        assert len(shared)==int(r['shared_frames'])
        assert min(shared)==int(r['shared_absolute_first']) and max(shared)==int(r['shared_absolute_last'])
        safe={t for t in shared if t-8>=min(shared) and t+8<=max(shared)}
        assert len(safe)==int(r['safe_shared_frame_count'])
        assert int(bool(safe))==int(r['safe_shared_eligible'])
        assert int(int(r['local_frame']) in safe)==int(r['current_frame_in_safe_shared'])
        if safe:
            assert int(r['safe_shared_absolute_first'])==min(safe) and int(r['safe_shared_absolute_last'])==max(safe)
        else:
            assert r['safe_shared_absolute_first']==r['safe_shared_absolute_last']==''
        assert int(r['global_input_start'])==max(0,min(a,b)-8)
        assert int(r['global_input_end'])==min(160,max(a,b)+48)
        assert int(r['global_halo_missing_before'])==max(0,8-min(a,b))
        assert int(r['global_halo_missing_after'])==max(0,max(a,b)+48-160)
        prev=original[key(r)]
        for t in ['a','b']:
            assert int(r[t+'_exclusive_frame_count'])==40-len(shared)
            assert int(r[t+'_peak_x_index'])==int(prev[t+'_peak_x_index'])
            assert int(r[t+'_peak_y_index'])==int(prev[t+'_peak_y_index'])
            assert abs(n(r,t+'_full_box_max_probability')-n(prev,t+'_replay_box_max'))<1e-5
            assert abs(n(r,t+'_full_box_max_logit')-n(prev,t+'_replay_box_max_logit'))<1e-4
            assert n(r,t+'_replay_abs_error')<=1e-4
            for th in [.1,.5,.9]:
                flag=int((n(r,t+'_full_box_max_probability')>=th)!=(n(r,t+'_saved_box_max')>=th))
                assert flag==int(r[t+'_replay_threshold_disagreement_'+str(th).replace('.','_')])
                disagreements+=flag
            for left,right in [('shared','lambda_000'),('full','lambda_100')]:
                for readout in ['box_max_probability','box_max_logit','a_peak_logit','b_peak_logit']:
                    error=abs(n(r,t+'_'+left+'_'+readout)-n(r,t+'_'+right+'_'+readout))
                    if readout.endswith('probability'):
                        assert error<=1e-4
                    else:
                        assert error<=1e-4+1e-4*abs(n(r,t+'_'+left+'_'+readout))
                    max_lambda_endpoint_error=max(max_lambda_endpoint_error,error)
            shared_p,full_p=n(r,t+'_shared_box_max_probability'),n(r,t+'_full_box_max_probability')
            assert int(r[t+'_shared_to_full_high_low_0_5'])==int(shared_p>=.5 and full_p<.5)
            assert int(r[t+'_shared_to_full_low_high_0_5'])==int(shared_p<.5 and full_p>=.5)
            assert int(r[t+'_shared_to_full_strong_high_low'])==int(shared_p>=.9 and full_p<.1)
            assert int(r[t+'_shared_to_full_strong_low_high'])==int(shared_p<.1 and full_p>=.9)
        for col,value in r.items():
            if col.endswith(('_probability','_logit')):
                if 'safe_shared' in col and not safe:
                    assert value==''
                    continue
                assert math.isfinite(float(value))
    assert disagreements==0
    max_errors={col:max(n(r,col) for r in rows) for col in rows[0] if 'error' in col}
    assert all(value<=1e-4 for col,value in max_errors.items() if 'probability' in col or 'replay_abs' in col)
    groups={
        'all':rows,
        'strong_flip':[r for r in rows if n(r,'a_saved_box_max')<.1 and n(r,'b_saved_box_max')>=.9],
        'cross_05':[r for r in rows if n(r,'a_saved_box_max')<.5<=n(r,'b_saved_box_max')],
        'all_low':[r for r in rows if n(r,'b_saved_box_max')<.5],
        'all_high':[r for r in rows if n(r,'a_saved_box_max')>=.5],
    }
    assert [len(groups[x]) for x in ['strong_flip','cross_05','all_low','all_high']]==[97,137,778,502]
    summaries={}
    for name,subset in groups.items():
        s={'n':len(subset),'endpoints':{}}
        for t in ['a','b']:
            suppressed=[r for r in subset if n(r,t+'_shared_box_max_probability')>=.9 and n(r,t+'_full_box_max_probability')<.1]
            enhanced=[r for r in subset if n(r,t+'_shared_box_max_probability')<.1 and n(r,t+'_full_box_max_probability')>=.9]
            suppressed05=[r for r in subset if n(r,t+'_shared_box_max_probability')>=.5 and n(r,t+'_full_box_max_probability')<.5]
            enhanced05=[r for r in subset if n(r,t+'_shared_box_max_probability')<.5 and n(r,t+'_full_box_max_probability')>=.5]
            monotonic=0
            for r in suppressed:
                values=[n(r,t+'_lambda_'+v+'_box_max_logit') for v in ['000','025','050','075','100']]
                monotonic+=int(all(y<=x+1e-4 for x,y in zip(values,values[1:])))
            s['endpoints'][t]={
                'shared_ge09_full_lt01':len(suppressed),'shared_lt01_full_ge09':len(enhanced),
                'shared_ge05_full_lt05':len(suppressed05),'shared_lt05_full_ge05':len(enhanced05),
                'suppressed09_track_ids':len({r['track_id'] for r in suppressed}),
                'suppressed09_sequences':dict(collections.Counter(r['sequence'] for r in suppressed)),
                'suppressed09_global_shared_ge09':sum(n(r,t+'_global_shared_box_max_probability')>=.9 for r in suppressed),
                'suppressed09_lambda_monotone_decreasing_logit':monotonic,
                'global_shared_ge09_full_lt01':sum(n(r,t+'_global_shared_box_max_probability')>=.9 and n(r,t+'_full_box_max_probability')<.1 for r in subset),
            }
            eligible=[r for r in subset if int(r['safe_shared_eligible'])]
            current_inside=[r for r in eligible if int(r['current_frame_in_safe_shared'])]
            for scope,ss in [('safe_eligible',eligible),('safe_current_inside',current_inside)]:
                s['endpoints'][t][scope]={
                    'n':len(ss),
                    'safe_ge09_full_lt01':sum(n(r,t+'_safe_shared_box_max_probability')>=.9 and n(r,t+'_full_box_max_probability')<.1 for r in ss),
                    'safe_ge05_full_lt05':sum(n(r,t+'_safe_shared_box_max_probability')>=.5 and n(r,t+'_full_box_max_probability')<.5 for r in ss),
                    'safe_lt01_full_ge09':sum(n(r,t+'_safe_shared_box_max_probability')<.1 and n(r,t+'_full_box_max_probability')>=.9 for r in ss),
                }
        s['any_endpoint_shared_ge09_full_lt01']=sum(any(n(r,t+'_shared_box_max_probability')>=.9 and n(r,t+'_full_box_max_probability')<.1 for t in ['a','b']) for r in subset)
        s['global_shared_two_slot_cross_05']=sum(min(n(r,t+'_global_shared_box_max_probability') for t in ['a','b'])<.5<=max(n(r,t+'_global_shared_box_max_probability') for t in ['a','b']) for r in subset)
        s['global_shared_two_slot_strong_flip']=sum(min(n(r,t+'_global_shared_box_max_probability') for t in ['a','b'])<.1 and max(n(r,t+'_global_shared_box_max_probability') for t in ['a','b'])>=.9 for r in subset)
        summaries[name]=s
    payload={'status':'PASS','scope':'Independent standard-library recheck of support-probe saved results. No main analysis import; no inference rerun.',
             'counts':shard_counts,'total':len(rows),'threshold_disagreements':disagreements,
             'maximum_errors':max_errors,'max_scalar_lambda_endpoint_error':max_lambda_endpoint_error,
             'summaries':summaries,'input_result_sha256':input_hashes,'script_sha256':sha(Path(__file__)),
             'wrapper_sha256':metadata[0]['wrapper_sha256'],'dependency_sha256':metadata[0]['frozen_dependency_sha256'],
             'limits':['Shared support refers to projected feature time slots; each frontend feature can include up to 8 original frames outside that shared interval.',
                       'Zeroing exclusive projected terms is an internal intervention potentially outside the natural feature distribution.',
                       'Global shared uses identical feature tensors with distinct original row coefficients and bias, and is still an ablated representation.',
                       'Response suppression is established within this frozen computation, not semantic target evidence sufficiency, visibility, deployable benefit, or false-alarm control.']}
    (OUT/'AUDIT.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    sf=summaries['strong_flip']['endpoints']['a']
    lines=['# 共享投影支持消融：独立审计','', '**结论：PASS。** 1,417 个目标帧及 min/max 窗选择与前一轮完全一致；三阈值重放零分歧；lambda=0 与 shared、lambda=1 与原完整输出的标量端点一致。','',
           '已核对 common absolute support、union+真实 halo、两端固定峰坐标、投影重构误差记录与原后端端点。完整张量未保存，张量级 gate 依据已独立审计代码及记录；本次独立复算保存的标量与计数。','',
           '97 个强翻转的低响应原窗口中：','',
           f'- 共享投影支持响应 ≥0.9，但加回全部独有投影项后原响应 <0.1：**{sf["shared_ge09_full_lt01"]}** 个目标帧，涉及 {sf["suppressed09_track_ids"]} 个轨迹 ID。',
           f'- 其中用统一真实前端特征后，共享支持响应仍 ≥0.9：**{sf["suppressed09_global_shared_ge09"]}** 个。',
           f'- 其中 5 个 lambda 采样点的框内最大 logit 单调不升：**{sf["suppressed09_lambda_monotone_decreasing_logit"]}** 个。',
           f'- 阈值 0.5 下，shared 高→full 低为 {sf["shared_ge05_full_lt05"]} 个。','',
           f'相反方向也很常见：同一 97 个强翻转的高响应窗中，shared <0.1 → full ≥0.9 为 **{summaries["strong_flip"]["endpoints"]["b"]["shared_lt01_full_ge09"]}** 个。', '',
           f'依赖安全控制 Ssafe 非空 {sf["safe_eligible"]["n"]}/97，当前帧位于 Ssafe 的仅 {sf["safe_current_inside"]["n"]} 个；其中 safe ≥0.9 → full <0.1 为 **{sf["safe_current_inside"]["safe_ge09_full_lt01"]}** 个。不能把普通 shared 的 24 个案例都写成只由共同原图支持的证据。', '',
           '必须同时报告相反方向；各端点、全部样本、始终低／高对照的促进与抑制计数见 AUDIT.json。','',
           '**解释边界：** shared 指投影特征的时间槽，而每个前端特征含 ±8 帧感受野。因此不能把结果写成“只看共同原始图像就足够”。删除独有投影项并非删除独有原图；该反事实可偏离自然表示分布。结果支持冻结计算内部的响应抑制，尚未证明语义证据可靠、物理可见性或误报受控的部署提升。','',
           '可复算脚本为 ../independent_support_audit.py。']
    (OUT/'AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'PASS','total':len(rows),'maximum_errors':max_errors,'summaries':summaries},indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
