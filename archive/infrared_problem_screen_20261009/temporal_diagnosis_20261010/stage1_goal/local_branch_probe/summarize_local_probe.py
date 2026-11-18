"""Summarize all prespecified conditions from saved local-branch probe rows."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
RUN=HERE/'full'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def val(row,field):
    return float(row[field])

def stats(rows,condition):
    result={'n':len(rows),'track_ids':len({r['track_id'] for r in rows}),'clips':dict(collections.Counter(r['sequence'] for r in rows)),'endpoints':{},'paired':{}}
    for tag in ['a','b']:
        prefix=tag+'_'+condition
        pv=[val(r,prefix+'_box_max_probability') for r in rows]
        ov=[val(r,tag+'_original_box_max_probability') for r in rows]
        st={'mean_probability':sum(pv)/len(pv) if pv else None,
            'mean_probability_difference':sum(x-y for x,y in zip(pv,ov))/len(pv) if pv else None,
            'mean_logit_difference':sum(val(r,prefix+'_box_max_logit')-val(r,tag+'_original_box_max_logit') for r in rows)/len(rows) if rows else None,
            'lt01':sum(x<.1 for x in pv),'ge09':sum(x>=.9 for x in pv),
            'strong_gain_from_original':sum(y<.1 and x>=.9 for x,y in zip(pv,ov)),
            'strong_loss_from_original':sum(y>=.9 and x<.1 for x,y in zip(pv,ov)),
            'fixed_original_b_peak_ge05':sum(val(r,prefix+'_b_peak_probability')>=.5 for r in rows)}
        for th in [.1,.5,.9]:
            st[str(th)]={'positive':sum(x>=th for x in pv),
                         'original_positive':sum(y>=th for y in ov),
                         'gain':sum(y<th<=x for x,y in zip(pv,ov)),
                         'loss':sum(x<th<=y for x,y in zip(pv,ov))}
        result['endpoints'][tag]=st
    for th in [.1,.5,.9]:
        aa=[val(r,'a_'+condition+'_box_max_probability')>=th for r in rows]
        bb=[val(r,'b_'+condition+'_box_max_probability')>=th for r in rows]
        result['paired'][str(th)]={'both_high':sum(a and b for a,b in zip(aa,bb)),
                                  'both_low':sum(not a and not b for a,b in zip(aa,bb)),
                                  'a_high_b_low':sum(a and not b for a,b in zip(aa,bb)),
                                  'a_low_b_high':sum(not a and b for a,b in zip(aa,bb))}
    return result

def main():
    meta=json.loads((RUN/'metadata.json').read_text())
    assert meta['status']=='COMPLETE' and meta['completed']==1417
    assert sha(RUN/'results.csv')==meta['result_sha256']
    assert sha(RUN/'selected_pairs.csv')==meta['selection_sha256']
    assert sha(HERE/'run_local_probe.py')==meta['script_sha256']
    rows=read(RUN/'results.csv')
    selection=read(RUN/'selected_pairs.csv')
    assert len(rows)==len(selection)==1417
    assert [int(r['pair_index']) for r in rows]==list(range(1417))
    assert len({(r['sequence'],r['local_frame'],r['track_id'],r['annotation_id']) for r in rows})==1417
    for r,s in zip(rows,selection):
        assert all(r[k]==v for k,v in s.items())
        for field,value in r.items():
            if field.endswith(('_probability','_logit','_error')):
                assert math.isfinite(float(value)),field
        for tag in ['a','b']:
            for radius in [0,4]:
                indices=[s for s in range(40) if abs(s-int(r[tag+'_slot']))<=radius]
                assert int(r[tag+'_r%d_local_feature_frame_count'%radius])==len(indices)
                assert int(r[tag+'_r%d_local_first_slot'%radius])==min(indices)
                assert int(r[tag+'_r%d_local_last_slot'%radius])==max(indices)
            for th in [.1,.5,.9]:
                flag=int((val(r,tag+'_original_box_max_probability')>=th)!=(val(r,tag+'_saved_box_max')>=th))
                assert flag==int(r[tag+'_replay_threshold_disagreement_'+str(th).replace('.','_')])==0
                assert (val(r,tag+'_full_box_max_probability')>=th)==(val(r,tag+'_original_box_max_probability')>=th)
    groups={'all':rows,'strong97':[r for r in rows if int(r['in_strong_97'])],
            'suppression24':[r for r in rows if int(r['in_suppression_24'])],
            'support42':[r for r in rows if int(r['in_support_42'])],
            'stable_high502':[r for r in rows if int(r['in_stable_high_502'])],
            'stable_low778':[r for r in rows if int(r['in_stable_low_778'])]}
    assert [len(v) for v in groups.values()]==[1417,97,24,42,502,778]
    conditions=['full','r0_g000','r0_g050','r4_g000','r4_g050']
    report={group:{condition:stats(subset,condition) for condition in conditions} for group,subset in groups.items()}
    for group in ['strong97','suppression24','support42']:
        report[group+'_by_clip']={seq:{condition:stats([r for r in groups[group] if r['sequence']==seq],condition) for condition in conditions} for seq in sorted({r['sequence'] for r in groups[group]})}
    report['maximum_errors']={field:max(val(r,field) for r in rows) for field in rows[0] if 'error' in field}
    report['threshold_disagreements']={field:sum(int(r[field]) for r in rows) for field in rows[0] if 'replay_threshold_disagreement' in field}
    report['validity_checks']={'rows_unique_and_complete':True,'selection_unchanged':True,'original_replay':True,'prespecified_strata_counts':True,'full_equals_original_thresholds':True,'raw_input_hashes_and_runtime_in_full_metadata':True}
    report['sources']={'results_sha256':sha(RUN/'results.csv'),'protocol_sha256':sha(HERE/'PROTOCOL.json'),'summarizer_sha256':sha(Path(__file__))}
    (HERE/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    columns=['stratum','n','condition','a_ge05','b_ge05','a_gain05','a_loss05','b_gain05','b_loss05','a_ge09','b_ge09','both_ge05','both_lt05']
    with (HERE/'SUMMARY.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for group in groups:
            for condition in conditions:
                s=report[group][condition];a=s['endpoints']['a'];b=s['endpoints']['b']
                writer.writerow(dict(stratum=group,n=s['n'],condition=condition,a_ge05=a['0.5']['positive'],b_ge05=b['0.5']['positive'],
                    a_gain05=a['0.5']['gain'],a_loss05=a['0.5']['loss'],b_gain05=b['0.5']['gain'],b_loss05=b['0.5']['loss'],
                    a_ge09=a['ge09'],b_ge09=b['ge09'],both_ge05=s['paired']['0.5']['both_high'],both_lt05=s['paired']['0.5']['both_low']))
    print((HERE/'SUMMARY.csv').read_text())
    print(json.dumps({'maximum_errors':report['maximum_errors'],'threshold_disagreements':report['threshold_disagreements']}))

if __name__=='__main__':
    main()
