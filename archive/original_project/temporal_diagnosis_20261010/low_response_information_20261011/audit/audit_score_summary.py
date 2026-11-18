#!/usr/bin/env python3
"""Independently reconstruct every field of Table 1 from all detailed rows."""
import csv,hashlib,json,math,time
from pathlib import Path

OUT=Path(__file__).resolve().parent;RUN=OUT.parent

def read(p):
    with p.open() as f:return list(csv.DictReader(f))

def quantile(v,q):
    if not v:return ''
    a=sorted(v);p=(len(a)-1)*q;i=int(p);r=p-i
    return a[i]*(1-r)+a[min(i+1,len(a)-1)]*r

def total_tracks(rows):return len({(r['sequence'],r['track_id']) for r in rows})
def total_clips(rows):return len({r['sequence'] for r in rows})

def main():
    rows=read(RUN/'analysis/LOW_RESPONSE_OBJECTS.csv');table=read(RUN/'analysis/SCORE_SEPARATION.csv')
    frozen=read(RUN/'COHORTS.csv');fk={(r['sequence'],r['frame'],r['annotation_id']):r for r in frozen}
    detailkeys=set();errors=[];checks=0;largest=0
    flags=['low_all_05','extreme_all_001','low_nonextreme','both_fixed_success','extra_only','high_unmatched']
    for row in rows:
        ident=(row['sequence'],row['frame'],row['annotation_id']);d=(*ident,row['arrangement']);assert d not in detailkeys;detailkeys.add(d)
        c=fk[ident]
        for field in flags+['track_id','source_frame','bbox_x','bbox_y','bbox_w','bbox_h']:
            assert float(row[field])==float(c[field]),(d,field)
        assert row['matched_at_05']==c['original_matched' if row['arrangement']=='O' else 'symmetric_matched']
        assert int(row['extreme_all_001'])<=int(row['low_all_05'])
    assert len(detailkeys)==10400 and {r[-1] for r in detailkeys}=={'O','S'}
    for actual in table:
        seq=actual['sequence'];plan=actual['arrangement'];group=actual['group']
        rs=[r for r in rows if r['arrangement']==plan and (seq=='pooled' or r['sequence']==seq) and (group=='all' or r[group]=='1')]
        both=[r for r in rs if min(int(r['local_valid_blocks']),int(r['global_valid_blocks']))>=20]
        nonempty=[r for r in rs if min(int(r['local_valid_blocks']),int(r['global_valid_blocks']))>0]
        exp={'arrangement':plan,'sequence':seq,'group':group,'target_frames':len(rs),'tracks':total_tracks(rs),'clips':total_clips(rs),
             'valid_both_backgrounds':len(both),'nonempty_both_backgrounds':len(nonempty),'insufficient_background_rows':len(rs)-len(both),
             'matched_at_05':sum(int(r['matched_at_05']) for r in rs)}
        for scope in ['local','global']:
            usable=[r for r in rs if int(r[scope+'_valid_blocks'])>=20]
            present=[r for r in rs if int(r[scope+'_valid_blocks'])>0]
            exp[scope+'_valid_ge20_rows']=len(usable);exp[scope+'_nonempty_rows']=len(present)
            exp[scope+'_insufficient_1_to_19_rows']=len(present)-len(usable);exp[scope+'_missing_0_rows']=len(rs)-len(present)
            for score in ['margin_q95','margin_q99','target_midrank_percentile','blocks_stronger_than_target']:
                vals=[float(r[scope+'_'+score]) for r in usable]
                for label,q in [('q25',.25),('median',.5),('q75',.75)]:exp[scope+'_'+score+'_'+label]=quantile(vals,q)
            for quant in ['q95','q99']:
                for label,rr in [('valid_ge20',usable),('all_nonempty',present)]:
                    n=sum(r[scope+'_advantage_'+quant]=='1' for r in rr)
                    exp[scope+'_advantage_'+quant+'_'+label+'_count']=n
                    exp[scope+'_advantage_'+quant+'_'+label+'_rate']=n/len(rr) if rr else ''
        categories={(1,1):'both',(1,0):'local_only',(0,1):'global_only',(0,0):'neither'}
        for category,label in categories.items():
            for qualifier,rr in [('valid_ge20',both),('all_nonempty',nonempty)]:
                selected=[r for r in rr if (int(r['local_advantage_q95']),int(r['global_advantage_q95']))==category]
                for measure,value in [('frames',len(selected)),('tracks',total_tracks(selected)),('clips',total_clips(selected))]:
                    exp[label+'_advantage_q95_'+qualifier+'_'+measure]=value
        for score in ['target_box_max_logit','global_background_peaks_stronger_than_target','global_background_peaks_not_lower_than_target']:
            vals=[float(r[score]) for r in rs]
            for label,q in [('q25',.25),('median',.5),('q75',.75)]:exp[score+'_'+label]=quantile(vals,q)
        assert set(exp)==set(actual),(set(exp)-set(actual),set(actual)-set(exp))
        for field,value in exp.items():
            checks+=1
            if isinstance(value,(int,float)):
                diff=abs(float(actual[field])-value);largest=max(largest,diff);ok=diff<1e-8
            else:ok=actual[field]==value
            if not ok:errors.append({'sequence':seq,'arrangement':plan,'group':group,'column':field,'actual':actual[field],'expected':value})
    assert len(table)==42
    report={'status':'PASS' if not errors else 'FAIL','detailed_rows_checked':len(rows),'frozen_cohort_identity_and_matching_checked':True,
            'summary_rows':len(table),'summary_cells':checks,'largest_numeric_difference':largest,'mismatches':errors,
            'main_denominator':'At least 20 blocks in each applicable scope; joint categories require both scopes >=20',
            'missing_denominators_reported':True,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),RUN/'analysis/LOW_RESPONSE_OBJECTS.csv',RUN/'analysis/SCORE_SEPARATION.csv',RUN/'COHORTS.csv']}}
    (OUT/'SCORE_SUMMARY_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'rows':len(table),'cells':checks,'mismatches':len(errors)}))
    if errors:raise AssertionError(errors[:5])

if __name__=='__main__':main()
