#!/usr/bin/env python3
"""Final arithmetic / scope review; no model execution or image-wide rescoring."""
from pathlib import Path
import collections,csv,hashlib,json,datetime
O=Path(__file__).resolve().parent;F=O.parent;Q=F.parent;V=Q/'problem_validity_20261011'
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def key(r):return r['sequence'],int(r['frame']),int(r['annotation_id'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plans={'O':[0,36,72,108,120],'I1':[0,36,72,96,120],'I2':[0,36,60,96,120],'S':[0,24,60,96,120]}
    for plan,starts in plans.items():
        coverage=[sum(s<=t<s+40 for s in starts) for t in range(160)]
        assert min(coverage)==1 and sum(coverage)==200
    assert [t for t in range(160) if not any(s<=t<s+40 for s in [0,24,72,108,120])]==list(range(64,72))
    prior={key(r):r for r in read(V/'control_summary/OBJECTS.csv') if r['mode']=='symmetric5_probmax' and r['small_geometry']=='1'}
    expected=json.loads((F/'COVERAGE_DIAGNOSTIC.json').read_text());coverage={}
    for group in ['gain','loss','both_success']:
        rows=[r for r in prior.values() if ((group=='gain' and r['gain']=='1') or (group=='loss' and r['loss']=='1') or (group=='both_success' and r['official_matched']=='1' and r['control_matched']=='1'))]
        counts=collections.Counter();latest=collections.Counter()
        for r in rows:
            t=int(r['frame']);orig=[s for s in plans['O'] if s<=t<s+40];sym=[s for s in plans['S'] if s<=t<s+40]
            counts[f'{len(orig)}->{len(sym)}']+=1;latest[str(max(sym)-max(orig))]+=1
        got={'n':len(rows),'coverage_transitions':dict(counts),'latest_access_delta':dict(latest)}
        assert got==expected[group],group;coverage[group]=got
    # All object table endpoint states agree with previous frozen control table.
    rows=read(F/'one_replacement/OBJECTS.csv');small=[r for r in rows if r['small_geometry']=='1'];assert len(small)==5200
    for r in small:
        p=prior[key(r)]
        assert r['O_matched']==p['official_matched'] and r['S_matched']==p['control_matched']
        assert r['four_states']==''.join(r[plan+'_matched'] for plan in plans)
        for plan,starts in plans.items():
            expected_starts=[s for s in starts if s<=int(r['frame'])<s+40]
            assert json.loads(r[plan+'_selected_starts'])==expected_starts
            assert int(r[plan+'_coverage'])==len(expected_starts)
    patterns=collections.Counter(r['four_states'] for r in small)
    assert patterns=={'1111':646,'0000':4426,'0111':32,'0011':16,'0001':26,'1000':19,'1100':21,'1110':14}
    steps=[]
    for before,after in [('O','I1'),('I1','I2'),('I2','S')]:
        assert len(set(plans[before])&set(plans[after]))==4
        gain=[r for r in small if r[before+'_matched']=='0' and r[after+'_matched']=='1']
        loss=[r for r in small if r[before+'_matched']=='1' and r[after+'_matched']=='0']
        steps.append({'before':before,'after':after,'gained':len(gain),'lost':len(loss),'net':len(gain)-len(loss)})
    assert [(r['gained'],r['lost']) for r in steps]==[(32,19),(16,21),(26,14)]
    table=read(F/'one_replacement/METRICS.csv')
    for r in table:
        rr=[o for o in rows if r['group']=='pooled' or o['sequence']==r['group']];ss=[o for o in rr if o['small_geometry']=='1']
        plan=r['plan'];ref=r['reference']
        assert len(rr)==int(r['gt_count']) and len(ss)==int(r['small_gt'])
        assert sum(int(o[plan+'_matched']) for o in rr)==int(r['matched_count'])
        assert sum(int(o[plan+'_matched']) for o in ss)==int(r['small_matched'])
        assert sum(o[plan+'_matched']=='1' and o[ref+'_matched']=='0' for o in ss)==int(r['gained'])
        assert sum(o[plan+'_matched']=='0' and o[ref+'_matched']=='1' for o in ss)==int(r['lost'])
        assert int(r['component_count'])-int(r['matched_count'])==int(r['unmatched_components'])
    trace=json.loads((F/'one_replacement/INTERMEDIATE_FUSION_TRACE.json').read_text())
    case=[r for r in small if r['sequence']=='DJI_0067_2' and r['frame']=='106' and r['track_id']=='7821902'];assert len(case)==1;case=case[0]
    assert case['four_states']=='1000'
    assert json.loads(case['O_successful_starts'])==[72] and json.loads(case['I1_successful_starts'])==[72]
    assert json.loads(case['S_successful_starts'])==[] and 72 not in json.loads(case['S_selected_starts'])
    assert trace['direct_multicomponent_merge'] and trace['source_foreground_preserved']
    assert trace['fused_match']['eligible_components']==0 and trace['fused_component']['assigned_track_id']==7701616
    # Explicit report scope gates; textual audit below accompanies numerical checks.
    report=(F/'REPORT.md').read_text();route=(F/'ROUTE_DECISION.md').read_text()
    required=['695例只有1个可用诊断窗','合计29条','这5例不能作为官方Pd因max融合下降的证据','不能称相同官方Fa下已经改善','不能扩写为任意安排都没有融合失败']
    assert all(s in report for s in required)
    assert '不是已证明的新模型idea' in route and '尚未证明' in route
    visual=json.loads((F/'visuals/ALL_FAILURE_REVIEW_RECORD.json').read_text())
    assert visual['full_failure_images_viewed']==16 and visual['stable_reference_images_viewed']==4
    assert len(read(F/'visuals/ALL_FAILURE_REVIEW.csv'))==16 and len(read(F/'visuals/STABLE_REFERENCE_REVIEW.csv'))==4
    assert '不由16例推算总体比例' in report
    result={'status':'PASS','reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'report_sha256':sha(F/'REPORT.md'),'route_decision_sha256':sha(F/'ROUTE_DECISION.md'),'scope':'Read-only final claim audit; no model execution or 640-frame rescoring; no plan or frozen reports modified','coverage_independent':coverage,'state_patterns':dict(patterns),'step_counts':steps,'all_18_metric_table_rows_object_counts_and_delta_fields_reconciled':True,'I1_retained_success_fusion_loss_and_endpoint_success_removed_consistent':True,'reviewed_case':case,'required_scientific_boundaries_present':required,'visual_claim_review':'Latest paragraph agrees with the 16+4 qualitative review ledger; no population/cause/visibility inference. This auditor read ledger and report, did not independently reopen images.','blocking_findings':[],'minor_wording_findings':(['49/33/36 share denominator 4391; recommend outputs/statistical scopes rather than denominators'] if '三个分母口径' in report else []),'source_hashes':{str(p):sha(p) for p in [V/'control_summary/OBJECTS.csv',F/'one_replacement/OBJECTS.csv',F/'one_replacement/METRICS.csv',F/'one_replacement/INTERMEDIATE_FUSION_TRACE.json',F/'COVERAGE_DIAGNOSTIC.json',F/'audit/AUDIT.json',F/'visuals/ALL_FAILURE_REVIEW.md',F/'visuals/ALL_FAILURE_REVIEW_RECORD.json',F/'visuals/ALL_FAILURE_REVIEW.csv',F/'visuals/STABLE_REFERENCE_REVIEW.csv']},'script_sha256':sha(Path(__file__))}
    (O/'CLAIM_REVIEW.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'PASS','patterns':dict(patterns),'steps':steps,'minor_wording_findings':result['minor_wording_findings']},ensure_ascii=False))
if __name__=='__main__':main()
