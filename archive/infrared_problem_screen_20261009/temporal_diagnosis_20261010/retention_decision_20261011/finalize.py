#!/usr/bin/env python3
import csv,datetime,hashlib,json,re
from pathlib import Path
D=Path(__file__).resolve().parent;C=D.parent/'combination_feasibility_20261011'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def main():
    checks=[]
    def check(name,v,detail=None):
        assert v,(name,detail)
        checks.append(dict(name=name,passed=bool(v),detail=detail))
    hist=read(C/'ARTIFACT_SHA256.json')['files']
    for r in hist:assert sha(C/r['path'])==r['sha256'],r['path']
    check('all_previous_combination_artifacts_unchanged',True,len(hist))
    r=read(D/'SUMMARY.json');a=read(D/'independent/SUMMARY.json');b=read(D/'background/SUMMARY.json')
    check('independent_source_hashes_unchanged',a['source_unchanged'] and b['sources_unchanged'])
    for rootkey,agentkey in [('shared','shared'),('per_clip','per_clip_hindsight')]:
        check(rootkey+'_high_min_loss',r[rootkey]['high_min_loss']==a['queries'][agentkey]['M_ge_720']['minimum_lost'])
        check(rootkey+'_zero_loss_max_M',r[rootkey]['zero_loss_max_M']==a['queries'][agentkey]['zero_loss']['maximum_M'])
    check('exact_background_U_span',r['max_exact_M_L_U_span']==a['maximum_exact_M_lost_group_background_ranges']['unmatched_components']['range']==14)
    check('exact_background_outside_span',r['max_exact_M_L_outside_span']==a['maximum_exact_M_lost_group_background_ranges']['positive_pixels_outside_boxes']['range']==320)
    check('same720_same_small_match_set',b['C06_C03_same_small_match_set_as_S'])
    claim=read(D/'independent/CLAIM_REVIEW.json')
    check('independent_claim_review',claim['verdict']=='PASS')
    report=(D/'REPORT.md').read_text()
    check('review_wording_suggestions_resolved','显著少丢几十个' not in report and '下界达到722及以上则无可行组合' in report)
    original={(x['choice_0026'],x['choice_0067']):x for x in rows(C/'PER_CLIP_COMBINATIONS.csv')}
    numeric=['small_matched','gained','lost','unmatched_components','positive_pixels_outside_boxes','positive_pixels','component_count','matched_count']
    checked=0
    for p in D.glob('*.csv'):
        for x in rows(p):
            if not x.get('choice_0026'):continue
            source=original[(x['choice_0026'],x['choice_0067'])]
            for k in numeric:
                if k in x:assert int(x[k])==int(source[k]),(p.name,k,x,source);checked+=1
            assert int(source['unmatched_components'])<=4015
    check('all_output_chosen_rows_exist_with_exact_metric_vector',True,checked)
    links=0
    for p in D.rglob('*.md'):
        for raw in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            s=raw.strip('<>')
            if s.startswith(('http:','https:','#')):continue
            s=re.sub(r':\d+$','',s.split('#')[0]);target=Path(s) if s.startswith('/') else p.parent/s
            assert target.exists() or (target.parent==D and target.name in ['FINAL_VALIDATION.json','ARTIFACT_SHA256.json']),(p,raw)
            links+=1
    check('all_local_report_links',True,links)
    # The only array-reading auxiliary run aggregated existing NPZ and area tables.
    status=dict(status='COMPLETE',task='retention and background decisions from existing exhaustive tables',
        threshold=.5,U_budget=4015,new_inference=False,new_fusion=False,new_training=False,new_threshold_scan=False,
        adaptive_selection_current_pool='closed_as_mainline',finite_retention_space_preserved=True,
        next_main_diagnosis='persistent-low-response target vs hard-background separation',next_logit_diagnosis_executed=False)
    (D/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    validation=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,
        independent_area_checks=b['distribution_metrics_crosschecks'],independent_object_checks=b['selected_object_npz_checks'],
        note='Deterministic queries on two development clips. No statistical-significance or deployability claim.')
    (D/'FINAL_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
    files=[dict(path=str(p.relative_to(D)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p!=D/'ARTIFACT_SHA256.json']
    (D/'ARTIFACT_SHA256.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files),indent=2)+'\n')
    print(json.dumps(dict(status='PASS',checks=len(checks),links=links,files=len(files),bytes=sum(x['bytes'] for x in files)),indent=2))
if __name__=='__main__':main()
