#!/usr/bin/env python3
"""Check frozen protocols, audits, links, evidence counts, and CPU job exits."""
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

F=Path(__file__).resolve().parent
V=F.parent/'problem_validity_20261011'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads((F/p).read_text())

checks={}
checks['main_protocol_frozen']=read('PROTOCOL_FREEZE.json')['sha256']==sha(F/'PROTOCOL.md')
checks['replacement_protocol_frozen']=read('ONE_REPLACEMENT_FREEZE.json')['sha256']==sha(F/'ONE_REPLACEMENT_PROTOCOL.md')
checks['adapted_metric_protocol_frozen']=read('metrics/PROTOCOL_FREEZE.json')['protocol_sha256']==sha(F/'metrics/ADAPTED_METRIC_PROTOCOL.md')
audits=['stages/VERIFICATION.json','conditions/INDEPENDENT_AUDIT.json','metrics/RAW_VERIFICATION.json',
        'audit/AUDIT.json','one_replacement/TABLE_CHECK.json','visuals/ALL_FAILURE_REVIEW_VALIDATION.json','audit/CLAIM_REVIEW.json']
for p in audits:checks[p]=read(p)['status']=='PASS'
claim=read('audit/CLAIM_REVIEW.json')
checks['reviewed_current_report']=claim['report_sha256']==sha(F/'REPORT.md')
checks['reviewed_current_route']=claim['route_decision_sha256']==sha(F/'ROUTE_DECISION.md')

rows=list(csv.DictReader((F/'stages/ERROR_STAGE_TABLE.csv').open()))
checks['5200_object_rows']=len(rows)==5200
observed={}
for cohort in ['gain','loss','both_success']:
    rr=[r for r in rows if r['cohort']==cohort]
    observed[cohort]={'n':len(rr),
        'coverage_transitions':dict(Counter(r['official_coverage']+'->'+r['symmetric_coverage'] for r in rr)),
        'latest_access_delta':dict(Counter(str(int(r['symmetric_latest_access_source'])-int(r['official_latest_access_source'])) for r in rr))}
checks['coverage_descriptions_recomputed']=observed==read('COVERAGE_DIAGNOSTIC.json')
checks['all128_atlas_review']=len(list(csv.DictReader((F/'visuals/EXCHANGE_VISUAL_REVIEW.csv').open())))==128
checks['all16_failure_visual_review']=len(list(csv.DictReader((F/'visuals/ALL_FAILURE_REVIEW.csv').open())))==16
checks['four_stable_visual_review']=len(list(csv.DictReader((F/'visuals/STABLE_REFERENCE_REVIEW.csv').open())))==4

missing=[];links=0
for p in F.rglob('*.md'):
    for url in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        if '://' in url or url.startswith('#'):continue
        raw=url.split('#',1)[0].strip('<>')
        if p.name=='MANIFEST.md' and raw in ['FINAL_VALIDATION.json','ARTIFACT_SHA256.json','STATUS.json']:continue
        target=Path(raw) if raw.startswith('/') else p.parent/raw
        links+=1
        if not target.exists():missing.append({'document':str(p.relative_to(F)),'target':url})
checks['all_markdown_local_links_exist']=not missing

old_changed=[]
for d in json.loads((V/'ARTIFACT_SHA256.json').read_text())['files']:
    p=V/d['path']
    if not p.exists() or sha(p)!=d['sha256']:old_changed.append(d['path'])
checks['previous_delivery_unchanged']=not old_changed

names=[
    'deeppro_failure_visuals_20261011','deeppro_failure_visuals_extra_20261011','deeppro_failure_visuals_stable_20261011',
    'deeppro_failure_stage_audit_20261011','deeppro_failure_stage_figures_20261011','deeppro_failure_stage_verification_20261011',
    'deeppro_failure_metric_audit_20261011','deeppro_failure_metric_verify_20261011',
    'deeppro_failure_one_replacement_20261011','deeppro_failure_replacement_trace_20261011','deeppro_failure_cross_audit_20261011']
jobs=[]
for n in names:
    d=json.loads(subprocess.check_output(['docker','inspect',n]))[0]
    state=d['State'];gpu=d['HostConfig'].get('DeviceRequests')
    jobs.append({'name':n,'state':state,'gpu':gpu,'image':d['Image'],'command':d['Config']['Cmd'],
        'pass':state['Status']=='exited' and state['ExitCode']==0 and not state['OOMKilled'] and not gpu})
checks['eleven_task_cpu_containers_exited_zero']=all(j['pass'] for j in jobs)
result={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS' if all(checks.values()) else 'FAIL',
        'checks':checks,'checked_links':links,'missing_links':missing,'changed_previous_artifacts':old_changed,'jobs':jobs,
        'report_sha256':sha(F/'REPORT.md'),'route_sha256':sha(F/'ROUTE_DECISION.md'),
        'scope':'Delivery consistency; numerical/scientific validity rests on linked raw data audits, not this status file.'}
(F/'FINAL_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['jobs']},ensure_ascii=False))
if result['status']!='PASS':raise SystemExit(1)
