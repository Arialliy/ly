#!/usr/bin/env python3
"""Check frozen artifacts, local links, claim evidence and CPU job completion."""
import csv, datetime, hashlib, json, re, subprocess
from pathlib import Path
C=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def csvread(p):return list(csv.DictReader(p.open()))
def main():
    checks=[]
    def check(name,condition,detail=None):
        checks.append(dict(check=name,pass_=bool(condition),detail=detail))
        assert condition,(name,detail)
    freeze=read(C/'PROTOCOL_FREEZE.json')
    check('primary_protocol_hash',sha(C/'PROTOCOL.md')==freeze['protocol_sha256'])
    check('legal_pool_hash',sha(C/'LEGAL_COMBINATIONS.json')==freeze['legal_combinations_sha256'])
    supplement=read(C/'LOW_RESPONSE_GRID_SUPPLEMENT_FREEZE.json')
    check('supplement_prerun_hash',sha(C/'LOW_RESPONSE_GRID_SUPPLEMENT.md')==supplement['sha256'])
    for base in [C.parent/'problem_validity_20261011',C.parent/'failure_attribution_20261011',C/'scoring',C/'support',C/'low_response',C/'low_response/supplement']:
        manifest=read(base/'ARTIFACT_SHA256.json');files=manifest['files']
        if isinstance(files,dict):files=[dict(path=k,**(v if isinstance(v,dict) else {'sha256':v})) for k,v in files.items()]
        for r in files:
            p=Path(r['path']);p=p if p.is_absolute() else base/p
            assert sha(p)==r['sha256'],str(p)
        check('frozen_manifest_'+str(base.relative_to(C.parent)),True,dict(files=len(files),bytes=sum((base/r['path']).stat().st_size for r in files)))
    for relative in ['scoring/VALIDATION.json','scoring/ROOT_AGGREGATE_AUDIT.json','support/VERIFICATION.json',
        'support/INDEPENDENT_VERIFICATION.json','support/CLAIM_REVIEW.json','low_response/INDEPENDENT_VALIDATION.json',
        'low_response/supplement/VALIDATION.json','CROSS_CHECK.json']:
        d=read(C/relative)
        check('audit_'+relative,d.get('status')=='PASS',d.get('status'))
    rows=csvread(C/'SHARED_COMBINATIONS.csv');pairs=csvread(C/'PER_CLIP_COMBINATIONS.csv')
    check('exhaustive_10_100',len(rows)==10 and len(pairs)==100)
    check('actual_max_720_721',max(int(r['small_matched']) for r in rows)==720 and max(int(r['small_matched']) for r in pairs)==721)
    check('actual_zero_loss_707',max(int(r['small_matched']) for r in pairs if r['U_budget_feasible']=='True' and int(r['lost'])==0)==707)
    a=[r for r in csvread(C/'low_response/supplement/THRESHOLD_CURVES.csv') if r['group']=='pooled' and r['mode']=='S' and float(r['threshold'])==.44][0]
    check('exploratory_S044',all(int(a[k])==v for k,v in dict(small_matched=727,low_cohort_recovered=6,gained_vs_original05=80,lost_vs_original05=53,unmatched_components=4011,positive_pixels_outside_boxes=39110).items()),a)
    # Inspect only this round's named CPU jobs; do not affect other running workloads.
    names=['deeppro-combination-aggregate-20261011','deeppro-combination-crosscheck-20261011','deeppro-combination-aggregate-final-20261011']
    jobs=[]
    for name in names:
        d=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
        jobs.append(dict(name=name,status=d['State']['Status'],exit_code=d['State']['ExitCode'],started=d['State']['StartedAt'],finished=d['State']['FinishedAt'],
            image=d['Config']['Image'],command=d['Config']['Cmd'],cpus=d['HostConfig']['NanoCpus']/1e9,memory_bytes=d['HostConfig']['Memory']))
    check('root_CPU_jobs_exited_zero',all(j['status']=='exited' and j['exit_code']==0 for j in jobs),jobs)
    (C/'ROOT_RUNTIME.json').write_text(json.dumps(dict(jobs=jobs,notes=['Second aggregation only corrected two plot label offsets and replaced a pending-audit metadata label; numerical outputs unchanged.','Subdirectory CPU run logs and validations are recorded in their own RUNTIME files.']),indent=2)+'\n')
    # Reserved artifacts are written below; references to those are deliberate.
    reserved={'FINAL_VALIDATION.json','ARTIFACT_SHA256.json'}
    links=0
    for p in C.rglob('*.md'):
        for raw in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            target=raw.strip('<>')
            if target.startswith(('http:','https:','#','mailto:')):continue
            target=target.split('#')[0];target=re.sub(r':\d+$','',target)
            q=Path(target) if target.startswith('/') else p.parent/target
            check_exists=q.exists() or (q.parent==C and q.name in reserved)
            assert check_exists,(str(p),raw)
            links+=1
    check('local_report_links',True,links)
    out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,
        primary_scope='fixed .5 and frozen 11-window pool; 10 shared / 100 per-clip combinations exhaustive',
        supplementary_scope='exploratory O/S thresholds .41-.49, 9 fixed new points; not continuous-threshold exhaustive',
        official_Pd_Fa=False,new_model_method_improvement=False,independent_confirmation=False)
    (C/'FINAL_VALIDATION.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    status=dict(status='COMPLETE',task='fixed-pool five-window actual-combination feasibility',
        next_priority='persistent-low-response target/background separability',
        shared_optimal_fixed05=720,per_clip_hindsight_optimal_fixed05=721,zero_loss_per_clip_fixed05=707,
        exploratory_S044_matches=727,exploratory_S044_U=4011,low_group_recovered=6,
        causal_content_vs_slot='unresolved',new_method_validated=False,official_metric_replication=False)
    (C/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    entries=[]
    for p in sorted(C.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p==C/'ARTIFACT_SHA256.json':continue
        entries.append(dict(path=str(p.relative_to(C)),bytes=p.stat().st_size,sha256=sha(p)))
    (C/'ARTIFACT_SHA256.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),excludes=['root ARTIFACT_SHA256.json','__pycache__'],files=entries),indent=2)+'\n')
    print(json.dumps(dict(status='PASS',checks=len(checks),links=links,artifacts=len(entries),bytes=sum(e['bytes'] for e in entries)),indent=2))
if __name__=='__main__':main()
