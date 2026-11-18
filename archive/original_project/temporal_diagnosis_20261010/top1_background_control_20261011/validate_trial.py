#!/usr/bin/env python3
"""Seal one completed controlled experiment without changing previous runs."""
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def readcsv(name):
    with (BASE/name).open() as f:
        return list(csv.DictReader(f))


def main():
    frozen=json.loads((BASE/'PROTOCOL_FREEZE.json').read_text())
    assert sha(BASE/'PROTOCOL.md')==frozen['protocol_sha256']
    checks={}
    for name in ['audit/LOSS_MASK_AUDIT.json','audit/COMPLETED_AUDIT.json']:
        obj=json.loads((BASE/name).read_text())
        assert obj['status']=='PASS',(name,obj.get('status'))
        checks[name]='PASS'
    run=json.loads((BASE/'training/run/RUN.json').read_text())
    assert run['final_step']==512 and run['seed']==46 and run['effective_extra_lambda']==0
    assert run['bn_initial_sha256']==run['bn_final_sha256']
    logs=[json.loads(l) for l in (BASE/'training/run/TRAIN.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs]==list(range(1,513))
    assert all(r['finite'] and abs(r['negative']-r['background_top1pct_mean'])<1e-6 for r in logs)
    sizes={name:len(readcsv('evaluation/'+name)) for name in ['SUMMARY.csv','SEQUENCE_SUMMARY.csv','OBJECT_COMPARISON.csv','TRACK_SUMMARY.csv','RECOVERY_CONCENTRATION.csv','FRAME_COUNTS.csv']}
    assert sizes=={'SUMMARY.csv':6,'SEQUENCE_SUMMARY.csv':12,'OBJECT_COMPARISON.csv':10400,'TRACK_SUMMARY.csv':780,'RECOVERY_CONCENTRATION.csv':24,'FRAME_COUNTS.csv':1920},sizes
    objects=readcsv('evaluation/OBJECT_COMPARISON.csv')
    assert len({(r['sequence'],r['annotation_id'],r['frame'],r['arrangement']) for r in objects})==10400
    for ar in ['O','S']:
        rr=[r for r in objects if r['arrangement']==ar]
        assert len(rr)==5200 and sum(int(r['low_all_05']) for r in rr)==4342 and sum(int(r['extreme_all_001']) for r in rr)==4170
    broken=[]
    for name in ['REPORT.md','FINAL_INTERPRETATION.md']:
        f=BASE/name
        for dest in re.findall(r'\]\(([^)]+)\)',f.read_text()):
            if dest.startswith(('http:','https:','#')):
                continue
            if not (f.parent/dest.split('#')[0]).resolve().exists():
                broken.append(dest)
    assert not broken,broken
    outcome=json.loads((BASE/'OUTCOME.json').read_text())
    status=dict(status='COMPLETE',completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        experiment='one fixed top-1% negative-aggregation control',training_runs=1,seed=46,updates=512,
        threshold=0.5,top_fraction=0.01,checkpoint='training/run/final_model.pth',
        validation=checks,rows=sizes,outcomes=outcome['outcomes'],
        scope='custom weak-box adaptation on two existing development clips',
        no_sweep=True,no_additional_seed=True,no_threshold_retuning=True,
        broken_report_links=broken)
    (BASE/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    manifest={}
    for p in sorted(BASE.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.md','.py','.json','.jsonl','.csv','.diff'] and p.name!='ARTIFACT_SHA256.json':
            manifest[str(p.relative_to(BASE))]=sha(p)
    (BASE/'ARTIFACT_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'COMPLETE','rows':sizes,'audits':checks,'artifacts':len(manifest)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
