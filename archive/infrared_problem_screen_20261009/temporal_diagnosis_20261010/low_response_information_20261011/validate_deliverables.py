#!/usr/bin/env python3
"""Final deliverable/source consistency checks; no re-inference or score changes."""
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent


def rows(name):
    with (BASE/name).open() as f:
        return list(csv.DictReader(f))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    expected={'LOW_RESPONSE_OBJECTS.csv':10400,'SCORE_SEPARATION.csv':42,
              'PEAK_LOCATION.csv':42,'TRACK_PERSISTENCE.csv':260,'TRACK_SUMMARY.csv':42}
    counts={name:len(rows(name)) for name in expected}
    assert counts==expected
    source=rows('analysis/LOW_RESPONSE_OBJECTS.csv')
    final=rows('LOW_RESPONSE_OBJECTS.csv')
    assert all(all(a[k]==b[k] for k in a) for a,b in zip(source,final))
    assert sum(r['visual_reviewed']=='1' for r in final)==144
    assert len({(r['sequence'],r['track_id'],r['frame']) for r in final if r['visual_reviewed']=='1'})==72
    assert (BASE/'SCORE_SEPARATION.csv').read_bytes()==(BASE/'analysis/SCORE_SEPARATION.csv').read_bytes()
    auditfiles=['analysis/VALIDATION.json','analysis/INDEPENDENT_BLOCK_AUDIT.json',
                'ASSEMBLY_VALIDATION.json','audit/OBJECT_RECOMPUTATION.json',
                'audit/FINAL_TABLES_AUDIT.json','analysis/INSUFFICIENT_BACKGROUND_ARCHIVE.json',
                'lambda0_control/audit/CONTROL_AUDIT.json']
    audit={}
    for name in auditfiles:
        d=json.loads((BASE/name).read_text())
        assert d['status']=='PASS', name
        audit[name]='PASS'
    broken=[]
    for name in ['REPORT.md','INTERVENTION_DECISION.md']:
        f=BASE/name
        for target in re.findall(r'\]\(([^)]+)\)',f.read_text()):
            if target.startswith(('http:','https:','#')):
                continue
            p=(f.parent/target.split('#')[0]).resolve()
            if not p.exists():
                broken.append(str(p))
    assert not broken, broken
    decision=json.loads((BASE/'NEXT_INTERVENTION.json').read_text())
    assert decision['selected_intervention_count']==1 and decision['status']=='selected_not_run'
    status=dict(status='COMPLETE',verified_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        rows=counts,numeric_values_unchanged=True,visual_unique_target_frames=72,visual_arrangement_rows=144,
        threshold=0.5,original_matches=700,best_shared_matches=720,
        independent_audits=audit,broken_report_links=broken,
        selected_intervention=decision['name'],selected_intervention_count=1,
        unclaimed=['official Pd/Fa','method gain','causal temporal mechanism','generalization beyond the two development clips'])
    (BASE/'FINAL_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    # Text/source manifests are small; raw logits/checkpoints have their own hashed manifests.
    manifest={}
    for p in sorted(BASE.rglob('*')):
        if p.is_file() and p.suffix in ['.md','.py','.json','.csv','.diff'] and '__pycache__' not in p.parts and p.name!='DELIVERABLE_SHA256.json':
            manifest[str(p.relative_to(BASE))]=sha(p)
    (BASE/'DELIVERABLE_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
