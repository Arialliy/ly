#!/usr/bin/env python3
"""Validate delivery consistency, without recomputing models or changing data."""
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = ROOT.parent / 'stage1_goal' / 'CVPR2027_RESEARCH_PLAN.md'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(rel):
    return json.loads((ROOT / rel).read_text())


checks = {}
freeze = read('PROTOCOL_FREEZE.json')
checks['frozen_protocol_hash'] = sha(ROOT / 'PROTOCOL.md') == freeze['protocol_sha256']
old = read('audit_old/SUMMARY.json')
checks['old_independent_join'] = all(x == 'PASS' for x in old['integrity_checks'].values())
for key, rel in [
    ('new_independent_join', 'audit_new/AUDIT.json'),
    ('control_independent_audit', 'audit_controls/SUMMARY.json'),
    ('supplemental_diagnostics', 'supplemental_diagnostics/RESULTS.json'),
    ('context_evidence', 'evidence_context/INTEGRITY_CHECK.json'),
    ('window_score_integrity', 'runtime/window_scoring/VERIFICATION.json'),
    ('control_score_integrity', 'simple_controls/OUTPUT_INTEGRITY_CHECK.json'),
]:
    checks[key] = read(rel)['status'] == 'PASS'
checks['new_audit_current_root_summary'] = read('audit_new/AUDIT.json')['root_summary_sha256'] == sha(ROOT / 'SUMMARY.json')
checks['root_summary_complete'] = read('SUMMARY.json')['status'] == 'COMPLETE'
claim = read('audit_controls/CLAIM_REVIEW.json')
checks['claim_review_pass'] = claim['status'] == 'PASS'
checks['claim_review_current_report'] = claim['root_report_sha256'] == sha(ROOT / 'REPORT.md')
checks['claim_review_current_plan'] = claim['plan_sha256'] == sha(PLAN)
checks['row_counts'] = (read('SUMMARY.json')['observations'], read('SUMMARY.json')['window_observations']) == (21270, 58686)
csv_names = ['TABLE.csv', 'OBJECTS.csv', 'WINDOWS.csv', 'TRACKS.csv', 'CONDITIONS.csv', 'REMAINING_OPPORTUNITIES.csv']
checks['cached_npz_revision_csv_identity'] = all(sha(ROOT / f) == sha(ROOT / 'join_v1' / f) for f in csv_names)

docs = [ROOT / 'REPORT.md', ROOT / 'MANIFEST.md', PLAN]
missing_links = []
checked_links = 0
for doc in docs:
    for link in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
        if '://' in link or link.startswith('#'):
            continue
        path = link.split('#', 1)[0].strip('<>')
        # Final manifest files are produced immediately after this check.
        if doc.name == 'MANIFEST.md' and path in ('FINAL_VALIDATION.json', 'ARTIFACT_SHA256.json'):
            continue
        target = Path(path) if path.startswith('/') else doc.parent / path
        checked_links += 1
        if not target.exists():
            missing_links.append({'doc': str(doc), 'link': link})
checks['local_document_links'] = not missing_links

names = [
    'deeppro_validity_window_score_dji_0026_2',
    'deeppro_validity_window_score_dji_0067_2',
    'deeppro_problem_validity_controls_dji_0026_2_20261011',
    'deeppro_problem_validity_controls_dji_0067_2_20261011',
    'deeppro_validity_join_20261011',
    'deeppro_validity_join_v2_20261011',
    'deeppro_validity_join_v3_20261011',
    'deeppro_validity_control_summary_20261011',
]
jobs = []
for name in names:
    d = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    expected_exit = 1 if name == 'deeppro_validity_join_v2_20261011' else 0
    state = d['State']
    devices = d['HostConfig'].get('DeviceRequests')
    passed = state['Status'] == 'exited' and state['ExitCode'] == expected_exit and not state['OOMKilled'] and not devices
    jobs.append({'name': name, 'status': state['Status'], 'exit': state['ExitCode'], 'expected_exit': expected_exit, 'gpu_devices': devices, 'pass': passed})
checks['task_containers_exited_as_recorded_cpu_only'] = all(j['pass'] for j in jobs)

report = {
    'utc': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS' if all(checks.values()) else 'FAIL',
    'checks': checks,
    'checked_local_links': checked_links,
    'missing_links': missing_links,
    'jobs': jobs,
    'report_sha256': sha(ROOT / 'REPORT.md'),
    'plan_sha256': sha(PLAN),
    'plan_archive_sha256': sha(PLAN.with_name('CVPR2027_RESEARCH_PLAN_before_problem_validity_20261011.md')),
    'scope': 'Consistency check; independent numerical audits remain the evidence for scientific counts. No new inference/training, no confirmation output evaluated.',
}
(ROOT / 'FINAL_VALIDATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'checks': checks, 'missing_links': missing_links}, ensure_ascii=False))
if report['status'] != 'PASS':
    raise SystemExit(1)
