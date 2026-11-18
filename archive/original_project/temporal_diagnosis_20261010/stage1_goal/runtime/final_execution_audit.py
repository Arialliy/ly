"""Read-only final execution audit; writes only its report artifacts."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

STAGE=Path(__file__).resolve().parent.parent
BASE=STAGE.parent.parent
RUNTIME=STAGE/'runtime'
SOURCE=BASE/'official_deeppro'
EXPECTED_CALIBRATION='2329b4d59dc1eaf2a0cfe71b743af7e8c7e5d682d175c3a88e03b1fae96dde8e'
EXPECTED_CHECKPOINT='4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def command(args):
    return subprocess.check_output(args,text=True).strip()

def state_record(obj):
    state=obj['State']
    return {'container_id':obj['Id'],'name':obj['Name'].lstrip('/'),'status':state['Status'],
            'running':state['Running'],'exit_code':state['ExitCode'],'oom_killed':state['OOMKilled'],
            'error':state.get('Error',''),'started_at':state['StartedAt'],'finished_at':state['FinishedAt']}

def main():
    snapshot_path=BASE/'official_reproduction/SOURCE_SNAPSHOT.json'
    snapshot=json.loads(snapshot_path.read_text())
    head=command(['git','-C',str(SOURCE),'rev-parse','HEAD'])
    git_status=command(['git','-C',str(SOURCE),'status','--porcelain'])
    mismatches=[]
    for name,digest in snapshot['tracked_sha256'].items():
        path=SOURCE/name
        observed=sha(path) if path.is_file() else None
        if observed!=digest:mismatches.append({'path':name,'expected':digest,'observed':observed})
    checkpoint=SOURCE/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
    checkpoint_sha=sha(checkpoint)
    calibration_sha=sha(STAGE/'summary/CALIBRATION.json')
    records=[]
    for r in json.loads((RUNTIME/'TRAIN_LAUNCHES.json').read_text()):
        records.append({'kind':'training','variant':r['strategy'],'container_id':r['container_id'],'launch_record':str(RUNTIME/'TRAIN_LAUNCHES.json')})
    for path in sorted((RUNTIME/'evaluation_jobs').glob('*/*/*/LAUNCH.json')):
        r=json.loads(path.read_text())
        records.append({'kind':'calibration_'+r['kind'],'variant':r['variant'],'sequence':r['sequence'],'container_id':r['container_id'],'launch_record':str(path)})
    for r in json.loads((RUNTIME/'DIAGNOSTIC_LAUNCHES.json').read_text()):
        records.append({'kind':'development_infer','variant':r['variant'],'container_id':r['container_id'],'launch_record':str(RUNTIME/'DIAGNOSTIC_LAUNCHES.json')})
    for path in sorted((RUNTIME/'diagnostic_score_jobs').glob('*/*/LAUNCH.json')):
        r=json.loads(path.read_text())
        records.append({'kind':'development_score','variant':r['variant'],'sequence':r['sequence'],'container_id':r['container_id'],'launch_record':str(path)})
    counts=dict(collections.Counter(r['kind'] for r in records))
    assert counts=={'training':4,'calibration_infer':10,'calibration_score':10,'development_infer':2,'development_score':8},counts
    assert len({r['container_id'] for r in records})==34
    actual=json.loads(command(['docker','inspect']+[r['container_id'] for r in records]))
    lookup={o['Id']:o for o in actual}
    for r in records:r['observed']=state_record(lookup[r['container_id']])
    named_ids=command(['docker','ps','-a','--no-trunc','--filter','name=deeppro_stage1','--format','{{.ID}}']).splitlines()
    named=[state_record(o) for o in json.loads(command(['docker','inspect']+named_ids))]
    expected_ids={r['container_id'] for r in records}
    additional=[r for r in named if r['container_id'] not in expected_ids]
    confirmation=json.loads((STAGE/'DATA_SELECTION.json').read_text())['splits']['confirmation']
    absence={}
    for root_name in ['predictions','evaluation','context_evaluation']:
        root=STAGE/root_name
        all_paths=list(root.rglob('*')) if root.exists() else []
        absence[root_name]={'root_exists':root.exists(),'sequences':{}}
        for seq in confirmation:
            matches=[p for p in all_paths if seq in p.relative_to(root).parts]
            files=[str(p.relative_to(STAGE)) for p in matches if p.is_file() or p.is_symlink()]
            absence[root_name]['sequences'][seq]={'matching_paths':[str(p.relative_to(STAGE)) for p in matches],
                                               'file_or_symlink_count':len(files),'files_or_symlinks':files}
    active_named=[r for r in named if r['running']]
    gpu_snapshot=command(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'])
    gpu_processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'])
    all_running=command(['docker','ps','--format','{{.Names}}|{{.ID}}|{{.Status}}'])
    formal_ok=all(r['observed']['status']=='exited' and r['observed']['exit_code']==0 and not r['observed']['running'] for r in records)
    no_confirm=all(v['file_or_symlink_count']==0 for root in absence.values() for v in root['sequences'].values())
    unchanged=(head==snapshot['git_commit'] and not git_status and not mismatches and checkpoint_sha==EXPECTED_CHECKPOINT)
    status='PASS' if formal_ok and no_confirm and unchanged and calibration_sha==EXPECTED_CALIBRATION and not active_named else 'FAIL'
    report={'status':status,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope':'Final read-only execution-state audit; no training, inference, scoring, stopping or cleanup performed',
            'official_source':{'git_head':head,'expected_git_head':snapshot['git_commit'],'git_status_porcelain':git_status,
                               'snapshot_path':str(snapshot_path),'snapshot_sha256':sha(snapshot_path),'tracked_files_checked':len(snapshot['tracked_sha256']),
                               'tracked_hash_mismatches':mismatches,'checkpoint_path':str(checkpoint),'checkpoint_sha256':checkpoint_sha,'unchanged':unchanged},
            'formal_job_counts':counts,'formal_job_count':len(records),'all_formal_jobs_exited_zero':formal_ok,'formal_jobs':records,
            'additional_stage1_named_containers':additional,'additional_terminal_failure_count':sum(r['status']=='exited' and r['exit_code']!=0 for r in additional),
            'confirmation_sequences':confirmation,'confirmation_output_scan':absence,'no_confirmation_model_output_files_in_three_roots':no_confirm,
            'confirmation_boundary':'Raw images and data were prepared and visually inspected. No confirmation model prediction/evaluation result outputs exist in the three requested output roots; do not claim that the whole confirmation dataset was completely unseen.',
            'calibration_sha256':calibration_sha,'expected_calibration_sha256':EXPECTED_CALIBRATION,'calibration_unchanged':calibration_sha==EXPECTED_CALIBRATION,
            'active_stage1_named_containers':active_named,'gpu_snapshot_csv':gpu_snapshot,'gpu_compute_process_snapshot_csv':gpu_processes,
            'all_currently_running_containers_snapshot':all_running,'unrelated_containers_untouched':True,
            'auditor_script_sha256':sha(Path(__file__))}
    (RUNTIME/'FINAL_EXECUTION_STATUS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':status,'formal_job_count':len(records),'all_formal_jobs_exited_zero':formal_ok,'official_unchanged':unchanged,
                      'tracked_files_checked':len(snapshot['tracked_sha256']),'no_confirmation_outputs':no_confirm,
                      'additional_terminal_failure_count':report['additional_terminal_failure_count'],'active_stage1_containers':len(active_named),
                      'calibration_unchanged':report['calibration_unchanged']}))
    assert status=='PASS'

if __name__=='__main__':main()
