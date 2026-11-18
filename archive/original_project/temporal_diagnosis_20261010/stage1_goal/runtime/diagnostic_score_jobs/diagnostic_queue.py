"""CPU-only old-development diagnostic scoring, no confirmation access."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent.parent
spec=importlib.util.spec_from_file_location('calibration_job_helpers',STAGE/'runtime/evaluation_jobs/calibration_queue.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
VARIANTS=['paired_mean','paired_max']
SEQUENCES=['DJI_0028_2','DJI_0028_4','DJI_0032_5','DJI_0047_3']
CALIBRATION_SHA='2329b4d59dc1eaf2a0cfe71b743af7e8c7e5d682d175c3a88e03b1fae96dde8e'

def event(kind,**kw):
    d={'utc':helpers.now(),'kind':kind,**kw}
    with (HERE/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
    print(json.dumps(d),flush=True)

def verify_predictions():
    assert helpers.sha(STAGE/'summary/CALIBRATION.json')==CALIBRATION_SHA
    launches=json.loads((STAGE/'runtime/DIAGNOSTIC_LAUNCHES.json').read_text())
    report={'status':'VERIFYING','calibration_sha256':CALIBRATION_SHA,'scope':'old development sequences only','containers':{},'predictions':[]}
    for launch in launches:
        variant=launch['variant'];assert variant in VARIANTS
        obs=helpers.inspect(launch['container_id'])
        assert 'inspection_error' not in obs and obs['state']['Status']=='exited' and obs['state']['ExitCode']==0
        marker=json.loads((STAGE/'runtime'/('DIAGNOSTIC_INFERENCE_'+variant+'.json')).read_text())
        assert marker['status']=='COMPLETE' and marker['completed']==SEQUENCES
        report['containers'][variant]=obs
        stdout=helpers.call(['docker','logs',launch['container_id']],check=False)
        (HERE/('inference_'+variant+'_stdout.log')).write_text(stdout.stdout)
        (HERE/('inference_'+variant+'_stderr.log')).write_text(stdout.stderr)
        expected_ckpt=helpers.sha(STAGE/'runs'/variant/'final_model.pth')
        for seq in SEQUENCES:
            pred=STAGE/'diagnostic_predictions'/variant/seq
            cp=json.loads((pred/'COMPLETE.json').read_text())
            assert cp['status']=='COMPLETE' and cp['sequence']==seq
            assert cp['checkpoint_sha256']==expected_ckpt
            assert cp['raw_sha256']==helpers.sha(STAGE/'data_diagnostics'/seq/'raw_frames.npy')
            assert cp['official_starts']==[0,36,72,108,120] and cp['diagnostic_starts']==list(range(0,121,12))
            assert len(cp['window_sha256'])==11
            assert cp['fused_sha256']==helpers.sha(pred/'official_fused.npz')
            for name,digest in cp['window_sha256'].items():assert digest==helpers.sha(pred/'windows'/name)
            report['predictions'].append({'variant':variant,'sequence':seq,'completion_sha256':helpers.sha(pred/'COMPLETE.json'),
                                          'checkpoint_sha256':expected_ckpt,'raw_sha256':cp['raw_sha256'],'all_window_and_fused_hashes_verified':True})
    assert set(report['containers'])==set(VARIANTS) and len(report['predictions'])==8
    report['status']='PASS';helpers.write(HERE/'INFERENCE_VERIFIED.json',report)
    event('inference_verification_pass',containers=2,predictions=8)

def launch(variant,seq,score_sha):
    assert helpers.sha(STAGE/'pilot/score_pilot.py')==score_sha
    out=STAGE/'diagnostic_evaluation'/variant/seq
    if out.exists() and any(out.iterdir()):raise RuntimeError('Existing score outputs require handle audit, no overwrite: '+str(out))
    folder=HERE/variant/seq;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'LAUNCH.json').exists():raise RuntimeError('Existing job handle; never restart automatically')
    name=f'deeppro_stage1_diag_score_{variant}_{seq.lower()}'
    cmd=helpers.docker_base(name)+[str(HERE/'score_pair.py'),'--variant',variant,'--sequence',seq,'--expected-scoring-sha256',score_sha]
    rec={'variant':variant,'sequence':seq,'command':cmd,'gpu':None,'created_utc':helpers.now(),'folder':str(folder),'status':'launching'}
    helpers.write(folder/'LAUNCH.json',rec)
    r=helpers.call(cmd,check=False)
    rec.update(launch_exit=r.returncode,stdout=r.stdout,stderr=r.stderr)
    if r.returncode:helpers.write(folder/'LAUNCH.json',rec);raise RuntimeError(rec)
    rec.update(container_id=r.stdout.strip(),status='running')
    helpers.write(folder/'LAUNCH.json',rec)
    event('launch',variant=variant,sequence=seq,container_id=rec['container_id'])
    return rec

def collect(job):
    folder=Path(job['folder']);obs=helpers.inspect(job['container_id']);helpers.write(folder/'LAST_INSPECT.json',obs)
    if 'inspection_error' in obs or obs['state']['Running']:return False
    logs=helpers.call(['docker','logs',job['container_id']],check=False)
    (folder/'stdout.log').write_text(logs.stdout);(folder/'stderr.log').write_text(logs.stderr)
    helpers.write(folder/'EXIT.json',obs)
    assert obs['state']['Status']=='exited' and obs['state']['ExitCode']==0
    out=STAGE/'diagnostic_evaluation'/job['variant']/job['sequence']
    expected=['official_fused']+[f'window_{s:03d}' for s in range(0,121,12)]
    completions={}
    for name in expected:
        cp=out/(name+'_COMPLETE.json');d=json.loads(cp.read_text())
        assert d['status']=='COMPLETE' and d['name']==name
        if name=='official_fused':assert len(d['thresholds'])==23 and d['official_fusion_exactly_reconstructed']
        else:assert d['thresholds']==[0.5]
        completions[name]=helpers.sha(cp)
    job.update(status='complete',exit_code=0,completions_sha256=completions)
    helpers.write(folder/'VERIFIED.json',job)
    event('complete',variant=job['variant'],sequence=job['sequence'],container_id=job['container_id'])
    return True

def main():
    if (HERE/'QUEUE.json').exists():raise RuntimeError('Existing queue state must be audited rather than overwritten')
    verify_predictions()
    score_sha=helpers.sha(STAGE/'pilot/score_pilot.py')
    pending=[[v,s] for v in VARIANTS for s in SEQUENCES]
    state={'status':'running','pid':os.getpid(),'created_utc':helpers.now(),'pending':pending,'active':[],'complete':[],
           'score_sha256':score_sha,'calibration_sha256':CALIBRATION_SHA,'context_threshold':0.5,'interpretation':'raw0.5 diagnostic, not calibrated operating point'}
    while True:
        for job in state['active'][:]:
            if collect(job):state['complete'].append(job);state['active'].remove(job)
        while state['pending'] and len(state['active'])<4:
            variant,seq=state['pending'].pop(0);state['active'].append(launch(variant,seq,score_sha))
        state['last_poll_utc']=helpers.now()
        if len(state['complete'])==8:
            assert helpers.sha(STAGE/'summary/CALIBRATION.json')==CALIBRATION_SHA
            state.update(status='complete',completed_utc=helpers.now());helpers.write(HERE/'QUEUE.json',state)
            event('queue_complete',jobs=8);return
        helpers.write(HERE/'QUEUE.json',state);time.sleep(5)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        helpers.write(HERE/'QUEUE_ERROR.json',{'utc':helpers.now(),'error':repr(error),'note':'No active container terminated/restarted. Inspect handles before recovery.'})
        raise
