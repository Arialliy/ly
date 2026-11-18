"""Verify fixed old-development scoring outputs, without choosing an operating point."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent.parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def main():
    state=json.loads((HERE/'QUEUE.json').read_text())
    assert state['status']=='complete' and len(state['complete'])==8
    assert not state['active'] and not state['pending']
    assert state['calibration_sha256']==sha(STAGE/'summary/CALIBRATION.json')
    inference=json.loads((HERE/'INFERENCE_VERIFIED.json').read_text())
    assert inference['status']=='PASS'
    for variant,obs in inference['containers'].items():
        actual=json.loads(subprocess.check_output(['docker','inspect',obs['id']],text=True))[0]
        assert actual['State']['Status']=='exited' and actual['State']['ExitCode']==0
    records=[]
    for job in state['complete']:
        variant,seq=job['variant'],job['sequence']
        actual=json.loads(subprocess.check_output(['docker','inspect',job['container_id']],text=True))[0]
        assert actual['State']['Status']=='exited' and actual['State']['ExitCode']==0
        assert not actual['HostConfig']['DeviceRequests']
        pred=STAGE/'diagnostic_predictions'/variant/seq
        source=json.loads((pred/'COMPLETE.json').read_text())
        matched=[r for r in inference['predictions'] if r['variant']==variant and r['sequence']==seq]
        assert len(matched)==1 and matched[0]['completion_sha256']==sha(pred/'COMPLETE.json')
        out=STAGE/'diagnostic_evaluation'/variant/seq
        for name,completion_hash in job['completions_sha256'].items():
            cp=out/(name+'_COMPLETE.json');assert completion_hash==sha(cp)
            d=json.loads(cp.read_text());assert d['status']=='COMPLETE'
            assert d['scoring_script_sha256']==state['score_sha256']==sha(STAGE/'pilot/score_pilot.py')
            assert d['annotations_sha256']==sha(STAGE/'data_diagnostics'/seq/'annotations.json')
            with (out/(name+'_curves.csv')).open(newline='') as f:curve=list(csv.DictReader(f))
            assert [float(r['threshold']) for r in curve]==d['thresholds']
            if name=='official_fused':
                assert len(curve)==23 and d['official_fusion_exactly_reconstructed']
                assert d['prediction_sha256']==source['fused_sha256']
            else:
                assert len(curve)==1 and d['thresholds']==[0.5]
                assert d['prediction_sha256']==source['window_sha256'][name+'.npz']
            records.append({'variant':variant,'sequence':seq,'name':name,'thresholds':d['thresholds'],
                            'curves_sha256':sha(out/(name+'_curves.csv')),'matches_sha256':sha(out/(name+'_matches.npz')),
                            'completion_sha256':completion_hash})
    assert len(records)==96
    report={'status':'PASS','scope':'Old development diagnostics only; raw0.5 contexts are not calibrated operating points',
            'inference_containers_exit0':2,'score_containers_exit0':8,'fused_grid_outputs':8,'context_outputs':88,
            'all_score_containers_cpu_only':True,'calibration_sha256_unchanged':state['calibration_sha256'],
            'all_recorded_completion_prediction_annotation_script_hashes_verified':True,'records':records,
            'verifier_sha256':sha(Path(__file__))}
    (HERE/'DIAGNOSTIC_SCORING_VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))

if __name__=='__main__':main()
