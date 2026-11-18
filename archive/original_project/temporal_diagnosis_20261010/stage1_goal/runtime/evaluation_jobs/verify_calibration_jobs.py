"""Verify completed calibration handles and artifacts; never starts inference."""
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
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    queue=json.loads((HERE/'QUEUE.json').read_text())
    assert queue['status']=='complete'
    variants=['official','paired_mean','paired_max','ohem','random_pair_max']
    sequences=json.loads((STAGE/'DATA_SELECTION.json').read_text())['splits']['calibration']
    expected={(v,s) for v in variants for s in sequences}
    assert set(map(tuple,queue['inference_completed']))==expected
    assert set(map(tuple,queue['scoring_completed']))==expected
    assert not queue['pending_variants'] and not queue['active_infer'] and not queue['active_score']
    assert set(queue['training_verified'])==set(variants[1:])
    for variant,training in queue['training_verified'].items():
        run=json.loads((STAGE/'runs'/variant/'RUN.json').read_text())
        assert training['final_step']==run['final_step']==512 and run['status']=='complete'
        assert training['bn_initial_sha256']==training['bn_final_sha256']==run['bn_final_sha256']
        assert training['checkpoint_sha256']==run['final_checkpoint_sha256']==sha(STAGE/'runs'/variant/'final_model.pth')
    for path,digest in queue['frozen_sources'].items():assert sha(Path(path))==digest
    rows=[]
    for variant,sequence in sorted(expected):
        pred=STAGE/'predictions'/variant/sequence
        eva=STAGE/'evaluation'/variant/sequence
        cp=json.loads((pred/'COMPLETE.json').read_text())
        sc=json.loads((eva/'official_fused_COMPLETE.json').read_text())
        assert cp['status']==sc['status']=='COMPLETE'
        assert cp['sequence']==sequence and cp['seed']==46
        assert cp['official_starts']==[0,36,72,108,120]
        assert cp['diagnostic_starts']==list(range(0,121,12))
        assert cp['raw_sha256']==sha(STAGE/'data'/sequence/'raw_frames.npy')
        assert cp['checkpoint_sha256']==sha(Path(cp['checkpoint']))
        expected_checkpoint=(json.loads((STAGE/'runs/paired_mean/RUN.json').read_text())['checkpoint_sha256']
                             if variant=='official' else queue['training_verified'][variant]['checkpoint_sha256'])
        assert cp['checkpoint_sha256']==expected_checkpoint
        assert cp['fused_sha256']==sha(pred/'official_fused.npz')==sc['prediction_sha256']
        assert sc['annotations_sha256']==sha(STAGE/'data'/sequence/'annotations.json')
        assert sc['scoring_script_sha256']==queue['frozen_sources'][str(STAGE/'pilot/score_pilot.py')]
        assert cp['script_sha256']==queue['frozen_sources'][str(STAGE/'pilot/infer_pilot.py')]
        assert sc['official_fusion_exactly_reconstructed'] and len(sc['thresholds'])==23
        if sequence==sequences[0]:assert len(cp['tiling_checks'])==3
        else:assert len(cp['tiling_checks'])==0
        for name,digest in cp['window_sha256'].items():assert sha(pred/'windows'/name)==digest
        with (eva/'official_fused_curves.csv').open(newline='') as f:curves=list(csv.DictReader(f))
        assert len(curves)==23 and [float(r['threshold']) for r in curves]==sc['thresholds']
        handles={}
        for kind in ['infer','score']:
            record=json.loads((HERE/variant/sequence/kind/'VERIFIED.json').read_text())
            assert record['status']=='complete' and record['exit_code']==0
            observed=json.loads(subprocess.check_output(['docker','inspect',record['container_id']],text=True))[0]
            assert observed['State']['Status']=='exited' and observed['State']['ExitCode']==0
            assert observed['Image']==queue['image_id']
            official_mounts=[m for m in observed['Mounts'] if m['Destination'].endswith('/official_deeppro')]
            assert len(official_mounts)==1 and not official_mounts[0]['RW']
            assert sha(Path(record['completion_path']))==record['completion_sha256']
            handles[kind]={'id':record['container_id'],'gpu':record['gpu'],'exit_code':0}
        rows.append({'variant':variant,'sequence':sequence,'checkpoint_sha256':cp['checkpoint_sha256'],
                     'inference_seconds':cp['elapsed_seconds'],'scoring_seconds':sc['elapsed_seconds_total'],
                     'tiling_checks':cp['tiling_checks'],'handles':handles,
                     'fused_curves_sha256':sha(eva/'official_fused_curves.csv'),
                     'fused_matches_sha256':sha(eva/'official_fused_matches.npz')})
    report={'status':'PASS','scope':'Calibration only; no confirmation inference or threshold selection in this queue',
            'inferences':10,'score_jobs':10,'threshold_count':23,'all_container_exits_zero':True,
            'all_training_verified_final_steps':512,'all_prediction_and_scoring_hashes_verified':True,
            'fused_exactly_reconstructed_by_scorer':True,'jobs':rows,
            'verifier_sha256':sha(Path(__file__))}
    (HERE/'CALIBRATION_EXECUTION_VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='jobs'}))

if __name__=='__main__':main()
