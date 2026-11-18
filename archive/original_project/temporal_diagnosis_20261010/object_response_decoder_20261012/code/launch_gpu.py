"""Isolated immutable-image invocation on user-authorized idle GPU1/2/3."""
import argparse,json,os,subprocess,time,hashlib
from pathlib import Path
N=Path(__file__).resolve().parents[1];R=N.parent.parent
IMAGE='sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['preflight','witness','train','infer'],required=True)
    p.add_argument('--gpu',type=int,choices=[1,2,3],required=True);p.add_argument('--condition',choices=['R0','R1','R2'])
    p.add_argument('--sequence',choices=['DJI_0026_2','DJI_0067_2']);p.add_argument('--attempt',default='1')
    a=p.parse_args()
    if a.kind in ['train','infer'] and a.condition is None:p.error('condition required')
    if a.kind=='infer' and a.sequence is None:p.error('sequence required')
    label='_'.join(str(x) for x in [a.kind,a.condition,a.sequence,a.attempt] if x)
    receipt=N/'runtime'/f'{label}.json';log=receipt.with_suffix('.log')
    if receipt.exists() or log.exists():raise FileExistsError(label)
    rows={int(s.split(',')[0]):[int(v.strip()) for v in s.split(',')[1:]] for s in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()}
    if rows[a.gpu][0]>=500:raise RuntimeError(f'GPU occupied {rows[a.gpu]}')
    if a.kind in ['train','infer']:
        f=json.loads((N/'PROTOCOL_FREEZE.json').read_text())
        for path,digest in f['code_sha256'].items():assert hashlib.sha256((N/path).read_bytes()).hexdigest()==digest,path
        assert hashlib.sha256((N/'PROTOCOL.md').read_bytes()).hexdigest()==f['protocol_sha256']
        assert json.loads((N/'checks/GPU_PREFLIGHT.json').read_text())['status']=='PASS'
    script={'preflight':'preflight.py','witness':'test_model.py','train':'train.py','infer':'infer.py'}[a.kind]
    call=[str(N/'code'/script)]
    if a.kind=='witness':call+=['--device','cuda']
    if a.kind in ['train','infer']:call+=['--variant',a.condition]
    if a.kind=='infer':call+=['--sequence',a.sequence]
    command=['docker','run','--rm','--name','ord_'+label.lower(),'--network','none','--gpus','all',
        '--user',f'{os.getuid()}:{os.getgid()}','--cpus','4','--memory','20g','--shm-size','2g',
        '-e',f'CUDA_VISIBLE_DEVICES={a.gpu}','-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2',
        '-e','OPENBLAS_NUM_THREADS=2','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
        '-e','CUBLAS_WORKSPACE_CONFIG=:4096:8','-e','MPLCONFIGDIR=/tmp/mpl',
        '-v',f'{R}:{R}:ro','-v',f'{N}:{N}:rw','-w',str(N),'--entrypoint','/opt/conda/bin/python',IMAGE,*call]
    record=dict(status='RUNNING',gpu=a.gpu,job=label,image=IMAGE,command=command,gpu_before=rows,start_unix=time.time())
    receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    with log.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    record.update(status='COMPLETE' if result.returncode==0 else 'FAILED',exit_code=result.returncode,end_unix=time.time())
    receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
