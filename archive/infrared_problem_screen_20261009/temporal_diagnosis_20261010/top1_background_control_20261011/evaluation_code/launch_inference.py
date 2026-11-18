from pathlib import Path
import argparse,datetime,hashlib,json,subprocess
L=Path(__file__).resolve().parents[1];R=L.parent.parent
p=argparse.ArgumentParser();p.add_argument('--sequence',required=True,choices=['DJI_0026_2','DJI_0067_2']);p.add_argument('--gpu',required=True,type=int);a=p.parse_args()
run=json.loads((L/'training/run/RUN.json').read_text());assert run['status']=='complete' and run['final_step']==512
checkpoint=L/'training/run/final_model.pth';assert checkpoint.is_file()
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
values={int(x.split(',')[0]):[int(y.strip()) for y in x.split(',')[1:]] for x in raw.splitlines()};assert values[a.gpu][0]<500,(a.gpu,values)
spec=R/'official_reproduction/env-spec.json';h=hashlib.sha256(json.dumps(json.loads(spec.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest();assert h=='de8a7eb93aac154354646ae9b59c7ba4086c2ad1d8a704894c9c1de86a4d6d86'
image='deeppro-official-eval:20261009';iid=subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip();assert iid=='sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92'
name='deeppro_top1_infer_'+a.sequence.lower()
cmd=['docker','run','-d','--name',name,'--user','1004:1004','--cpus','4','--memory','12g','--gpus','all']
for env in [f'CUDA_VISIBLE_DEVICES={a.gpu}','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONDONTWRITEBYTECODE=1','PYTHONUNBUFFERED=1','MPLCONFIGDIR=/tmp/deeppro-matplotlib']:cmd+=['-e',env]
cmd+=['-v','/home/ly:/home/ly','-v',str(R/'official_deeppro')+':'+str(R/'official_deeppro')+':ro','--entrypoint','python',image,str(L/'evaluation_code/infer_top1.py'),'--source',str(R/'official_deeppro'),'--reference-checkpoint',str(R/'official_deeppro/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'),'--checkpoint',str(checkpoint),'--dataset-root',str(L.parent/'stage1_goal/data'),'--sequence',a.sequence,'--output',str(L/'predictions'/a.sequence)]
r=subprocess.run(cmd,text=True,capture_output=True)
(L/'predictions'/(name+'_LAUNCH.json')).write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,stdout=r.stdout,stderr=r.stderr,exit_code=r.returncode,image_id=iid,spec_sha256=h,warm_reuse=True,gpu_preflight_csv=raw),indent=2)+'\n')
print(r.stdout,r.stderr);assert r.returncode==0
