import sys,time,json,os
from pathlib import Path
import numpy as np
import torch
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E))
from train_modules.region_contrast_loss import pool_regions,region_contrast_loss
from train_modules.reliable_keep_loss import reliable_keep_loss
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.manual_seed(46);torch.cuda.manual_seed_all(46)
base=torch.randn(2,8,40,24,24,device='cuda');regs=np.array([[0,0,2,2,7,7],[0,0,4,4,9,9],[1,4,2,3,6,8]])
grads=[]
for _ in range(2):
 f=base.clone().requires_grad_();u=pool_regions(f,regs);(u*torch.arange(8,device='cuda')).sum().backward();grads.append(f.grad.clone())
assert torch.equal(grads[0],grads[1])
del f,base,grads
f=torch.randn(8,8,40,128,128,device='cuda',requires_grad=True)
target=[];pos=[];bg=[]
for i in range(128):
 b=i%4;t=i%36;x=16+(i%16)*5;y=16+(i//16)*7;target.append([b,t,x,y,x+4,y+4]);pos.append([b,t+4,x,y,x+4,y+4]);bg.append([b,t,100,100,104,104])
bg += [[r[0]+4,*r[1:]] for r in bg]
pairs=np.stack([np.arange(128),128+np.arange(128)],1);neg=np.stack([np.arange(128),128+np.arange(128)],1)
torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
loss,stats=region_contrast_loss(f,target+pos,pairs,bg,neg);loss.backward();torch.cuda.synchronize();seconds=time.perf_counter()-start
assert torch.isfinite(f.grad).all() and f.grad.abs().sum()>0
result={'status':'PASS','deterministic_algorithms':True,'overlapping_gather_repeated_gradient_bit_identical':True,'real_size_feature_shape':list(f.shape),'pairs':128,'loss':float(loss),'forward_backward_seconds':seconds,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'nonzero_gradient':True,'stats':stats}
(E/'resources/LOSS_GPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
