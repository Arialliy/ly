"""Frozen TEP-SAM execution helpers; never modify author source/model operations."""
import hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

STAGE=Path(__file__).resolve().parents[1]
SOURCE=STAGE/'author_source'
CHECKPOINT=STAGE/'weights/model_NUDT-MIRSDT_SAM1-B.pth'
CHECKPOINT_SHA='1125ea28b4dfe0745be1f33500f4e5463f1cbd8e837a3c1c733b34bb0de9e6f0'
sys.path.insert(0,str(SOURCE))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def seed():
    random.seed(46);np.random.seed(46);torch.manual_seed(46);torch.cuda.manual_seed_all(46)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def source_check():
    hashes=json.load(open(STAGE/'runtime/SOURCE_HASHES.json'))
    assert all(sha(SOURCE/name)==value for name,value in hashes.items())
    return {'tracked_file_count':len(hashes),'hashes_verified':True}

def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def load_config():
    return yaml.safe_load(open(SOURCE/'configs/sam-vit-b-token-NUDT-MIRSDT.yaml'))

def load_model():
    import models
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    m=models.make(load_config()['model']).cuda()
    sd=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    if 'state_dict' in sd: sd=sd['state_dict']
    if 'model' in sd: sd=sd['model']
    r=m.load_state_dict(sd,strict=True)
    assert not r.missing_keys and not r.unexpected_keys
    m.eval().requires_grad_(False)
    return m

class Recorder:
    def __init__(self,model,shape,count):
        self.model=model;self.shape=tuple(shape);self.count=count
        self.prob=np.empty((count,*shape),np.float32)
        self.aux=np.empty_like(self.prob);self.calls=0;self.forward_seconds=[];self.equality_pixels=[]
        self.raw_output_shapes=set();self.aux_output_shapes=set()
        self.handle=model.image_encoder.register_forward_hook(self.capture)
        self.aux_tensor=None
    def capture(self,module,args,output):
        assert len(output)==3
        self.aux_tensor=output[2]
    def invoke(self,original,inp,*args,**kwargs):
        assert inp.shape==(1,7,3,1024,1024)
        assert inp.dtype==torch.float32
        self.aux_tensor=None
        torch.cuda.synchronize();start=time.perf_counter()
        out=original(inp,*args,**kwargs)
        torch.cuda.synchronize();self.forward_seconds.append(time.perf_counter()-start)
        assert out.shape==(1,1,1024,1024) and torch.isfinite(out).all()
        aux=self.aux_tensor
        assert aux is not None and aux.shape==(1,1,256,256) and torch.isfinite(aux).all()
        # Main: author infer already performed its bilinear->crop->bilinear chain.
        p=F.interpolate(torch.sigmoid(out),size=self.shape,mode='area')[0,0]
        # Auxiliary: project diagnostic adapter, exactly reusing author postprocess.
        # The author provides no auxiliary test output or auxiliary metric protocol.
        aux_full=self.model.postprocess_masks(aux,self.model.inp_size,self.model.inp_size)
        a=F.interpolate(torch.sigmoid(aux_full),size=self.shape,mode='area')[0,0]
        assert torch.isfinite(p).all() and torch.isfinite(a).all()
        self.prob[self.calls]=p.cpu().numpy();self.aux[self.calls]=a.cpu().numpy()
        self.equality_pixels.append([int((p==.5).sum()),int((a==.5).sum())])
        self.raw_output_shapes.add(tuple(out.shape));self.aux_output_shapes.add(tuple(aux.shape))
        self.calls+=1
        return out
    def finish(self,outdir,frame_indices,source_frames):
        assert self.calls==self.count
        outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
        kw={'frame_indices':np.asarray(frame_indices,np.int64),'source_frames':np.asarray(source_frames,np.int64)}
        np.savez_compressed(outdir/'probabilities.npz',probabilities=self.prob,**kw)
        np.savez_compressed(outdir/'auxiliary_probabilities.npz',probabilities=self.aux,**kw)
        return {'forward_calls':self.calls,'forward_seconds_total':sum(self.forward_seconds),'forward_seconds':self.forward_seconds,
                'raw_main_shapes':sorted(self.raw_output_shapes),'raw_aux_shapes':sorted(self.aux_output_shapes),
                'exact_0_5_pixels_main_aux':np.sum(self.equality_pixels,axis=0).tolist(),
                'main_range':[float(self.prob.min()),float(self.prob.max())],'aux_range':[float(self.aux.min()),float(self.aux.max())],
                'probabilities_sha256':sha(outdir/'probabilities.npz'),'auxiliary_probabilities_sha256':sha(outdir/'auxiliary_probabilities.npz')}
