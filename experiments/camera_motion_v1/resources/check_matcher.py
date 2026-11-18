import sys,time,json
from pathlib import Path
import cv2,numpy as np,torch
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/"code"))
from estimate_motion import *
cv2.setNumThreads(1);torch.set_num_threads(2)
rng=np.random.default_rng(17);src=rng.uniform([10,10],[620,490],(500,2));Ht=np.array([[1.,-.003,2.375],[.002,1.,-1.625],[0.000001,-0.000001,1.]])
dst=project(Ht,src);H,row,*_=fit_validate(src,dst,640,512,123)
assert row["trusted"],row
assert np.max(np.abs(project(H,src)-dst))<1e-3
image=cv2.imread(str(E.parents[1]/"data/public/NUDT-MIRSDT/Sequence1/images/00001.png"),0)
assert image.shape==(157,209)
Ht=np.array([[1,0,2.375],[0,1,-1.625],[0,0,1]],float)
out=cv2.warpPerspective(image,Ht,(209,157),flags=cv2.INTER_LINEAR)
back=cv2.warpPerspective(out,np.linalg.inv(Ht),(209,157),flags=cv2.INTER_LINEAR)
assert np.array_equal(cv2.warpPerspective(image,np.eye(3),(209,157)),image)
model=GIMMatcher();t=time.perf_counter();src,dst,conf=model(image,out,123);dt=time.perf_counter()-t
H,row,*_=fit_validate(src,dst,209,157,123)
result={"exact_correspondence_homography_check":"pass","NUDT_source":"Sequence1/images/00001.png","NUDT_shape":list(image.shape),"identity_image_exact":True,"noninteger_translation":[2.375,-1.625],"noninteger_roundtrip_mean_absolute_intensity_difference":float(np.abs(image.astype(float)-back).mean()),"matcher_seconds":dt,"matches":len(src),"matcher_geometry":row,"known_transform_match_error_median":float(np.median(np.linalg.norm(project(Ht,src)-dst,axis=1))) if len(src) else None,"peak_gpu_gb":torch.cuda.max_memory_allocated()/1e9}
(E/"motion/MATCHER_CHECK.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)
