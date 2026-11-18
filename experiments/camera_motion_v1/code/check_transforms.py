"""Critical coordinate tests on fixed NUDT training source and analytic signals."""
import numpy as np
from PIL import Image
from geometry import warp,prepare,restore,apply_fallback
from runtime import B,E,save,sha

path=B/'data/public/NUDT-MIRSDT/Sequence1/images/00001.png'
a=np.asarray(Image.open(path),dtype=np.float32)
paths=[path.with_name(f'{i:05d}.png') for i in range(1,41)]
raw=np.stack([np.asarray(Image.open(p),dtype=np.float32) for p in paths])
I=np.repeat(np.eye(3)[None],40,axis=0)
c1,c2,v=prepare(raw,I)
assert np.array_equal(c1,raw) and np.array_equal(c2,raw)
assert np.array_equal(restore(raw,I),raw)
assert v[:,9:-9,9:-9].all()

# Analytic impulse: source (25,25) maps to (25.5,25.25) in destination.
p=np.zeros((64,64),np.float32);p[25,25]=1
H=np.array([[1,0,.5],[0,1,.25],[0,0,1]],np.float64)
w=warp(p,H)
expected=np.zeros_like(p);expected[25,25:27]=.375;expected[26,25:27]=.125
np.testing.assert_array_equal(w,expected)
assert np.count_nonzero(w)==4 and w.sum()==1
rt=warp(w,np.linalg.inv(H))
assert not np.array_equal(rt,p) and rt[25,25]<1

# Translation direction and recovering a per-frame prediction before fusion.
J=np.array([[1,0,4],[0,1,-3],[0,0,1]],np.float64)
q=warp(p,J);assert q[22,29]==1
np.testing.assert_array_equal(warp(q,np.linalg.inv(J)),p)
Z=np.repeat(J[None],40,axis=0);_,_,valid=prepare(raw,Z)
base=np.full(raw.shape,.7,np.float32);cand=np.full(raw.shape,.2,np.float32)
out=apply_fallback(cand,base,valid)
assert np.array_equal(out[~valid],base[~valid]) and np.array_equal(out[valid],cand[valid])
assert np.all(apply_fallback(cand,base,np.zeros_like(valid))==base)
# Never fuse predictions that still live in different anchor coordinates.
rp=restore(np.stack([warp(p,J),warp(p,H)]),np.stack([J,H]))
assert np.unravel_index(np.maximum.reduce(rp).argmax(),p.shape)==(25,25)
save(E/'checks/TRANSFORM_TESTS.json',{'status':'PASS','source':str(path.relative_to(B)),
 'source_sha256':sha(path),'source_frame_sha256':{p.name:sha(p) for p in paths},
 'source_split':'NUDT official train Sequence1 first40','source_shape':list(raw.shape),
 'tests':['identity image/prediction','noninteger bilinear impulse weights and roundtrip attenuation',
          'source-to-anchor/inverse direction','geometry-only fallback including rejected window',
          'restore before cross-window fusion'],
 'mask_rule':'40-frame intersection; 9-pixel spatial receptive-field erosion; C1/C2 support intersection',
 'code_sha256':{p.name:sha(p) for p in [E/'code/geometry.py',E/'code/check_transforms.py']}})
print('Transform tests PASS')
