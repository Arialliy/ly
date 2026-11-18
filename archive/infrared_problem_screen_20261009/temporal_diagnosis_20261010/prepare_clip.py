import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('clip',type=Path);a=p.parse_args();m=json.loads((a.clip/'FRAME_MANIFEST.json').read_text());arr=[]
for f in m['frames']:
 path=a.clip/f['local_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==f['sha256'];im=Image.open(path);assert im.mode=='L' and im.size==(640,512);arr.append(np.asarray(im,dtype=np.uint8))
x=np.stack(arr);assert x.shape==(160,512,640);np.save(a.clip/'raw_frames.npy',x);print(json.dumps({'input':str(a.clip/'raw_frames.npy'),'shape':list(x.shape),'sha256':hashlib.sha256((a.clip/'raw_frames.npy').read_bytes()).hexdigest()}))
