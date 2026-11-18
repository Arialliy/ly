from dataclasses import replace
import hashlib,json,time
from pathlib import Path
import torch
from experiments.irstd_independent_v1.engine.checkpoints import configure_determinism
from experiments.irstd_independent_v1.data.dataset import MaskClipDataset
from experiments.irstd_independent_v1.models.hypothesis_detector import HypothesisEvidenceDetector
from experiments.irstd_independent_v1.evaluation.s2b_froc import _frame_peaks
root=Path('/project')
exp=root/"experiments/irstd_independent_v1"
out=exp/"reports/s4_runtime"
pre=exp/"reports/s3_runs/seed46_core64_20261118_01/interaction_step4.pt"
sd=torch.load(pre, map_location='cpu',weights_only=True)['model']
configure_determinism()
data=MaskClipDataset(json.loads((exp/'manifests/nudt_mirsdt_mask17.json').read_bytes()),root,'train')
clip=data.inference_item(data.centers.index(('Sequence1',9)))
clip=replace(clip,frames=clip.frames.cuda(),pixel_valid=clip.pixel_valid.cuda(),time_offsets=clip.time_offsets.cuda())
m=HypothesisEvidenceDetector().cuda().eval()
m.load_state_dict(sd)
def peaks(a):
 p=_frame_peaks('Sequence1/00009',a.detach().cpu().numpy()[0,0],[])
 return set(map(tuple,p.yx[p.scores.argsort()[::-1][:100]].tolist()))
with torch.inference_mode():
 records=[]
 ref=None
 for q,v,full in [(64,8,False),(64,8,True),(128,81,True),(256,81,True),(512,81,True),(1024,81,True),(256,81,False)]:
  m.video_aggregator.query_chunk=q
  m.video_aggregator.velocity_chunk=v
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
  try:
   z=(m(clip) if full else m.predict_tiled(clip)).logits
  except torch.OutOfMemoryError:
   print("OOM",q,v,flush=True);torch.cuda.empty_cache();continue
  torch.cuda.synchronize();dt=time.perf_counter()-start
  if ref is None:ref=z.clone();pr=peaks(ref)
  p=peaks(z)
  row={'query':q,'velocity':v,'full':full,'id':clip.output_frame_id[0],'hw':list(z.shape[-2:]),'seconds':dt,'peak_mib':torch.cuda.max_memory_allocated()/2**20,'max_diff':float((ref-z).abs().max()),'bad':int((~torch.isclose(z,ref,rtol=1e-4,atol=1e-5)).sum()),'top100_intersect':len(p&pr)}
  print(row,flush=True);records.append(row)
  (out/'benchmark.json').write_text(json.dumps(records,indent=2))
