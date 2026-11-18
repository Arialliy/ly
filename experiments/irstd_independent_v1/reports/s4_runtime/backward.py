import json,time,torch,numpy as np
from pathlib import Path
from dataclasses import replace
from experiments.irstd_independent_v1.engine.checkpoints import configure_determinism
from experiments.irstd_independent_v1.data.dataset import MaskClipDataset
from experiments.irstd_independent_v1.models.hypothesis_detector import HypothesisEvidenceDetector
from experiments.irstd_independent_v1.evaluation.s2b_froc import _frame_peaks
from experiments.irstd_independent_v1.losses.detection import detection_loss
root=Path('/project')
exp=root/"experiments/irstd_independent_v1"
out=exp/"reports/s4_runtime"
pre=exp/"reports/s3_runs/seed46_core64_20261118_01/interaction_step4.pt"
sd=torch.load(pre,map_location='cpu',weights_only=True)['model']
configure_determinism()
data=MaskClipDataset(json.loads((exp/'manifests/nudt_mirsdt_mask17.json').read_bytes()),root,'train')
clip,gt=data[data.centers.index(('Sequence82',43))]
clip=replace(clip,frames=clip.frames.cuda(),pixel_valid=clip.pixel_valid.cuda(),time_offsets=clip.time_offsets.cuda())
gt=replace(gt,target_map=gt.target_map[...,64:128,64:128].cuda(),ignore_map=gt.ignore_map[...,64:128,64:128].cuda())
m=HypothesisEvidenceDetector().cuda().train()
m.load_state_dict(sd)
tile=next(tile for tile in m.tiles(clip.scene_meta[0].original_hw) if tile.core_xyxy==(64,64,128,128))
results=[]
ref=None
for q,v,c in [(64,8,True),(256,81,True),(512,81,True),(1024,81,True),(256,81,False),(512,81,False),(1024,81,False)]:
 m.zero_grad(set_to_none=True)
 m.video_aggregator.query_chunk=q;m.video_aggregator.velocity_chunk=v;m.video_aggregator.checkpoint_chunks=c
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 p=m.forward_core(clip,tile);l=detection_loss(p,gt,clip.pixel_valid[:,8,:,64:128,64:128]);torch.cuda.synchronize();mid=time.perf_counter()
 l.backward();torch.cuda.synchronize();end=time.perf_counter()
 grad=torch.cat([p.grad.detach().flatten() for p in m.parameters()]).cpu()
 z=p.logits.detach().cpu().clone()
 if ref is None:ref=(z.clone(),grad.clone())
 diff=(grad-ref[1]).abs()
 row=dict(query=q,velocity=v,checkpoint=c,forward_s=mid-start,backward_s=end-mid,total_s=end-start,peak_MiB=torch.cuda.max_memory_allocated()/2**20,loss=l.item(),logits_max_abs=(ref[0]-z).abs().max().item(),grad_max_abs=diff.max().item(),grad_relative_norm=float(torch.linalg.vector_norm(diff.double())/torch.linalg.vector_norm(ref[1].double())),grad_outside_tolerance=int((~torch.isclose(grad,ref[1],rtol=1e-4,atol=1e-5)).sum()))
 print(row,flush=True);results.append(row);(out/'backward.json').write_text(json.dumps(results,indent=2))
 del l,p,z,grad
