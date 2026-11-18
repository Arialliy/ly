from pathlib import Path
import json,torch
import run_factorial_frozen as f
q=Path(__file__).resolve().parent
b=q.parent.parent
source=b/'official_deeppro';checkpoint=source/'log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth'
torch.set_num_threads(2)
model,_,_,_=f.model_setup(source,checkpoint,torch.device('cpu'))
r=[]
with torch.inference_mode():
 for slot in range(40):
  _,bias=f.row_weights(model,slot)
  logit=float(f.phi(model,bias[None,:,None,None,None]).item())
  r.append({'slot':slot,'logit':logit,'probability':float(torch.sigmoid(torch.tensor(logit)).item())})
out={'purpose':'bias-only zero-feature control; no inference claim','rows':r,'max_probability':max(x['probability'] for x in r),'min_probability':min(x['probability'] for x in r)}
(q/'BIAS_ONLY_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
