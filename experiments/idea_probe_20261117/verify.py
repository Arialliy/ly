import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
E=Path(__file__).resolve().parent;torch.set_num_threads(2)
f=np.load(E/'FEATURES.npz');p=np.load(E/'PREDICTIONS.npz');r=json.loads((E/'RESULTS.json').read_text())
m=list(csv.DictReader((E/'SAMPLE_MANIFEST.csv').open()));plan=np.load(E/'EXPOSURE_PLAN.npy')
assert plan.shape==(100,256)
assert all(m[int(i)]['split']=='train' for i in plan.ravel())
assert all(np.array_equal(f['labels'][row],np.tile([1,0],128)) for row in plan)
out={'exposure_plan_shape':list(plan.shape),'all_exposures_training_only':True,'all_steps_balanced_pairs':True,'prediction_replay':{},'auc_independent_pairwise':{}}
for name in ['real','current']:
    x=f['raw'] if name=='real' else np.repeat(f['raw'][:,2:3],5,axis=1)
    x=((x-f['mu'])/f['sd']).reshape(len(x),45)
    model=nn.Sequential(nn.Linear(45,32),nn.ReLU(),nn.Linear(32,1));model.load_state_dict(torch.load(E/(name+'_step100.pt'),map_location='cpu',weights_only=True));model.eval()
    with torch.no_grad():s=model(torch.from_numpy(x)).squeeze(1).numpy()
    assert np.array_equal(s,p[name]);out['prediction_replay'][name]='bitwise_equal'
    for seq in r['sources_dev']:
        ids=np.array([i for i,z in enumerate(m) if z['sequence']==seq]);pos=s[ids[f['labels'][ids]==1]];neg=s[ids[f['labels'][ids]==0]]
        a=float(((pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg)).mean());assert abs(a-r['summary'][seq][name]['auc'])<1e-12
        out['auc_independent_pairwise'][seq+'/'+name]=a
out['verification_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();out['status']='PASS'
(E/'REPLAY_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
