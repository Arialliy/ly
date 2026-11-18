"""Read-only civil traffic training diagnostic for the frozen four arms."""
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent/'motion_joint_factorial'
sys.path.insert(0,str(STAGE))
from experiment import FIELDS, model_trainables, train_data
from model import FactorialDetector, load_official_detector
from protocol import CENTERS, TRAIN, frame_path, metadata, setup


def auc(positive, negative):
    n, k = len(positive),len(negative)
    scores=np.r_[positive,negative]
    r=rankdata(scores)
    return float((r[:n].sum()-n*(n+1)/2)/(n*k))


@torch.inference_mode()
def main():
    setup()
    hard=[]
    counts=Counter()
    for seq in TRAIN:
        meta=metadata(seq)
        selected={im['id'] for im in meta['images'] if im['frame_id'] in CENTERS}
        names={c['id']:c['name'] for c in meta['categories']}
        for a in meta['annotations']:
            _,_,w,h=a['bbox']
            if (a['image_id'] in selected and a['category_id'] in range(5) and
                    not (a.get('ignore') or a.get('iscrowd')) and 0<w*h<=144 and max(w,h)<=16):
                counts[names[a['category_id']]]+=1
        for t in CENTERS:
            with np.load(frame_path('backbone',seq,t)) as rec:
                target=rec['target']
                n=int(target.sum())
                assert target[:n].all() and not target[n:].any() and len(target)-n==256
                hard.append((np.arange(len(target))>=n) & (np.arange(len(target))<n+128))
    hard=np.concatenate(hard)
    detector=load_official_detector('cuda')
    result={}
    for selection,letters in [('full','AB'),('split','CD')]:
        buf=train_data(selection)
        y=buf['target'].cpu().numpy()
        for arm in letters:
            m=FactorialDetector(detector,arm).eval()
            p=model_trainables(m)
            state=torch.load(STAGE/'runs/models'/f'{arm}.pt',weights_only=True,map_location='cpu')
            assert set(p)==set(state)
            for key,parameter in p.items():
                parameter.copy_(state[key])
            outputs=[]
            for start in range(0,len(y),512):
                block=slice(start,start+512)
                ev={f:buf[f][block] for f in FIELDS}
                outputs.append(m.forward_points(buf['features'][block],ev).cpu())
            scores=torch.cat(outputs).double()
            bce=(F.softplus(-scores[y]).mean()+F.softplus(scores[~y]).mean())/2
            s=scores.numpy()
            result[arm]={'balanced_BCE':bce.item(),'positive_vs_hard_negative_AUC':auc(s[y],s[hard])}
    report={'train_classes':counts,'total_saved_points':len(y),'positive':int(y.sum()),
            'original_high_scoring_negative_peaks':int(hard.sum()),'values':result}
    print(json.dumps(report,indent=2),flush=True)
    dest=HERE/'runs/training_fit.json'
    tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(report,indent=2)+'\n')
    tmp.replace(dest)


if __name__=='__main__':
    main()
