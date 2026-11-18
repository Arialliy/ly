"""Paired subpixel texture worlds and frozen DUF evidence-arrival readouts."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
V1 = HERE.parents[1] / 'experiments' / 'vsr_geometry_model_v1'
sys.path.insert(0, str(V1))
from backbone import load_backbone, WEIGHTS_SHA256


AMP = .05
INNER = np.s_[4:-4, 4:-4]
RGB_WEIGHTS = np.array([.299, .587, .114])
MODES = ('neutral', 'slot3', 'slot4')
CUTOFFS = {'early': ('neutral', 'slot3', 'slot3'), 'late': ('neutral', 'neutral', 'slot4')}


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


def coeff(img, carrier):
    y = img[INNER].astype(np.float64) @ RGB_WEIGHTS
    c = carrier[INNER]
    return float(np.sum(c*y) / np.sum(c*c))


def scene(record, phase):
    path = V1/'data'/record['path']
    assert record['split']=='train' and record['center']==record['frame']==32
    assert sha(path.read_bytes())==record['sha256']
    with Image.open(path) as f:
        w,h=f.size
        rgb=np.asarray(f.convert('RGB').crop((w//2-32,h//2-32,w//2+32,h//2+32)),dtype=np.float64)/255.
    y,x=np.indices((32,32))
    carrier=np.cos(2*np.pi*(3*x+y)/32+phase)
    carrier-=carrier[INNER].mean()
    for py in [0,1]:
        for px in [0,1]:
            view=rgb[py::2,px::2]
            view-=coeff(view,carrier)*carrier[...,None]
    rgb=.15+.7*(rgb-rgb.min())/(rgb.max()-rgb.min())
    assert max(abs(coeff(rgb[py::2,px::2],carrier)) for py in [0,1] for px in [0,1])<1e-12
    return rgb,carrier


def make_world(base, carrier, polarity, fact):
    hr=base.copy()
    for py in (0,1):
        hr[py::2,::2]+=polarity*AMP*carrier[...,None]
        hr[py::2,1::2]+=polarity*fact*AMP*carrier[...,None]
    assert hr.min()>=0 and hr.max()<=1
    hr=hr.astype(np.float32)
    return hr[::2,::2].copy(),hr[::2,1::2].copy()


def class_score(value, target):
    signed=value*target
    return float(signed>0) if signed!=0 else .5


@torch.inference_mode()
def measure(model, record, i):
    bg,carrier=scene(record,.37+i*.51)
    worlds={}
    inputs=[]
    rows=[]
    for s in (-1,1):
        for b in (-1,1):
            ref,reveal=make_world(bg,carrier,s,b)
            worlds[(s,b)]=(ref,reveal)
            cr,cv=coeff(ref,carrier),coeff(reveal,carrier)
            assert abs(cr-s*AMP)<1e-6 and abs(cv-s*b*AMP)<1e-6
            oracle=class_score(cr*cv,b)
            assert oracle==1
            rows.append(dict(sequence=record['sequence'],s=s,b=b,reference_coefficient=cr,
                             true_odd_coefficient=cv,analytic=oracle))
            for mode,slot in zip(MODES,(None,3,4)):
                slots=[ref]*5
                if slot is not None: slots[slot]=reveal
                inputs.append(np.stack(slots).transpose(0,3,1,2))
    for s in (-1,1):
        np.testing.assert_array_equal(worlds[s,1][0],worlds[s,-1][0])
        np.testing.assert_array_equal(worlds[s,1][1],worlds[-s,-1][1])
    # Common batch size for equivalent contexts and stable CPU comparisons.
    tensors=torch.from_numpy(np.stack(inputs))
    outs=torch.cat([model(t) for t in tensors.split(3)])
    assert outs.shape==(12,3,64,64) and torch.isfinite(outs).all()
    images=outs.permute(0,2,3,1).numpy()
    for j,row in enumerate(rows):
        row['outputs']={}
        for k,mode in enumerate(MODES):
            pred=coeff(images[j*3+k,::2,1::2],carrier)
            row['outputs'][mode]=dict(coefficient=pred,
                                      fact_accuracy=class_score(row['reference_coefficient']*pred,row['b']))
    for j in (0,2):
        np.testing.assert_array_equal(images[3*j],images[3*(j+1)])
    return rows


def aggregate(rows):
    output={}
    for mode in MODES:
        values=[r['outputs'][mode] for r in rows]
        changes=[r['s']*r['b']*(v['coefficient']-r['outputs']['neutral']['coefficient'])/AMP
                 for r,v in zip(rows,values)]
        paired=[]
        for i in range(0,len(rows),2):
            minus,plus=rows[i:i+2]
            assert (minus['s'],minus['sequence'],minus['b'])==(plus['s'],plus['sequence'],-1) and plus['b']==1
            delta=plus['outputs'][mode]['coefficient']-minus['outputs'][mode]['coefficient']
            paired.append(delta*plus['s']/(2*AMP))
        output[mode]=dict(worlds=len(values), fact_accuracy=float(np.mean([p['fact_accuracy'] for p in values])),
                          mean_fraction_true_paired_difference=float(np.mean(paired)),
                          pairs_moving_in_true_direction=int(np.sum(np.array(paired)>0)), pairs=len(paired),
                          signed_mean_change_amplitude_fraction=float(np.mean(changes)),
                          signed_change_fraction_bounds=[min(changes),max(changes)],
                          world_signed_changes_positive=int(np.sum(np.array(changes)>0)),
                          mean_coefficient_by_true_sign={str(sign):float(np.mean([v['coefficient'] for r,v in zip(rows,values)
                                                                                if r['b']*r['s']==sign])) for sign in (-1,1)})
    return output


def main():
    torch.set_num_threads(2)
    mpath=V1/'data/MANIFEST.json'
    records=sorted((r for r in json.loads(mpath.read_text())['files']
                    if r['split']=='train' and r['frame']==r['center']==32),key=lambda r:r['sequence'])[:8]
    assert len(records)==len({r['sequence'] for r in records})==8
    model=load_backbone()
    rows=[w for i,r in enumerate(records) for w in measure(model,r,i)]
    summary=aggregate(rows)
    assert summary['neutral']['fact_accuracy']==.5 and summary['neutral']['mean_fraction_true_paired_difference']==0
    aggregate_prefix={name:[dict(horizon=h, mode=k, oracle_known_carrier=1 if k!='neutral' else .5,
                                 duf=summary[k]['fact_accuracy']) for h,k in enumerate(modes)]
                      for name,modes in CUTOFFS.items()}
    data=dict(input_manifest_sha256=sha(mpath.read_bytes()),weights_sha256=WEIGHTS_SHA256,
              measured_sequences=[r['sequence'] for r in records],amplitude=AMP,
              zero_phase_observations_exactly_matched=True,reveal_only_marginals_exactly_matched=True,
              balanced_worlds=32,ref_alone_optimal_accuracy=.5,reveal_only_optimal_accuracy=.5,
              summary=summary,arrival=aggregate_prefix,rows=rows)
    (HERE/'RESULTS.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(summary=summary,arrival=aggregate_prefix),indent=2))


if __name__=='__main__':
    main()
