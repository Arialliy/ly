"""Post hoc physics controls. Carrier/fact never enter estimate_image()."""
import json
import sys
from pathlib import Path

import numpy as np

import witness as w


ROOT = w.HERE.parents[1]
PREVIOUS = ROOT/'experiments/vsr_sampling_x2_20261118'
sys.path.insert(0, str(PREVIOUS))
import registration
import solver

METHODS = ('known_copy', 'known_fourier', 'estimated_fourier')


def bilinear_center(lr):
    h,width=lr.shape
    out=np.empty((2*h,2*width))
    out[::2,::2]=lr
    out[::2,1::2]=(lr+np.roll(lr,-1,axis=1))/2
    out[1::2]=(out[::2]+np.roll(out[::2],-1,axis=0))/2
    return out


def estimate_image(observations, known_offsets, method):
    luminance=observations.astype(np.float64)@w.RGB_WEIGHTS
    if method=='known_copy':
        out=bilinear_center(luminance[2])
        for frame,(dx,dy) in zip(luminance,known_offsets):
            assert dy==0 and dx in (0,.5)
            out[::2,int(2*dx)::2]=frame
        return out,known_offsets
    offsets=(known_offsets if method=='known_fourier' else
             registration.estimate_shifts(observations,reference=2))
    assert method in ('known_fourier', 'estimated_fourier')
    out=solver.reconstruct(luminance,offsets,ridge=.05,blur_sigma=0.,reference=2)['classic']
    return out,offsets


def main():
    manifest=json.loads((w.V1/'data/MANIFEST.json').read_text())
    records=sorted((r for r in manifest['files'] if r['split']=='train' and r['center']==r['frame']==32),
                   key=lambda r:r['sequence'])[:8]
    rows={method:[] for method in METHODS}
    for i,r in enumerate(records):
        base,carrier=w.scene(r,.37+i*.51)
        for s in (-1,1):
            for b in (-1,1):
                ref,reveal=w.make_world(base,carrier,s,b)
                cr=w.coeff(ref,carrier)
                for method in METHODS:
                    entry=dict(sequence=r['sequence'],s=s,b=b,outputs={})
                    for mode,slot in zip(w.MODES,(None,3,4)):
                        obs=np.stack([ref]*5)
                        truth=np.zeros((5,2))
                        if slot is not None:
                            obs[slot]=reveal
                            truth[slot,0]=.5
                        pred,offset=estimate_image(obs,truth if method!='estimated_fourier' else None,method)
                        # Equal RGB channels give exactly the same luminance coefficient.
                        g=w.coeff(np.repeat(pred[::2,1::2,None],3,axis=-1),carrier)
                        entry['outputs'][mode]=dict(coefficient=g,fact_accuracy=w.class_score(cr*g,b),
                                                  shift=offset[slot].tolist() if slot is not None else None,
                                                  epe=float(np.linalg.norm(offset[slot]-truth[slot])) if slot is not None else None)
                    rows[method].append(entry)
    summary={m:w.aggregate(rs) for m,rs in rows.items()}
    assert all(summary[m]['neutral']['fact_accuracy']==.5 for m in METHODS)
    assert summary['known_copy']['slot3']['fact_accuracy']==summary['known_copy']['slot4']['fact_accuracy']==1.
    estimates=np.array([r['outputs']['slot4']['shift'] for r in rows['estimated_fourier']])
    epe=np.linalg.norm(estimates-np.array([.5,0.]),axis=1)
    score=dict(worlds=len(epe),mean=float(epe.mean()),median=float(np.median(epe)),max=float(epe.max()),
               mean_estimated_dxdy=estimates.mean(axis=0).tolist(),
               EPE_by_fact={str(sign):float(np.mean([r['outputs']['slot4']['epe'] for r in rows['estimated_fourier']
                                                      if r['b']==sign])) for sign in (-1,1)})
    out=dict(summary=summary,registration=score,rows=rows,ridge=.05,truth_offsets_at_reveal=[.5,0],
             prior_sha256=w.sha((w.HERE/'RESULTS.json').read_bytes()),solver_sha256=w.sha((PREVIOUS/'solver.py').read_bytes()),
             registration_sha256=w.sha((PREVIOUS/'registration.py').read_bytes()))
    (w.HERE/'GEOMETRY_RESULTS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(score)
    for m in METHODS:
        print(m,[(k,summary[m][k]['fact_accuracy'],summary[m][k]['mean_fraction_true_paired_difference'])
                 for k in w.MODES])


if __name__=='__main__':
    main()
