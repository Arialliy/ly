"""Post hoc support sensitivity: nested LR fields, identical shifts and scoring pixels."""
import json

import numpy as np

import registration
import run
import solver


def evaluate(frames, sequence, j):
    truth = solver.prepare_target(frames[2])[72:-72,72:-72]
    zero = np.zeros((1,2))
    wide = np.stack([solver.observe(frame,zero,.8)[0] for frame in frames])
    wide += np.random.default_rng(561000 + 10*sequence + j).normal(0,1/255,wide.shape)
    narrow = wide[:,32:-32,32:-32]
    offsets = registration.estimate_shifts(narrow,reference=2)
    row = dict(seq=sequence,patch=j)
    for name,obs,border in [('narrow',narrow,8),('wide',wide,72)]:
        result = solver.reconstruct(obs,offsets,.05,.8,reference=2)['classic']
        single = solver.reconstruct(np.repeat(obs[2:3],5,axis=0),np.zeros_like(offsets),.05,.8)['classic']
        row[name] = {}
        for method,prediction in [('classic',result),('repeat',single)]:
            center = prediction[border:-border,border:-border]
            assert center.shape==truth.shape==(112,112)
            row[name][method] = float(np.mean((center-truth)**2))
    return row


def main():
    manifest = json.loads((run.INPUT/'MANIFEST.json').read_text())
    rows = []
    for seq in run.SEQUENCES:
        frames = run.footage(seq,manifest)
        _, h, w = frames.shape
        j = 0
        for yf in (1/3,2/3):
            for xf in (.25,.75):
                x,y = round(xf*w)-128,round(yf*h)-128
                rows.append(evaluate(frames[:,y:y+256,x:x+256],seq,j))
                j+=1
    assert len(rows)==32
    run.write('HALO_METRICS.json',rows)


if __name__=='__main__':
    main()
