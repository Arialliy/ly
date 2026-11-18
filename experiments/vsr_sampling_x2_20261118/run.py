"""Reproduce the paired sampling-direction experiment without fitting a model."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import registration
import solver


ROOT = Path(__file__).resolve().parent
SEQUENCES = [1, 31, 61, 91, 121, 151, 181, 211]
SHIFTS = np.array([[v, 0] for v in (0, .25, -.25, .5, -.5)])
INPUT = ROOT / 'inputs'
SIZE = 128
CROP = 8


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, obj):
    out = ROOT / name
    temp = out.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(out)


def diagnostics(truth, pred):
    diff = (pred - truth)[CROP:-CROP, CROP:-CROP]
    h, w = diff.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing='ij')
    power = abs(np.fft.fft2(diff * np.hanning(h)[:,None] * np.hanning(w)[None,:], norm='forward')) ** 2
    band_x = (abs(fx) > .25) & (abs(fy) < .25)
    band_y = (abs(fy) > .25) & (abs(fx) < .25)
    return dict(mse=float(np.mean(diff**2)), x=float(power[band_x].sum()), y=float(power[band_y].sum()))


def compare(truth, lrs, offsets, sigma, reference, repeated):
    images = solver.reconstruct(lrs, offsets, ridge=.05, blur_sigma=sigma, reference=reference)
    result = {name:diagnostics(truth, im) for name, im in images.items()}
    result['repeat'] = repeated
    return result


def single(truth, reference, sigma):
    repeated = np.repeat(reference[None], 5, axis=0)
    pic = solver.reconstruct(repeated, np.zeros_like(SHIFTS), ridge=.05, blur_sigma=sigma)['classic']
    return diagnostics(truth, pic)


def controlled(image, seq, patch_id, condition, seed):
    sigma, noise_sd = (0.,0.) if condition == 'clean' else (.8,1/255)
    truth = solver.prepare_target(image)
    noise = np.random.default_rng(seed).normal(0,noise_sd,(5,SIZE//2,SIZE//2))
    rows = []
    repeated = None
    for axis, shifts in [('H',SHIFTS), ('V',SHIFTS[:,::-1])]:
        lrs = solver.observe(truth, shifts, sigma) + noise
        if repeated is None:
            ref = lrs[0]
            repeated = single(truth, ref, sigma)
        else:
            np.testing.assert_allclose(ref, lrs[0], atol=1e-12)
        estimates = registration.estimate_shifts(lrs)
        errors = np.linalg.norm(estimates[1:]-shifts[1:],axis=-1)
        row = dict(seq=seq, patch=patch_id, degradation=condition, axis=axis,
                   estimated=estimates.tolist(), epe=float(errors.mean()), max_epe=float(errors.max()))
        for mode, offsets in [('known',shifts),('estimated',estimates)]:
            row[mode] = compare(truth,lrs,offsets,sigma,0,repeated)
        rows.append(row)
    return rows


def native(sequence, patch_id, frames, seed):
    truth = solver.prepare_target(frames[2])
    zero = np.zeros((1,2))
    obs = np.stack([solver.observe(f,zero,.8)[0] for f in frames])
    obs += np.random.default_rng(seed).normal(0,1/255,obs.shape)
    estimated = registration.estimate_shifts(obs,reference=2)
    return dict(seq=sequence, patch=patch_id, estimated=estimated.tolist(),
                scores=compare(truth,obs,estimated,.8,2,single(truth,obs[2],.8)))


def chart(phase):
    y, x = np.indices((SIZE, SIZE)) / SIZE
    p = 2*np.pi
    base = .5 + .075*np.cos(p*(4*x + 5*y)) + .06*np.cos(p*(7*x-3*y) + .4)
    base += .04*np.cos(p*(2*x + 7*y) + 1.2)
    return base + .14*np.cos(p*(43*x)+phase) + .06*np.sin(p*(49*x)+phase/2)


def footage(seq, manifest):
    entries = sorted((r for r in manifest['files'] if r['sequence']==seq), key=lambda r:r['frame'])
    assert [r['frame'] for r in entries] == list(range(48,53))
    raw = []
    for entry in entries:
        path = INPUT / entry['path']
        assert digest(path) == entry['sha256']
        with Image.open(path) as source:
            assert source.mode=='RGB' and source.size==(1280,720)
            rgb = np.asarray(source, dtype=np.float64) / 255
            raw.append(rgb @ np.array([.299,.587,.114]))
    return np.stack(raw)


def main():
    manifest = json.loads((INPUT/'MANIFEST.json').read_text())
    checks = dict(solver=solver.selfcheck(), registration=registration.self_check(),
                  manifest_sha256=digest(INPUT/'MANIFEST.json'), numpy=np.__version__)
    chart_rows, controlled_rows, real_rows = [], [], []
    for j, phi in enumerate([0,.7,1.4,2.1]):
        c = chart(phi)
        for label, frame in [('x',c),('y',c.T)]:
            chart_rows.extend(controlled(frame,label,j,'clean',100+j))
    for seq in SEQUENCES:
        frames = footage(seq,manifest)
        _,h,w = frames.shape
        j = 0
        for yf in (1/3,2/3):
            for xf in (.25,.75):
                x,y = round(xf*w)-SIZE//2,round(yf*h)-SIZE//2
                stack = frames[:, y:y+SIZE, x:x+SIZE]
                seed = 461000+seq*10+j
                for cond in ['clean','blur_noise']:
                    controlled_rows.extend(controlled(stack[2],seq,j,cond,seed))
                real_rows.append(native(seq,j,stack,seed+100000))
                j+=1
        print(f'Completed sequence {seq:03}',flush=True)
    assert (len(chart_rows),len(controlled_rows),len(real_rows)) == (16,128,32)
    write('METRICS.json',dict(checks=checks, charts=chart_rows, controlled=controlled_rows, native=real_rows))


if __name__ == '__main__':
    main()
