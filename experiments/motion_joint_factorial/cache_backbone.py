"""Cache identical native TPro center features and official output scores for all arms."""
import argparse
import time

import numpy as np
import torch

from protocol import CENTERS, DEV, TRAIN, frame_path, raw_frames, save_npz, setup, training_points, workers_frames
from model import load_official_detector, normalize_frames


@torch.inference_mode()
def native_center(detector, window):
    x = normalize_frames(torch.from_numpy(window.copy()).cuda())[None,None]
    h,w = window.shape[-2:]
    features = np.empty((h,w,32), dtype='float32')
    logits = np.empty((h,w), dtype='float32')
    slot = CENTERS[0]
    captured = []
    def capture(_layer, args):
        captured.append(args[0][0,:,slot])
    handle = detector.conv_out1.register_forward_pre_hook(capture)
    try:
        for y in range(0,h,128):
            for z in range(0,w,128):
                bottom, right = min(y+128,h), min(z+128,w)
                ya, za = max(0,y-16), max(0,z-16)
                tile = x[...,ya:min(bottom+16,h), za:min(right+16,w)].contiguous()
                out = detector(tile)[1][0,slot]
                sy, sx = slice(y-ya,bottom-ya), slice(z-za,right-za)
                features[y:bottom,z:right] = captured.pop()[:,sy,sx].permute(1,2,0).cpu().numpy()
                logits[y:bottom,z:right] = out[sy,sx].cpu().numpy()
    finally:
        handle.remove()
    if not np.isfinite(features).all() or not np.isfinite(logits).all():
        raise FloatingPointError('nonfinite native output')
    return features,logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    setup()
    detector = load_official_detector("cuda")
    for seq,idx in workers_frames(TRAIN+DEV,args.shard,args.workers):
        dest = frame_path('backbone',seq,idx)
        if dest.exists():
            continue
        started=time.perf_counter()
        volume=raw_frames(seq)
        features,base=native_center(detector,volume[idx-19:idx+21])
        if seq in TRAIN:
            xy,target=training_points(seq,idx,base)
            ys,xs=xy[:,1],xy[:,0]
            save_npz(dest, features=features[ys,xs],base=base[ys,xs],xy=xy,target=target)
        else:
            save_npz(dest,features=features,base=base)
        print(seq,idx,'native seconds',round(time.perf_counter()-started,2),flush=True)


if __name__ == '__main__':
    main()
