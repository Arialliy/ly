"""Attribute position misses to center evidence, image margin and tile budget."""
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from extract import ROOT, RUN, signal, positions


def upper_bound(xy, boxes):
    xy = xy.T
    candidates, objects = [], []
    for i, (x,y,w,h) in enumerate(boxes):
        ids = np.flatnonzero((xy[0]>=x-1) & (xy[0]<=x+w+1) &
                             (xy[1]>=y-1) & (xy[1]<=y+h+1))
        candidates.extend(ids)
        objects.extend([i]*len(ids))
    graph = csr_matrix((np.ones(len(objects)),(objects,candidates)),shape=(len(boxes),xy.shape[1]))
    return int(np.sum(maximum_bipartite_matching(graph,perm_type='column')>=0))


def main():
    cv2.setNumThreads(1)
    meta=json.loads((RUN/'labels.json').read_text())['frames']
    out={}
    for seq in sorted({r['sequence'] for r in meta if r['split']=='calibration'}):
        frames=np.load(ROOT/'data/stage1'/seq/'raw_frames.npy',mmap_mode='r')
        counts=Counter()
        for r in meta:
            if r['sequence']!=seq:
                continue
            z,_=signal(frames[r['index']])
            strength=np.abs(z)
            yy,xx=np.where((strength==cv2.dilate(strength,np.ones((5,5),np.uint8))) & (strength>0))
            pts=np.stack((xx,yy),axis=1)
            height,width=z.shape
            interior=(xx>=25)&(xx<width-25)&(yy>=25)&(yy<height-25)
            counts.update(dict(frames=1, targets=len(r['targets']),
                any_center_peak=upper_bound(pts,r['targets']),
                center_peak_after_margin=upper_bound(pts[interior],r['targets']),
                final_budget=upper_bound(positions(z),r['targets'])))
        out[seq]=dict(counts)
    total=sum((Counter(v) for v in out.values()),Counter())
    out['all_dev']=dict(total)
    scores=json.loads((RUN/'scores.json').read_text())['methods']['AB_multi_full']
    for name,counts in out.items():
        assert counts['final_budget']==scores[name]['candidate_oracle_tp']
        assert counts['targets']>=counts['any_center_peak']>=counts['center_peak_after_margin']>=counts['final_budget']
    (RUN/'coverage.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out)


if __name__=='__main__':
    main()
