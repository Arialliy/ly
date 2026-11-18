"""Post-hoc coverage and per-annotation C/D matching on saved thermal traffic maps."""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

HERE = Path(__file__).resolve().parent
CONF = HERE.parent / "motion_joint_confirmation"
sys.path.insert(0, str(CONF))
from locked_protocol import CENTERS, COHORT, ROOT, frame_record
from verify_counts import annotated_boxes, incidence

sys.path.insert(0, str(HERE.parent / "motion_joint_factorial"))
from evaluate import peaks

MODELS = ("original", "A", "B", "C", "D")
BUDGETS = ("0.5", "1", "5")
OPS = (*BUDGETS, "development_threshold")
SIZE = (512, 640)


def table(seq, idx, metadata):
    frame = frame_record(seq, idx)
    image, = (im for im in metadata['images'] if im['frame_id'] == idx)
    rows = []
    for ann in metadata['annotations']:
        if ann['image_id'] != image['id']:
            continue
        x, y, w, h = ann['bbox']
        if (ann['category_id'] in range(5) and not ann.get('ignore') and not ann.get('iscrowd')
                and 0 < w*h <= 144 and max(w,h) <= 16):
            rows.append(ann)
    assert [r['bbox'] for r in rows] == frame['targets']
    good, ignore = annotated_boxes(metadata, idx)
    return rows, good, ignore


def cardinality(edges):
    """Calculate the exact maximum one-to-one set size."""
    reduced = edges[edges.any(axis=1)]
    if not reduced.size:
        return 0
    assignment = maximum_bipartite_matching(csr_matrix(reduced), perm_type='column')
    return int(np.count_nonzero(assignment >= 0))


def coverage(edges):
    cap = cardinality(edges)
    return dict(matchable=cap, boxes_touched=int(edges.any(axis=0).sum()),
                candidates_touching_box=int(edges.any(axis=1).sum()))


def pixels_in_boxes(boxes, margin):
    """Enumerate the only pixel rows that can affect the one-to-one upper bound."""
    height,width=SIZE
    bitmap = np.zeros(SIZE, dtype=bool)
    for x0,y0,x1,y1 in boxes:
        xa=max(margin,int(np.ceil(x0)))
        xb=min(width-1-margin,int(np.floor(x1)))
        ya=max(margin,int(np.ceil(y0)))
        yb=min(height-1-margin,int(np.floor(y1)))
        if xa <= xb and ya <= yb:
            bitmap[ya:yb+1,xa:xb+1] = True
    y,x=np.nonzero(bitmap)
    return np.column_stack((x,y))


def target_sets(edges, ids):
    """Report one maximum matching, possible and forced matched target IDs.

    Every nonisolated target can be covered by some maximum matching: if it is
    currently unmatched, exchange one incident candidate for that candidate's
    previous target. A target is forced if omitting it lowers the maximum size.
    """
    e=edges[edges.any(axis=1)]
    matched=np.zeros(len(ids), dtype=bool)
    if e.size:
        assignment=maximum_bipartite_matching(csr_matrix(e), perm_type='row')
        matched=assignment >= 0
    assert len(matched)==len(ids)
    n=int(matched.sum())
    possible=e.any(axis=0)
    must=matched.copy()
    if n != possible.sum():
        for b in np.flatnonzero(matched):
            must[b]=cardinality(np.delete(e,b,axis=1)) < n
    assert np.all(~must|matched) and np.all(~matched|possible)
    return dict(witness=ids[matched].tolist(), forced=ids[must].tolist(), possible=ids[possible].tolist())


def overlap_extrema(c, d):
    """Exact minimum/maximum common GTs over all C/D maximum-cardinality pairings."""
    bounds=[]
    edges=[]
    for graph in (c,d):
        p,b=np.nonzero(graph)
        edges.append((p,b))
        bounds.append(cardinality(graph))
    M,N=len(edges[0][0]),len(edges[1][0])
    T=c.shape[1]
    rows=[]; lower=[]; upper=[]
    def add(row,lo,hi):
        rows.append(row);lower.append(lo);upper.append(hi)
    indicators=[]
    for offset,((p,b),k) in zip((0,M),zip(edges,bounds)):
        ar=np.arange(len(p))+offset
        for i in np.unique(p):
            row=np.zeros(M+N+T); row[ar[p==i]]=1
            add(row,0,1)
        idrows=[]
        for j in range(T):
            row=np.zeros(M+N+T); row[ar[b==j]]=1
            add(row,0,1); idrows.append(row)
        indicators.append(idrows)
        row=np.zeros(M+N+T); row[ar]=1
        add(row,k,k)
    for j,(u,v) in enumerate(zip(*indicators)):
        z=np.zeros(M+N+T); z[M+N+j]=1
        add(z-u,-np.inf,0);add(z-v,-np.inf,0);add(z-u-v,-1,np.inf)
    objective=np.zeros(M+N+T);objective[M+N:]=1
    constraint=LinearConstraint(csr_matrix(np.asarray(rows)),lower,upper)
    answer=[]
    for sign in (1,-1):
        opt=milp(sign*objective,integrality=np.ones(len(objective)),bounds=Bounds(0,1),constraints=constraint)
        if not opt.success:
            raise RuntimeError(opt.message)
        answer.append(int(round(sign*opt.fun)))
    return answer


def paired(c,d, graphs):
    sets=lambda obj,key: set(obj[key])
    cw,dw=sets(c,'witness'),sets(d,'witness')
    changeable=(sets(c,'forced')!=sets(c,'possible') or sets(d,'forced')!=sets(d,'possible'))
    bounds=overlap_extrema(*graphs) if changeable else [len(cw&dw)]*2
    return dict(common=sorted(cw&dw),C_only=sorted(cw-dw),D_only=sorted(dw-cw),
                common_min_max=bounds,C_only_min_max=[len(cw)-bounds[1],len(cw)-bounds[0]],
                D_only_min_max=[len(dw)-bounds[1],len(dw)-bounds[0]],
                C_optional=sorted(sets(c,'possible')-sets(c,'forced')),
                D_optional=sorted(sets(d,'possible')-sets(d,'forced')))


def summarize_pairs(records):
    out={}
    for key in records[0]:
        if key.endswith('min_max'):
            out[key]=np.sum([r[key] for r in records],axis=0).tolist()
        else:
            out[key]=sorted(i for r in records for i in r[key])
    return out


def thresholds(ref):
    return {op:ref['froc'][op]['logit_threshold'] for op in BUDGETS} | {
        'development_threshold':ref['fixed_development_primary']['logit_threshold']}


def main():
    cv2.setNumThreads(1)
    archived=json.loads((CONF/'runs/scores.json').read_text())
    report={'unit':'annotation_id per source and frame; track_id is a supplementary attribute',
            'scope':'post-hoc analysis of twelve previously reported civilian confirmation frames',
            'thresholds':{m:thresholds(archived['methods'][m]['all_confirmation']) for m in MODELS}}
    counts={m:Counter() for m in MODELS}
    dense=Counter()
    raw_rows=[]
    identity={}
    for seq in COHORT:
        meta=json.loads((ROOT/'data/stage1'/seq/'annotations.json').read_text())
        for idx in CENTERS:
            anns,good,ignore=table(seq,idx,meta)
            ids=np.array([a['id'] for a in anns],dtype=int)
            for ann in anns:
                assert ann['id'] not in identity
                identity[ann['id']]=dict(source=seq,frame=idx,track_id=ann['track_id'],
                                         category_id=ann['category_id'],bbox=ann['bbox'])
            frame={'source':seq,'frame':idx,'n_gt':len(good)}
            dense_row={}
            for name,margin in [('all_pixels',0),('motion_usable_pixels',1),('conservative_all_paths_pixels',25)]:
                points=pixels_in_boxes(good,margin)
                cov=coverage(incidence(points,good))
                dense_row[name]=[cov['boxes_touched'],cov['matchable']]
                dense[name]+=cov['matchable']
            dense['n_gt']+=len(good)
            frame['dense_upper_bounds_touched_and_matched']=dense_row
            model_rows={}
            ops_graphs={k:{} for k in OPS}
            with np.load(CONF/'runs/predictions'/f'{seq}_{idx:03d}.npz') as cache:
                for m in MODELS:
                    im=cache[m]
                    py,px=np.nonzero(im==cv2.dilate(im,np.ones((5,5),dtype='uint8')))
                    before=np.column_stack((px,py))
                    xy,score=peaks(im)
                    e=incidence(xy,good)
                    ignored=(~e.any(axis=1)) & incidence(xy,ignore).any(axis=1)
                    row={'local_peaks':len(before),'nms_peaks':len(xy),
                         'pre_nms_oracle':cardinality(incidence(before,good)),
                         'post_nms_oracle':cardinality(e),'post_nms_peaks_touching_GT':int(e.any(axis=1).sum()),
                         'post_nms_GT_with_any_peak':int(e.any(axis=0).sum()),'ignored_peaks':int(ignored.sum())}
                    model_rows[m]={'stages':row,'ops':{}}
                    counts[m].update(row)
                    for op,t in report['thresholds'][m].items():
                        chosen=(score>=t)&~ignored if t is not None else np.zeros(len(score),dtype=bool)
                        g=e[chosen]
                        n=cardinality(g)
                        details={'tp':n,'fp':int(chosen.sum())-n}
                        if m in ('C','D'):
                            details.update(target_sets(g,ids))
                            ops_graphs[op][m]=g
                        model_rows[m]['ops'][op]=details
            frame['models']=model_rows
            frame['paired']={op:paired(model_rows['C']['ops'][op],model_rows['D']['ops'][op],
                                       (ops_graphs[op]['C'],ops_graphs[op]['D'])) for op in OPS}
            raw_rows.append(frame)
    assert len(raw_rows)==12 and len(identity)==519
    for m in MODELS:
        ref=archived['methods'][m]['all_confirmation']
        for k,v in [('post_nms_oracle','candidate_oracle_tp'),('nms_peaks','candidates'),('ignored_peaks','ignored_candidates')]:
            assert counts[m][k]==ref[v],(m,k)
        for op in OPS:
            row=ref['froc'][op] if op in BUDGETS else ref['fixed_development_primary']
            for key in ('tp','fp'):
                assert sum(fr['models'][m]['ops'][op][key] for fr in raw_rows)==row[key],(m,op,key)
    report.update(dense=dict(dense),coverage_by_model={m:dict(v) for m,v in counts.items()},
                  pairs_by_group={k:{op:summarize_pairs([r['paired'][op] for r in raw_rows
                    if k=='pooled' or r['source']==k]) for op in OPS} for k in ('pooled',*COHORT)},
                  annotation_lookup=identity,frames=raw_rows)
    path=HERE/'runs/identity_coverage.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(report,separators=(',',':'),allow_nan=False)+'\n')
    temp.replace(path)
    print('dense',report['dense'])
    print('coverage',report['coverage_by_model'])
    for group,r in report['pairs_by_group'].items():
        print(group,{k:{key:v[key] for key in ('common_min_max','C_only_min_max','D_only_min_max')} for k,v in r.items()})
    print(path)


if __name__=='__main__':
    main()
