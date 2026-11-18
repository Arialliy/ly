"""Descriptive image-motion measurements; no model-result-based grouping."""
import csv
import json
from pathlib import Path
import numpy as np
from runtime import E,SEQS,STARTS,save,sha

def read(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def trusted(r):return str(r.get('trusted')).lower() in ['true','1']
def describe(rows,key,prefix,absolute=False):
    vals=[float(r[key]) for r in rows if r.get(key,'')!='' and np.isfinite(float(r[key]))]
    if absolute:vals=list(map(abs,vals))
    return {prefix+'_n':len(vals),**{prefix+'_'+label:float(np.quantile(vals,q)) if vals else ''
            for label,q in [('median',.5),('p90',.9),('p95',.95),('max',1.)]}}

def main():
    summaries=[];windows=[];sources={}
    for folder in sorted((E/'motion').glob('DJI*')):
        p=folder/'adjacent.csv';rows=read(p);sources[str(p.relative_to(E))]=sha(p)
        assert len(rows)==159
        valid=[r for r in rows if trusted(r)]
        row={'sequence':folder.name,'split':'development' if folder.name in SEQS else 'training',
             'pairs':len(rows),'trusted_pairs':len(valid),'invalid_pairs':len(rows)-len(valid)}
        for key,label,ab in [('displacement_px','trusted_center_shift_px',False),('rotation_deg','trusted_abs_rotation_deg',True),
            ('scale','trusted_scale',False),('heldout_before_median_px','trusted_heldout_before_median_px',False),
            ('heldout_after_median_px','trusted_heldout_after_median_px',False)]:
            row.update(describe(valid,key,label,ab))
        summaries.append(row)
        if folder.name in SEQS:
            p=folder/'windows.csv';allrows=read(p);sources[str(p.relative_to(E))]=sha(p)
            for start in STARTS:
                rr=[r for r in allrows if int(r['window_start'])==start and int(r['source_frame'])!=start+19]
                assert len(rr)==39
                tr=[r for r in rr if trusted(r)]
                out={'sequence':folder.name,'start':start,'anchor':start+19,'pairs':39,'trusted_pairs':len(tr),
                    'window_trusted':int(len(tr)==39),'failure_reasons':';'.join(sorted({r['reason'] for r in rr if not trusted(r)}))}
                for key,label in [('displacement_px','center_shift_px'),('heldout_before_median_px','before_residual_px'),
                                  ('heldout_after_median_px','after_residual_px')]:out.update(describe(tr,key,label))
                windows.append(out)
    assert len(summaries)==10 and len(windows)==16
    write(E/'evaluation/MOTION_SUMMARY.csv',summaries)
    write(E/'evaluation/WINDOW_GEOMETRY_SUMMARY.csv',windows)
    save(E/'evaluation/MOTION_SUMMARY_COMPLETE.json',{'status':'COMPLETE','source_sha256':sources,
         'script_sha256':sha(__file__),'scope':'8 train and 2 development; invalid pairs excluded from numerical motion summary, counted explicitly',
         'adjacent_pairs':1590,'direct_anchor_pairs':624,'direct_anchor_trusted':sum(r['trusted_pairs'] for r in windows),
         'trusted_windows':sum(r['window_trusted'] for r in windows)})
if __name__=='__main__':main()
