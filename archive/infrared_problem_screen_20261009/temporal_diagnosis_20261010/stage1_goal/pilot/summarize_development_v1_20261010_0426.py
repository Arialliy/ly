#!/usr/bin/env python3
"""Secondary raw-.5 diagnostics on the unchanged, prespecified old cohorts.

Calibration failed: these outputs cannot establish a feasible operating point.
No cohort is selected from candidate responses; no confirmation is accessed.
"""
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

P = Path(__file__).resolve().parents[1]
Q = P.parent
VARIANTS = ['paired_mean', 'paired_max']
SEQ = ['DJI_0028_2', 'DJI_0028_4', 'DJI_0032_5', 'DJI_0047_3']


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return (int(row.get('local_frame', row.get('frame'))), int(row['annotation_id']))


def old_values(row):
    return dict(matched=int(row['matched']), box_max=float(row['box_max']),
                peak_x=float(row['peak_x']) if row['peak_x'] else None,
                peak_y=float(row['peak_y']) if row['peak_y'] else None,
                track_id=int(row['track_id']), source_frame=int(row['source_frame']),
                bbox=[float(row['bbox_'+a]) for a in ['x', 'y', 'w', 'h']])


def load_new(variant, seq, name):
    d = P/'diagnostic_evaluation'/variant/seq
    meta = json.loads((d/(name+'_COMPLETE.json')).read_text())
    assert meta['status'] == 'COMPLETE'
    assert meta['name'] == name
    with np.load(d/(name+'_matches.npz'), allow_pickle=False) as z:
        ts = z['thresholds'].tolist()
        ti = ts.index(.5)
        assert ts == meta['thresholds']
        assert z['matched'].shape[1] == len(meta['objects'])
        rows = {}
        for i, obj in enumerate(meta['objects']):
            k = key(obj)
            assert k not in rows
            rows[k] = dict(obj, matched=int(z['matched'][ti, i]),
                          box_max=float(z['box_max'][i]),
                          peak_x=float(z['peak_x'][ti, i]) if z['matched'][ti, i] else None,
                          peak_y=float(z['peak_y'][ti, i]) if z['matched'][ti, i] else None)
    return rows


def longest_miss(rows, field):
    run = best = 0
    previous = None
    for r in sorted(rows, key=lambda r: r['source_frame']):
        f = r['source_frame']
        if previous is None or f != previous + 1 or r[field]:
            run = 0
        if not r[field]:
            run += 1
            best = max(best, run)
        previous = f
    return best


def total(rows):
    summed = ['original_fused_match', 'candidate_fused_match', 'fused_gain', 'fused_loss',
              'window_gain', 'window_loss', 'window_count', 'original_window_matches',
              'candidate_window_matches', 'any_to_none', 'none_to_any',
              'original_all_match', 'all_to_not_all', 'protected_original', 'protected_lost',
              'a_original_match', 'b_original_match', 'a_candidate_match', 'b_candidate_match',
              'a_candidate_ge05', 'b_candidate_ge05', 'a_candidate_ge09', 'b_candidate_ge09',
              'candidate_all_box_low', 'candidate_all_box_high', 'candidate_box_flip',
              'response_flip_to_both_low']
    out = {k: sum(r[k] for r in rows) for k in summed}
    out.update(n=len(rows), tracks=len({(r['sequence'], r['track_id']) for r in rows}),
               fused_net=out['fused_gain']-out['fused_loss'],
               protected_loss_rate=(out['protected_lost']/out['protected_original']
                                    if out['protected_original'] else None),
               protection_satisfied=(bool(out['protected_original']) and
                                      100*out['protected_lost'] <= out['protected_original']))
    disp = [r['fused_joint_peak_shift'] for r in rows if r['fused_joint_peak_shift'] is not None]
    out['joint_fused_peak_displacement'] = dict(n=len(disp),
        mean=float(np.mean(disp)) if disp else None, median=float(np.median(disp)) if disp else None,
        p95=float(np.quantile(disp, .95)) if disp else None)
    return out


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    out = P/'summary/development'; out.mkdir(parents=True, exist_ok=True)
    dest = out/'DEVELOPMENT.json'
    if dest.exists():
        raise FileExistsError('Completed development analysis must not be silently replaced')
    selection = P/'local_branch_probe/full/selected_pairs.csv'
    cohorts = read_csv(selection)
    assert len(cohorts) == 1417
    group_flags = {'strong97':'in_strong_97', 'suppression24':'in_suppression_24',
                   'support42':'in_support_42', 'stable_high502':'in_stable_high_502',
                   'stable_low778':'in_stable_low_778'}
    assert [sum(int(r[k]) for r in cohorts) for k in group_flags.values()] == [97,24,42,502,778]
    rows=[]; frame_totals=[]; fingerprints={str(selection):digest(selection)}
    for seq in SEQ:
        op=Q/'analysis'/seq/'per_track_frame.csv'
        cp=Q/'window_offset_experiment/analysis'/seq/'per_context_all_tracks.csv'
        fp=Q/'analysis'/seq/'per_frame_counts.csv'
        for p in [op,cp,fp]: fingerprints[str(p)]=digest(p)
        old_fused={key(r):old_values(r) for r in read_csv(op) if float(r['threshold'])==.5}
        old_ctx={(int(r['window_start']),*key(r)):old_values(r) for r in read_csv(cp)
                 if float(r['threshold'])==.5}
        counts=[r for r in read_csv(fp) if float(r['threshold'])==.5]
        assert len(counts)==160
        frame_totals.append(dict(variant='official',sequence=seq,threshold=.5,frames=160,
            matched=sum(int(r['matched_count']) for r in counts),
            components=sum(int(r['component_count']) for r in counts),
            unmatched_components=sum(int(r['unmatched_components']) for r in counts),
            positive_pixels_outside_boxes=sum(int(r['positive_pixels_outside_boxes']) for r in counts)))
        for v in VARIANTS:
            fused=load_new(v,seq,'official_fused')
            contexts={s:load_new(v,seq,f'window_{s:03d}') for s in range(0,121,12)}
            assert set(fused)==set(old_fused)
            for k, ov in old_fused.items():
                nv=fused[k]
                assert ov['track_id']==nv['track_id'] and ov['source_frame']==nv['source_frame']
                assert ov['bbox']==nv['bbox']
            d=P/'diagnostic_evaluation'/v/seq
            for p in d.glob('*'):
                if p.suffix in ['.npz','.json','.csv']: fingerprints[str(p)]=digest(p)
            cr=next(r for r in read_csv(d/'official_fused_curves.csv') if float(r['threshold'])==.5)
            frame_totals.append(dict(variant=v,sequence=seq,threshold=.5,frames=160,
                **{k:int(cr[k]) for k in ['matched','components','unmatched_components','positive_pixels_outside_boxes']}))
            for c in [r for r in cohorts if r['sequence']==seq]:
                k=key(c); frame=k[0]
                starts=[s for s in range(0,121,12) if s<=frame<s+40]
                assert len(starts)==int(c['context_count'])>=2
                oo=[old_ctx[(s,*k)] for s in starts]; nn=[contexts[s][k] for s in starts]
                for a,b in zip(oo,nn):
                    assert a['bbox']==b['bbox'] and a['track_id']==b['track_id'] and a['source_frame']==b['source_frame']
                om=np.array([r['matched'] for r in oo],bool); nm=np.array([r['matched'] for r in nn],bool)
                ob=np.array([r['box_max'] for r in oo]); nb=np.array([r['box_max'] for r in nn])
                of=old_fused[k]; nf=fused[k]
                assert of['matched']==int(c['original_point'])
                assert abs(ob.min()-float(c['context_boxmax_min']))<1e-7
                assert abs(ob.max()-float(c['context_boxmax_max']))<1e-7
                ia=starts.index(int(c['a_start']));ib=starts.index(int(c['b_start']))
                protected=bool(om.all() and of['matched'])
                joint=bool(of['matched'] and nf['matched'])
                rows.append(dict(variant=v,sequence=seq,local_frame=frame,annotation_id=k[1],
                    track_id=int(c['track_id']),source_frame=int(c['source_frame']),
                    **{name:int(c[flag]) for name,flag in group_flags.items()},
                    original_fused_match=of['matched'], candidate_fused_match=nf['matched'],
                    fused_gain=int(not of['matched'] and nf['matched']),
                    fused_loss=int(of['matched'] and not nf['matched']),
                    window_count=len(starts),original_window_matches=int(om.sum()),candidate_window_matches=int(nm.sum()),
                    window_gain=int((~om&nm).sum()),window_loss=int((om&~nm).sum()),
                    any_to_none=int(om.any() and not nm.any()),none_to_any=int(not om.any() and nm.any()),
                    original_all_match=int(om.all()),all_to_not_all=int(om.all() and not nm.all()),
                    protected_original=int(protected),protected_lost=int(protected and not nf['matched']),
                    a_start=starts[ia],b_start=starts[ib],a_original_match=int(om[ia]),b_original_match=int(om[ib]),
                    a_candidate_match=int(nm[ia]),b_candidate_match=int(nm[ib]),
                    a_candidate_ge05=int(nb[ia]>=.5),b_candidate_ge05=int(nb[ib]>=.5),
                    a_candidate_ge09=int(nb[ia]>=.9),b_candidate_ge09=int(nb[ib]>=.9),
                    candidate_all_box_low=int(nb.max()<.5),candidate_all_box_high=int(nb.min()>=.5),
                    candidate_box_flip=int(nb.min()<.5<=nb.max()),
                    response_flip_to_both_low=int(ob.min()<.5<=ob.max() and nb.max()<.5),
                    fused_joint_peak_shift=float(np.hypot(nf['peak_x']-of['peak_x'],nf['peak_y']-of['peak_y'])) if joint else None))
    assert len(rows)==2834
    summary={}
    byseq=[];tracks=[]
    for v in VARIANTS:
        rr=[r for r in rows if r['variant']==v]
        summary[v]={'all1417':total(rr),**{g:total([r for r in rr if r[g]]) for g in group_flags}}
        for seq in SEQ:
            st=total([r for r in rr if r['sequence']==seq])
            byseq.append(dict(variant=v,sequence=seq,**{k:val for k,val in st.items() if not isinstance(val,dict)}))
        tt=defaultdict(list)
        for r in rr:tt[(r['sequence'],r['track_id'])].append(r)
        for (seq,track),rrr in sorted(tt.items()):
            tracks.append(dict(variant=v,sequence=seq,track_id=track,n=len(rrr),
                original_longest_miss=longest_miss(rrr,'original_fused_match'),
                candidate_longest_miss=longest_miss(rrr,'candidate_fused_match'),
                fused_gain=sum(r['fused_gain'] for r in rrr),fused_loss=sum(r['fused_loss'] for r in rrr)))
    aggregate={v:{k:sum(r[k] for r in frame_totals if r['variant']==v)
                   for k in ['frames','matched','components','unmatched_components','positive_pixels_outside_boxes']}
               for v in ['official']+VARIANTS}
    payload=dict(status='COMPLETE_SECONDARY_UNCALIBRATED_DIAGNOSIS',threshold=.5,
        interpretation='Calibration failed. Old development cohorts and raw .5 only; these are not equal-budget improvements or independent confirmation.',
        cohorts=summary,all640_frames=aggregate,original_baseline_1417_match_count=549,
        fingerprints=fingerprints,script_sha256=digest(Path(__file__)))
    assert all(summary[v]['all1417']['original_fused_match']==549 for v in VARIANTS)
    save_csv(out/'OBJECTS.csv',rows);save_csv(out/'SEQUENCES.csv',byseq)
    save_csv(out/'TRACKS.csv',tracks);save_csv(out/'FRAME_COUNTS.csv',frame_totals)
    with dest.open('x') as f:json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=payload['status'],cohorts=summary,all640_frames=aggregate),ensure_ascii=False))


if __name__=='__main__':main()
