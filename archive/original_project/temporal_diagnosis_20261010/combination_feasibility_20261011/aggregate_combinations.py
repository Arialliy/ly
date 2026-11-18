#!/usr/bin/env python3
"""Enumerate every shared/per-clip choice from already rescored legal plans."""
import csv, hashlib, importlib.util, itertools, json
from pathlib import Path
from collections import defaultdict
import numpy as np

C = Path(__file__).resolve().parent
Q = C.parent
SEQS = ['DJI_0026_2', 'DJI_0067_2']
FIELDS = ['gt_count', 'matched_count', 'component_count', 'unmatched_components',
          'positive_pixels', 'positive_pixels_outside_boxes']
STARTS = json.loads((C/'LEGAL_COMBINATIONS.json').read_text())['legal_full_coverage_5']
PLANS = [f'C{i:02}' for i in range(10)] + ['ALL11']
U_BUDGET = 4015

def dumpcsv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()

def main():
    z = {}
    for seq in SEQS:
        with np.load(C/'scoring'/f'{seq}_scores.npz', allow_pickle=False) as f:
            z[seq] = {k:f[k] for k in f.files}
        assert z[seq]['plans'].tolist() == PLANS
        assert z[seq]['frame_counts'].shape == (11,160,6)
        assert (z[seq]['coverage'] > 0).all()
        assert (z[seq]['coverage'][:10].sum(1) == 200).all()
    assert sum(int(z[s]['small'].sum()) for s in SEQS) == 5200
    assert sum(int(z[s]['matched'][9,z[s]['small']].sum()) for s in SEQS) == 700

    def score(choices, scope):
        row = dict(scope=scope, choice_0026=PLANS[choices[0]], choice_0067=PLANS[choices[1]],
                   starts_0026=json.dumps(STARTS[choices[0]] if choices[0]<10 else list(range(0,121,12))),
                   starts_0067=json.dumps(STARTS[choices[1]] if choices[1]<10 else list(range(0,121,12))),
                   small_gt=5200, small_matched=0, gained=0, lost=0, net=0,
                   original_sym_union_retained=0, symmetric_gains_retained=0, beyond_original_sym_union=0,
                   both_sequence_U_within_original=True)
        totals = np.zeros(6, np.int64)
        for seq, pi, ub in zip(SEQS, choices, [741,3274]):
            a=z[seq]; sm=a['small'].astype(bool); orig=a['matched'][9].astype(bool)
            sym=a['matched'][3].astype(bool); cur=a['matched'][pi].astype(bool)
            row['small_matched'] += int((sm & cur).sum())
            row['gained'] += int((sm & cur & ~orig).sum())
            row['lost'] += int((sm & ~cur & orig).sum())
            row['original_sym_union_retained'] += int((sm & cur & (orig|sym)).sum())
            row['symmetric_gains_retained'] += int((sm & cur & sym & ~orig).sum())
            row['beyond_original_sym_union'] += int((sm & cur & ~(orig|sym)).sum())
            fc=a['frame_counts'][pi].sum(0); totals+=fc
            row['both_sequence_U_within_original'] &= bool(fc[3]<=ub)
        row['net']=row['gained']-row['lost']
        assert row['small_matched']==700+row['net']
        row.update({f:int(v) for f,v in zip(FIELDS,totals)})
        row['U_budget_feasible']=bool(totals[3]<=U_BUDGET)
        row['match_percent']=100*row['small_matched']/5200
        row['delta_percentage_points']=100*row['net']/5200
        return row

    shared=[score((i,i),'shared') for i in range(10)]
    pairs=[score((i,j),'per_clip_hindsight') for i,j in itertools.product(range(10),repeat=2)]
    all11=score((10,10),'all11_extra_budget_reference')
    dumpcsv(C/'SHARED_COMBINATIONS.csv', shared)
    dumpcsv(C/'PER_CLIP_COMBINATIONS.csv', pairs)
    dumpcsv(C/'ALL11_REFERENCE.csv', [all11])

    def rank(r):
        return (-r['small_matched'],r['lost'],r['unmatched_components'],r['positive_pixels_outside_boxes'],
                r['choice_0026'],r['choice_0067'])
    selections=[]
    for scope,rows in [('shared',shared),('per_clip_hindsight',pairs)]:
        for use_budget in [True,False]:
            for limit in [0,7,35,None]:
                eligible=[r for r in rows if (not use_budget or r['U_budget_feasible'])
                          and (limit is None or r['lost']<=limit)]
                assert eligible
                best=min(eligible,key=rank)
                selections.append(dict(require_U_budget=use_budget,loss_limit='unlimited' if limit is None else limit,
                                       feasible_count=len(eligible),**best))
    dumpcsv(C/'TRADEOFF_SELECTIONS.csv', selections)
    front=[]
    for scope,rows in [('shared',shared),('per_clip_hindsight',pairs)]:
        for r in rows:
            dominates=lambda s: (s['gained']>=r['gained'] and s['lost']<=r['lost'] and
                s['unmatched_components']<=r['unmatched_components'] and
                (s['gained']>r['gained'] or s['lost']<r['lost'] or s['unmatched_components']<r['unmatched_components']))
            if not any(dominates(s) for s in rows): front.append(r)
    dumpcsv(C/'PARETO_GAIN_LOSS_U.csv', front)

    # Reconstruct metadata with the exact frozen annotation order; no prediction recomputation.
    path=Q/'stage1_goal/pilot/score_pilot.py'
    spec=importlib.util.spec_from_file_location('pilot_score',path)
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    helper,_=scorer.helper()
    meta={s:scorer.gt_data(Q/'stage1_goal/data'/s/'annotations.json',helper)[1] for s in SEQS}
    selected_pairs=sorted(set((r['choice_0026'],r['choice_0067']) for r in selections if r['require_U_budget'])|{('C09','C09'),('C03','C03'),('ALL11','ALL11')})
    trackrows=[]; objectrows=[]
    for p0,p1 in selected_pairs:
        for seq,plan in zip(SEQS,[p0,p1]):
            a=z[seq];pi=PLANS.index(plan); tracks=defaultdict(lambda:dict(n=0,original_matched=0,matched=0,gained=0,lost=0))
            for i,m in enumerate(meta[seq]):
                assert bool(m['strict_small'])==bool(a['small'][i])
                if not a['small'][i]:continue
                om,cm=bool(a['matched'][9,i]),bool(a['matched'][pi,i])
                t=tracks[m['track_id']];t['n']+=1;t['original_matched']+=int(om);t['matched']+=int(cm)
                t['gained']+=int(cm and not om);t['lost']+=int(om and not cm)
                objectrows.append(dict(choice_0026=p0,choice_0067=p1,sequence=seq,frame=m['frame'],
                    source_frame=m['source_frame'],annotation_id=m['annotation_id'],track_id=m['track_id'],
                    original_matched=int(om),matched=int(cm),gained=int(cm and not om),lost=int(om and not cm)))
            trackrows.extend(dict(choice_0026=p0,choice_0067=p1,sequence=seq,track_id=tid,**t,net=t['gained']-t['lost']) for tid,t in sorted(tracks.items()))
    dumpcsv(C/'SELECTED_TRACK_RESULTS.csv',trackrows)
    dumpcsv(C/'SELECTED_OBJECT_RESULTS.csv',objectrows)

    summary=dict(status='COMPLETE',count_shared=len(shared),count_per_clip=len(pairs),threshold=.5,
        U_budget=U_BUDGET,shared_feasible=sum(r['U_budget_feasible'] for r in shared),
        per_clip_feasible=sum(r['U_budget_feasible'] for r in pairs),
        maximum_original_sym_union_retained_shared=max(r['original_sym_union_retained'] for r in shared),
        maximum_original_sym_union_retained_per_clip=max(r['original_sym_union_retained'] for r in pairs),
        selections=selections,all11=all11,shared=shared,
        support_cross_check='CROSS_CHECK.json (separate identity-joined verification)',
        sources={str(C/'scoring'/f'{s}_scores.npz'):sha(C/'scoring'/f'{s}_scores.npz') for s in SEQS},
        limitations=['Ground-truth-assisted post hoc selection on two previously used development clips.',
                     'Background constraint is unmatched component count; outside-box pixels are descriptive.',
                     'These are diagnostic peak-in-box matches, not official Pd/Fa.',
                     'ALL11 uses 2.2x the window-frame budget; larger-budget subsets were not actually fused exhaustively.'])
    (C/'AGGREGATE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5.5),constrained_layout=True)
    for feasible,color,label in [(False,'#c6cbd0','Per-clip hindsight: U > 4015'),(True,'#7badd0','Per-clip hindsight: U <= 4015')]:
        rr=[r for r in pairs if r['U_budget_feasible']==feasible]
        ax.scatter([r['lost'] for r in rr],[r['gained'] for r in rr],s=28,c=color,alpha=.7,label=label,zorder=2)
    ax.scatter([r['lost'] for r in shared],[r['gained'] for r in shared],marker='D',s=46,facecolors='none',edgecolors='#172f46',label='Shared arrangements',zorder=3)
    for r in shared:
        offset={'C04':(8,-10),'C05':(-25,7)}.get(r['choice_0026'],(4,3))
        ax.annotate(r['choice_0026'],(r['lost'],r['gained']),xytext=offset,textcoords='offset points',fontsize=8)
    maxxy=max(max(r['lost'],r['gained']) for r in pairs)+8
    ax.plot([0,maxxy],[0,maxxy],ls='--',color='#555',lw=.8,label='Zero net gain',zorder=1)
    ax.set(xlim=(-3,maxxy),ylim=(-3,maxxy),xlabel='Original successes lost (target-frames)',ylabel='New matches (target-frames)',
           title='Exhaustive legal five-window choices | 2 development clips, 5200 small targets')
    ax.grid(alpha=.15);ax.legend(fontsize=8,loc='upper left')
    fig.savefig(C/'GAIN_LOSS_BUDGET.png',dpi=180);fig.savefig(C/'GAIN_LOSS_BUDGET.svg');plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['count_shared','count_per_clip','shared_feasible','per_clip_feasible','selections','all11']},indent=2))

if __name__=='__main__':main()
