"""Tabulate results with videos as the aggregation unit."""
import json
import math
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parent
db=lambda ratio: 10*math.log10(ratio)
METHODS=('prior','repeat','scalar','classic','swapped')


def summarize_dataset(rows, group, controlled):
    out={}
    for method in METHODS:
        vals=[r[group][method] for r in rows]
        item={'psnr':-db(mean(v['mse'] for v in vals)),
              'mean_patch_psnr':mean(-db(v['mse']) for v in vals)}
        if controlled:
            matched=sum(v['x' if r['axis']=='H' else 'y'] for r,v in zip(rows,vals))
            other=sum(v['y' if r['axis']=='H' else 'x'] for r,v in zip(rows,vals))
            item['direction_db']=db(other/matched)
        out[method]=item
    return out


def averages(by_seq):
    first=next(iter(by_seq.values()))
    return {method:{key:mean(v[method][key] for v in by_seq.values()) for key in first[method]}
            for method in METHODS}


def main():
    metrics=json.loads((ROOT/'METRICS.json').read_text())
    halo=json.loads((ROOT/'HALO_METRICS.json').read_text())
    seqs=sorted({r['seq'] for r in metrics['native']})
    output={'controlled':{},'chart_direction_db':{}}
    for mode in ['known','estimated']:
        charts=metrics['charts']
        relevant=lambda r: r[mode]['classic'][r['seq']]
        good=sum(relevant(r) for r in charts if (r['axis']=='H')==(r['seq']=='x'))
        bad=sum(relevant(r) for r in charts if (r['axis']=='H')!=(r['seq']=='x'))
        output['chart_direction_db'][mode]=db(bad/good)
    for condition in ['clean','blur_noise']:
        rows=[r for r in metrics['controlled'] if r['degradation']==condition]
        box={'epe_lr':mean(r['epe'] for r in rows)}
        for mode in ['known','estimated']:
            by_seq={s:summarize_dataset([r for r in rows if r['seq']==s],mode,True) for s in seqs}
            box[mode]=dict(per_sequence=by_seq,mean=averages(by_seq),
                           positive=sum(v['classic']['direction_db']>0 for v in by_seq.values()))
        output['controlled'][condition]=box
    native=metrics['native']
    by_seq={s:summarize_dataset([r for r in native if r['seq']==s],'scores',False) for s in seqs}
    output['native']=dict(per_sequence=by_seq,mean=averages(by_seq),
                          wins=sum(v['classic']['psnr']>v['repeat']['psnr'] for v in by_seq.values()),
                          mean_patch_wins=sum(v['classic']['mean_patch_psnr']>v['repeat']['mean_patch_psnr'] for v in by_seq.values()))
    support={}
    for size in ['narrow','wide']:
        by_seq={s:{m:-db(mean(r[size][m] for r in halo if r['seq']==s)) for m in ('classic','repeat')} for s in seqs}
        support[size]={'per_sequence':by_seq, 'mean':{m:mean(v[m] for v in by_seq.values()) for m in ('classic','repeat')},
                       'mean_patch':{m:mean(-db(r[size][m]) for r in halo) for m in ('classic','repeat')},
                       'wins':sum(v['classic']>v['repeat'] for v in by_seq.values())}
    output['halo']=support
    (ROOT/'SUMMARY.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
