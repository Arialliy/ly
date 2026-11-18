"""Independent summaries of completed paired runs from saved MSEs and trajectories."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(outer):
    outer=outer.resolve()
    complete=json.loads((outer/'COMPLETE.json').read_text())
    assert complete['status']=='COMPLETE'
    assert digest(outer/'patch_metrics.csv')==complete['patch_metrics_sha256']
    assert digest(outer/'sequence_metrics.json')==complete['sequence_metrics_sha256']
    with (outer/'patch_metrics.csv').open() as f:
        patch=list(csv.DictReader(f))
    stored={(r['split'],r['seed'],r['sequence']):r for r in json.loads((outer/'sequence_metrics.json').read_text())}
    summary={}
    all_loss={}
    for split in ['train','evaluation']:
        rows=[r for r in patch if r['split']==split]
        seeds=sorted({int(r['seed']) for r in rows})
        seqs=sorted({int(r['sequence']) for r in rows})
        assert len(seeds)==3 and len(seqs)==(12 if split=='train' else 8)
        groups={}
        for seed in seeds:
            rec=json.loads((outer/f'seed_{seed}.json').read_text())
            assert rec['updates']==240 and rec['parameters_per_head']==52456
            assert rec['initialization_sha256']['B']==rec['initialization_sha256']['C']
            assert rec['final_sha256']['B']!=rec['final_sha256']['C']
            assert digest(outer/f'seed_{seed}.pt')==rec['weights_sha256']
            for e in range(15):
                assert sorted(i for b in rec['batches'][16*e:16*(e+1)] for i in b)==list(range(96))
            all_loss[seed]={m:[mean(x[j] for x in rec['loss_by_update'][:16]),
                               mean(x[j] for x in rec['loss_by_update'][-16:])] for j,m in enumerate('BC')}
            for seq in seqs:
                group=[r for r in rows if int(r['seed'])==seed and int(r['sequence'])==seq]
                assert len(group)==(8 if split=='train' else 4)
                psnr={m:-10*math.log10(mean(float(r[f'mse_{m}']) for r in group)) for m in 'ABC'}
                assert all(abs(psnr[m]-stored[split,seed,seq][f'psnr_{m}'])<1e-10 for m in 'ABC')
                psnr['C_minus_B']=psnr['C']-psnr['B']
                groups[seed,seq]=psnr
        by_seed={s:{m:mean(groups[s,k][m] for k in seqs) for m in ('A','B','C','C_minus_B')} for s in seeds}
        by_seq={k:{m:mean(groups[s,k][m] for s in seeds) for m in ('A','B','C','C_minus_B')} for k in seqs}
        all_psnr={m:mean(r[m] for r in by_seq.values()) for m in ('A','B','C','C_minus_B')}
        delta=[r['C_minus_B'] for r in by_seq.values()]
        summary[split]=dict(aggregate=all_psnr, by_seed=by_seed,by_sequence=by_seq,
                            winning_seqs=sum(x>0 for x in delta),
                            leave_one_seq_out_range=[min((sum(delta)-d)/(len(delta)-1) for d in delta),
                                                     max((sum(delta)-d)/(len(delta)-1) for d in delta)])
    summary['loss_first_last_epoch']=all_loss
    (outer/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('out',type=Path)
    s=summarize(parser.parse_args().out)
    for split in ['train','evaluation']:
        print(split,s[split]['aggregate'],s[split]['winning_seqs'],s[split]['leave_one_seq_out_range'])
        print('by_seed',s[split]['by_seed'])
