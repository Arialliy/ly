"""Compare the local negative kernel with a TRUSTED local DEIM author file.

Usage:
  python check_official_kernel.py --source /path/to/DEIM/engine/deim/deim_criterion.py

This reads and executes two methods from the supplied source. Use only a trusted,
reviewed author checkout, not arbitrary uploaded code. No network access required.
It compares the all-background branch with caller-chosen equal normalization;
NOT whole MAL, detector training, quality targets, or author performance.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F
from readout_weighting import readout_terms


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('OFFICIAL_KERNEL_CHECK.json'))
    args=ap.parse_args()
    src=args.source.resolve()
    raw=src.read_bytes()
    tree=ast.parse(raw.decode('utf-8'))
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DEIMCriterion']
    if len(classes)!=1:
        raise ValueError('Expected exactly one DEIMCriterion class.')
    wanted={'loss_labels_mal','_get_src_permutation_idx'}
    methods=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    if len(methods)!=2:
        raise ValueError('Required author methods are missing; inspect version differences.')
    ns={'torch':torch, 'F':F}
    for m in methods:
        exec(compile(ast.fix_missing_locations(ast.Module(body=[m],type_ignores=[])), str(src),'exec'),ns)
    Kernel=type('AuthorKernel',(),{name:ns[name] for name in wanted})
    kernel=Kernel()
    kernel.gamma=2.0; kernel.mal_alpha=None; kernel.num_classes=1
    checked=[]
    for batch,queries in [(1,1),(1,7),(2,11)]:
        torch.manual_seed(batch*100+queries)
        z=torch.randn(batch,queries,dtype=torch.float64).mul(5).requires_grad_()
        refz=z.detach().clone().requires_grad_()
        empty=torch.zeros(0,dtype=torch.int64)
        indices=[(empty.clone(),empty.clone()) for _ in range(batch)]
        targets=[{'labels':empty.clone()} for _ in range(batch)]
        author=kernel.loss_labels_mal(
            {'pred_logits':refz.unsqueeze(-1), 'pred_boxes':torch.zeros(batch,queries,4,dtype=torch.float64)},
            targets,indices,num_boxes=batch*queries,values=torch.empty(0,dtype=torch.float64),
        )['loss_mal']
        ours=readout_terms(torch.empty(batch,0,dtype=torch.float64),z,mode='confidence').negative.mean()
        ga,=torch.autograd.grad(author,refz)
        go,=torch.autograd.grad(ours,z)
        torch.testing.assert_close(ours,author,rtol=1e-10,atol=1e-12)
        torch.testing.assert_close(go,ga,rtol=1e-10,atol=1e-12)
        checked.append({'batch':batch,'queries':queries,'value_and_gradient':'passed'})
    proc=subprocess.run(['git','-C',str(src.parent),'rev-parse','HEAD'],capture_output=True,text=True)
    report={
        'scope':'All-background DEIM MAL kernel, gamma=2, mal_alpha=None, equal mean normalization only',
        'author_source':str(src),'source_sha256':hashlib.sha256(raw).hexdigest(),
        'git_commit':proc.stdout.strip() if proc.returncode==0 else None,
        'tests':checked,'whole_detector_or_training_verified':False,
    }
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
