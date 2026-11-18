"""Verify frozen train/infer checkpoint identity before calling the unchanged scorer."""
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def parse_condition(condition):
    m=re.fullmatch(r'(E[012])(?:_(mass|current)|_step(128|256|384))?',condition)
    assert m,condition
    variant,diagnostic,step=m.groups()
    assert diagnostic is None or variant=='E2'
    return variant,diagnostic or 'none',int(step or 512)


def verify_identity(base,condition,sequence,predictions_root):
    import torch
    variant,diagnostic,step=parse_condition(condition)
    q=base.parent
    runpath=base/'training'/variant/'run/RUN.json'
    infpath=predictions_root/sequence/'COMPLETE.json'
    run=json.loads(runpath.read_text());inf=json.loads(infpath.read_text())
    freeze_path=base/'PROTOCOL_FREEZE.json'
    freeze=json.loads(freeze_path.read_text())
    assert sha(base/'PROTOCOL.md')==freeze['protocol_sha256']
    for path,expected in freeze['code_sha256'].items():assert sha(base/path)==expected,('Frozen code changed',path)
    assert run['status']=='complete' and run['condition']==run['variant']==variant
    assert run['steps']==run['final_step']==512 and run['radius']==2
    assert run['protocol']=='positive_masked_official_SoftIoU_empty_old_top1pct'
    assert run['bn_initial_sha256']==run['bn_final_sha256']
    assert inf['status']=='COMPLETE' and inf['sequence']==sequence and inf['condition']==condition
    assert inf['training_variant']==variant and inf['diagnostic']==diagnostic and inf['checkpoint_step']==step
    checkpoint_path=Path(inf['checkpoint'])
    assert checkpoint_path.parent.resolve()==runpath.parent.resolve()
    assert checkpoint_path.name in ([f'snapshot_{step:03d}.pth','final_model.pth'] if step==512 else [f'snapshot_{step:03d}.pth'])
    snapshot=next(r for r in run['snapshots'] if r['step']==step)
    cp_sha=sha(checkpoint_path)
    assert snapshot['sha256']==inf['checkpoint_sha256']==cp_sha
    if step==512:assert run['final_checkpoint_sha256']==cp_sha
    cp=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    assert cp['condition']==cp['variant']==variant and cp['step']==step
    assert cp['initial_checkpoint_sha256']==run['checkpoint_sha256']==inf['reference_checkpoint_sha256']
    prior=json.loads((q/'top1_background_control_20261011/training/run/RUN.json').read_text())
    for key in ['checkpoint_sha256','plan_sha256','selection_sha256','model_source_sha256']:
        assert run[key]==prior[key],('Original setup changed',key)
    assert cp['plan_sha256']==run['plan_sha256']
    assert cp['label_config']==run['label_config']==inf['label_config']
    paths={'train_source':base/'code/train_mutual.py','module_source':base/'code/module.py',
           'label_source':q/'supervision_alignment_control_20261012/code/gaussian_labels.py',
           'top1_loss_source':q/'top1_background_control_20261011/code/losses.py',
           'model_source':Path(inf['model_source'])}
    sources={str(runpath):sha(runpath),str(infpath):sha(infpath),str(checkpoint_path):cp_sha,
             str(freeze_path):sha(freeze_path),str(base/'PROTOCOL.md'):sha(base/'PROTOCOL.md')}
    for name,path in paths.items():
        actual=sha(path);assert cp[name+'_sha256']==run[name+'_sha256']==inf[name+'_sha256']==actual
        sources[str(path)]=actual
    script=base/'code/infer_mutual.py';assert sha(script)==inf['script_sha256'];sources[str(script)]=sha(script)
    assert inf['normalization']==[85.0799,47.4845]
    assert inf['official_starts']==[0,36,72,108,120]
    assert inf['diagnostic_starts']==[0,24,36,60,72,96,108,120]
    assert inf['tile_core']==128 and inf['tile_halo']==20
    assert inf['parameters']==run['parameter_count']==(70913 if variant=='E0' else 72194)
    assert inf['model_labels_access'] is False
    assert bool(inf['read_only_query_labels_access'])==(condition=='E2')
    assert len(inf['tiling_checks'])==3
    assert all(w['threshold05_disagreements']==0 for w in inf['tiling_checks'])
    assert isinstance(cp['model_state_dict'],dict) and cp['model_state_dict']
    assert 'TPro.gamma' not in cp['model_state_dict'],'BC checkpoint forbidden'
    newkeys=[k for k in cp['model_state_dict'] if k.startswith('temporal_aggregation.')]
    assert bool(newkeys)==(variant!='E0')
    if condition=='E2':
        provenance=predictions_root/sequence/'PROVENANCE_COMPLETE.json'
        record=json.loads(provenance.read_text());assert record['status']=='COMPLETE'
        assert sha(record['rows_path'])==record['rows_sha256']
        sources[str(provenance)]=sha(provenance)
    return dict(identity_source_sha256=sources,checkpoint_sha256=cp_sha,training_variant=variant,
                checkpoint_step=step,main_endpoint=step==512,diagnostic=diagnostic,
                checkpoint_cpu_weights_only_metadata_checked=True)
