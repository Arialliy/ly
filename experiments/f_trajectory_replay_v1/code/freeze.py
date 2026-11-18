"""Seal the registered paired continuation after input/cache checks pass."""
from pathlib import Path
import torch
import runtime as rt
import replay_data as replay
import effect_common as common


def freeze():
    out=rt.E/'PROTOCOL_FREEZE.json'
    assert not out.exists()
    preparation=rt.read_json(rt.E/'PREPARATION_FREEZE.json')
    # Cache manifests bind every changed ordinary-training cache to its sources.
    manifest=rt.read_json(rt.E/'CACHE_MANIFEST.json')
    assert manifest['status']=='COMPLETE' and manifest['jobs']==512
    for path,value in manifest['source_hashes'].items():assert rt.sha(path)==value,path
    for row in manifest['files']:assert rt.sha(row['path'])==row['sha256'],row['path']
    assert replay.schedule()['exposure_gate']['status']=='PASS_NONTRIVIAL_EXPOSURE_CHANGE'
    caches={}
    for condition in ('ORIGINAL','REPLAY'):
        for index in range(512):
            for path in replay.cache_paths(index,condition):
                if str(path) not in caches:caches[str(path)]=rt.sha(path)
    # Match all inherited C/FRAME caches against the previous frozen protocols.
    oldc=rt.read_json(rt.V1/'PROTOCOL_FREEZE.json')['cache_sha256']
    oldf=rt.read_json(rt.V2/'PROTOCOL_FREEZE.json')['cache_sha256']
    for i,value in oldc.items():assert caches[str(rt.V1/'cache'/f'entry_{int(i):03d}.npz')]==value
    for i,value in oldf.items():assert caches[str(rt.V2/'cache'/f'entry_{int(i):03d}.npz')]==value
    cp=torch.load(rt.INIT_CP,map_location='cpu',weights_only=True)
    sources=dict(replay.schedule()['sources'])
    sources.update(manifest['source_hashes'])
    sources.update({str(p):rt.sha(p) for p in [rt.INIT_CP,rt.CP,rt.MODEL_SOURCE,
        rt.V1/'code/train.py',rt.V2/'code/train.py',rt.V2/'code/common.py',rt.V2/'CALIBRATION.json']})
    sources.update(replay.original_data.source_fingerprints())
    files=list((rt.E/'code').glob('*.py'))
    files += [rt.E/name for name in ['PROTOCOL.md','PREPARATION_FREEZE.json','SCHEDULE.json',
        'ORIGINAL_SCHEDULE.csv','REPLAY_SCHEDULE.csv','EXPOSURE_COMPARISON.csv',
        'TRAIN_DIAGNOSTIC_PLAN.json','TRAIN_EVAL_PLAN.json','DEV_EVALUATION_PLAN.json',
        'CACHE_MANIFEST.json','CACHE_EXPOSURE.json',
        'evaluation/SAMPLER_SMOKE.json','evaluation/TRAINER_CODE_REVIEW.json',
        'evaluation/TRAIN_EVAL_CODE_REVIEW.json','evaluation/TRAIN_EVAL_CHECK.json','evaluation/DEV_REFERENCE_CHECK.json']]
    rt.save_json(out,dict(status='FROZEN_BEFORE_TRAINING',files={str(p.relative_to(rt.E)):rt.sha(p) for p in files},
        reused_sources=sources,cache_sha256=caches,initial_checkpoint_sha256=rt.sha(rt.INIT_CP),
        initial_optimizer_sha256=common.optimizer_hash(cp['optimizer_state_dict']),
        initial_model_state_sha256=rt.tensor_hash(cp['model_state_dict']),seed=46,
        conditions=['ORIGINAL','REPLAY'],additional_steps=512,intermediate_dev_evaluation=False,
        configuration_or_new_gradient_updates=False))
    print({'status':'FROZEN','sha256':rt.sha(out),'input_cache_files':len(caches)},flush=True)


if __name__=='__main__':freeze()
