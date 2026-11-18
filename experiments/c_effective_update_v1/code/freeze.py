"""Freeze the training-only effect plan before new teacher/effect measurements."""
import time
from pathlib import Path
from runtime import E,V1,V2,ROOT,CP,INIT_CP,MODEL_SOURCE,sha,read_json,save_json


def main():
    destination=E/'PROTOCOL_FREEZE.json'
    if destination.exists():raise FileExistsError(destination)
    assert not list((E/'cohorts').glob('entry_*.json')) and not list((E/'pairs').glob('pair_*.json'))
    plan=read_json(E/'PAIR_PLAN.json');pairs=plan['pairs']
    assert len(pairs)==16 and sorted(p['pair_id'] for p in pairs)==list(range(16))
    indices=sorted({p[k] for p in pairs for k in ('update_index','observer_index')})
    assert len(indices)==32
    source_paths=[INIT_CP,CP,MODEL_SOURCE,V2/'CALIBRATION.json',V2/'PROTOCOL_FREEZE.json',
        V2/'VALIDATION.json',V2/'training/FRAME/run/RUN.json',V2/'train_modules/temporal_risk.py',
        ROOT/'deeppro_c_v2_reference/temporal_risk.py',V2/'SEGMENT_PLAN.json']
    source_paths += [Path(x) for x in read_json(V1/'PROTOCOL_FREEZE.json')['training_sources']]
    source_paths += [folder/'cache'/f'entry_{i:03d}.npz' for folder in (V1,V2) for i in indices]
    source_paths += [folder/'cache'/f'entry_{i:03d}.json' for folder in (V1,V2) for i in indices]
    sources={str(p):sha(p) for p in source_paths}
    files=[E/'PROTOCOL.md',E/'PAIR_PLAN.json']
    for folder in ('code','resources','tests'):
        files += [p for p in (E/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    save_json(destination,{'status':'FROZEN_BEFORE_EFFECT_MEASUREMENTS','created_unix':time.time(),
        'initial_checkpoint_sha256':sha(INIT_CP),'release_checkpoint_sha256':sha(CP),'initial_step':2304,
        'copies_accumulated':False,'pairs':16,'conditions':['full_F','no_keep','no_track'],
        'config_requires_stage_one_support':True,'paired256_requires_candidate_support':True,
        'uses_development':False,'plan_sha256':sha(E/'PAIR_PLAN.json'),
        'files':{str(p.relative_to(E)):sha(p) for p in files},'reused_sources':sources})
    print('FROZEN',sha(destination),flush=True)


if __name__=='__main__':main()
