"""Seal the short-revision protocol after training-only preparation."""
import time
from runtime import E,V1,CP,INIT_CP,sha,save_json,read_json
from common import fingerprints


def main():
    target=E/'PROTOCOL_FREEZE.json'
    if target.exists():raise FileExistsError(target)
    assert read_json(E/'CALIBRATION.json')['status']=='complete'
    assert all(read_json(E/'cache'/f'SHARD_{i}.json')['status']=='COMPLETE' for i in range(3))
    files=[E/'PROTOCOL.md',E/'SEGMENT_PLAN.json',E/'SEGMENT_AUDIT.json',E/'CALIBRATION.json']
    files+=sorted((E/'code').glob('*.py'))+sorted((E/'train_modules').glob('*.py'))
    files += [E/'resources/REFERENCE_TESTS.xml',E/'resources/INTEGRATION_TESTS.xml']
    save_json(target,{'status':'frozen_before_256step_revision','created_unix':time.time(),
        'initial_checkpoint_sha256':sha(INIT_CP),'release_checkpoint_sha256':sha(CP),
        'initial_v1_protocol_sha256':sha(V1/'PROTOCOL_FREEZE.json'),
        'conditions':['CONT','FRAME','GAP'],'start_step':2048,'additional_steps':256,'main_step':2304,
        'files':{str(p.relative_to(E)):sha(p) for p in files},'training_sources':fingerprints(),
        'calibration_sha256':sha(E/'CALIBRATION.json'),
        'cache_sha256':{str(i):sha(E/'cache'/f'entry_{i:03d}.npz') for i in range(256)},
        'v1_cache_sha256':{str(i):sha(V1/'cache'/f'entry_{i:03d}.npz') for i in range(256)}})
    print('FROZEN',sha(target),flush=True)

if __name__=='__main__':main()
