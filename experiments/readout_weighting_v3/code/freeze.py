"""Seal reviewed sources and reused inputs before either 256-step revision."""
import time
from runtime import E,V1,V2,ROOT,CP,INIT_CP,sha,save_json,read_json
from common import fingerprints


def main():
    target=E/'PROTOCOL_FREEZE.json'
    if target.exists():raise FileExistsError(target)
    for name in ('F_REUSE_AUDIT.json','REAL_WEIGHTING_CHECK.json','TRAINING_CODE_REVIEW.json'):
        assert read_json(E/'resources'/name)['status']=='PASS',name
    assert read_json(V2/'VALIDATION.json')['status']=='PASS'
    assert not any((E/'training').glob('*/run/RUN.json'))
    previous=read_json(V2/'PROTOCOL_FREEZE.json')
    for i in range(256):
        assert sha(V1/'cache'/f'entry_{i:03d}.npz')==previous['v1_cache_sha256'][str(i)]
        assert sha(V2/'cache'/f'entry_{i:03d}.npz')==previous['cache_sha256'][str(i)]
    files=[E/'PROTOCOL.md']
    files+=sorted((E/'code').glob('*.py'))+sorted((E/'train_modules').glob('*.py'))
    files+=sorted((E/'resources').glob('*'))
    files+=sorted((E/'reference/DeepProPlus_C_readout_v3').glob('*'))
    files=[p for p in files if p.is_file()]
    save_json(target,dict(status='frozen_before_256step_revision',created_unix=time.time(),
        conditions=['D','N'],reused_conditions={'release':'release','C_init':'C_init','CONT':'CONT','F':'FRAME'},
        start_step=2048,additional_steps=256,main_step=2304,
        initial_checkpoint_sha256=sha(INIT_CP),release_checkpoint_sha256=sha(CP),
        initial_v1_protocol_sha256=sha(V1/'PROTOCOL_FREEZE.json'),
        initial_v2_protocol_sha256=sha(V2/'PROTOCOL_FREEZE.json'),
        calibration_path=str(V2/'CALIBRATION.json'),calibration_sha256=sha(V2/'CALIBRATION.json'),
        reference_zip_sha256=sha(ROOT/'DeepProPlus_C_readout_v3.zip'),
        files={str(p.relative_to(E)):sha(p) for p in files},training_sources=fingerprints(),
        v1_cache_sha256=previous['v1_cache_sha256'],v2_cache_sha256=previous['cache_sha256'],
        gamma=2,normalization_group='existing_segment8',new_calibration=False,F_retrained=False))
    print('FROZEN',sha(target),flush=True)


if __name__=='__main__':main()
