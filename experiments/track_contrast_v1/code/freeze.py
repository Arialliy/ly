"""Close numerical calibration and lock all inputs before formal A/B/C runs."""
import json
import time
from collections import Counter
from runtime import E,CP,MODEL_SOURCE,Q,sha,save_json
from data import source_fingerprints


def main():
    path=E/'PROTOCOL_FREEZE.json'
    if path.exists():raise FileExistsError(path)
    calibration=json.loads((E/'CALIBRATION.json').read_text())
    assert calibration['status']=='complete' and not calibration['uses_development']
    files=['PROTOCOL.md','PAIR_PLAN.json','SAMPLER_AUDIT.json','CALIBRATION.json',
        'code/train.py','code/data.py','code/prepare.py','code/calibrate.py','code/runtime.py',
        'code/infer.py','code/evaluate.py','code/launch.py','code/pipeline.py','code/freeze.py','train_modules/track_pair_sampler.py',
        'train_modules/region_contrast_loss.py','train_modules/reliable_keep_loss.py',
        'resources/REFERENCES.json']
    counts=Counter();cache_hashes={}
    for i in range(512):
        p=E/'cache'/f'entry_{i:03d}.npz';r=json.loads(p.with_suffix('.json').read_text())
        assert r['index']==i and r['sha256']==sha(p)
        assert all(sha(source)==value for source,value in r['sources'].items())
        cache_hashes[str(i)]=r['sha256'];counts['pairs_requested']+=r['pairs']
        counts['pairs_with_two_negatives']+=r['pairs_with_two_negatives']
        counts['entries_with_pairs']+=int(r['pairs']>0)
        counts['entries_with_two_negative_pairs']+=int(r['pairs_with_two_negatives']>0)
    save_json(path,dict(status='frozen_before_formal_training',created_unix=time.time(),
        files={name:sha(E/name) for name in files},training_sources=source_fingerprints(),
        calibration_sha256=sha(E/'CALIBRATION.json'),cache_sha256=cache_hashes,cache_counts=dict(counts),
        release_checkpoint_sha256=sha(CP),model_source_sha256=sha(MODEL_SOURCE),
        original_plan_sha256=sha(Q/'stage1_goal/pilot/TRAIN_PLAN.json'),
        main_endpoint=2048,seed=46,conditions=['A','B','C'],settings=calibration['fixed_settings']))
    print(json.dumps({'protocol_sha256':sha(path),'counts':dict(counts),'settings':calibration['fixed_settings']}))

if __name__=='__main__':main()
