"""Fixed assets and membership of the first held-out civil traffic confirmation."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE_ONE = HERE.parent / 'motion_joint_factorial'
OLD_RUN = STAGE_ONE / 'runs'
RUN = HERE / 'runs'
COHORT = ('DJI_0075_3', 'DJI_0139_2', 'DJI_0343_3')
CENTERS = (19, 59, 99, 139)
METHODS = ('original', 'A', 'B', 'C', 'D')
sys.path.insert(0, str(STAGE_ONE))


def frame_record(seq, idx):
    from protocol import frame_record as old_record
    if seq not in COHORT or idx not in CENTERS:
        raise ValueError((seq,idx))
    return {**old_record(seq,idx), 'split':'confirmation'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def build_lock():
    source_split = ROOT / 'results/references/DATA_SELECTION.json'
    metadata = json.loads(source_split.read_text())
    confirm = metadata['splits']['confirmation']
    clips = {c['sequence']:c for c in metadata['clips']}
    eligible = [s for s in confirm if clips[s]['dominant_group'] in ('road','ship')]
    if set(eligible)!=set(COHORT) or any(clips[s]['checkpoint_train_prefix_overlap'] or
            clips[s]['developed_prefix_overlap'] for s in COHORT):
        raise RuntimeError('Unexpected source split or source prefix overlap')
    assets = json.loads((OLD_RUN/'artifacts.json').read_text())
    paths = [ROOT / assets['release']]
    ref = {paths[0]:assets['release_sha256']}
    for m in METHODS[1:]:
        rel=f'runs/models/{m}.pt'
        ref[STAGE_ONE/rel]=assets['weights_and_predictions'][rel]
    for path, expected in ref.items():
        if digest(path)!=expected:
            raise RuntimeError(f'Weight differs from stage one: {path.name}')
    scores_path = OLD_RUN/'scores.json'
    dev = json.loads(scores_path.read_text())['methods']
    thresholds = {m:dev[m]['all_dev']['froc']['1']['logit_threshold'] for m in METHODS}
    deps = [source_split,scores_path,STAGE_ONE/'model.py',STAGE_ONE/'motion_evidence.py',
            STAGE_ONE/'cache_backbone.py',STAGE_ONE/'experiment.py',STAGE_ONE/'protocol.py',
            STAGE_ONE/'evaluate.py',HERE.parent/'motion_hypothesis_traffic_pilot_20261118'/'readout.py',
            HERE.parent/'motion_hypothesis_traffic_pilot_20261118'/'extract.py',
            HERE/'PROTOCOL.md',HERE/'locked_protocol.py',HERE/'infer_locked.py']
    for seq in COHORT:
        deps.extend(ROOT/'data/stage1'/seq/name for name in ('annotations.json','raw_frames.npy'))
    return dict(cohort=COHORT, centers=CENTERS, excluded_confirmation=sorted(set(confirm)-set(COHORT)),
                resolution=[512,640], training_updates=0, dev_logit_threshold=thresholds,
                raw_hashes={str(p.relative_to(ROOT)):digest(p) for p in deps},
                weight_hashes={str(p.relative_to(ROOT)):h for p,h in ref.items()})


def verify_lock():
    saved=json.loads((RUN/'LOCK.json').read_text())
    current=json.loads(json.dumps(build_lock()))
    if saved!=current:
        raise RuntimeError('Inputs no longer match the confirmation lock')
    return saved


if __name__=='__main__':
    value=json.dumps(build_lock(),indent=2,ensure_ascii=False)+'\n'
    path=RUN/'LOCK.json'
    if path.exists():
        verify_lock()
    else:
        with path.open('x') as stream:
            stream.write(value)
    print(path)
