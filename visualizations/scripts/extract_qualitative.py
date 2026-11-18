"""Extract fixed cached frames for reproducible figures, without model execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]
V = BASE/'visualizations'
RESULTS = BASE/'results/tlfp_pilot_20261012'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--original-stage1',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads((V/'QUALITATIVE_SELECTION.json').read_text())
    objects=list(csv.DictReader((RESULTS/'evaluation/OBJECT_COMPARISON.csv').open()))
    (V/'data').mkdir(exist_ok=True)
    hashes={}; examples=[]
    for seq in plan['sequences']:
        raw_path=args.original_stage1/'data'/seq/'raw_frames.npy'
        release_path=args.original_stage1/'predictions/official'/seq/'official_fused.npz'
        candidate_path=RESULTS/'predictions/T'/seq/'official_fused.npz'
        receipt_path=RESULTS/'evaluation/T'/seq/'COMPLETE.json'
        meta=json.loads(receipt_path.read_text())
        by_id={int(x['annotation_id']):x for x in meta['objects']}
        raw=np.load(raw_path,mmap_mode='r',allow_pickle=False)
        ids=np.asarray(plan['frames'],dtype=np.int32)
        sample={'raw':np.array(raw[ids],copy=True),'frames':ids}
        assert raw.shape==(160,512,640) and raw.dtype==np.uint8
        for label,path in [('release',release_path),('T',candidate_path)]:
            with np.load(path,allow_pickle=False) as cache:
                full=cache['probabilities']
                assert full.shape==raw.shape and full.dtype==np.float32
                sample[label]=np.array(full[ids],copy=True)
                del full
        np.savez_compressed(V/'data'/f'{seq}_selected_frames.npz',**sample)
        for row in objects:
            if row['sequence']==seq and row['arrangement']=='O' and int(row['frame']) in plan['frames']:
                entry={key:row[key] for key in ['sequence','frame','source_frame','track_id','annotation_id','low_all_05','extreme_all_001']}
                entry['bbox']=by_id[int(row['annotation_id'])]['bbox']
                for condition in ['release','S','T','T_current','T_bypass']:
                    for suffix in ['matched','box_max','peak_x','peak_y']:
                        entry[condition+'_'+suffix]=row[condition+'_'+suffix]
                examples.append(entry)
        for path in [raw_path,release_path,candidate_path,receipt_path]:hashes[str(path)]=digest(path)
    (V/'data/QUALITATIVE_OBJECTS.json').write_text(json.dumps(examples,indent=2)+'\n')
    for path in [V/'QUALITATIVE_SELECTION.json',RESULTS/'evaluation/OBJECT_COMPARISON.csv']:hashes[str(path)]=digest(path)
    (V/'QUALITATIVE_EXTRACTION.json').write_text(json.dumps(dict(status='COMPLETE',model_runs=0,source_sha256=hashes,
        frame_count=len(plan['frames'])*len(plan['sequences']),arrangement='O',probabilities='existing sigmoid probability caches; no rescaling or new evaluation',
        raw_pixels='unmodified uint8 raw frames',sampling_plan='QUALITATIVE_SELECTION.json'),indent=2)+'\n')
    print('Extracted six fixed raw/probability frames from caches')

if __name__=='__main__':main()
