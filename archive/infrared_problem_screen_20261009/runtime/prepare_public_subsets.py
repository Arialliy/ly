"""Materialize already selected public PNG/JPEG frames without transformation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

p = Path(__file__).resolve().parent
vt = p / 'public_subset_vt'
manifest = json.loads((vt / 'FRAME_MANIFEST.json').read_text())
datasets = {
    'vt_small_clip': [vt / f['archive_member'] for f in manifest['frames']],
    'nudt_native_clip': sorted((p / 'public_subset_nudt/NUDT-MIRSDT/Sequence85/images').glob('*')),
}
for name, paths in datasets.items():
    frames = []
    records = []
    for path in paths:
        with Image.open(path) as image:
            a = np.asarray(image)
            if a.ndim == 3:
                a = a[:, :, 0]
            frames.append(a.copy())
            records.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'shape':list(a.shape)})
    frames = np.stack(frames)
    assert frames.shape[0] >= 40
    out = p / name
    out.mkdir(exist_ok=True)
    np.save(out / 'raw_frames.npy', frames)
    Image.fromarray(frames[0]).save(out / 'original_frame0.png')
    meta = {'name':name,'shape':list(frames.shape),'dtype':str(frames.dtype),
            'raw_sha256':hashlib.sha256(frames.tobytes()).hexdigest(),
            'transforms':'none; retain official loader first-channel convention',
            'records':records,
            'interpretation':'VT real additional development sequence; NUDT synthesized official first test sequence partial window'}
    (out / 'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:v for k,v in meta.items() if k!='records'}),flush=True)
