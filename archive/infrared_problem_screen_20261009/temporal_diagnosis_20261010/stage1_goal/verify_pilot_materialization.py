"""Independent all-frame check against frozen selection and unfiltered source JSON."""
from pathlib import Path
import collections, datetime, gzip, hashlib, json, struct, zlib
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    sel=json.loads((HERE/'DATA_SELECTION.json').read_text())
    assert sha(HERE/'DATA_SELECTION.json')==(HERE/'DATA_SELECTION.sha256').read_text().split()[0]
    src=json.load(gzip.open(ROOT/'runtime/VT_TRAIN_01_ANNOTATIONS.json.gz','rt'))
    images={im['file_name']:im for im in src['images']}
    ann=collections.defaultdict(list)
    for a in src['annotations']:ann[a['image_id']].append(a)
    central=(HERE/'data/ARCHIVE_CENTRAL_DIRECTORY.bin').read_bytes()
    authority=json.loads((HERE/'data/CENTRAL_DIRECTORY_PROVENANCE.json').read_text())
    assert hashlib.sha256(central).hexdigest()==authority['sha256']
    central_crcs={};pos=0
    while pos+46<=len(central):
        h=struct.unpack_from('<4s6H3L5H2L',central,pos)
        assert h[0]==b'PK\x01\x02'
        nl,el,cl=h[10:13]
        name=central[pos+46:pos+46+nl].decode('utf-8' if h[3]&0x800 else 'cp437')
        central_crcs[name]=h[7];pos+=46+nl+el+cl
    assert pos==len(central) and len(central_crcs)==94270
    rows=[]
    for c in sel['clips']:
        d=HERE/'data'/c['sequence']
        labels=json.loads((d/'annotations.json').read_text())
        fm=json.loads((d/'FRAME_MANIFEST.json').read_text())
        ar=np.load(d/'raw_frames.npy',mmap_mode='r',allow_pickle=False)
        assert ar.shape==(160,512,640) and ar.dtype==np.uint8
        expected_images=[images[n.split('VT-Tiny-MOT/train2017/',1)[1]] for n in c['selected_members']]
        expected_annotations=[a for im in expected_images for a in ann[im['id']]]
        assert labels['images']==expected_images and labels['annotations']==expected_annotations
        assert labels['categories']==src['categories']
        assert [im['id'] for im in expected_images]==c['selected_image_ids']
        assert sha(d/'raw_frames.npy')==fm['array_sha256']
        assert sha(d/'annotations.json')==fm['annotations_sha256']
        exact=0
        for i,(im,n) in enumerate(zip(expected_images,c['selected_members'])):
            p=d/'images'/Path(n).name
            assert sha(p)==fm['frames'][i]['sha256']
            assert zlib.crc32(p.read_bytes())&0xffffffff==central_crcs[n]
            with Image.open(p) as raw:
                raw.load();assert raw.mode=='L' and raw.size==(640,512)
                assert np.array_equal(np.asarray(raw),ar[i]);exact+=1
            assert fm['frames'][i]['source_image_id']==im['id']
        assert exact==160
        rows.append({'sequence':c['sequence'],'array_jpeg_exact_frames':exact,'retained_all_source_annotations':len(expected_annotations),'frame_manifest_sha256':sha(d/'FRAME_MANIFEST.json'),'annotations_sha256':sha(d/'annotations.json')})
    state=json.loads((HERE/'data/ACQUISITION_REPORT.json').read_text())
    assert state['frames']==sum(r['array_jpeg_exact_frames'] for r in rows)==2240
    assert state['all_crc_verified']
    report={'status':'PASS','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_sha256':sha(HERE/'DATA_SELECTION.json'),'central_directory_sha256':authority['sha256'],'checks':['All 2240 native uint8 frames exactly equal Pillow JPEG decoding in frozen numeric order.','All original source image records and every corresponding annotation retained field-for-field.','All category records retained; no mask or pseudo-label constructed.','All array, label and original JPEG hashes match manifests.','All 2240 original JPEG CRC32 values independently recomputed and equal the public ZIP central directory.'],'rows':rows,'verified_frames':2240,'verified_annotations':sum(r['retained_all_source_annotations'] for r in rows),'limits':'This validates materialization and retention, not semantic box quality, target visibility, original DeepPro masks, or physical scene independence.'}
    (HERE/'data/DATA_INTEGRITY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','verified_frames','verified_annotations']},ensure_ascii=False))

if __name__=='__main__':main()
