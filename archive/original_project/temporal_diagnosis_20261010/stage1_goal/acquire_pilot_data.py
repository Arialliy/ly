"""Acquire frozen pilot IR JPEGs using public ZIP ranges, then materialize uint8 clips.

Budget is cumulative across reruns, conservatively debited before every body read.
No model inference, mask creation, credentials, resizing of experiment inputs, or full ZIP download.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import collections, datetime, gzip, hashlib, importlib.util, json, os, struct, urllib.parse, zlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / 'data'
LIMIT = 250_000_000

def sha(data): return hashlib.sha256(data).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); tmp.replace(path)
def print_event(**kwargs): print(json.dumps(kwargs,ensure_ascii=False),flush=True)

def main():
    selection_path=HERE/'DATA_SELECTION.json'
    selection=json.loads(selection_path.read_text())
    expected_sha=(HERE/'DATA_SELECTION.sha256').read_text().split()[0]
    assert sha(selection_path.read_bytes())==expected_sha
    assert selection['expected_frames']==2240 and sum(c['frames'] for c in selection['clips'])==2240
    source_path=ROOT/'VT_TINY_MOT_PUBLIC_ZIP_MANIFEST.json'
    source=json.loads(source_path.read_text()); entries={e['name']:e for e in source['entries']}
    chosen=[entries[n] for c in selection['clips'] for n in c['selected_members']]
    assert len(chosen)==len({e['name'] for e in chosen})==2240
    assert sum(e['compressed_bytes'] for e in chosen)==selection['expected_compressed_image_bytes']
    DATA.mkdir(exist_ok=True)
    selected_manifest={k:v for k,v in source.items() if k!='entries'}
    selected_manifest.update(entries=chosen,entries_count=len(chosen),source_manifest_sha256=sha(source_path.read_bytes()),selection_sha256=expected_sha)
    write_json(DATA/'SELECTED_ZIP_MANIFEST.json',selected_manifest)
    ledger_path=DATA/'NETWORK_TRANSFER.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'started_utc':now(),'body_budget_bytes':LIMIT,'conservative_debited_bytes':0,'actual_body_bytes_completed_reads':0,'requests':[],'scope':'All form/range/retry bodies; reserve requested bytes before read, never refunded after interruption; no transport headers.'}
    assert ledger['body_budget_bytes']==LIMIT
    records_path=DATA/'MEMBER_RECORDS.json'
    records=json.loads(records_path.read_text()) if records_path.exists() else {}
    helper_path=ROOT/'runtime/fetch_public_zip_subset.py'
    spec=importlib.util.spec_from_file_location('frozen_pilot_public_zip_helper',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    native_urlopen=helper.urllib.request.urlopen

    class CountedResponse:
        def __init__(self,response,request):
            self.response=response
            url=request if isinstance(request,str) else request.full_url
            parsed=urllib.parse.urlsplit(url)
            self.record={'endpoint':parsed.scheme+'://'+parsed.netloc+parsed.path,'range':None if isinstance(request,str) else request.get_header('Range'),'status':response.status,'content_range':response.headers.get('Content-Range'),'completed_body_bytes':0,'debited_bytes':0}
            ledger['requests'].append(self.record);write_json(ledger_path,ledger)
        def __getattr__(self,name):return getattr(self.response,name)
        def __enter__(self):return self
        def __exit__(self,*args):self.response.close()
        def read(self,size=-1):
            if size<0 or ledger['conservative_debited_bytes']+size>LIMIT:raise RuntimeError('Cumulative public HTTP response-body budget would be exceeded')
            ledger['conservative_debited_bytes']+=size;self.record['debited_bytes']+=size
            write_json(ledger_path,ledger)
            body=self.response.read(size)
            ledger['actual_body_bytes_completed_reads']+=len(body);self.record['completed_body_bytes']+=len(body)
            ledger['last_read_utc']=now();write_json(ledger_path,ledger)
            return body
    def counted_urlopen(request,**kwargs):return CountedResponse(native_urlopen(request,**kwargs),request)

    helper.urllib.request.urlopen=counted_urlopen
    archive=None
    try:
        for clip in selection['clips']:
            seq=clip['sequence']; dest=DATA/seq; dest.mkdir(exist_ok=True)
            rawroot=dest/'source_archive'
            missing=[]
            for name in clip['selected_members']:
                path=rawroot/name;rec=records.get(name)
                if not (path.is_file() and rec and path.stat().st_size==entries[name]['uncompressed_bytes'] and sha(path.read_bytes())==rec['sha256']):missing.append(entries[name])
            if missing:
                if archive is None:archive=helper.PublicArchive(selected_manifest,budget=LIMIT)
                before=len(archive.records)
                try: archive.extract(missing,rawroot)
                finally:
                    for rec in archive.records[before:]: records[rec['member']]=rec
                    write_json(records_path,records)
            assert all(n in records and (rawroot/n).is_file() for n in clip['selected_members'])
            write_json(dest/'DOWNLOAD_PROVENANCE.json',{'sequence':seq,'source_url':'https://drive.google.com/file/d/'+source['file_id']+'/view','archive_bytes':source['archive_bytes'],'range_only':True,'selection_sha256':expected_sha,'helper_sha256':sha(helper_path.read_bytes()),'script_sha256':sha(Path(__file__).read_bytes()),'cumulative_network_ledger':'../NETWORK_TRANSFER.json','raw_bytes_preserved':True,'members':[records[n] for n in clip['selected_members']],'completed_utc':now()})
            print_event(event='download_verified',sequence=seq,frames=160,cumulative_debit=ledger['conservative_debited_bytes'])
        # VT uses data descriptors, so its local headers lack authoritative CRCs.
        # Read only ZIP metadata and check the central CRC for every selected member.
        central_path=DATA/'ARCHIVE_CENTRAL_DIRECTORY.bin'
        if not central_path.exists():
            if archive is None:archive=helper.PublicArchive(selected_manifest,budget=LIMIT)
            size=source['archive_bytes']
            tail=archive.read_range(size-65557,size-1)
            offset=tail.rfind(b'PK\x05\x06')
            assert offset>=0
            footer=struct.unpack_from('<4s4H2LH',tail,offset)
            central_size,central_offset=footer[5:7]
            assert central_size==source['central_directory_bytes'] and central_size<=16_000_000
            central=archive.read_range(central_offset,central_offset+central_size-1)
            tmp=central_path.with_suffix('.tmp');tmp.write_bytes(central);tmp.replace(central_path)
            write_json(DATA/'CENTRAL_DIRECTORY_PROVENANCE.json',{'source_url':'https://drive.google.com/file/d/'+source['file_id']+'/view','offset':central_offset,'bytes':central_size,'sha256':sha(central),'scope':'Only ZIP central directory metadata, no additional image members.','retrieved_utc':now()})
        central=central_path.read_bytes();central_meta=json.loads((DATA/'CENTRAL_DIRECTORY_PROVENANCE.json').read_text())
        assert sha(central)==central_meta['sha256']
        pos=0;central_entries={}
        while pos+46<=len(central):
            f=struct.unpack_from('<4s6H3L5H2L',central,pos);assert f[0]==b'PK\x01\x02'
            flags,method,crc,compressed,uncompressed=f[3],f[4],f[7],f[8],f[9]
            nl,el,cl=f[10:13];name=central[pos+46:pos+46+nl].decode('utf-8' if flags&0x800 else 'cp437')
            central_entries[name]={'crc32':crc,'compressed_bytes':compressed,'uncompressed_bytes':uncompressed,'method':method,'local_header_offset':f[16]}
            pos+=46+nl+el+cl
        assert pos==len(central) and len(central_entries)==source['entries_count']
        for clip in selection['clips']:
            dest=DATA/clip['sequence']
            for n in clip['selected_members']:
                ce=central_entries[n]
                assert all(entries[n][k]==ce[k] for k in ['compressed_bytes','uncompressed_bytes','method','local_header_offset'])
                crc=zlib.crc32((dest/'source_archive'/n).read_bytes())&0xffffffff
                assert crc==ce['crc32']
                records[n].update(crc_verified=True,crc32=crc,crc_authority='public ZIP central directory',central_directory_sha256=central_meta['sha256'])
            provenance=json.loads((dest/'DOWNLOAD_PROVENANCE.json').read_text())
            provenance.update(members=[records[n] for n in clip['selected_members']],central_directory_provenance='../CENTRAL_DIRECTORY_PROVENANCE.json',script_sha256=sha(Path(__file__).read_bytes()),crc_verification_completed_utc=now())
            write_json(dest/'DOWNLOAD_PROVENANCE.json',provenance)
        write_json(records_path,records)
        print_event(event='central_crc_verified',members=2240,cumulative_debit=ledger['conservative_debited_bytes'])
    finally:
        helper.urllib.request.urlopen=native_urlopen
        ledger['last_exit_utc']=now();write_json(ledger_path,ledger)

    annotation_path=ROOT/'runtime/VT_TRAIN_01_ANNOTATIONS.json.gz'
    source_annotations=json.load(gzip.open(annotation_path,'rt'))
    images_by_id={im['id']:im for im in source_annotations['images']}
    annotations_by_image=collections.defaultdict(list)
    for a in source_annotations['annotations']:annotations_by_image[a['image_id']].append(a)
    split_by_seq={s:split for split,seqs in selection['splits'].items() for s in seqs}
    clips_out=[]; image_hash_index=collections.defaultdict(list)
    for clip in selection['clips']:
        seq=clip['sequence'];dest=DATA/seq;rawroot=dest/'source_archive'
        ims=[images_by_id[i] for i in clip['selected_image_ids']]
        assert [int(Path(im['file_name']).stem) for im in ims]==list(range(int(Path(clip['first_filename']).stem),int(Path(clip['last_filename']).stem)+1))
        selected_ids=set(clip['selected_image_ids']);video_ids={im['video_id'] for im in ims}
        aa=[a for i in clip['selected_image_ids'] for a in annotations_by_image[i]]
        trackids={a['track_id'] for a in aa}
        # Preserve all original annotation fields; source tracks are video-level metadata.
        subset={'categories':source_annotations['categories'],'images':ims,'annotations':aa,'videos':[v for v in source_annotations['videos'] if v['id'] in video_ids],'source_tracks_metadata':[v for v in source_annotations.get('tracks',[]) if v.get('id') in video_ids],'original_track_ids':sorted(trackids),'subset_provenance':{'source_path':str(annotation_path),'source_sha256':sha(annotation_path.read_bytes()),'selection_sha256':expected_sha,'retention':'All original annotations, categories, ignore/crowd fields and bbox polygons for each selected image; no filtering, pseudo mask or inferred visibility.','split':split_by_seq[seq],'sequence':seq}}
        write_json(dest/'annotations.json',subset)
        array=np.empty((160,512,640),dtype=np.uint8);frames=[]
        for i,(im,name) in enumerate(zip(ims,clip['selected_members'])):
            path=rawroot/name;raw=path.read_bytes();rec=records[name]
            assert name=='VT-Tiny-MOT/train2017/'+im['file_name']
            assert len(raw)==entries[name]['uncompressed_bytes']==rec['bytes'] and sha(raw)==rec['sha256']
            with Image.open(path) as image:
                image.load();assert image.mode=='L' and image.size==(640,512) and image.format=='JPEG'
                array[i]=np.asarray(image)
            item={'index':i,'source_image_id':im['id'],'source_video_id':im['video_id'],'source_file_name':im['file_name'],'raw_filename_number':int(Path(im['file_name']).stem),'source_frame_id':im['frame_id'],'source_mot_frame_id':im['mot_frame_id'],'archive_member':name,'relative_path':str(path.relative_to(dest)),'sha256':rec['sha256'],'bytes':len(raw),'shape':[512,640],'dtype':'uint8','native_mode':'L','annotation_ids':[a['id'] for a in annotations_by_image[im['id']]],'crc_verified':rec['crc_verified']}
            frames.append(item);image_hash_index[rec['sha256']].append({'sequence':seq,'index':i,'split':split_by_seq[seq]})
        tmp=dest/'raw_frames.npy.tmp'
        with tmp.open('wb') as f:np.save(f,array,allow_pickle=False)
        tmp.replace(dest/'raw_frames.npy')
        linked=dest/'images';target=rawroot/'VT-Tiny-MOT/train2017'/seq/'01'
        if not linked.exists():linked.symlink_to(os.path.relpath(target,dest),target_is_directory=True)
        assert linked.resolve()==target.resolve()
        manifest={'sequence':seq,'split':split_by_seq[seq],'frame_count':160,'raw_frames_array':'raw_frames.npy','shape':[160,512,640],'dtype':'uint8','array_sha256':sha((dest/'raw_frames.npy').read_bytes()),'images_link':'images','frame_order':'selected_image_ids from frozen DATA_SELECTION; native numeric file name increasing','raw_inputs_resized':False,'annotations_sha256':sha((dest/'annotations.json').read_bytes()),'frames':frames}
        write_json(dest/'FRAME_MANIFEST.json',manifest)
        clips_out.append({'sequence':seq,'split':split_by_seq[seq],'frames':160,'annotations':len(aa),'tracks':len(trackids),'native_shape':[512,640],'raw_frames_npy':str(dest/'raw_frames.npy'),'array_sha256':manifest['array_sha256'],'frame_manifest_sha256':sha((dest/'FRAME_MANIFEST.json').read_bytes()),'all_crc_verified':all(x['crc_verified'] for x in frames)})
        print_event(event='materialized',sequence=seq,annotations=len(aa),shape=[160,512,640])

    # Inspection-only contact sheets: first/middle/last raw frames, 2x downsample display.
    font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()
    def sheet(name,seqs):
        canvas=Image.new('RGB',(960,len(seqs)*292),'white');draw=ImageDraw.Draw(canvas);panel_info=[]
        for row,seq in enumerate(seqs):
            mm=json.loads((DATA/seq/'FRAME_MANIFEST.json').read_text())
            for col,index in enumerate([0,79,159]):
                f=mm['frames'][index]
                with Image.open(DATA/seq/f['relative_path']) as im:vis=im.convert('RGB').resize((320,256),Image.Resampling.BOX)
                x,y=col*320,row*292
                draw.text((x+3,y+3),f'{seq} {split_by_seq[seq]}',fill='black',font=font)
                draw.text((x+3,y+20),f"index {index} / file {f['raw_filename_number']:05d}",fill='black',font=font)
                canvas.paste(vis,(x,y+36));panel_info.append({'row':row,'column':col,'sequence':seq,'index':index,'image_sha256':f['sha256']})
        canvas.save(DATA/name)
        write_json(DATA/(name+'.json'),{'inspection_only':True,'display_resize':[320,256],'original_size':[640,512],'panels':panel_info})
    sheet('CONTACT_TRAIN.png',selection['splits']['train'])
    sheet('CONTACT_HELDOUT.png',selection['splits']['calibration']+selection['splits']['confirmation'])
    sheet('CONTACT_ALL.png',[c['sequence'] for c in selection['clips']])
    cross_seq_dupes=[v for v in image_hash_index.values() if len({r['sequence'] for r in v})>1]
    summary={'completed_utc':now(),'status':'complete','selection_sha256':expected_sha,'selection_policy':selection['selection_policy'],'frames':sum(c['frames'] for c in clips_out),'sequence_count':len(clips_out),'raw_input_format':'uint8 native grayscale [160,512,640], no resize','clips':clips_out,'network':{'body_budget_bytes':LIMIT,'conservative_debited_bytes':ledger['conservative_debited_bytes'],'actual_body_bytes_completed_reads':ledger['actual_body_bytes_completed_reads']},'all_crc_verified':all(c['all_crc_verified'] for c in clips_out),'cross_sequence_exact_jpeg_hash_duplicates':cross_seq_dupes,'independence_limit':'No duplicate bytes does not prove different scenes. Review contact sheets; prefixes are only grouping proxy. Confirmation images viewed only for annotation/scene checking; no model predictions.'}
    assert summary['frames']==2240 and ledger['conservative_debited_bytes']<=LIMIT
    write_json(DATA/'ACQUISITION_REPORT.json',summary)
    print_event(event='complete',frames=2240,sequences=14,actual_response_bytes=ledger['actual_body_bytes_completed_reads'],conservative_debit=ledger['conservative_debited_bytes'])

if __name__=='__main__':main()
