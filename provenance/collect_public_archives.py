#!/usr/bin/env python3
"""Download public source ZIPs, verify their frozen member manifest and CRCs, and safely extract."""
from pathlib import Path, PurePosixPath
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, json, hashlib, urllib.request, urllib.parse, time, zipfile, stat, shutil, datetime, argparse, collections

B=Path('/home/ly/DeepProPlus_TLFP_Code')
A=B/'archive/infrared_problem_screen_20261009'
MANIFESTS={'NUDT-MIRSDT':'NUDT_PUBLIC_ZIP_MANIFEST.json','VT-Tiny-MOT':'VT_TINY_MOT_PUBLIC_ZIP_MANIFEST.json'}
CHUNK=32*1024*1024
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
class Form(HTMLParser):
    def __init__(self):super().__init__();self.action=None;self.fields={}
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=='form':self.action=d.get('action')
        if tag=='input' and d.get('name'):self.fields[d['name']]=d.get('value','')
def public_url(fid):
    with urllib.request.urlopen('https://drive.google.com/uc?export=download&id='+fid,timeout=45) as r:
        p=Form();p.feed(r.read(50000).decode('utf-8'))
    if not p.action:raise RuntimeError('Public download form unavailable')
    action=urllib.parse.urlparse(p.action)
    if action.scheme!='https' or not (action.hostname=='drive.usercontent.google.com' or action.hostname=='drive.google.com'):
        raise RuntimeError('Unexpected public download host')
    return p.action+'?'+urllib.parse.urlencode(p.fields)
def download(name,manifest,directory,record):
    directory.mkdir(parents=True,exist_ok=True);dest=directory/(name+'.zip');part=dest.with_suffix('.zip.part');statepath=dest.with_suffix('.download.json')
    size=manifest['archive_bytes'];record['download_started_utc']=now()
    if dest.exists():
        if dest.stat().st_size!=size:raise RuntimeError('Existing ZIP has wrong size')
        record['download_mode']='reuse_existing_complete_zip';return dest
    state=json.loads(statepath.read_text()) if statepath.exists() else {'source_url':record['source_url'],'archive_bytes':size,'chunks':{},'created_utc':now()}
    if state['archive_bytes']!=size or state['source_url']!=record['source_url']:raise RuntimeError('Partial file source mismatch')
    fd=os.open(part,os.O_RDWR|os.O_CREAT,0o644);os.ftruncate(fd,size)
    try:
        # Authenticate no user; only the publicly offered download form is used.
        url=public_url(manifest['file_id'])
        jobs=[]
        for start in range(0,size,CHUNK):
            end=min(size,start+CHUNK)-1;old=state['chunks'].get(str(start))
            if old and old['end']==end and hashlib.sha256(os.pread(fd,end-start+1,start)).hexdigest()==old['sha256']:continue
            jobs.append((start,end))
        def fetch(pair):
            start,end=pair;expected=end-start+1;failures=[]
            for attempt in range(3):
                try:
                    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
                    with urllib.request.urlopen(request,timeout=90) as r:
                        if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise RuntimeError('HTTP range/total mismatch')
                        data=r.read(expected+1)
                    if len(data)!=expected:raise RuntimeError('Short/long response')
                    offset=0
                    while offset<len(data):offset+=os.pwrite(fd,data[offset:],start+offset)
                    return str(start),{'end':end,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'attempts':attempt+1,'transient_failure_types':failures}
                except Exception as e:
                    failures.append(type(e).__name__)
                    if attempt==2:raise RuntimeError(f'Chunk {start}-{end} failed after 3 attempts: {type(e).__name__}') from None
                    time.sleep(min(2**attempt,4))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(fetch,x) for x in jobs]
            for f in as_completed(futures):
                key,row=f.result();state['chunks'][key]=row;state['updated_utc']=now();save(statepath,state)
                print(json.dumps({'dataset':name,'chunks_complete':len(state['chunks']),'chunks_total':(size+CHUNK-1)//CHUNK,'downloaded_bytes':sum(x['bytes'] for x in state['chunks'].values())}),flush=True)
        os.fsync(fd)
    finally:os.close(fd)
    if sum(x['bytes'] for x in state['chunks'].values())!=size:raise RuntimeError('Missing completed ranges')
    part.replace(dest);record['download_completed_utc']=now();record['download_mode']='public_form_ranged_complete_archive';return dest
def safe_name(info):
    name=info.filename;p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or '\x00' in name:raise RuntimeError('Unsafe ZIP member path')
    mode=(info.external_attr>>16)&0xffff
    if stat.S_ISLNK(mode):raise RuntimeError('ZIP symlink disallowed')
    if info.flag_bits&1:raise RuntimeError('Encrypted ZIP member disallowed')
    return p
def main(name):
    path=A/MANIFESTS[name];manifest=json.loads(path.read_text());recordpath=B/'provenance'/('PUBLIC_ARCHIVE_'+name+'.json')
    record={'dataset':name,'status':'RUNNING','created_utc':now(),'source_url':'https://drive.google.com/file/d/'+manifest['file_id']+'/view','manifest_path':str(path.relative_to(B)),'manifest_sha256':sha(path),'expected_archive_bytes':manifest['archive_bytes'],'expected_uncompressed_bytes':sum(x['uncompressed_bytes'] for x in manifest['entries']),'policy':'Public data archive only. No upstream code execution. Original project untouched. No split changes or use of confirmation examples for training.'}
    save(recordpath,record)
    try:
        free=shutil.disk_usage(B).free
        need=record['expected_archive_bytes']+record['expected_uncompressed_bytes']+1024**3
        record['disk_available_bytes_before']=free;record['conservative_required_bytes']=need
        if free<need:raise RuntimeError('Insufficient space')
        zpath=download(name,manifest,B/'data/public_archives',record);record['zip_path']=str(zpath.relative_to(B));record['archive_bytes']=zpath.stat().st_size;record['archive_sha256']=sha(zpath);save(recordpath,record)
        final=B/'data/public'/name;staging=B/'data/public'/('.'+name+'.extracting');staging.mkdir(parents=True,exist_ok=True)
        rows=[];expected={x['name']:x for x in manifest['entries']}
        with zipfile.ZipFile(zpath) as z:
            entries=z.infolist();names=[x.filename for x in entries]
            if len(names)!=len(set(names)) or len(names)!=len(expected) or set(names)!=set(expected):raise RuntimeError('ZIP member set differs from frozen manifest')
            for info in entries:
                p=safe_name(info);e=expected[info.filename]
                if p.parts[0]!=name:raise RuntimeError('Unexpected archive top-level directory')
                if (info.file_size,info.compress_size,info.compress_type,info.header_offset)!=(e['uncompressed_bytes'],e['compressed_bytes'],e['method'],e['local_header_offset']):raise RuntimeError('ZIP member metadata differs from frozen manifest')
            record.update(member_count=len(entries),path_safety='PASS',frozen_manifest_match='PASS');save(recordpath,record)
            bad=z.testzip()
            if bad is not None:raise RuntimeError('ZIP CRC failure')
            record['full_zip_crc']='PASS';record['crc_checked_utc']=now();save(recordpath,record)
            for count,info in enumerate(entries):
                p=safe_name(info);dst=staging.joinpath(*p.parts)
                if info.is_dir():dst.mkdir(parents=True,exist_ok=True);continue
                dst.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
                with z.open(info) as source,dst.open('wb') as out:
                    for chunk in iter(lambda:source.read(1024*1024),b''):out.write(chunk);h.update(chunk)
                if dst.stat().st_size!=info.file_size:raise RuntimeError('Extracted size mismatch')
                rows.append({'member':info.filename,'size':info.file_size,'crc32':f'{info.CRC:08x}','sha256':h.hexdigest()})
                if count%10000==0:print(json.dumps({'dataset':name,'extracted_members':count,'total':len(entries)}),flush=True)
        if final.exists():raise FileExistsError('Extraction destination already exists; no overwrite')
        staging.joinpath(name).replace(final);staging.rmdir()
        memberpath=B/'provenance'/('PUBLIC_MEMBERS_'+name+'.json');save(memberpath,{'dataset':name,'files':rows})
        suffix=collections.Counter(Path(x['member']).suffix.lower() for x in rows)
        record.update(status='COMPLETE',completed_utc=now(),extracted_directory=str(final.relative_to(B)),files=len(rows),uncompressed_bytes=sum(x['size'] for x in rows),extension_counts=dict(suffix),member_hash_manifest=str(memberpath.relative_to(B)),member_hash_manifest_sha256=sha(memberpath),transient_signed_urls_persisted=False)
        save(recordpath,record);print(json.dumps({k:v for k,v in record.items() if k not in ['policy']},ensure_ascii=False),flush=True)
    except Exception as e:
        record.update(status='FAILED',failed_utc=now(),error_type=type(e).__name__,error_message=str(e) if not any(s in str(e) for s in ['http://','https://','token=']) else 'URL-bearing error omitted; see error type')
        save(recordpath,record);print(json.dumps(record,ensure_ascii=False),flush=True);raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=list(MANIFESTS));main(p.parse_args().dataset)
