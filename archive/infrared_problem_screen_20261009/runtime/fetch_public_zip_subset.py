"""Bounded member extraction from an officially linked public Drive ZIP.
Uses the website's public download form; requires HTTP 206 and checks range offsets.
No credentials are accepted/stored. Rejects unsafe archive paths. No full-archive fetch.
"""
import sys, json, struct, urllib.request, urllib.parse, zlib, hashlib, time
from pathlib import Path, PurePosixPath
from html.parser import HTMLParser
class Form(HTMLParser):
    def __init__(self): super().__init__(); self.action=None; self.fields={}
    def handle_starttag(self, tag, attrs):
        d=dict(attrs)
        if tag=='form': self.action=d.get('action')
        if tag=='input' and d.get('name'): self.fields[d['name']]=d.get('value','')
def safe_path(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('Unsafe archive member path')
    return p
class PublicArchive:
    def __init__(self, manifest, budget=100_000_000):
        self.manifest=manifest;self.budget=budget;self.transferred=0;self.records=[]
        fid=manifest['file_id'];self.source_url='https://drive.google.com/file/d/'+fid+'/view'
        with urllib.request.urlopen('https://drive.google.com/uc?export=download&id='+fid,timeout=30) as r:
            p=Form();p.feed(r.read(20000).decode())
        if not p.action: raise RuntimeError('Public download form unavailable')
        self.url=p.action+'?'+urllib.parse.urlencode(p.fields)
    def read_range(self, start, end):
        if end<start:return b''
        n=end-start+1
        if self.transferred+n>self.budget:raise RuntimeError('Subset transfer budget exceeded')
        for attempt in range(3):
            try:
                req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=45) as r:
                    expected=f'bytes {start}-{end}/{self.manifest["archive_bytes"]}'
                    if r.status!=206 or r.headers.get('Content-Range')!=expected:
                        raise RuntimeError('Range unsupported or wrong response; stopped before body')
                    b=r.read(n+1);self.transferred+=len(b)
                if len(b)!=n:raise RuntimeError('Incomplete ranged response')
                return b
            except Exception:
                if attempt==2:raise
                time.sleep(1)
    def member_from_blob(self, entry, blob, start):
        safe_path(entry['name']);off=entry['local_header_offset']-start
        h=struct.unpack_from('<4s5H3L2H',blob,off)
        if h[0]!=b'PK\x03\x04':raise RuntimeError('Invalid local ZIP header')
        flags,method,crc=h[2],h[3],h[6];nl,el=h[9],h[10]
        if flags&1:raise RuntimeError('Encrypted archive not supported')
        begin=off+30+nl+el;compressed=blob[begin:begin+entry['compressed_bytes']]
        if len(compressed)!=entry['compressed_bytes']:raise RuntimeError('Insufficient range for ZIP member')
        data=zlib.decompress(compressed,-15) if method==8 else compressed if method==0 else None
        if data is None:raise RuntimeError('Unsupported ZIP compression')
        if len(data)!=entry['uncompressed_bytes']:raise RuntimeError('Uncompressed size mismatch')
        if not flags&8 and zlib.crc32(data)&0xffffffff!=crc:raise RuntimeError('CRC mismatch')
        self.records.append({'member':entry['name'],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'crc_verified':not bool(flags&8)})
        return data
    def member(self, entry):
        off=entry['local_header_offset'];header=self.read_range(off,off+29)
        h=struct.unpack('<4s5H3L2H',header);end=off+30+h[9]+h[10]+entry['compressed_bytes']-1
        return self.member_from_blob(entry,header+self.read_range(off+30,end),off)
    def extract(self, entries, destination):
        entries=sorted([x for x in entries if not x['name'].endswith('/')],key=lambda x:x['local_header_offset'])
        destination=Path(destination).resolve();groups=[]
        for e in entries:
            safe_path(e['name']);start=e['local_header_offset'];end=min(self.manifest['archive_bytes']-1,start+30+len(e['name'].encode())+65535+e['compressed_bytes'])
            if groups and start-groups[-1][1]<65536 and end-groups[-1][0]<=8_000_000:
                groups[-1][1]=max(end,groups[-1][1]);groups[-1][2].append(e)
            else:groups.append([start,end,[e]])
        for start,end,items in groups:
            b=self.read_range(start,end)
            for e in items:
                data=self.member_from_blob(e,b,start);p=destination.joinpath(*safe_path(e['name']).parts)
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            print(json.dumps({'extracted_members':len(items),'transferred_bytes':self.transferred}),flush=True)
    def provenance(self):
        return {'source_url':self.source_url,'official_public_form_used':True,'range_only':True,'archive_bytes':self.manifest['archive_bytes'],'transferred_bytes':self.transferred,'transfer_budget':self.budget,'members':self.records}
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--prefix',action='append',required=True);p.add_argument('--destination',required=True);p.add_argument('--budget',type=int,default=100_000_000);a=p.parse_args()
    m=json.loads(Path(a.manifest).read_text());arc=PublicArchive(m,a.budget);chosen=[x for x in m['entries'] if any(x['name'].startswith(v) for v in a.prefix)]
    arc.extract(chosen,a.destination);Path(a.destination,'DOWNLOAD_PROVENANCE.json').write_text(json.dumps(arc.provenance(),indent=2))
