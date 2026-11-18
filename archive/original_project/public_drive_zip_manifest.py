"""Inspect an officially linked public ZIP using its normal Drive download form and HTTP Range.
No archive extraction; reads only 32 bytes, ZIP footer, and central-directory metadata.
"""
import urllib.request, urllib.parse, struct, json, sys, collections
from html.parser import HTMLParser
class Form(HTMLParser):
    def __init__(self):
        super().__init__(); self.action=None; self.fields={}
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag=='form': self.action=a.get('action')
        if tag=='input' and a.get('name'): self.fields[a['name']]=a.get('value','')
def public_download_url(fid):
    with urllib.request.urlopen('https://drive.google.com/uc?export=download&id='+fid, timeout=20) as r:
        p=Form(); p.feed(r.read(20000).decode())
    if not p.action: raise RuntimeError('Expected publicly offered Drive download form')
    return p.action+'?'+urllib.parse.urlencode(p.fields)
def get_range(url, first, last):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={first}-{last}'}),timeout=30) as r:
        if r.status!=206: raise RuntimeError('Server did not honor Range; stopped before reading body')
        data=r.read(last-first+2)
        if len(data)!=last-first+1: raise RuntimeError('Range size mismatch')
        return data,r.headers['Content-Range']
def inspect(fid):
    url=public_download_url(fid);head,cr=get_range(url,0,31);size=int(cr.split('/')[-1])
    tail,_=get_range(url,max(0,size-65557),size-1);off=tail.rfind(b'PK\x05\x06')
    if off<0: raise RuntimeError('ZIP footer missing')
    sig,disk,cd_disk,ndisk,ntotal,cd_size,cd_offset,comment=struct.unpack_from('<4s4H2LH',tail,off)
    if cd_size>16000000: raise RuntimeError(f'Central directory {cd_size} exceeds 16 MB metadata cap')
    central,_=get_range(url,cd_offset,cd_offset+cd_size-1);pos=0;entries=[]
    while pos+46<=len(central):
        fields=struct.unpack_from('<4s6H3L5H2L',central,pos)
        if fields[0]!=b'PK\x01\x02': raise RuntimeError('Invalid central directory')
        flags,method=fields[3],fields[4];cs,us=fields[8],fields[9];nl,el,cl=fields[10:13];lo=fields[16]
        name=central[pos+46:pos+46+nl].decode('utf-8' if flags&0x800 else 'cp437')
        entries.append({'name':name,'compressed_bytes':cs,'uncompressed_bytes':us,'method':method,'local_header_offset':lo})
        pos+=46+nl+el+cl
    return {'file_id':fid,'archive_bytes':size,'central_directory_bytes':cd_size,'metadata_bytes_read':32+len(tail)+len(central),'entries_count':len(entries),'entries':entries}
if __name__=='__main__':
    r=inspect(sys.argv[1]);open(sys.argv[2],'w').write(json.dumps(r,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k!='entries'},indent=2));print('sample_names',[x['name'] for x in r['entries'][:20]])
