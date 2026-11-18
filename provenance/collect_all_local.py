#!/usr/bin/env python3
"""Collect the existing project without changing any source file contents."""
import os, json, hashlib, shutil, datetime, argparse
from pathlib import Path

R = Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009')
B = Path('/home/ly/DeepProPlus_TLFP_Code')
A = B/'archive/infrared_problem_screen_20261009'
P = B/'provenance/ALL_LOCAL_COLLECTION.json'
EXCLUDED = {'__pycache__', '.pytest_cache', '.git'}

def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(x):
    tmp=P.with_suffix('.json.tmp'); tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(P)

def collect():
    rows=[]; excluded=[]; symlinks=[]
    def walk(src,dst,parents=()):
        resolved=src.resolve(strict=True)
        if resolved in parents: raise RuntimeError(f'Symlink cycle: {src}')
        if src.is_symlink(): symlinks.append({'source':str(src),'target':os.readlink(src),'resolved':str(resolved),'destination':str(dst.relative_to(B)), 'handling':'materialized as directory or regular hardlinked file'})
        if src.is_dir():
            dst.mkdir(parents=True,exist_ok=True)
            for child in sorted(src.iterdir()):
                if child.name in EXCLUDED or child.suffix=='.pyc' or (child.name=='site' and child.parent.name=='dependencies'):
                    excluded.append(str(child));continue
                walk(child,dst/child.name,(*parents,resolved))
            return
        if not src.is_file(): raise RuntimeError(f'Unsupported file: {src}')
        stat=resolved.stat()
        if dst.exists():
            if os.path.samefile(resolved,dst): mode='hardlink_preexisting'
            elif dst.stat().st_size == stat.st_size and stat.st_uid != os.getuid():
                def digest(path):
                    h=hashlib.sha256()
                    with path.open('rb') as f:
                        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
                    return h.digest()
                if digest(resolved)!=digest(dst):raise FileExistsError(dst)
                mode='copy_permission_fallback_preexisting'
            else: raise FileExistsError(dst)
        else:
            try: os.link(resolved,dst);mode='hardlink'
            except PermissionError:
                shutil.copy2(resolved,dst);mode='copy_permission_fallback'
        rows.append({'source':str(src),'resolved_source':str(resolved),'destination':str(dst.relative_to(B)),
                     'size':stat.st_size,'source_mtime_ns':stat.st_mtime_ns,'source_device':stat.st_dev,
                     'source_inode':stat.st_ino,'mode':mode})
    walk(R,A)
    walk(R/'temporal_diagnosis_20261010/stage1_goal/data',B/'data/stage1')
    result={'created_utc':stamp(),'status':'LINKS_COMPLETE_HASH_PENDING','source_root':str(R),'archive_root':str(A),
            'policy':'Existing local project; preserve all result/weight/data files. Materialize symlinks. Exclude Git internals, bytecode/test caches and dependencies/site. Original file contents never edited. Hardlinked snapshots must not be edited in place.',
            'exclusions':excluded,'source_symlinks':symlinks,'files':rows,
            'file_count':len(rows),'logical_bytes':sum(x['size'] for x in rows)}
    save(result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','exclusions','source_symlinks']},ensure_ascii=False),flush=True)

def hashes():
    result=json.loads(P.read_text());cache={};done=0
    for row in result['files']:
        src=Path(row['resolved_source']);dst=B/row['destination'];s=src.stat()
        if (s.st_size,s.st_mtime_ns)!=(row['size'],row['source_mtime_ns']):raise RuntimeError(f'Source changed: {src}')
        key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if key not in cache:
            h=hashlib.sha256()
            with src.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            cache[key]=h.hexdigest()
        row['sha256']=cache[key]
        if row['mode'].startswith('hardlink'):
            if not os.path.samefile(src,dst):raise RuntimeError(f'Hardlink mismatch: {dst}')
        else:
            h=hashlib.sha256()
            with dst.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=row['sha256']:raise RuntimeError(f'Copy mismatch: {dst}')
        done+=1
        if done%1000==0: print(json.dumps({'hashed_files':done,'total':len(result['files']),'unique_inodes':len(cache)}),flush=True)
    result.update(status='COMPLETE',verified_utc=stamp(),unique_source_inodes=len(cache),all_hashes_verified=True)
    save(result); print(json.dumps({'status':'COMPLETE','files':len(result['files']),'bytes':result['logical_bytes'],'unique_inodes':len(cache)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--hash-only',action='store_true');a=parser.parse_args()
    hashes() if a.hash_only else collect()
