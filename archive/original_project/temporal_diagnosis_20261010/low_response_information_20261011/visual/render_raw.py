import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image,ImageDraw,ImageFont

Q=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
D=Q/'low_response_information_20261011/visual'; P=Q/'stage1_goal'
cases=list(csv.DictReader((D/'SELECTED_CASES.csv').open())); rows=list(csv.DictReader((D/'SELECTED_FRAMES.csv').open()))
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
except OSError:font=ImageFont.load_default()
raws={}; annotations={}
for seq in sorted({x['sequence'] for x in cases}):
    raws[seq]=np.load(P/'data'/seq/'raw_frames.npy',mmap_mode='r')
    ann=json.loads((P/'data'/seq/'annotations.json').read_text());byid={r['id']:i for i,r in enumerate(ann['images'])}; aa=defaultdict(list)
    for a in ann['annotations']:aa[byid[a['image_id']]].append(a)
    annotations[seq]=aa
def view(seq,r,extent,scale,cx,cy):
    im=Image.fromarray(raws[seq][int(r['frame'])],mode='L').convert('RGB')
    left=int(cx-extent//2);top=int(cy-extent//2)
    im=im.crop((left,top,left+extent,top+extent)).resize((extent*scale,extent*scale),Image.Resampling.NEAREST);dr=ImageDraw.Draw(im)
    for a in annotations[seq][int(r['frame'])]:
        x,y,w,h=a['bbox'];target=str(a['id'])==r['annotation_id']
        if x+w<left or y+h<top or x>=left+extent or y>=top+extent:continue
        color=(255,220,0) if target else (0,150,220)
        dr.rectangle(((x-left)*scale,(y-top)*scale,(x+w-left)*scale-1,(y+h-top)*scale-1),outline=color,width=2 if target else 1)
    return im
out=[]
for c in cases:
    rr=[r for r in rows if r['case_id']==c['case_id']];seq=c['sequence']
    cx=round(float(np.median([float(r['bbox_x'])+float(r['bbox_w'])/2 for r in rr])));cy=round(float(np.median([float(r['bbox_y'])+float(r['bbox_h'])/2 for r in rr])))
    im=Image.new('RGB',(max(1000,len(rr)*300),742),'#15191f');dr=ImageDraw.Draw(im)
    dr.text((10,8),c['case_id'],font=font,fill='white');dr.text((10,28),'Raw fixed [0,255]; yellow=focus GT, cyan=all other GT; TOP: fixed scene crop, BOTTOM: target-centered zoom',font=font,fill='white')
    for col,r in enumerate(rr):
        x=col*300;dr.text((x+4,54),f'f={r["frame"]} source={r["source_frame"]} ann={r["annotation_id"]}',font=font,fill='white')
        im.paste(view(seq,r,96,3,cx,cy),(x+4,80));dr.text((x+4,374),f'group={r["visual_group"]}',font=font,fill='white')
        tx=round(float(r['bbox_x'])+float(r['bbox_w'])/2);ty=round(float(r['bbox_y'])+float(r['bbox_h'])/2)
        im.paste(view(seq,r,32,9,tx,ty),(x+4,400));dr.text((x+4,692),f'O/S match {r["official_matched"]}/{r["symmetric_matched"]}',font=font,fill='white')
        dr.text((x+4,714),f'pool box max={float(r["context_box_max"]):.3g}',font=font,fill='white')
    path=D/(c['case_id']+'_raw.png');im.save(path)
    out.append(dict(case_id=c['case_id'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),center_x=cx,center_y=cy,n_frames=len(rr)))
(D/'RAW_RENDER_MANIFEST.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'rendered_cases':len(out),'case_frames':len(rows)}))
