import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
Q=Path('/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010')
D=Q/'low_response_information_20261011/visual';R=D.parent;P=Q/'stage1_goal'
cases=list(csv.DictReader((D/'SELECTED_CASES.csv').open())); rows=list(csv.DictReader((D/'SELECTED_FRAMES.csv').open()))
detail_path=R/'analysis/LOW_RESPONSE_OBJECTS.csv';details=list(csv.DictReader(detail_path.open()));dd={(r['sequence'],r['annotation_id'],r['arrangement']):r for r in details}
font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),13)
raws={};annotations={};zz={}
for seq in sorted({x['sequence'] for x in cases}):
    raws[seq]=np.load(P/'data'/seq/'raw_frames.npy',mmap_mode='r')
    ann=json.loads((P/'data'/seq/'annotations.json').read_text());byid={r['id']:i for i,r in enumerate(ann['images'])};aa=defaultdict(list)
    for a in ann['annotations']:aa[byid[a['image_id']]].append(a)
    annotations[seq]=aa
    for arr in ['O','S']:zz[seq,arr]=np.load(R/'logits'/seq/f'fused_{arr}_logits.npy',mmap_mode='r')
cmap=matplotlib.colormaps['magma'];extent=96;scale=3
def cropview(seq,r,arr=None):
    cx=round(float(r['bbox_x'])+float(r['bbox_w'])/2);cy=round(float(r['bbox_y'])+float(r['bbox_h'])/2)
    left=cx-extent//2;top=cy-extent//2
    if arr is None:im=Image.fromarray(raws[seq][int(r['frame'])],mode='L').convert('RGB')
    else:
        z=zz[seq,arr][int(r['frame'])];im=Image.fromarray((cmap(np.clip((z+20)/40,0,1))[...,:3]*255).astype(np.uint8))
    im=im.crop((left,top,left+extent,top+extent)).resize((extent*scale,extent*scale),Image.Resampling.NEAREST);dr=ImageDraw.Draw(im)
    for a in annotations[seq][int(r['frame'])]:
        x,y,w,h=a['bbox'];target=str(a['id'])==r['annotation_id']
        if x+w<left or y+h<top or x>=left+extent or y>=top+extent:continue
        dr.rectangle(((x-left)*scale,(y-top)*scale,(x+w-left)*scale-1,(y+h-top)*scale-1),outline='yellow' if target else '#00aadd',width=2 if target else 1)
    for a in ['O','S'] if arr is None else [arr]:
        d=dd[seq,r['annotation_id'],a];x=(float(d['neighborhood_peak_x'])-left)*scale;y=(float(d['neighborhood_peak_y'])-top)*scale
        color='#ff6040' if a=='O' else '#40ff70'
        if a=='O':dr.line((x-5,y,x+5,y),fill=color,width=2);dr.line((x,y-5,x,y+5),fill=color,width=2)
        else:dr.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
        if arr:
            nx0,ny0,nx1,ny1=[int(d[k]) for k in ('neighborhood_x0','neighborhood_y0','neighborhood_x1','neighborhood_y1')]
            dr.rectangle(((nx0-left)*scale,(ny0-top)*scale,(nx1-left)*scale-1,(ny1-top)*scale-1),outline='#b0b0b0',width=1)
            if d['matched_peak_x']:
                mx=(float(d['matched_peak_x'])-left)*scale;my=(float(d['matched_peak_y'])-top)*scale
                dr.rectangle((mx-4,my-4,mx+4,my+4),outline='white',width=1)
    return im
out=[]
for c in cases:
    rr=[r for r in rows if r['case_id']==c['case_id']];seq=c['sequence'];width=max(1000,len(rr)*300)
    im=Image.new('RGB',(width,1112),'#15191f');dr=ImageDraw.Draw(im)
    dr.text((8,6),c['case_id'],font=font,fill='white')
    dr.text((8,25),'Target-centered raw [0,255]; yellow=focus GT, cyan=other GT; O peak=orange +, S peak=green circle; white box=actual matched peak',font=font,fill='white')
    dr.text((8,44),'Logit maps fixed [-20,20], magma black=-20, purple=-10, red=0, orange=10, pale=20 (clipped only for display). Gray box=search area.',font=font,fill='white')
    for col,r in enumerate(rr):
        x=col*300;dr.text((x+4,65),f'f={r["frame"]} ann={r["annotation_id"]} group={r["visual_group"]}',font=font,fill='white');im.paste(cropview(seq,r),(x+4,86))
        for ai,arr in enumerate(['O','S']):
            d=dd[seq,r['annotation_id'],arr];y0=382+ai*360
            dr.text((x+4,y0),f'{arr}: M={d["matched_at_05"]} target z={float(d["target_box_max_logit"]):.2f} p={float(d["target_box_max_probability"]):.2g}',font=font,fill='white')
            im.paste(cropview(seq,r,arr),(x+4,y0+22))
            dr.text((x+4,y0+313),f'peak z={float(d["neighborhood_peak_logit"]):.2f} dist={float(d["neighborhood_peak_distance_to_box"]):.2f}px inside={d["neighborhood_peak_inside_target"]}',font=font,fill='white')
            def fm(k):return f'{float(d[k]):.2f}' if d[k] else 'NA'
            dr.text((x+4,y0+332),f'local/global d95={fm("local_margin_q95")}/{fm("global_margin_q95")}',font=font,fill='white')
    path=D/(c['case_id']+'_peaks.png');im.save(path);out.append(dict(case_id=c['case_id'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(D/'PEAK_RENDER_MANIFEST.json').write_text(json.dumps(dict(detail_csv=str(detail_path),detail_csv_sha256=hashlib.sha256(detail_path.read_bytes()).hexdigest(),maps_fixed_range=[-20,20],panels=out),indent=2)+'\n')
print(json.dumps({'rendered_peak_panels':len(out)}))
