#!/usr/bin/env python3
"""Deterministic raw-pixel analytical crops; no contrast alteration/interpolation."""
import csv
import hashlib
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(FONT, 17)
small = ImageFont.truetype(FONT, 15)
title = ImageFont.truetype(FONT, 25)

def read(p):
    return list(csv.DictReader(p.open()))

strong = read(ROOT / 'result_stratification/context/strong_flips_strict_half_le16.csv')
all_pairs = read(OUT / 'all_strong_window_pairs.csv')
first = {}
for r in sorted(strong, key=lambda r:(r['sequence'],int(r['track_id']),int(r['local_frame']))):
    first.setdefault((r['sequence'],r['track_id']),r)
assert len(first) == 18
rows = []
for n, r in enumerate(first.values(),1):
    p = next(x for x in all_pairs if x['sequence']==r['sequence'] and x['track_id']==r['track_id'] and x['local_frame']==r['local_frame'])
    rows.append(dict(panel=n, **p))

def source(r):
    d = next((ROOT/'additional_clips').glob(r['sequence']+'__*'))
    manifest = json.loads((d/'FRAME_MANIFEST.json').read_text())
    entry = next(x for x in manifest['frames'] if int(x['raw_frame'])==int(r['source_frame']))
    p = d / entry['local_path']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256']
    im = Image.open(p).convert('L')
    assert im.size==(640,512)
    return im, p, entry['sha256']

def region(r, im, zoom=4, size=64, boxed=True, color='#3CFF93'):
    x,y,w,h = [float(r[k]) for k in ['bbox_x','bbox_y','bbox_w','bbox_h']]
    x0 = min(max(int(math.floor(x+w/2-size/2)),0),im.width-size)
    y0 = min(max(int(math.floor(y+h/2-size/2)),0),im.height-size)
    raw = im.crop((x0,y0,x0+size,y0+size)).resize((size*zoom,size*zoom),Image.Resampling.NEAREST).convert('RGB')
    if boxed:
        d = ImageDraw.Draw(raw)
        d.rectangle(((x-x0)*zoom,(y-y0)*zoom,(x+w-x0)*zoom,(y+h-y0)*zoom),outline=color,width=2)
    return raw,(x0,y0,x0+size,y0+size)

mapping = []
for page in range(2):
    canvas = Image.new('RGB',(1680,1170),'#101923')
    d = ImageDraw.Draw(canvas)
    d.text((20,12),f'First strong-flip frame per track | page {page+1}/2',font=title,fill='white')
    d.text((20,47),'Raw JPEG pixels; 64 x 64 crops; 4x nearest-neighbor. Left: raw. Right: GT box overlay. No contrast stretch.',font=font,fill='#C7D2DC')
    for i,r in enumerate(rows[page*9:(page+1)*9]):
        x=20+(i%3)*554; y=86+(i//3)*355
        im,p,sha=source(r)
        raw,rect=region(r,im,boxed=False)
        boxed,_=region(r,im)
        d.text((x,y),f"#{r['panel']:02d} {r['sequence']} | {r['category']}",font=font,fill='white')
        d.text((x,y+23),f"tid {r['track_id']} | src {r['source_frame']} | local {r['local_frame']}",font=small,fill='#C7D2DC')
        d.text((x,y+44),f"box {float(r['bbox_w']):g}x{float(r['bbox_h']):g}; p={float(r['a_boxmax']):.3g} / {float(r['b_boxmax']):.3g}",font=small,fill='#C7D2DC')
        canvas.paste(raw,(x,y+69)); canvas.paste(boxed,(x+270,y+69))
        mapping.append(dict(panel=r['panel'],contact_sheet=f'visibility_first_flip_page{page+1}.png',
                            sequence=r['sequence'],track_id=r['track_id'],category=r['category'],
                            local_frame=int(r['local_frame']),source_frame=int(r['source_frame']),
                            bbox_xywh=[float(r[k]) for k in ['bbox_x','bbox_y','bbox_w','bbox_h']],
                            crop_xyxy=list(rect),original_image=str(p),original_sha256=sha,
                            selection='First strong-flip local frame for this track; numerical ID ascending within sequence.'))
    canvas.save(OUT/f'visibility_first_flip_page{page+1}.png')

opposite = read(OUT/'same_frame_opposite_direction.csv')
pair_mapping=[]
for fi in [17,45]:
    rr=[r for r in opposite if r['sequence']=='DJI_0047_3' and int(r['local_frame'])==fi]
    assert len(rr)==2
    im,p,sha=source(rr[0])
    whole=im.convert('RGB')
    dw=ImageDraw.Draw(whole)
    colors=['#FFC14A','#40DBFF']
    for r,c in zip(rr,colors):
        x,y,w,h=[float(r[k]) for k in ['bbox_x','bbox_y','bbox_w','bbox_h']]
        dw.rectangle((x,y,x+w,y+h),outline=c,width=2)
        label=f"{r['category']} {r['track_id']}"
        tx=min(max(int(x)-40,0),400); ty=int(y)-23 if y>35 else int(y)+int(h)+4
        dw.rectangle((tx,ty,tx+230,ty+21),fill='#101923')
        dw.text((tx+3,ty),label,font=small,fill=c)
    canvas=Image.new('RGB',(1110,1030),'#101923'); d=ImageDraw.Draw(canvas)
    d.text((20,15),f'DJI_0047_3 | current raw frame {fi} | opposite strong responses',font=title,fill='white')
    d.text((20,54),f"Window starts {rr[0]['a_start']} -> {rr[0]['b_start']}; slots {rr[0]['a_slot']} -> {rr[0]['b_slot']}",font=font,fill='white')
    canvas.paste(whole,(20,100))
    d.text((20,630),'Full frame at native 640 x 512.',font=font,fill='#C7D2DC')
    d.text((20,660),'Right: 64 x 64 raw crops, 6x nearest-neighbor.',font=font,fill='#C7D2DC')
    d.text((20,690),'Only GT outline added; no contrast stretching.',font=font,fill='#C7D2DC')
    d.text((20,736),'Boxes indicate annotation locations.',font=font,fill='white')
    d.text((20,765),'A box alone does not establish target visibility.',font=font,fill='white')
    for j,(r,c) in enumerate(zip(rr,colors)):
        y=100+j*455
        crop,rect=region(r,im,zoom=6,color=c)
        canvas.paste(crop,(700,y+60))
        d.text((700,y),f"{r['category']} {r['track_id']}",font=font,fill=c)
        d.text((700,y+21),f"p {float(r['a_boxmax']):.6g} -> {float(r['b_boxmax']):.6g}",font=small,fill='white')
        d.text((700,y+41),f"point match {r['a_matched']} -> {r['b_matched']}",font=small,fill='white')
        pair_mapping.append(dict(figure=f'opposite_frame{fi:03d}_whole_and_crops.png',sequence=r['sequence'],
                                 local_frame=fi,source_frame=int(r['source_frame']),track_id=r['track_id'],category=r['category'],
                                 bbox_xywh=[float(r[k]) for k in ['bbox_x','bbox_y','bbox_w','bbox_h']],
                                 crop_xyxy=list(rect),original_image=str(p),original_sha256=sha,
                                 a_start=int(r['a_start']),b_start=int(r['b_start']),
                                 a_boxmax=float(r['a_boxmax']),b_boxmax=float(r['b_boxmax'])))
    canvas.save(OUT/f'opposite_frame{fi:03d}_whole_and_crops.png')
(OUT/'VISIBILITY_MAPPING.json').write_text(json.dumps(dict(selection='Earliest strong-flip frame per track, all 18 track IDs; plus predetermined opposite-response frames17/45',
    rendering='JPEG L pixels unmodified, no histogram normalization, nearest-neighbor zoom; box overlays shown alongside original raw crops on contact sheets',
    first_flip=mapping,opposite_pairs=pair_mapping),ensure_ascii=False,indent=2)+'\n')
print('\n'.join(str(OUT/f) for f in ['visibility_first_flip_page1.png','visibility_first_flip_page2.png',
    'opposite_frame017_whole_and_crops.png','opposite_frame045_whole_and_crops.png','VISIBILITY_MAPPING.json']))
