from pathlib import Path
import json,hashlib,datetime
import numpy as np
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent;Q=P.parent.parent;S=Q/'additional_clips/DJI_0047_3__4621500__00000__00159';manifest=json.loads((S/'FRAME_MANIFEST.json').read_text());ann=json.loads((S/'annotations.json').read_text());imgids={x['raw_frame']:x['source_image_id']for x in manifest['frames']};selected={x['image_id']:x for x in ann['annotations']if x['track_id']==4621500}
protocol={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Post hoc explanatory visual review after parent reported model response loss; not blind confirmatory annotation and not proof of visibility.','frames':[list(range(0,40)),list(range(40,160,10))],'selection':'fixed complete first40 frames plus stride10 afterwards as requested; no prediction arrays opened','raw_context_px':48,'display_scale':3,'resize':'nearest neighbor','intensity':'native0-255, no contrast normalization','bbox':'thin red outline on crop, original coordinates unmodified','original_registration_rules_unchanged':True}
(P/'FIXED_VISUAL_REVIEW_PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
for name,frames,ncols in [('FIXED_FIRST40_CONTEXT.png',list(range(40)),5),('FIXED_LATER_CONTEXT.png',list(range(40,160,10)),4)]:
 cw,ch=300,183;nrows=(len(frames)+ncols-1)//ncols;out=Image.new('RGB',(cw*ncols,ch*nrows),'white');draw=ImageDraw.Draw(out)
 for j,f in enumerate(frames):
  imrow=manifest['frames'][f];p=S/imrow['local_path'];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==imrow['sha256'];im=Image.open(p).convert('L');a=selected[imgids[f]];x,y,w,h=a['bbox'];cx=x+w/2;cy=y+h/2;x0=max(0,min(640-48,int(cx)-24));y0=max(0,min(512-48,int(cy)-24));crop=im.crop((x0,y0,x0+48,y0+48)).resize((144,144),Image.Resampling.NEAREST).convert('RGB');marked=crop.copy();d=ImageDraw.Draw(marked);d.rectangle(((x-x0)*3,(y-y0)*3,(x+w-x0)*3-1,(y+h-y0)*3-1),outline='red',width=1)
  ox=(j%ncols)*cw;oy=(j//ncols)*ch;out.paste(crop,(ox,oy+20));out.paste(marked,(ox+148,oy+20));draw.text((ox+3,oy+3),f'frame {f:03d} | raw / bbox | {x},{y},{w},{h}',fill='black');draw.text((ox+3,oy+166),f'48x48 crop, 3x nearest, origin {x0},{y0}',fill='black')
 out.save(P/name)
print('fixed visual sheets saved')
