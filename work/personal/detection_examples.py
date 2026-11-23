from pathlib import Path
from ultralytics import YOLO
import numpy as np, json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
P=Path(__file__).resolve().parent;root=P/'balanced_split';m=YOLO(P/'yolo_balanced/weights/best.pt')
classes=['person','car','cup'];counts={}
for split in ['train','val','test']:
 files=sorted((root/'labels'/split).glob('*.txt'));labs=[[int(l.split()[0]) for l in f.read_text().splitlines()] for f in files]
 counts[split]=[[sum(k in row for row in labs),sum(row.count(k) for row in labs)] for k in range(3)]
files=sorted((root/'images/test').glob('*.jpg'));rows=[]
for f in files:
 im=np.array(Image.open(f).convert('RGB'));h,w=im.shape[:2];gt=[]
 for line in (root/'labels/test'/(f.stem+'.txt')).read_text().splitlines():
  c,x,y,bh,bv=map(float,line.split());gt.append([int(c),(x-bh/2)*w,(y-bv/2)*h,(x+bh/2)*w,(y+bv/2)*h])
 out=m.predict(str(f),imgsz=320,conf=.25,verbose=False)[0].boxes
 pred=[[int(c),*xy.tolist(),float(conf)] for xy,conf,c in zip(out.xyxy.cpu(),out.conf.cpu(),out.cls.cpu())];rows.append(dict(id=f.stem,gt=gt,pred=pred))
(P/'det_examples.json').write_text(json.dumps(dict(counts=counts,rows=rows),indent=2))
examples=[next(i for i,r in enumerate(rows) if any(b[0]==2 for b in r['gt'])),max(range(len(rows)),key=lambda i:len(rows[i]['pred']))]
if examples[0]==examples[1]:examples[1]=max((i for i in range(len(rows)) if i!=examples[0]),key=lambda i:len(rows[i]['pred']))
fig,axes=plt.subplots(2,2,figsize=(9,6.2))
for j,i in enumerate(examples):
 r=rows[i];im=Image.open(files[i])
 for ax,items,title in zip(axes[j],[r['gt'],r['pred']],['Ground truth','Prediction']):
  ax.imshow(im);ax.axis('off');ax.set_title(r['id']+' | '+title,fontsize=10)
  for c,x,y,x2,y2,*conf in items:
   color=['#1fdca7','#ffe04d','#ff6767'][c];ax.add_patch(Rectangle((x,y),x2-x,y2-y,fill=False,edgecolor=color,lw=1));ax.text(x,y,classes[c]+(f' {conf[0]:.2f}' if conf else ''),fontsize=6,color='black',bbox=dict(facecolor=color,edgecolor='none',pad=.3))
fig.tight_layout();fig.savefig(P/'det_examples.png',dpi=190)
r=list(csv.DictReader((P/'yolo_balanced/results.csv').open()));fig,ax=plt.subplots(1,2,figsize=(9,3))
for k in ['box_loss','cls_loss','dfl_loss']:ax[0].plot([int(z['epoch']) for z in r],[float(z['train/'+k]) for z in r],label=k)
for k in ['mAP50(B)','mAP50-95(B)']:ax[1].plot([int(z['epoch']) for z in r],[float(z['metrics/'+k]) for z in r],label=k)
for a in ax:a.legend(fontsize=8);a.set_xlabel('Epoch')
ax[0].set_ylabel('Training loss');ax[1].set_ylabel('Validation AP');fig.tight_layout();fig.savefig(P/'det_curve.png',dpi=170)
