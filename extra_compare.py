"""Run from the extracted experiment bundle root with the cv environment."""
from pathlib import Path
import io,json,tarfile
import cv2
import numpy as np
from PIL import Image
from scipy.stats import binomtest

p=Path('work/personal')
archives=Path('work/provenance/downloads')
names=json.loads((p/'seg_split.json').read_text())[2]
images={}
with tarfile.open(archives/'images.tar.gz.incomplete','r|gz') as tar:
    try:
        for member in tar:
            key=Path(member.name).stem
            if member.name.endswith('.jpg') and key in names:
                image=Image.open(io.BytesIO(tar.extractfile(member).read())).convert('RGB')
                images[key]=np.asarray(image.resize((128,128),Image.Resampling.BILINEAR))
            if len(images)==len(names):break
    except (tarfile.ReadError,EOFError):pass
assert set(images)==set(names)
totals={k:np.zeros((2,2),np.int64) for k in ['Otsu','GrabCut']}
with tarfile.open(archives/'annotations-full.tar.gz') as tar:
    for n in names:
        rgb=images[n]; gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        _,bright=cv2.threshold(gray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        border=np.ones(gray.shape,bool);border[6:-6,6:-6]=False
        otsu=1-bright if bright[border].mean()>=0.5 else bright
        m=np.zeros(gray.shape,np.uint8);cv2.setRNGSeed(2026)
        cv2.grabCut(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),m,(6,6,116,116),np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_RECT)
        mask=Image.open(io.BytesIO(tar.extractfile('annotations/trimaps/'+n+'.png').read()))
        y=np.asarray(mask.resize((128,128),Image.Resampling.NEAREST))
        valid=y!=3
        for k,result in [('Otsu',otsu),('GrabCut',m%2)]:
            totals[k]+=np.bincount(2*(y[valid]==1).astype(int)+result[valid],minlength=4).reshape(2,2)
seg={}
for k,c in totals.items():
    assert c.sum()==838514
    iou=c.diagonal()/(c.sum(0)+c.sum(1)-c.diagonal())
    seg[k]=[float(c.trace()/c.sum()),float(iou[1]),float(iou.mean())]
classes='airplane automobile bird cat deer dog frog horse ship truck'.split()
y=np.array([classes.index(s.split('/')[1]) for s in json.loads((p/'classification_split.json').read_text())[2]])
r=json.loads((p/'seed2026/predictions.json').read_text())
base=np.array(r['Base'])==y
tests={}
for k in ('SE','CBAM','ECA'):
    new=np.array(r[k])==y
    b=int((base&~new).sum());c=int((~base&new).sum())
    tests[k]=[b,c,binomtest(c,b+c,.5).pvalue]
prev=0
for i,k in enumerate(sorted(tests,key=lambda k:tests[k][2])):
    prev=max(prev,min(1,(3-i)*tests[k][2]))
    tests[k].append(prev)
print(json.dumps(dict(seg=seg,tests=tests),indent=2))
