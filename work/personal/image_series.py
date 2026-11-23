from pathlib import Path
import json
from skimage import data
from skimage.metrics import structural_similarity as ssim
import cv2 as cv,numpy as np
P=Path(__file__).resolve().parent
rows=[]
for name,rgb in [('coffee',data.coffee()),('astronaut',data.astronaut()),('chelsea',data.chelsea()),('rocket',data.rocket())]:
 g=cv.cvtColor(rgb,cv.COLOR_RGB2GRAY);rng=np.random.default_rng(2026)
 z=np.uint8(np.clip(g.astype(float)+rng.normal(0,25,g.shape),0,255));s=g.copy();idx=rng.choice(g.size,g.size//20,replace=False);s.flat[idx[:len(idx)//2]]=0;s.flat[idx[len(idx)//2:]]=255
 for n,img in [('Gaussian',z),('Impulse',s)]:
  for k in (3,5,7):
   for method,out in [('Mean',cv.blur(img,(k,k))),('Gaussian',cv.GaussianBlur(img,(k,k),0)),('Median',cv.medianBlur(img,k))]:
    rows.append(dict(image=name,noise=n,k=k,method=method,psnr=cv.PSNR(g,out),ssim=ssim(g,out,data_range=255)))
(P/'image_series.json').write_text(json.dumps(rows,indent=2))
