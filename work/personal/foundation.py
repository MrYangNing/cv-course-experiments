from pathlib import Path
import cv2 as cv,numpy as np,json
from skimage import data
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
cv.setNumThreads(1);cv.setRNGSeed(2026);rng=np.random.default_rng(2026)
rgb=data.coffee();g=cv.cvtColor(rgb,cv.COLOR_RGB2GRAY);hsv=cv.cvtColor(rgb,cv.COLOR_RGB2HSV)
def plate(name,items,n):
 fig,axes=plt.subplots((len(items)+n-1)//n,n,figsize=((8,6) if name=='matches' else (3*n,2.4*((len(items)+n-1)//n))),squeeze=False)
 for ax in axes.flat:ax.axis('off')
 for ax,(title,im) in zip(axes.flat,items):ax.imshow(im,cmap='gray',vmin=0,vmax=255);ax.set_title(title,fontsize=10)
 fig.tight_layout();fig.savefig(P/(name+'.png'),dpi=165);plt.close(fig)
plate('channels',list(zip(['RGB','Red','Green','Blue','Luminance','Hue (0-179)','Saturation','Value'],[rgb,*cv.split(rgb),g,*cv.split(hsv)])),4)
normal=np.uint8(np.clip(g.astype(float)+rng.normal(0,25,g.shape),0,255));salt=g.copy();idx=rng.choice(g.size,g.size//20,replace=False);salt.flat[idx[:len(idx)//2]]=0;salt.flat[idx[len(idx)//2:]]=255
rows=[];panels=[]
for label,noisy in [('Gaussian',normal),('Impulse',salt)]:
 for k in [3,5,7]:
  methods=[('Noisy',noisy),('Mean',cv.blur(noisy,(k,k))),('Gaussian',cv.GaussianBlur(noisy,(k,k),0)),('Median',cv.medianBlur(noisy,k))]
  for method,z in methods:
   if method=='Noisy' and k!=5:continue
   rows.append([label,k,method,cv.PSNR(g,z),ssim(g,z,data_range=255)])
   if k==5:panels.append((label+' / '+method,z))
plate('filters',panels,4)
dx=cv.Sobel(g,cv.CV_64F,1,0);dy=cv.Sobel(g,cv.CV_64F,0,1);lap=np.abs(cv.Laplacian(g,cv.CV_64F,ksize=3));eq=cv.equalizeHist(g)
norm=lambda x:cv.normalize(x,None,0,255,cv.NORM_MINMAX)
edges=[cv.Canny(g,a,b) for a,b in [(30,90),(60,180),(90,270)]]
plate('edges',[('Luminance',g),('Sobel magnitude',norm(np.hypot(dx,dy))),('Absolute Laplacian',norm(lap)),*zip(['Canny 30/90','Canny 60/180','Canny 90/270'],edges)],3)
fig,ax=plt.subplots(2,2,figsize=(8,5))
for i,(s,img) in enumerate([('Original',g),('Equalized',eq)]):
 ax[0,i].imshow(img,cmap='gray',vmin=0,vmax=255);ax[0,i].axis('off');ax[0,i].set_title(s);ax[1,i].plot(np.bincount(img.ravel(),minlength=256));ax[1,i].set(xlabel='Intensity',ylabel='Pixels')
fig.tight_layout();fig.savefig(P/'hist.png',dpi=155)
h=cv.cornerHarris(g.astype('float32'),2,3,.04);points=(h==cv.dilate(h,None))&(h>.01*h.max());marked=rgb.copy()
for y,x in zip(*np.where(points)):cv.circle(marked,(int(x),int(y)),3,(0,255,180),1)
matches=[];vis=[('Harris maxima',marked)];lines=[]
for name,det,normtype in [('SIFT',cv.SIFT_create(),cv.NORM_L2),('ORB',cv.ORB_create(nfeatures=1000),cv.NORM_HAMMING)]:
 k,d=det.detectAndCompute(g,None);vis.append((name,cv.drawKeypoints(rgb,k,None,color=(0,255,180))))
 for angle in [0,15,35]:
  M=cv.getRotationMatrix2D((g.shape[1]/2,g.shape[0]/2),angle,1);M[:,2]+=[20,10]
  w=cv.warpAffine(g,M,(g.shape[1],g.shape[0]));q,f=det.detectAndCompute(w,None)
  good=[a for a,b in cv.BFMatcher(normtype).knnMatch(d,f,k=2) if a.distance<.75*b.distance]
  src=np.float64([k[a.queryIdx].pt for a in good]);dst=np.float64([q[a.trainIdx].pt for a in good]);err=np.linalg.norm(src@M[:,:2].T+M[:,2]-dst,axis=1)
  matches.append([name,angle,len(k),len(q),len(good),int((err<=3).sum()),float(np.median(err))])
  if angle==35:lines.append((name+' / 35 deg',cv.drawMatches(g,k,w,q,sorted(good,key=lambda a:a.distance)[:40],None,matchColor=(25,190,120),flags=2)))
plate('keypoints',vis,3);plate('matches',lines,1)
(P/'foundation.json').write_text(json.dumps(dict(shape=rgb.shape,filters=rows,edges=[int(np.count_nonzero(a)) for a in edges],std=[float(np.std(a)) for a in [g,eq]],levels=[len(np.unique(a)) for a in [g,eq]],harris=int(points.sum()),matches=matches),indent=2))
