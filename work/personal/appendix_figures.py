from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
P=Path(__file__).resolve().parent
out=P/'expanded';out.mkdir(exist_ok=True)
def read(name):return json.loads((P/(name+'.json')).read_text())
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=190,bbox_inches='tight');plt.close(fig)
rows=read('image_series')
fig,axs=plt.subplots(1,2,figsize=(8.3,3.1))
for ax,noise,method,title in zip(axs,['Gaussian','Impulse'],['Gaussian','Median'],['高斯噪声 · 高斯滤波','椒盐噪声 · 中值滤波']):
    for im in ['coffee','astronaut','chelsea','rocket']:
        selected=sorted([r for r in rows if (r['image'],r['noise'],r['method'])==(im,noise,method)],key=lambda r:r['k'])
        ax.plot([r['k'] for r in selected],[r['psnr'] for r in selected],marker='o',label=im)
    ax.set(xticks=[3,5,7],xlabel='窗口边长（像素）',ylabel='PSNR（dB）',title=title);ax.grid(axis='y',alpha=.2)
axs[1].legend(fontsize=8,ncol=2);save(fig,'window')
r=next(r for r in read('det_examples')['rows'] if r['id']=='000000000110')
gt=[g[1:] for g in r['gt'] if g[0]==2];pr=[b[1:] for b in r['pred'] if b[0]==2]
im=Image.open(P/'balanced_split/images/test/000000000110.jpg')
fig,ax=plt.subplots(figsize=(8,2.7));ax.imshow(im);ax.set_xlim(142,458);ax.set_ylim(125,43);ax.axis('off')
for col,boxes in [('#13d8bd',gt),('#ff973a',pr)]:
    for i,(x,y,xx,yy,*_) in enumerate(boxes):
        ax.add_patch(Rectangle((x,y),xx-x,yy-y,fill=False,edgecolor=col,lw=1.6))
        if _:ax.text(x,y-2-(i%2)*6,f'P{i+1}',color='black',fontsize=8,bbox=dict(facecolor=col,pad=1,edgecolor='none'))
save(fig,'cups')
def iou(a,b):
    x,y,xx,yy=a[:4];l,t,r,bottom=b
    inter=max(0,min(xx,r)-max(x,l))*max(0,min(yy,bottom)-max(y,t))
    return inter/((xx-x)*(yy-y)+(r-l)*(bottom-t)-inter)
vals=[max(iou(a,b) for b in gt) for a in pr]
classes='airplane automobile bird cat deer dog frog horse ship truck'.split()
cn='飞机 汽车 鸟 猫 鹿 狗 蛙 马 船 卡车'.split()
labels=np.array([classes.index(s.split('/')[1]) for s in read('classification_split')[2]])
preds=read('seed2026/predictions');ds=np.array(preds['DS']);kd=np.array(preds['KD'])
gain=[int(((labels==i)&(ds!=i)&(kd==i)).sum()) for i in range(10)]
loss=[int(((labels==i)&(ds==i)&(kd!=i)).sum()) for i in range(10)]
fig,ax=plt.subplots(figsize=(8,2.9));x=np.arange(10)
ax.bar(x-.18,gain,.36,label='蒸馏后改对',color='#197674');ax.bar(x+.18,loss,.36,label='蒸馏后改错',color='#b96132')
ax.set(xticks=x,xticklabels=cn,ylabel='图像数（张）');ax.legend(ncol=2);save(fig,'paired')
(out/'derived.json').write_text(json.dumps(dict(iou=vals,gain=gain,loss=loss,confidence=[a[-1] for a in pr])))
