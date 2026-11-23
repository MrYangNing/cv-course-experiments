from pathlib import Path
import io,tarfile,random,json,copy
import torch,numpy as np
from torch import nn
from torch.nn import functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
D=Path('work/provenance/downloads')
imgs={}
with tarfile.open(D/'images.tar.gz.incomplete','r|gz') as t:
 try:
  for m in t:
   if m.name.endswith('.jpg') and not Path(m.name).name.startswith('.'):
    imgs[Path(m.name).stem]=np.array(Image.open(io.BytesIO(t.extractfile(m).read())).convert('RGB').resize((128,128),Image.Resampling.BILINEAR))
 except (EOFError,tarfile.ReadError):pass
with tarfile.open(D/'annotations-full.tar.gz') as t:
 def ids(s):return set(line.split()[0] for line in t.extractfile('annotations/'+s+'.txt').read().decode().splitlines() if line.strip())
 tr=sorted(ids('trainval')&imgs.keys());te=sorted(ids('test')&imgs.keys())
 masks={i:np.array(Image.open(io.BytesIO(t.extractfile('annotations/trimaps/'+i+'.png').read())).resize((128,128),Image.Resampling.NEAREST)) for i in tr+te}
random.Random(2026).shuffle(tr);n=max(1,len(tr)//5);splits=[tr[n:],tr[:n],te]
(P/'seg_split.json').write_text(json.dumps(splits))
def dataset(names):
 x=torch.tensor(np.stack([imgs[i] for i in names]),device='cuda').permute(0,3,1,2).float()/255
 y=torch.tensor(np.stack([masks[i] for i in names]),device='cuda').long()
 return x,torch.where(y==3,255,(y==1).long())
data=list(map(dataset,splits))
def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU())
class Net(nn.Module):
 def __init__(self,skip):
  super().__init__();self.skip=skip;self.a=block(3,16);self.b=block(16,32);self.c=block(32,64);self.d=block(64+32*skip,32);self.e=block(32+16*skip,16);self.f=nn.Conv2d(16,2,1)
 def forward(self,x):
  a=self.a(x);b=self.b(F.max_pool2d(a,2));c=self.c(F.max_pool2d(b,2))
  def up(x,side):
   z=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
   return torch.cat([z,side],1) if self.skip else z
  return self.f(self.e(up(self.d(up(c,b)),a)))
def scores(p,y):
 v=y!=255;cm=torch.bincount(2*y[v]+p[v],minlength=4).reshape(2,2).cpu().numpy();d=cm.diagonal();iou=d/(cm.sum(1)+cm.sum(0)-d)
 return dict(cm=cm.tolist(),iou=iou.tolist(),miou=float(iou.mean()),accuracy=float(d.sum()/cm.sum()),correct=int(d.sum()),pixels=int(cm.sum()))
results={};preds={}
for name,skip,dice in [('EncoderDecoder',False,False),('UNet',True,False),('UNetDice',True,True)]:
 torch.manual_seed(2026);m=Net(skip).cuda();opt=torch.optim.Adam(m.parameters(),lr=.001);curve=[];best=-1
 x,y=data[0]
 for ep in range(25):
  m.train()
  for idx in torch.randperm(len(x)).split(8):
   o=m(x[idx]);target=y[idx];loss=F.cross_entropy(o,target,ignore_index=255)
   if dice:
    a=o.softmax(1)[:,1]*(target!=255);b=target==1;loss=loss+1-(2*(a*b).sum()+1)/(a.sum()+b.sum()+1)
   opt.zero_grad();loss.backward();opt.step()
  m.eval()
  with torch.no_grad():v=scores(m(data[1][0]).argmax(1),data[1][1])['miou']
  curve.append(v)
  if v>best:best=v;state=copy.deepcopy(m.state_dict())
 m.load_state_dict(state)
 with torch.no_grad():pred=torch.cat([m(z).argmax(1) for z in data[2][0].split(16)])
 preds[name]=pred;results[name]=dict(**scores(pred,data[2][1]),epoch=curve.index(best)+1,curve=curve,params=sum(p.numel() for p in m.parameters()))
 print(name,results[name],flush=True)
(P/'segmentation.json').write_text(json.dumps(results,indent=2))
bestname=max(results,key=lambda k:max(results[k]['curve']));pred=preds[bestname];y=data[2][1]
order=sorted(range(len(te)),key=lambda i:scores(pred[i],y[i])['miou'])
fig,axes=plt.subplots(3,4,figsize=(9,7))
examples=[]
for row,i in enumerate([order[0],order[len(te)//2],order[-1]]):
 examples.append([te[i],scores(pred[i],y[i])])
 for ax,im,title in zip(axes[row],[imgs[te[i]],np.where(masks[te[i]]==3,.5,masks[te[i]]==1),pred[i].cpu(),((pred[i]!=y[i])&(y[i]!=255)).cpu()],['Input','Ground truth','Prediction','Error']):
  ax.imshow(im,cmap='gray',vmin=0,vmax=1 if title!='Input' else 255);ax.set_title(title if row==0 else '',fontsize=10);ax.axis('off')
fig.tight_layout();fig.savefig(P/'seg_examples.png',dpi=160)
(P/'seg_examples.json').write_text(json.dumps(dict(model=bestname,examples=examples)))
for k,v in results.items():plt.figure(2);plt.plot(range(1,26),v['curve'],label=k)
plt.xlabel('Epoch');plt.ylabel('Validation mIoU');plt.legend();plt.tight_layout();plt.savefig(P/'seg_curve.png',dpi=165)
