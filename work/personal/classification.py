from pathlib import Path
import io, tarfile, json, random, copy, csv
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
rng=random.Random(2026)
classes='airplane automobile bird cat deer dog frog horse ship truck'.split()
archive='work/provenance/downloads/cifar-fastai.tgz'
groups={}
with tarfile.open(archive,'r|gz') as t:
 for m in t:
  if m.isfile() and m.name.endswith('.png'):
   _,split,cls,file=m.name.split('/')
   groups.setdefault((split,cls),[]).append((file,t.extractfile(m).read()))
print([(k,len(v)) for k,v in groups.items()],flush=True)
sets=[[],[],[]]
for (sp,cls),items in sorted(groups.items()):
 rng.shuffle(items)
 for i,(file,raw) in enumerate(items[:1200 if sp=='train' else 200]):
  k=(0 if i<1000 else 1) if sp=='train' else 2
  sets[k].append((np.array(Image.open(io.BytesIO(raw)).convert('RGB')),classes.index(cls),f'{sp}/{cls}/{file}'))
(P/'classification_split.json').write_text(json.dumps([[r[2] for r in s] for s in sets]))
def tensor(s):
 x=torch.tensor(np.stack([r[0] for r in s]),device='cuda').permute(0,3,1,2).float()/255
 return (x-.5)/.5,torch.tensor([r[1] for r in s],device='cuda')
sets=list(map(tensor,sets))
class Attention(nn.Module):
 def __init__(self,c,kind):
  super().__init__()
  self.kind=kind
  if kind=='ECA':self.g=nn.Conv1d(1,1,3,padding=1,bias=False)
  else:self.g=nn.Sequential(nn.Conv2d(c,max(1,c//16),1),nn.ReLU(),nn.Conv2d(max(1,c//16),c,1))
  if kind=='CBAM':self.s=nn.Conv2d(2,1,7,padding=3,bias=False)
 def forward(self,x):
  avg=x.mean((2,3),keepdim=True)
  if self.kind=='ECA':a=self.g(avg.flatten(2).transpose(1,2)).transpose(1,2).unsqueeze(-1)
  else:
   a=self.g(avg)
   if self.kind=='CBAM':a=a+self.g(x.amax((2,3),keepdim=True))
  x=x*a.sigmoid()
  if self.kind=='CBAM':x=x*self.s(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1)).sigmoid()
  return x
def model(kind):
 if kind in ['STD','DS','KD']:
  layers=[];c=3
  for o in [32,64,128]:
   if kind=='STD':layers.append(nn.Conv2d(c,o,3,padding=1,bias=False))
   else:layers.extend([nn.Conv2d(c,c,3,padding=1,groups=c,bias=False),nn.Conv2d(c,o,1,bias=False)])
   layers.extend([nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2)]);c=o
  return nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,10))
 m=resnet18(num_classes=10);m.conv1=nn.Conv2d(3,64,3,padding=1,bias=False);m.maxpool=nn.Identity()
 if kind!='Base':
  for stage in [m.layer1,m.layer2,m.layer3,m.layer4]:
   for block in stage:block.bn2=nn.Sequential(block.bn2,Attention(block.conv2.out_channels,kind))
 return m
def complexity(m):
 count=[];hooks=[]
 def op(l,i,o):count.append(o.numel()*l.weight[0].numel())
 for l in m.modules():
  if isinstance(l,(nn.Conv1d,nn.Conv2d,nn.Linear)):hooks.append(l.register_forward_hook(op))
 m.eval()
 with torch.no_grad():m(sets[0][0][:1])
 for h in hooks:h.remove()
 return sum(count)*2
def infer(m,x):
 return torch.cat([m(z) for z in x.split(128)])
results={}; allpred={};torch.set_num_threads(2)
import os,time,statistics
seed=int(os.environ.get('CV_SEED','2026'))
target=P/f'seed{seed}';target.mkdir(exist_ok=True)
kinds=['Base','SE','CBAM','ECA','STD','DS','KD']
if seed==2026:kinds.extend(['T1','T2','T6','T8','T10','A01','A03','A07','A09'])
for kind in kinds:
 distill=kind=='KD' or kind.startswith(('T','A'))
 temp=int(kind[1:]) if kind.startswith('T') else 4
 alpha=int(kind[1:])/10 if kind.startswith('A') else .5
 torch.manual_seed(seed);m=model('KD' if distill else kind).cuda();flops=complexity(m);opt=torch.optim.AdamW(m.parameters(),lr=.001);best=-1;curve=[]
 gen=torch.Generator(device='cuda').manual_seed(seed)
 x,y=sets[0]
 for ep in range(6):
  m.train()
  for idx in torch.randperm(len(y),device='cuda',generator=gen).split(128):
   z=x[idx];out=m(z);loss=F.cross_entropy(out,y[idx])
   if distill:
    with torch.no_grad():soft=F.softmax(teacher(z)/temp,1)
    loss=(1-alpha)*loss+alpha*temp**2*F.kl_div(F.log_softmax(out/temp,1),soft,reduction='batchmean')
   opt.zero_grad();loss.backward();opt.step()
  m.eval()
  with torch.no_grad():n=int((infer(m,sets[1][0]).argmax(1)==sets[1][1]).sum())
  curve.append(n)
  if n>best:best=n;state=copy.deepcopy(m.state_dict())
 m.load_state_dict(state);m.eval()
 if kind=='Base':teacher=m
 with torch.no_grad():
  logits=infer(m,sets[2][0]);pred=logits.argmax(1)
  top5=int((logits.topk(5,1).indices==sets[2][1][:,None]).any(1).sum())
  single=sets[2][0][:1]
  for _ in range(20):m(single)
  timing=[]
  for _ in range(80):
   torch.cuda.synchronize();start=time.perf_counter_ns();m(single);torch.cuda.synchronize();timing.append((time.perf_counter_ns()-start)/1e6)
 latency=statistics.median(timing)
 if kind in ['Base','DS','KD'] and seed==2026:torch.save(state,target/(kind+'.pt'))
 correct=int((pred==sets[2][1]).sum());allpred[kind]=pred.cpu().tolist()
 results[kind]=dict(top5=top5,latency_ms=latency,parameters=sum(p.numel() for p in m.parameters()),flops=flops,validation=curve,test_correct=correct,test_n=len(pred),best_epoch=curve.index(best)+1)
 if kind in ['Base','SE','CBAM','ECA']:
  # Grad-CAM for ground truth at the last residual stage.
  z=sets[2][0][::200][:4];feat={}
  def record(mod,inp,out):out.retain_grad();feat['map']=out
  hook=m.layer4.register_forward_hook(record);out=m(z);out.gather(1,sets[2][1][::200][:4,None]).sum().backward();hook.remove()
  a=feat['map'];cam=(a.grad.mean((2,3),keepdim=True)*a).sum(1).relu().detach().cpu()
  fig,axes=plt.subplots(1,4,figsize=(8,2.4))
  for i,ax in enumerate(axes):
   ax.imshow(((z[i].cpu().permute(1,2,0)+1)/2).numpy());ax.imshow(cam[i],cmap='inferno',alpha=.5,extent=(-.5,31.5,31.5,-.5),interpolation='bilinear');ax.axis('off');ax.set_title(classes[int(out[i].argmax())],fontsize=9)
  fig.tight_layout();fig.savefig(target/(kind+'_cam.png'),dpi=145);plt.close(fig)
 (target/'classification.json').write_text(json.dumps(results,indent=2));print(kind,results[kind],flush=True)
(target/'predictions.json').write_text(json.dumps(allpred))
for k,r in results.items():plt.plot(range(1,7),np.array(r['validation'])/20,marker='.',label=k)
plt.ylabel('Validation accuracy (%)');plt.xlabel('Epoch');plt.legend(ncol=4);plt.tight_layout();plt.savefig(target/'classification_curve.png',dpi=180)
