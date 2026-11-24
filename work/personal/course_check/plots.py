from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Patch
P=Path(__file__).parent
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False})
def save(fig,n):
    fig.savefig(P/(n+'.png'),dpi=190,bbox_inches='tight',pad_inches=.09);plt.close(fig)
def diagram():
    fig,ax=plt.subplots(figsize=(9.2,3.5));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    return fig,ax
def box(ax,x,y,txt,color,w=2.25,h=.65):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.04',fc=color,ec='#466071'));ax.text(x,y,txt,ha='center',va='center',fontsize=10)
def arrow(ax,a,b,color='#4a5d68'):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=color,lw=1.8))
fig,ax=diagram()
for x,y,t,col in [(1.8,3.3,'编码 1　16 通道\n128 × 128','#dceaf5'),(1.8,1.95,'编码 2　32 通道\n64 × 64','#dceaf5'),(5,.68,'瓶颈　64 通道\n32 × 32','#e1edf0'),(8.2,1.95,'解码 1　32 通道\n64 × 64','#f8e5cd'),(8.2,3.3,'解码 2　16 通道\n128 × 128','#f8e5cd')]:box(ax,x,y,t,col)
arrow(ax,(1.8,2.94),(1.8,2.32));arrow(ax,(2.65,1.57),(4.1,1.04));arrow(ax,(5.91,1.04),(7.32,1.57));arrow(ax,(8.2,2.32),(8.2,2.94))
for y in [1.95,3.3]:
    arrow(ax,(3,y),(7,y),'#197056');ax.text(5,y+.12,'同尺度特征 → 通道拼接',ha='center',color='#197056')
ax.text(.52,2.63,'池化',fontsize=9);ax.text(8.65,2.6,'上采样',fontsize=9)
save(fig,'unet')
fig,ax=diagram()
for x,t,c in [(1,'输入 x','#e3edf4'),(3,'卷积组¹','#e3edf4'),(5.1,'注意力门控','#cde9e0'),(7.3,'逐元素加法','#f8e5cd'),(9.15,'ReLU','#e3edf4')]:
    box(ax,x,1.6,t,c,1.5)
for x,y in [(1.8,2.2),(3.8,4.3),(5.9,6.5),(8.1,8.35)]:arrow(ax,(x,1.6),(y,1.6))
ax.plot([1,1,7.3],[1.98,3,3],color='#47748c',lw=1.6);arrow(ax,(7.3,3.04),(7.3,1.98),'#47748c')
ax.text(4.3,3.18,'捷径：维度相同直接传递，维度改变时投影',ha='center')
ax.text(5,.63,'¹  3×3 卷积 → BN → ReLU → 3×3 卷积 → BN',ha='center')
save(fig,'residual')
dat=np.load(P/'examples.npz');truth,pred,ids=dat['truth'],dat['pred'].astype(bool),dat['ids'];cms=[]
for a,b in zip(truth,pred):
    v=a!=3;cms.append(np.bincount(2*(a[v]==1).astype(int)+b[v],minlength=4).reshape(2,2))
cms=np.stack(cms);fg=cms[:,1,1]/(cms[:,1,1]+cms[:,0,1]+cms[:,1,0]);rank=np.argsort(fg)
fig,axs=plt.subplots(3,4,figsize=(8.6,6.5))
green=[35,158,110];red=[215,49,72];orange=[247,163,29];gray=[165]*3;blue=[32,55,84]
examples=[]
for j,i in enumerate([rank[0],np.argmax(cms[:,0,1]),rank[-1]]):
    gt=truth[i];pr=pred[i];rgb=dat['rgb'][i];v=gt!=3
    label=np.zeros_like(rgb);label[:]=blue;label[gt==1]=green;label[~v]=gray
    prediction=np.zeros_like(rgb);prediction[:]=blue;prediction[pr]=green
    over=rgb.copy()
    for cond,col in [(v&pr&(gt==1),green),(v&~pr&(gt==1),red),(v&pr&(gt==2),orange),(~v,gray)]:
        over[cond]=.35*rgb[cond]+.65*np.array(col)
    for ax,im,t in zip(axs[j],[rgb,label,prediction,over],['原图','人工标注','预测掩码','逐像素叠加']):
        ax.imshow(im,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_visible(False)
        if j==0:ax.set_title(t)
    axs[j,0].set_ylabel(f'{ids[i]}\n前景 IoU {fg[i]:.3f}',fontsize=9)
    examples.append(dict(id=str(ids[i]),iou=float(fg[i]),cm=cms[i].tolist()))
fig.legend(handles=[Patch(color=np.array(col)/255,label=t) for col,t in [(green,'命中 TP'),(red,'漏检 FN'),(orange,'误检 FP'),(gray,'忽略边界')]],loc='lower center',ncol=4,frameon=False)
fig.tight_layout(rect=(0,.065,1,1));save(fig,'error_gallery')
(P/'analysis.json').write_text(json.dumps({'cm':cms.sum(axis=0).tolist(),'examples':examples},indent=2))

