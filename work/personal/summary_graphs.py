from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent
r=json.loads((P/'seed2026/classification.json').read_text());keys=['Base','SE','CBAM','ECA','STD','DS','KD']
fig,ax=plt.subplots(1,2,figsize=(9,3.4),sharey=True)
for a,ks,title in zip(ax,[keys[:4],keys[4:]],['Residual networks','Compact networks']):
 for k in ks:a.plot(range(1,7),np.array(r[k]['validation'])/20,marker='o',markersize=3,label=k)
 a.set(xlabel='Epoch',title=title,xticks=range(1,7));a.legend(fontsize=8);a.grid(alpha=.18)
ax[0].set_ylabel('Validation accuracy (%)');fig.tight_layout();fig.savefig(P/'curve_main.png',dpi=175)
fig,ax=plt.subplots(figsize=(10,3.4));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
lines=[['SE','Global mean','FC → ReLU → FC','Channel gate','Scale X'],['CBAM','Mean & max','Shared MLP','Channel gate','7×7 spatial gate'],['ECA','Global mean','1D conv (k=3)','Channel gate','Scale X']]
for i,line in enumerate(lines):
 y=3-i*1.15
 for j,txt in enumerate(line):
  x=j*2+.05;ax.add_patch(FancyBboxPatch((x,y-.3),1.64,.6,boxstyle='round,pad=0.06',facecolor='#dceaf0' if j else '#285671',edgecolor='white'));ax.text(x+.82,y,txt,ha='center',va='center',fontsize=9,color='black' if j else 'white')
  if j<4:ax.annotate('',xy=(x+1.94,y),xytext=(x+1.7,y),arrowprops=dict(arrowstyle='->',color='#47636f'))
fig.tight_layout();fig.savefig(P/'modules.png',dpi=165,bbox_inches='tight')
