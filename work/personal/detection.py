from pathlib import Path
import io,zipfile,random,json,os
from ultralytics import YOLO
from PIL import Image
P=Path(__file__).resolve().parent; root=P/'balanced_split';z=zipfile.ZipFile('work/provenance/downloads/coco128.zip'); rows=[]
for name in sorted(z.namelist()):
 if '/labels/' not in name or not name.endswith('.txt'):continue
 boxes=[line.split() for line in z.read(name).decode().splitlines() if len(line.split())==5 and int(line.split()[0]) in [0,2,41]]
 if boxes and 'coco128/images/train2017/'+Path(name).stem+'.jpg' in z.namelist():rows.append((Path(name).stem,boxes))
rng=random.Random(2026)
def coverage(candidate):
 a=int(.65*len(candidate));b=int(.82*len(candidate))
 return min(sum(any(int(box[0])==c for box in boxes) for _,boxes in subset) for subset in (candidate[:a],candidate[a:b],candidate[b:]) for c in (0,2,41))
rows=max((rng.sample(rows,len(rows)) for _ in range(500)),key=coverage)
for i,(stem,boxes) in enumerate(rows):
 sp='train' if i<int(.65*len(rows)) else 'val' if i<int(.82*len(rows)) else 'test'
 for kind in ['images','labels']:(root/kind/sp).mkdir(parents=True,exist_ok=True)
 (root/'images'/sp/(stem+'.jpg')).write_bytes(z.read('coco128/images/train2017/'+stem+'.jpg'))
 (root/'labels'/sp/(stem+'.txt')).write_text('\n'.join(str([0,2,41].index(int(b[0])))+' '+' '.join(b[1:]) for b in boxes))
cfg=root/'data.yaml';cfg.write_text('path: '+root.as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [person, car, cup]\n')
if __name__=='__main__':
 model=YOLO('yolov8n.pt');model.train(data=str(cfg),epochs=12,imgsz=320,batch=8,device=0,workers=0,amp=False,optimizer='AdamW',lr0=.001,seed=2026,project=str(P),name='yolo_balanced',exist_ok=True,plots=False)
 best=YOLO(str(P/'yolo_balanced/weights/best.pt'));res=best.val(data=str(cfg),split='test',device=0,imgsz=320,batch=8,workers=0,plots=False,project=str(P),name='det_test',exist_ok=True)
 (P/'detection.json').write_text(json.dumps({**res.results_dict,'ap50':res.box.ap50.tolist(),'names':res.names,'split':[len(list((root/'images'/s).glob('*.jpg'))) for s in ['train','val','test']]},indent=2))
 for i,p in enumerate(sorted((root/'images/test').glob('*.jpg'))[:4]):
  pred=best.predict(str(p),imgsz=320,conf=.15,verbose=False)[0];Image.fromarray(pred.plot()[:,:,::-1]).save(P/f'prediction{i}.jpg')
