from pathlib import Path
import json
from ultralytics import YOLO
P=Path(__file__).resolve().parent
for seed in [2024,2025]:
 name='det_seed'+str(seed);cfg=str(P/'balanced_split/data.yaml')
 YOLO('yolov8n.pt').train(data=cfg,epochs=12,imgsz=320,batch=8,device=0,workers=0,amp=False,optimizer='AdamW',lr0=.001,seed=seed,project=str(P),name=name,exist_ok=True,plots=False)
 result=YOLO(str(P/name/'weights/best.pt')).val(data=cfg,split='test',device=0,imgsz=320,batch=8,workers=0,plots=False,project=str(P),name=name+'_test',exist_ok=True)
 (P/(name+'.json')).write_text(json.dumps(dict(metrics=result.results_dict,ap50=result.box.ap50.tolist()),indent=2))
