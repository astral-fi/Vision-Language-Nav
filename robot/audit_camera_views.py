"""Collect representative saved JetRacer views for calibration-target inspection."""
from pathlib import Path
import cv2
import numpy as np

root=Path(__file__).resolve().parent
out=root/'calib/jetracer_audit'
out.mkdir(parents=True,exist_ok=True)
views=[]
for name in ['20261020T124752Z_bot1_19f056','20261020T125821Z_bot1_d0067e']:
    cap=cv2.VideoCapture(str(root/'runs'/name/'camera.mp4'))
    length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(4):
        index=int(length*i/4)
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,f=cap.read()
        if not ok:continue
        path=out/f'{name}_{index:05d}.jpg'
        cv2.imwrite(str(path),f)
        thumb=cv2.resize(f,(640,420))
        cv2.putText(thumb,f'{name[9:15]} frame {index}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),1)
        views.append(thumb)
    cap.release()
if views:
    cv2.imwrite(str(out/'contact.jpg'),np.concatenate([np.concatenate(views[i:i+2],axis=1) for i in range(0,len(views),2)],axis=0))
print(out)
