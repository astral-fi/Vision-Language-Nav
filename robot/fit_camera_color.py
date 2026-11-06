"""Estimate the magenta shading field from neutral surfaces in varied saved views.

This is an empirical neutral-surface correction, not a color-chart calibration.
It does not estimate geometric intrinsics or modify the robot camera stack.
"""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from camera_color import ColorCorrection, basis

root=Path(__file__).resolve().parent
paths=sorted((root/'calib/jetracer_audit').glob('2026*.jpg'))
samples=[]
for path in paths:
    f=cv2.imread(str(path)).astype(np.float32)+1
    h,w=f.shape[:2]
    b,g,r=cv2.split(f)
    gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    lr,lb=np.log(r/g),np.log(b/g)
    mask=(f.min(-1)>45)&(f.max(-1)<235)&(np.abs(np.log(r/b))<.15)&(lr>-.2)&(lr<.5)&(lb>-.2)&(lb<.5)&(gradient<25)
    features=basis(w,h)
    for yy in range(0,h,28):
        for xx in range(0,w,32):
            valid=mask[yy:yy+28,xx:xx+32]
            if valid.sum()<80: continue
            z=features[yy:yy+28,xx:xx+32][valid]
            samples.append((np.median(z,axis=0),np.median(lb[yy:yy+28,xx:xx+32][valid]),np.median(lr[yy:yy+28,xx:xx+32][valid])))
X=np.array([x[0] for x in samples]); Y=np.array([x[1:] for x in samples])
assert len(samples)>150,'Insufficient neutral surface coverage'
coeff=np.linalg.lstsq(X,Y,rcond=None)[0]
for _ in range(12):
    residual=Y-X@coeff
    weights=np.minimum(1,.035/np.maximum(np.abs(residual),1e-6))
    for c in range(2):
        s=np.sqrt(weights[:,c]); coeff[:,c]=np.linalg.lstsq(X*s[:,None],Y[:,c]*s,rcond=None)[0]
profile={'version':1,'method':'robust neutral-surface log-chromaticity field',
         'basis':['1','x','y','r2','r4'],'log_blue_over_green':coeff[:,0].tolist(),
         'log_red_over_green':coeff[:,1].tolist(),'width':w,'height':h,
         'neutral_cells':len(samples),'median_abs_fit_log_error':np.median(np.abs(Y-X@coeff),axis=0).tolist(),
         'source_images':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
         'limitation':'Estimated from assumed neutral room surfaces; no color chart or spectral response measurement.'}
out=root/'calib/jetracer_color.json';out.write_text(json.dumps(profile,indent=2))
correct=ColorCorrection(out)
for path in [paths[0],Path('/tmp/jetracer_calibration_position.jpg')]:
    f=cv2.imread(str(path))
    if f is None:continue
    pair=np.concatenate([f,correct(f)],axis=1)
    cv2.imwrite(str(root/'calib/jetracer_audit'/('color_'+path.name)),pair)
print(json.dumps({k:v for k,v in profile.items() if k!='source_images'},indent=2))
