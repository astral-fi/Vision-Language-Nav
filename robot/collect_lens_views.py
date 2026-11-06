"""Find diverse real checkerboard observations in existing bot1 recordings."""
import json
from pathlib import Path
import cv2
import numpy as np

root=Path(__file__).resolve().parent
out=root/'calib/jetracer_audit/lens_views';out.mkdir(parents=True,exist_ok=True)
pattern=(7,5)
views=[]
def detect(f):
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    g=cv2.resize(g,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    ok,c=cv2.findChessboardCornersSB(g,pattern,flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    return c.astype(np.float64)/2 if ok else None
for run in sorted((root/'runs').glob('*bot1*')):
    first=cv2.imread(str(run/'first.jpg'))
    if first is None or first.shape[1]!=640: continue
    c=detect(first)
    if c is None: continue
    print('Target present:',run.name,flush=True)
    cap=cv2.VideoCapture(str(run/'camera.mp4'))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for index in range(0,min(total,300),5):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,f=cap.read()
        if not ok: break
        c=detect(f)
        if c is None: continue
        # Reject repeated stationary observations, even across different runs.
        if any(np.sqrt(np.mean((c-np.array(v['corners']))**2))<1.5 for v in views):continue
        area=cv2.contourArea(cv2.convexHull(c.astype(np.float32)))
        if area<120:continue
        path=out/f'{run.name}_{index:05d}.png';cv2.imwrite(str(path),f)
        views.append({'path':str(path),'run':run.name,'frame':index,'corners':c.tolist(),'area_px':area})
        print('Kept',len(views),run.name,index,'area',round(area),flush=True)
        (out/'observations.json').write_text(json.dumps({'pattern':pattern,'width':640,'height':420,'views':views},indent=2))
        if len(views)>=40:break
    cap.release()
    if len(views)>=40:break
(out/'observations.json').write_text(json.dumps({'pattern':pattern,'width':640,'height':420,'views':views},indent=2))
print('TOTAL',len(views),flush=True)
