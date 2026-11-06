"""Look for actual checkerboard observations; never synthesize lens coefficients."""
from pathlib import Path
import json
import cv2

root=Path(__file__).resolve().parent/'calib/jetracer_audit'
results=[]
for path in sorted(root.glob('2026*.jpg'))[:4]:
    f=cv2.imread(str(path));gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    # Wall board in the initial room view; retain source pixel coordinates.
    if path.name.endswith('_00000.jpg'):
        crop=f[125:190,380:445]
        cv2.imwrite(str(root/'wall_target_zoom.png'),cv2.resize(crop,None,fx=8,fy=8,interpolation=cv2.INTER_NEAREST))
    scaled=cv2.resize(gray,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    for pattern in [(7,7),(7,5),(6,4),(9,6)]:
        ok,corners=cv2.findChessboardCornersSB(scaled,pattern,flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners=corners/2
            results.append({'path':str(path),'pattern':pattern,'corners':corners.tolist()})
            print(path.name,pattern,'FOUND',flush=True)
            break
    else:
        print(path.name,'no checkerboard detected',flush=True)
(root/'target_detections.json').write_text(json.dumps(results,indent=2))
print('usable_views',len(results),'aruco_available',hasattr(cv2,'aruco'))
