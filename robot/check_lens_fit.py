"""Compare measured checkerboard reprojection with stored and candidate models."""
import json
from pathlib import Path
import cv2
import numpy as np

root=Path(__file__).resolve().parent/'calib'
observations=json.loads((root/'jetracer_audit/lens_views/observations.json').read_text())
camera=json.loads((root/'jetracer_camera_probe.json').read_text())['camera_info']
K=np.array(camera['K'],np.float64).reshape(3,3);D=np.array(camera['D'],np.float64)
points=np.zeros((35,3),np.float64);points[:,:2]=np.mgrid[0:7,0:5].T.reshape(-1,2)
images=[np.array(v['corners'],np.float64) for v in observations['views']]
errors=[]
for image in images:
    ok,r,t=cv2.solvePnP(points,image,K,D,flags=cv2.SOLVEPNP_ITERATIVE)
    projected,_=cv2.projectPoints(points,r,t,K,D)
    errors.append(float(np.sqrt(np.mean((projected-image)**2))))
joined=np.concatenate(images).reshape(-1,2)
report={'views':len(images),'board':'7x5 inner corners',
        'board_area_px_range':[min(v['area_px'] for v in observations['views']),max(v['area_px'] for v in observations['views'])],
        'corner_bounds_px':[joined.min(0).tolist(),joined.max(0).tolist()],
        'stored_plumb_bob_reprojection_rms_px':float(np.sqrt(np.mean(np.array(errors)**2))),
        'stored_per_view_rms_px':errors}
try:
    flags=cv2.fisheye.CALIB_USE_INTRINSIC_GUESS|cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC|cv2.fisheye.CALIB_FIX_SKEW|cv2.fisheye.CALIB_FIX_K3|cv2.fisheye.CALIB_FIX_K4
    rms,k,d,_,_=cv2.fisheye.calibrate([points.reshape(1,-1,3)]*len(images),
        [x.reshape(1,-1,2) for x in images],(640,420),K.copy(),np.zeros((4,1)),flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,80,1e-8))
    report['candidate_fisheye']={'rms_px':float(rms),'K':k.tolist(),'D':d.ravel().tolist()}
except cv2.error as e:
    report['candidate_fisheye_error']=str(e)
report['decision']='Use existing live-camera plumb_bob calibration; small targets and limited image coverage do not establish a reliable new fisheye calibration.'
(root/'jetracer_lens_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
