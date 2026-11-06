"""Finalize the validated targetless lens model and inspect its usable view."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from camera_color import ColorCorrection

root=Path(__file__).resolve().parent
fit=json.loads((root/'calib/jetracer_targetless_validation.json').read_text())
assert fit['selected']=='fisheye'
chosen=fit['candidates']['fisheye']
assert chosen['minimum_radial_derivative']>.05
assert chosen['validation_rms_px']<=fit['stored_validation_rms_px']
K=np.array(chosen['K']);D=np.array(chosen['D'])
validities={}
for hfov in np.arange(79,49,-.5):
    f=320/np.tan(np.deg2rad(hfov/2))
    target=np.array([[f,0,320],[0,f,240],[0,0,1.]])
    mx,my=cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),target,(640,480),cv2.CV_32FC1)
    fraction=float(((mx>=1)&(mx<638)&(my>=1)&(my<418)).mean())
    validities[str(float(hfov))]=fraction
    if fraction>=.995:break
profile={'K':K.tolist(),'D':D.tolist(),'width':640,'height':420,
         'distortion_model':'opencv_fisheye','virtual_hfov_deg':float(hfov),
         'target_size':[640,480],'valid_source_fraction':fraction,
         'method':fit['method'],'validation_report':'jetracer_targetless_validation.json',
         'raw_edge_rms_px':fit['raw_validation_rms_px'],
         'validation_edge_rms_px':chosen['validation_rms_px'],
         'limitation':fit['limitation']}
(root/'calib/jetracer_fisheye.json').write_text(json.dumps(profile,indent=2))
source=cv2.imread('/tmp/jetracer_calibration_position.jpg')
color=ColorCorrection(root/'calib/jetracer_color.json')
corrected=color(source)
rectified=cv2.remap(corrected,mx,my,cv2.INTER_LINEAR)
cv2.imwrite(str(root/'calib/jetracer_audit/final_color.jpg'),corrected)
cv2.imwrite(str(root/'calib/jetracer_audit/final_model_view.jpg'),rectified)
cv2.imwrite(str(root/'calib/jetracer_audit/final_before_after.jpg'),np.concatenate([source,corrected],axis=1))
times=[]
for _ in range(100):
    started=time.perf_counter();cv2.remap(color(source),mx,my,cv2.INTER_LINEAR)
    times.append((time.perf_counter()-started)*1000)
report={'virtual_hfov_deg':float(hfov),'valid_source_fraction':fraction,
        'valid_source_fraction_at_79deg':validities['79.0'],
        'median_processing_ms':float(np.median(times)),'p95_processing_ms':float(np.percentile(times,95))}
(root/'calib/jetracer_camera_processing_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
