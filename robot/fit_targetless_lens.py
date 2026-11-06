"""Refine lens distortion from observed room edges, with separate validation views.

Existing CameraInfo anchors focal length and principal point. Long architectural
edges supply independent straightness constraints; no calibration board is used.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

root=Path(__file__).resolve().parent/'calib'
camera=json.loads((root/'jetracer_camera_probe.json').read_text())['camera_info']
K=np.array(camera['K'],float).reshape(3,3);D=np.array(camera['D'],float)
size=(camera['width'],camera['height'])
paths=sorted((root/'jetracer_audit').glob('2026*.jpg'))
paths += sorted((root.parent/'runs/20261020T131402Z_bot1_e6e282').glob('*.jpg'))
mx,my=cv2.initUndistortRectifyMap(K,D,np.eye(3),K,size,cv2.CV_32FC1)
groups=[]
lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
def straight_residual(points):
    centered=points-points.mean(0)
    _,_,v=np.linalg.svd(centered,full_matrices=False)
    return centered@v[-1]
for index,path in enumerate(paths):
    image=cv2.imread(str(path))
    if image is None:continue
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rect=cv2.remap(gray,mx,my,cv2.INTER_LINEAR)
    lines=lsd.detect(rect)[0]
    if lines is None:continue
    gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    count=0
    for line in sorted(lines[:,0],key=lambda x:-np.hypot(x[2]-x[0],x[3]-x[1])):
        a,b=line[:2],line[2:]
        length=np.linalg.norm(b-a)
        if length<75:continue
        p=a[None]+np.linspace(.04,.96,min(90,int(length/2)))[:,None]*(b-a)[None]
        def sample(m,x,y):
            return cv2.remap(m,np.array(x,np.float32).reshape(1,-1),np.array(y,np.float32).reshape(1,-1),cv2.INTER_LINEAR).ravel()
        raw=np.column_stack([sample(mx,p[:,0],p[:,1]),sample(my,p[:,0],p[:,1])])
        tangent=np.gradient(raw,axis=0)
        normal=np.column_stack([-tangent[:,1],tangent[:,0]])
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-6)
        offsets=np.arange(-2,2.01,.25)
        probes=raw[:,None,:]+normal[:,None,:]*offsets[None,:,None]
        strengths=sample(gradient,probes[:,:,0].ravel(),probes[:,:,1].ravel()).reshape(len(raw),-1)
        best=strengths.argmax(1)
        good=(best>0)&(best<len(offsets)-1)&(strengths.max(1)>40)
        adjusted=raw+normal*offsets[best,None]
        good &= (adjusted[:,0]>3)&(adjusted[:,0]<size[0]-3)&(adjusted[:,1]>3)&(adjusted[:,1]<size[1]-3)
        adjusted=adjusted[good]
        if len(adjusted)<18:continue
        corrected=cv2.undistortPoints(adjusted.reshape(-1,1,2),K,D,P=K).reshape(-1,2)
        if np.sqrt(np.mean(straight_residual(corrected)**2))>1.1:continue
        groups.append({'source':str(path),'validation':index%3==2,'points':adjusted})
        count+=1
        if count>=24:break
    print(path.name,'edges',count,flush=True)
assert len(groups)>30,'Insufficient independent straight edges'
train=[g for g in groups if not g['validation']];test=[g for g in groups if g['validation']]
assert len(test)>8
def residual(params,model,selected):
    d=D.copy() if model=='plumb_bob' else fish0.copy()
    d[:2]=params[:2]
    intrinsics=K.copy()
    if len(params)>2:intrinsics[0,2],intrinsics[1,2]=params[2:4]
    allpoints=np.concatenate([g['points'] for g in selected])
    if model=='plumb_bob':
        corrected=cv2.undistortPoints(allpoints.reshape(-1,1,2),intrinsics,d,P=K).reshape(-1,2)
    else:
        corrected=cv2.fisheye.undistortPoints(allpoints.reshape(-1,1,2),intrinsics,d,P=K).reshape(-1,2)
    out=[];start=0
    for g in selected:
        n=len(g['points']);out.extend(straight_residual(corrected[start:start+n]));start+=n
    return np.nan_to_num(np.array(out),nan=1000,posinf=1000,neginf=-1000)
theta=np.linspace(.01,.95,200)
t=np.tan(theta);radial=t*(1+D[0]*t*t+D[1]*t**4+D[4]*t**6)
fish0=np.linalg.lstsq(np.column_stack([theta**3,theta**5,theta**7,theta**9]),radial-theta,rcond=None)[0]
models={}
for model,initial in [('plumb_bob',D[:2]),('fisheye',fish0[:2])]:
    initial=np.r_[initial,K[0,2],K[1,2]]
    fit=least_squares(lambda p:residual(p,model,train),initial,
                      bounds=(initial-np.array([.25,.25,70,70]),initial+np.array([.25,.25,70,70])),
                      loss='soft_l1',f_scale=.3,max_nfev=80,diff_step=1e-4)
    d=D.copy() if model=='plumb_bob' else fish0.copy();d[:2]=fit.x[:2]
    intrinsics=K.copy();intrinsics[0,2],intrinsics[1,2]=fit.x[2:4]
    radius=np.linspace(0,1.5 if model=='plumb_bob' else 1.1,300)
    derivative=1+sum((2*i+3)*value*radius**(2*i+2) for i,value in enumerate([d[0],d[1],d[4]] if model=='plumb_bob' else d))
    models[model]={'K':intrinsics.tolist(),'D':d.tolist(),'minimum_radial_derivative':float(derivative.min()),
                   'training_rms_px':float(np.sqrt(np.mean(residual(fit.x,model,train)**2))),
                   'validation_rms_px':float(np.sqrt(np.mean(residual(fit.x,model,test)**2))),
                   'evaluations':int(fit.nfev)}
prior=float(np.sqrt(np.mean(residual(D[:2],'plumb_bob',test)**2)))
valid=[name for name in models if models[name]['minimum_radial_derivative']>.05]
assert valid,'Both models fold within the validation angular range'
selected=min(valid,key=lambda name:models[name]['validation_rms_px'])
report={'method':'targetless architectural-edge straightness with CameraInfo intrinsic prior',
        'K':K.tolist(),'width':size[0],'height':size[1],
        'training_edges':len(train),'validation_edges':len(test),
        'training_points':sum(len(g['points']) for g in train),'validation_points':sum(len(g['points']) for g in test),
        'raw_validation_rms_px':float(np.sqrt(np.mean(np.concatenate([straight_residual(g['points']) for g in test])**2))),
        'stored_validation_rms_px':prior,'candidates':models,'selected':selected,
        'limitation':'Focal lengths anchored to live CameraInfo; distortion center and first two radial coefficients refined. Higher-order coefficients retain the prior model representation.',
        'sources':[str(p) for p in paths]}
(root/'jetracer_targetless_validation.json').write_text(json.dumps(report,indent=2))
np.savez_compressed(root/'jetracer_audit/targetless_edges.npz',**{f'edge_{i}':g['points'] for i,g in enumerate(groups)})
print(json.dumps(report,indent=2),flush=True)
