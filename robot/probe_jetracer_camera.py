"""Read live camera metadata and graph without publishing commands."""
import json
from pathlib import Path
import time
from transport import ZenohLink, TYPES

data = {}
with ZenohLink('127.0.0.1:7447') as link:
    def info(raw):
        m = TYPES.deserialize_cdr(raw, 'sensor_msgs/msg/CameraInfo')
        data['camera_info'] = {'width':m.width,'height':m.height,'frame':m.header.frame_id,
                               'distortion_model':m.distortion_model,'K':m.k.tolist(),
                               'D':m.d.tolist(),'P':m.p.tolist()}
    link.subscribe('bot1/csi_cam_0/camera_info', info)
    time.sleep(2)
    nodes=[]
    for reply in link.session.get('@/**/ros2/node/**', timeout=3):
        try:
            sample=reply.ok
            key=str(sample.key_expr)
            payload=sample.payload.to_bytes().decode()
            if 'csi_cam' in key:
                nodes.append({'key':key,'payload':json.loads(payload)})
        except Exception:
            pass
data['nodes']=nodes
p=Path(__file__).resolve().parent/'calib/jetracer_camera_probe.json'
p.write_text(json.dumps(data,indent=2))
print(json.dumps(data.get('camera_info'),indent=2))
print('camera_nodes',len(nodes))
