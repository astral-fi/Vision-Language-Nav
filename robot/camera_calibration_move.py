"""One bounded JetRacer calibration movement, with camera watchdog and recording."""
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import signal
import time
import cv2
from config import Settings
from recording import RunRecording
from transport import ZenohLink, decode_image, decode_odom

p=ArgumentParser(description=__doc__)
p.add_argument('--linear',type=float,default=.08)
p.add_argument('--angular',type=float,default=0.)
p.add_argument('--duration',type=float,default=3.)
a=p.parse_args()
assert abs(a.linear)<=.08 and abs(a.angular)<=.26 and 0<a.duration<=3
s=Settings(robot='jetracer').resolve()
args=Namespace(runs_dir=str(Path(__file__).resolve().parent/'runs'),video_fps=10.,
               instruction='Manual camera calibration movement',dry_run=False)
with ZenohLink(s.router) as link:
    frames=link.subscribe(s.camera_topic,lambda raw:decode_image(raw,True))
    odom=link.subscribe('bot1/odom',decode_odom)
    drive=link.publisher(s.cmd_vel_topic)
    def halt(*_):
        drive.stop(repeat=10)
        raise SystemExit(1)
    signal.signal(signal.SIGINT,halt)
    signal.signal(signal.SIGTERM,halt)
    try:
        drive.stop()
        assert frames.wait(5,max_age=1) is not None,'No fresh camera'
        with RunRecording(frames,args,s) as r:
            r.odom=odom
            r.metadata['activations']={'applicable':False,'reason':'manual calibration; no inference'}
            r.metadata['calibration_motion']=vars(a)
            start=time.monotonic()
            while time.monotonic()-start<a.duration:
                assert frames.age<.5,'Camera stale; stopping'
                drive.send(a.linear,a.angular)
                time.sleep(.05)
            drive.stop(repeat=10)
            time.sleep(.3)
            r.metadata['stop_reason']='calibration_pulse_complete'
            cv2.imwrite(str(r.path/'calibration_view.jpg'),frames.get(.5))
            print(json.dumps({'run':str(r.path),'odom':odom.get(1)}),flush=True)
    finally:
        drive.stop(repeat=10)
