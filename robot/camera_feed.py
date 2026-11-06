"""Local-only live MJPEG camera preview; command videos are saved by client.py."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time

import cv2
import numpy as np
from camera_color import ColorCorrection
from view import FisheyeViewAdapter
from transport import ZenohLink, decode_image


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == '/health':
            age = self.server.frames.age
            data = json.dumps({'ready': age < 1, 'camera_age_s': age if age < 1e6 else None}).encode()
            self.send_response(200 if age < 1 else 503)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path not in ('/', '/stream.mjpg', '/latest.jpg', '/unrectified.jpg'):
            self.send_error(404)
            return
        frame = self.server.frames.get(max_age=1)
        if frame is None and self.path.endswith('.jpg'):
            self.send_error(503, 'No fresh camera frame')
            return
        if frame is None:
            frame = np.zeros((420, 640, 3), np.uint8)
        self.send_response(200)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Type', 'image/jpeg' if self.path.endswith('.jpg')
                         else 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                fresh = self.server.frames.get(max_age=1)
                if fresh is not None:
                    frame = fresh
                view = frame if self.path == '/unrectified.jpg' else self.server.adapter(frame)
                if fresh is None:
                    view = view.copy()
                    cv2.putText(view, 'Camera reconnecting...', (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
                ok, image = cv2.imencode('.jpg', view, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    break
                if self.path.endswith('.jpg'):
                    self.wfile.write(image.tobytes())
                    return
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                 str(len(image)).encode() + b'\r\n\r\n' + image.tobytes() + b'\r\n')
                self.wfile.flush()
                time.sleep(.1)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--router', default='127.0.0.1:7447')
    parser.add_argument('--camera-topic', default='bot1/csi_cam_0/image_raw/compressed')
    parser.add_argument('--port', default=8766, type=int)
    parser.add_argument('--color-profile', default=None)
    parser.add_argument('--lens-profile', default=None)
    args = parser.parse_args()
    with ZenohLink(args.router) as link:
        color = ColorCorrection(args.color_profile)
        frames = link.subscribe(args.camera_topic, lambda raw: color(decode_image(raw, args.camera_topic.endswith('/compressed'))))
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
        server.daemon_threads = True
        server.frames = frames
        server.adapter = FisheyeViewAdapter(args.lens_profile) if args.lens_profile else lambda frame: frame
        print(f'Live camera: http://127.0.0.1:{args.port}/stream.mjpg', flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
