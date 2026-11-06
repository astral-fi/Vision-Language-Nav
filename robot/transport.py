"""Generic zenoh I/O: latest-wins subscriptions and a Twist publisher.

Nothing here knows about FutureNav or about one particular robot. Every key
expression is an argument, so the same classes serve any ROS 2 stack reachable
through a zenoh bridge. Messages are (de)serialized as CDR with rosbags, the
same way tools/zenoh_robot.py and tools/zenoh_probe.py do it.
"""

from __future__ import annotations

import json
import math
import threading
import time

import cv2
import numpy as np
from rosbags.typesys import Stores, get_typestore
import zenoh

TYPES = get_typestore(Stores.ROS2_HUMBLE)
# PIL-free decode table for raw sensor_msgs/Image; compressed frames go via cv2.
RAW_CHANNELS = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}


def endpoint(router: str) -> str:
    """Accept 'ip', 'ip:port' or a full 'tcp/ip:port' zenoh endpoint."""
    if '/' in router:
        return router
    return f'tcp/{router}' if ':' in router else f'tcp/{router}:7447'


class Latest:
    """Holds only the newest sample. A stale observation is worse than none, so
    every read can demand a maximum age and get None instead of old data."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = None
        self._stamp = 0.0

    def set(self, value) -> None:
        with self._lock:
            self._value, self._stamp = value, time.monotonic()

    def get(self, max_age: float | None = None):
        with self._lock:
            if self._value is None:
                return None
            if max_age is not None and time.monotonic() - self._stamp > max_age:
                return None
            return self._value

    def wait(self, timeout: float, max_age: float | None = None, poll: float = 0.01):
        deadline = time.monotonic() + timeout
        while True:
            value = self.get(max_age)
            if value is not None or time.monotonic() >= deadline:
                return value
            time.sleep(poll)

    @property
    def age(self) -> float:
        with self._lock:
            return math.inf if self._value is None else time.monotonic() - self._stamp


def decode_image(payload: bytes, compressed: bool) -> np.ndarray:
    """-> BGR uint8, the layout OpenCV and FutureNav's camera adapter expect."""
    if compressed:
        msg = TYPES.deserialize_cdr(payload, 'sensor_msgs/msg/CompressedImage')
        frame = cv2.imdecode(np.frombuffer(msg.data.tobytes(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f'undecodable {msg.format} frame')
        return frame
    msg = TYPES.deserialize_cdr(payload, 'sensor_msgs/msg/Image')
    if msg.encoding not in RAW_CHANNELS:
        raise ValueError(f'unsupported encoding {msg.encoding}')
    channels = RAW_CHANNELS[msg.encoding]
    frame = np.frombuffer(msg.data.tobytes(), np.uint8)
    frame = frame.reshape(msg.height, msg.step)[:, :msg.width * channels]
    frame = frame.reshape(msg.height, msg.width, channels)
    if msg.encoding.startswith('rgb'):
        frame = frame[:, :, ::-1] if channels == 3 else frame[:, :, [2, 1, 0, 3]]
    elif msg.encoding == 'mono8':
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(frame)


def decode_odom(payload: bytes) -> tuple[float, float, float]:
    """-> (x, y, yaw) in the odometry frame, yaw in radians."""
    msg = TYPES.deserialize_cdr(payload, 'nav_msgs/msg/Odometry')
    position, q = msg.pose.pose.position, msg.pose.pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return float(position.x), float(position.y), yaw


def decode_scan(payload: bytes):
    return TYPES.deserialize_cdr(payload, 'sensor_msgs/msg/LaserScan')


def forward_clearance(scan, half_width_rad: float = math.radians(30),
                      laser_yaw: float = math.pi,
                      chassis_x: float = 0.12, chassis_y: float = 0.10,
                      backward: bool = False) -> float:
    """Nearest valid return in the travel-direction sector, in metres (inf if none).

    laser_yaw rotates the scan into base_footprint -- the JetRacer's lidar is
    mounted backwards (yaw pi), as config/chassis_filter.yaml and the static TF
    in jetracer_launch.py both record. Returns inside the chassis box are the
    robot seeing itself and are dropped.
    """
    centre = math.pi if backward else 0.0
    nearest = math.inf
    for i, value in enumerate(scan.ranges):
        r = float(value)
        if not (math.isfinite(r) and scan.range_min <= r <= scan.range_max):
            continue
        angle = scan.angle_min + i * scan.angle_increment + laser_yaw
        if abs(r * math.cos(angle)) <= chassis_x and abs(r * math.sin(angle)) <= chassis_y:
            continue
        offset = math.atan2(math.sin(angle - centre), math.cos(angle - centre))
        if abs(offset) <= half_width_rad and r < nearest:
            nearest = r
    return nearest


def twist_payload(linear: float, angular: float) -> bytes:
    vector = TYPES.types['geometry_msgs/msg/Vector3']
    twist = TYPES.types['geometry_msgs/msg/Twist']
    msg = twist(vector(float(linear), 0.0, 0.0), vector(0.0, 0.0, float(angular)))
    return bytes(TYPES.serialize_cdr(msg, msg.__msgtype__))


class TwistPublisher:
    """cmd_vel writer. stop() repeats the zero command because a single dropped
    put would leave the robot holding its last velocity."""

    def __init__(self, session, key: str, dry_run: bool = False):
        self.key, self.dry_run = key, dry_run
        self._publisher = None if dry_run else session.declare_publisher(key)

    def send(self, linear: float, angular: float) -> None:
        if self._publisher is not None:
            self._publisher.put(twist_payload(linear, angular))

    def stop(self, repeat: int = 3, gap: float = 0.05) -> None:
        for _ in range(repeat):
            self.send(0.0, 0.0)
            time.sleep(gap)

    def close(self) -> None:
        if self._publisher is not None:
            self.stop()
            self._publisher.undeclare()
            self._publisher = None


class ZenohLink:
    """One client session, plus the subscriptions and publisher made from it."""

    def __init__(self, router: str, connect_timeout: float = 5.0):
        self.endpoint = endpoint(router)
        config = zenoh.Config.from_json5(json.dumps({
            'mode': 'client',
            'connect': {'endpoints': [self.endpoint],
                        'timeout_ms': int(connect_timeout * 1000)},
            'scouting': {'multicast': {'enabled': False}},
        }))
        try:
            zenoh.init_log_from_env_or('error')
        except AttributeError:
            pass
        self.session = zenoh.open(config)
        self._subscribers = []
        self._publishers = []

    def subscribe(self, key: str, decode) -> Latest:
        """Decode in the subscriber callback so the newest sample is always ready,
        and never let a bad frame kill the zenoh thread."""
        latest = Latest()

        def on_sample(sample):
            try:
                latest.set(decode(sample.payload.to_bytes()))
            except Exception as exc:
                print(f'[zenoh] {key}: {exc}', flush=True)

        self._subscribers.append(self.session.declare_subscriber(key, on_sample))
        return latest

    def publisher(self, key: str, dry_run: bool = False) -> TwistPublisher:
        publisher = TwistPublisher(self.session, key, dry_run=dry_run)
        self._publishers.append(publisher)
        return publisher

    def close(self) -> None:
        for publisher in self._publishers:
            publisher.close()
        for subscriber in self._subscribers:
            subscriber.undeclare()
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
