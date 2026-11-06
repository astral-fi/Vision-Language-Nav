"""Read-only Zenoh camera receipt rates; no robot publisher is declared."""
import json
import time
from transport import ZenohLink

counts = {}
with ZenohLink('127.0.0.1:7447') as link:
    for key in ('tb3/flir_camera/image_raw/compressed', 'tb3/flir_camera/image_raw',
                'bot1/csi_cam_0/image_raw/compressed'):
        counts[key] = {'samples': 0, 'bytes': 0, 'first': None, 'last': None}
        def receive(raw, topic=key):
            now = time.monotonic()
            item = counts[topic]
            item['samples'] += 1
            item['bytes'] += len(raw)
            item['first'] = item['first'] or now
            item['last'] = now
            return None
        link.subscribe(key, receive)
    time.sleep(6)
    for value in counts.values():
        value['last_age_s'] = None if value['last'] is None else time.monotonic() - value['last']
print(json.dumps(counts, indent=2))
