"""Synthetic smoke test for deploy/server.py. No Habitat, no MP3D scenes.

    python deploy/smoke_test.py [steps]

Feeds the server generated frames and reports what analysis alone cannot give:
measured latency, measured VRAM, and whether the VGGT cache grows exactly one
frame per step.

The ACTIONS IT RETURNS ARE MEANINGLESS. The frames are crude synthetic
geometry, far outside the Matterport3D distribution the model was trained on;
expect it to emit the same action over and over. This validates that the
pipeline runs, not that the policy works. For that you need the real
evaluation -- see deploy/README.md section 3a.
"""
import base64, io, json, os, sys, time, urllib.request
import numpy as np
from PIL import Image, ImageDraw

SERVER = os.environ.get("FUTURENAV_SERVER", "http://127.0.0.1:8642")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def post(path, payload, timeout=600):
    req = urllib.request.Request(SERVER + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def frame(i):
    """A crude 640x480 'corridor' that slides sideways, so successive frames have
    real parallax for VGGT rather than being identical noise."""
    img = Image.new("RGB", (640, 480), (196, 190, 178))
    d = ImageDraw.Draw(img)
    off = (i * 26) % 640
    d.rectangle([0, 360, 640, 480], fill=(122, 100, 78))        # floor
    d.rectangle([0, 0, 640, 90], fill=(232, 230, 224))          # ceiling
    for k in range(-1, 6):                                       # wall panels
        x = k * 150 - off
        d.rectangle([x, 90, x + 140, 360], outline=(140, 136, 128), width=3)
    d.rectangle([320 - off // 3, 170, 420 - off // 3, 360], fill=(70, 92, 120))  # door
    return np.asarray(img)


print(f"health: {json.load(urllib.request.urlopen(SERVER + '/health'))}")
post("/reset", {"session": "smoke", "instruction":
                "walk down the hallway and stop at the blue door"})

lat = []
for i in range(STEPS):
    buf = io.BytesIO()
    Image.fromarray(frame(i)).save(buf, format="PNG")
    t0 = time.time()
    r = post("/step", {"session": "smoke", "step": i,
                       "instruction": "walk down the hallway and stop at the blue door",
                       "image": base64.b64encode(buf.getvalue()).decode()})
    dt = time.time() - t0
    lat.append(dt)
    v = r.get("vram", {})
    print(f"step {i:2d}  {r['action_name']:<13} {dt:5.2f}s  cache={r['vggt_cache_frames']:2d}f "
          f"ok={r['vggt_cache_ok']}  vram alloc={v.get('alloc_gb')} peak={v.get('peak_gb')} "
          f"reserved={v.get('reserved_gb')}  raw={r['raw']!r}")

warm = lat[3:] or lat
print(f"\nlatency: first={lat[0]:.2f}s  warm mean={sum(warm)/len(warm):.2f}s  "
      f"min={min(warm):.2f}s max={max(warm):.2f}s")
print("\nNote: latency and VRAM both keep rising until the VGGT cache saturates at\n"
      "56 frames. Extrapolate from the last few steps, do not read the mean above\n"
      "as the steady-state cost.")
