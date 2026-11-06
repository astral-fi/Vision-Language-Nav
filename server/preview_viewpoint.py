"""Render the same episode start pose across camera heights and pitches.

    unset DISPLAY
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL.so.1:/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0
    ~/miniforge3/envs/habitat/bin/python deploy/preview_viewpoint.py --episode 0

Answers "what does the model actually see at 0.5 m, and would tilting up help?"
by showing it, instead of reasoning about frustums.

Geometry worth knowing before reading the output: for a LEVEL camera the horizon
sits on the optical axis, i.e. the exact vertical centre of the frame, at EVERY
height. So lowering a level camera does not increase the share of the frame
showing floor -- it stays ~50%. What changes is (a) the floor you see is nearer
(0.81 m at the bottom edge from 0.5 m, vs 2.02 m from 1.25 m) and (b) you can no
longer see over furniture. Tilting up does move the horizon down and recover
distant/high structure, but it cannot undo occlusion.
"""

import argparse
import gzip
import json
import math
import os

import magnum as mn
import numpy as np
from PIL import Image, ImageDraw

import habitat_sim

R2R = "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
HFOV = 79.0
W, H = 640, 480


def episode(idx):
    with gzip.open(R2R, "rt") as fh:
        eps = json.load(fh)["episodes"]
    eps.sort(key=lambda e: e["episode_id"])
    # Match eval/run.py's seeded 50-episode selection so previews line up with
    # the episodes the ablations actually ran.
    np.random.seed(42)
    sel = sorted(np.random.choice(len(eps), 50, replace=False))
    return eps[sel[idx]]


def render(sim, agent, pos, rot, height, pitch_deg):
    st = habitat_sim.AgentState()
    st.position = np.array(pos, dtype=np.float32)
    st.rotation = np.quaternion(rot[3], rot[0], rot[1], rot[2])
    agent.set_state(st)
    for uuid, s in sim._sensors.items():
        node = s._sensor_object.node
        node.reset_transformation()
        node.translate(np.array([0.0, height, 0.0], dtype=np.float32))
        # positive pitch_deg = nose UP. magnum.Rad, not habitat_sim.geo.Rad,
        # which does not exist in habitat-sim 0.1.7.
        node.rotate_x_local(mn.Rad(math.radians(pitch_deg)))
    return sim.get_sensor_observations()["rgb"][:, :, :3]


def label(img, text, sub=None):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 34 if sub else 22], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 220, 90))
    if sub:
        d.text((6, 19), sub, fill=(170, 200, 255))
    # horizon line for a level camera sits at the vertical centre; draw where it
    # actually falls for this pitch so the floor share is readable at a glance.
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--heights", default="1.25,0.75,0.5,0.3,0.12")
    ap.add_argument("--pitches", default="0,10,20",
                    help="degrees nose-UP; 0 = level (training default)")
    ap.add_argument("--out", default="outputs/viewpoint_grid.jpg")
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()

    ep = episode(a.episode)
    scene = os.path.join("data/scene_datasets", ep["scene_id"])
    print(f"episode {ep['episode_id']}  scene {ep['scene_id'].split('/')[-2]}")
    print(f"instruction: {ep['instruction']['instruction_text'].strip()}")

    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = scene
    cfg.gpu_device_id = 0
    ac = habitat_sim.AgentConfiguration()
    spec = habitat_sim.SensorSpec()
    spec.uuid = "rgb"
    spec.resolution = [H, W]
    spec.parameters["hfov"] = str(HFOV)
    ac.sensor_specifications = [spec]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ac]))
    agent = sim.get_agent(0)

    heights = [float(x) for x in a.heights.split(",")]
    pitches = [float(x) for x in a.pitches.split(",")]
    tiles = []
    for p in pitches:
        row = []
        for h in heights:
            img = render(sim, agent, ep["start_position"], ep["start_rotation"], h, p)
            near = h / math.tan(math.radians(31.7 + p)) if (31.7 + p) > 0 else float("inf")
            row.append(label(img, f"h={h:.2f}m  pitch=+{p:.0f}deg",
                             f"floor from {near:.2f}m"))
        tiles.append(row)
    sim.close()

    tw = int(W * a.scale)
    th = int(H * a.scale)
    grid = Image.new("RGB", (tw * len(heights), th * len(pitches)), (20, 20, 20))
    for r, row in enumerate(tiles):
        for c, im in enumerate(row):
            grid.paste(im.resize((tw, th), Image.LANCZOS), (c * tw, r * th))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    grid.save(a.out, quality=92)
    print(f"\nsaved {a.out}  ({grid.width}x{grid.height})")
    print("columns = camera height (left = training default 1.25 m)")
    print("rows    = pitch, 0 deg = level = what the model was trained on")


if __name__ == "__main__":
    main()
