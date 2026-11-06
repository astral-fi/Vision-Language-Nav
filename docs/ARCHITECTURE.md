# Running FutureNav on a JetRacer

Short version: **FutureNav cannot run on the JetRacer's own compute.** It is a
5.48 B-parameter model that needs ~23 GB of VRAM and ~49 TFLOP per navigation
step. Put the model on a workstation GPU and make the car a thin client. The
code in this directory does that, plus the sensor and actuator adaptation the
sim-to-real gap requires.

---

## 1. What the model actually is

From `llxs/FutureNav → FutureNav-4B-Base/config.json` and the code in
`src/qwen_vl/model/`:

| Component | Detail |
|---|---|
| Backbone | Qwen3-VL-4B (`qwen3_vl`), text: 36 layers, hidden 2560, 32 Q / 8 KV heads, head_dim 128 |
| Vision tower | Qwen3-VL ViT, 24 layers, hidden 1024, patch 16, spatial merge 2, deepstack at layers 5/11/17 |
| Geometry tower | VGGT aggregator (DINOv2-L/14 patch embed + 24 alternating frame/global blocks, dim 1024, 16 heads), `camera_head` and `track_head` deleted |
| Fusion | `merger_vggt` (`VGGTMerger`, 2048·4 → 4096 → 2560), injected as `image_embeds += lam * image_embeds_3d` with `lam = 0.2` |
| Aux heads | `forward_head`, `inverse_head` — training-only, unused at inference |
| Weights | 5,481,222,666 params, 11.74 GB bf16, 3 safetensors shards |
| Output | free-text generation, `max_new_tokens=24`, greedy; the string is matched against `{STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT}` |

Two heads not in the checkpoint (`gen_head`, `inverse_next_proj`) are
constructed in `modeling_qwen3_vl_obshead.py` and will be randomly initialised
with a warning on load. They are never called at inference — ignore the warning.

### Per-step data flow

```
RGB 640x480
  └─ load_and_preprocess_images (target 644 wide, /14 grid)  →  3 x 490 x 644
       ├─ VGGT tower  (current frame only)  35 x 46 = 1610 patches + 5 specials = 1615 tokens
       │    └─ aggregated_tokens[-2] → 2048-dim → 2x2 merge → 17 x 23 = 391 tokens
       │         └─ merger_vggt → 2560-dim, linearly interpolated 391 → 2700 positions
       └─ Qwen tower: already a multiple of 32, no trim at this input size
            └─ Qwen processor smart_resize → 480 x 640 → (30x40)/4 = 300 tokens/image
                 x 9 images (8 history + current) = 2700 image tokens
                 + system prompt + instruction ≈ 2800–3000 total context tokens
```

Note the 391 → 2700 step: the geometry features come from the **current frame
only**, and are linearly interpolated along the sequence axis to cover all nine
images' token positions before the `+ 0.2 *` fusion. The geometric signal is
therefore spread over the whole visual context, not aligned per-frame.

The **VGGT tower runs on one frame per step**. Earlier frames persist as a KV
cache on `model.past_key_values_vggt`, trimmed by `StartRecentKVCache(start=8,
recent=48)` — 56 frames of attention-sink-style memory. That cache is the
model's spatial memory of the episode and it is append-only:

* replaying a step inserts a frame twice,
* dropping a step leaves a hole,
* both silently degrade every later decision.

`deploy/server.py` enforces a monotonic step counter for this reason.

### Compute and memory budget per step

| Item | Cost |
|---|---|
| Qwen ViT (9 images × 1200 patch tokens) | ~7.8 TFLOP |
| LLM prefill (~2800 tokens, 3.63 B non-embedding params, no cross-step reuse) | ~23.2 TFLOP |
| VGGT DINOv2 patch embed + frame blocks | ~3.2 TFLOP |
| VGGT global blocks (1615 queries × up to 90,440 cached keys) | ~14.4 TFLOP |
| **Total** | **~49 TFLOP / step** |
| Weights (bf16) | 11.74 GB |
| VGGT KV cache | ~159 MB/frame × 56 = **~8.9 GB** |
| Activations (eager LLM attention over ~2800 tokens) | ~1–3 GB |
| **Peak VRAM** | **~22–24 GB** |

The LLM KV cache is *not* reused between steps — the image set changes every
step, so the whole ~2800-token prefix is recomputed. That is where most of the
FLOPs go, and it is why per-step latency is roughly constant.

### Hardware reality check

| Platform | Memory | Verdict |
|---|---|---|
| Jetson Nano 4 GB (stock JetRacer / JetRacer Pro) | 4 GB shared | **Impossible.** 3× short on memory even at INT4; Maxwell sm_53 has no bf16, no tensor cores, and JetPack 4.6 caps you at Python 3.6 / PyTorch 1.10 while this code needs Python ≥3.9, `transformers ≥ 4.57`, and `torch.amp.autocast` (torch ≥ 2.4). Compute alone would be ~7 minutes per 0.25 m step. |
| Orin Nano 8 GB | 8 GB shared | Not viable. Weights alone at INT4 are ~3.2 GB, but there is no INT4 path for the VGGT tower or its custom KV cache, and you would have to cut the cache to ~4 frames. ~15–30 s/step if you got it running. |
| AGX Orin 64 GB | 64 GB shared | The only Jetson that fits. ~3–7 s/step in FP16. Costs more than the whole JetRacer, draws 15–60 W, and does not physically mount on a 1/10 chassis. |
| RTX 3090 / 4090 (24 GB) | 24 GB | Works. The stock 8+48 cache needs ~21.6 GB with `--attn sdpa` — fits on a **headless** card, OOMs if a desktop session shares the GPU. `--vggt-recent-frames 24` drops it to ~17 GB. ~1.0–1.5 s/step on a 4090. |
| A100 / H100 / RTX 6000 Ada (40 GB+) | 40 GB+ | Comfortable, ~0.5–1.0 s/step. |

So: **off-board inference, on-board control.**

---

## 2. Architecture

```
JetRacer (Jetson Nano)                    Workstation GPU (>= 24 GB)
┌────────────────────────────┐            ┌──────────────────────────────┐
│ CSI camera (IMX219-160)    │            │ deploy/server.py             │
│   └ camera_adapter.py      │  JPEG      │   └ futurenav_core.py        │
│       fisheye → 79° pinhole│ ─────────► │       FutureNavPolicy        │
│                            │   WiFi     │       (VGGT KV cache lives   │
│ jetracer_client.py         │ ◄───────── │        here, one episode)    │
│   └ motion.py              │  action    └──────────────────────────────┘
│       Ackermann primitives │
│   └ safety gates           │
└────────────────────────────┘
```

Round trip on a decent 5 GHz link: ~30–80 ms network + 0.6–1.5 s inference.
Each cycle is stop-and-go — the car is stationary while the model thinks, which
is exactly what Habitat does during training and also what keeps a ~1 s latency
off the critical path of a moving vehicle.

Effective speed: ~0.25 m per ~2 s ≈ **0.12 m/s**.

---

## 3. Setup

### Workstation

```bash
conda create -n futurenav python=3.10 -y && conda activate futurenav
pip install -r requirements.txt      # habitat is NOT needed for deployment
huggingface-cli download llxs/FutureNav --include "FutureNav-4B-Base/*" \
    --local-dir pretrain_models

export PYTHONPATH=$PWD/src:$PWD
python deploy/server.py \
    --checkpoint pretrain_models/FutureNav-4B-Base \
    --attn sdpa --port 8642
```

On a 24 GB card, start there and watch the `vram=alloc/peak` figure the server
prints each step. If peak creeps past ~22 GB, add `--vggt-recent-frames 24`.
Free the GPU of anything else first — `nvidia-smi` should show ~0 MB before you
start, and a desktop session or a browser will cost you 1–2 GB you need.

The server also prints `cache=Nf` per step. That number must equal
`min(step, start+recent)`. If it climbs faster, the vision branch is being
re-entered during token decode, the spatial memory is being corrupted, and the
server flags it with `!!`.

### JetRacer

Only needs `opencv-python`, `numpy`, `pillow`, `jetcam`, `jetracer` — no torch,
no transformers.

```bash
python3 deploy/calibrate_camera.py --capture --out deploy/fisheye_calib.json
python3 deploy/calibrate_motion.py --stage speed
python3 deploy/calibrate_motion.py --stage radius
python3 deploy/calibrate_motion.py --stage verify

python3 deploy/jetracer_client.py \
    --server http://<workstation-ip>:8642 \
    --instruction "walk out of the bedroom and stop next to the kitchen table" \
    --mount-pitch 8.0 --dry-run          # drop --dry-run once you trust it
```

Always do a first run with `--dry-run --save-dir /tmp/frames` and look at the
rectified frames. If they do not look like a chest-height, level, ~79° view of
a room, fix that before interpreting any navigation behaviour.

---

## 3a. Running the Habitat evaluation

### The environment conflict, and why there are two envs

The top-level README says to create one Python 3.9 environment and
`conda install habitat-sim=0.1.7=py3.9_headless_linux_...`. **That build does
not exist.** The `aihabitat` channel publishes habitat-sim 0.1.7 for Python
3.6, 3.7 and 3.8 only. And the model cannot run on 3.8:

| | requires |
|---|---|
| habitat-sim 0.1.7 | Python ≤ 3.8 |
| transformers ≥ 4.57 (first release with Qwen3-VL) | Python ≥ 3.9 |

So the simulator and the model get one environment each and talk over the same
HTTP protocol the JetRacer uses. A side benefit: the number you measure in sim
is a number about the exact serving path that will drive the car.

```
conda env "habitat"   py3.8    conda env "futurenav"  py3.10
  habitat-sim 0.1.7            torch 2.6 + cu124
  habitat-lab 0.1.7    HTTP    transformers >= 4.57
  VLN_CE            <------->  deploy/server.py
  eval/run.py --server         (holds the VGGT cache)
```

### Step 1 — environments

```bash
bash deploy/setup_model_env.sh      # conda env "futurenav", py3.10
bash deploy/setup_habitat_env.sh    # conda env "habitat",   py3.8
```

Both are idempotent. Each ends with an import check that fails loudly rather
than leaving you to discover the problem 40 minutes into a run.

Two pins in the habitat env are load-bearing:

* **`numpy==1.23.5`.** VLN_CE still uses `np.float` and `np.int`, removed in
  numpy 1.24 (`habitat_extensions/sensors.py`, `actions.py`). Upgrading numpy
  breaks the import, not the results.
* **`opencv-python==4.8.1.78`.** Newer wheels are built against numpy 2 and
  fail to import against 1.23.

The setup script deliberately does *not* install `VLN_CE/requirements.txt`
as-is: it pins `tensorflow==1.13.1`, which has no Python 3.8 wheel and is never
imported by the eval path. It also skips habitat-lab's ddppo requirements,
which would install their own torch.

### Step 2 — data

```bash
bash deploy/fetch_checkpoint.sh     # 11.7 GB from HuggingFace
bash deploy/fetch_episodes.sh       # R2R + RxR episode json.gz, public
```

**The Matterport3D scenes are gated and cannot be scripted.** Sign the Terms of
Use at <https://niessner.github.io/Matterport/>; the maintainers email back a
`download_mp.py`. Then:

```bash
python2 download_mp.py --task habitat -o data/scene_datasets/
cd data/scene_datasets && unzip v1/tasks/mp3d_habitat.zip
```

Use `--task habitat` specifically. The full Matterport3D release is ~1 TB and
does **not** include the `.navmesh` files the simulator needs to path; the
habitat task subset is ~15 GB and does. You should end with
`data/scene_datasets/mp3d/{scene_id}/{scene_id}.glb` plus a matching `.navmesh`.

**You almost certainly do not need all 90 scenes.** R2R `val_unseen` — the
split the published numbers are reported on — touches only 11:

```
2azQ1b91cZZ  8194nk5LbLH  EU6Fwq7SyZv  QUCTc6BB5sX  TbHJrupSAjP  X7HyMhZNoso
Z6MFQCViBuw  oLBMNvg9in8  pLe4wQe7qrG  x8F5xyUWy9e  zsNo4HB9uLZ
```

RxR `val_unseen` uses **the same 11 scenes**, so one ~2 GB download covers both
benchmarks' unseen splits. `download_mp.py` accepts `--id`, and `check_data.py`
prints the exact per-scene commands when the scene directory is missing.

`val_seen` is the exception — it spans 53 scenes, so pull the full task set if
you want that split too.

Verify before running anything:

```bash
python deploy/check_data.py
```

It cross-references the scenes each episode actually references against what is
on disk, so a partial scene download is caught up front rather than as a crash
in episode 200.

### Step 3 — start the model server

In the **futurenav** env:

```bash
conda activate futurenav
export PYTHONPATH=$PWD/src:$PWD
python deploy/server.py \
    --checkpoint pretrain_models/FutureNav-4B-Base \
    --attn sdpa --port 8642
```

Wait for `[server] ready`. Loading 11.7 GB takes a minute or two.

### Step 4 — run the evaluation

In a second terminal, the **habitat** env:

```bash
conda activate habitat
bash eval/eval_r2r_remote.sh val_unseen 50
```

Arguments are `[split] [max_episodes] [server_url]`; `0` episodes means all
1,839 of val_unseen. Start with 50 — at roughly 1.2 s per step and ~55 steps
per episode that is about an hour, and it is enough to tell you whether the
pipeline is sound.

Episode selection uses `np.random.seed(42)`, so two runs with the same
`max_episodes` see the same episodes and are directly comparable.

`eval/run.py` skips episodes it finds in `<output>/log/`, which makes runs
resumable — but it also means a second run with different settings will silently
reuse the first run's results. **Set `TAG` or clear the directory between
configurations.**

```bash
TAG=baseline  bash eval/eval_r2r_remote.sh val_unseen 50
TAG=cache24   bash eval/eval_r2r_remote.sh val_unseen 50   # after restarting the
                                                           # server with
                                                           # --vggt-recent-frames 24
```

### Measured on an RTX 4090

The figures elsewhere in this document are analytic. These are not — they come
from `deploy/smoke_test.py` against the real checkpoint on a 4090:

| | Predicted | Measured |
|---|---|---|
| Weights resident | 11.74 GB | **11.27 GB** (tied embeddings are stored once in memory) |
| VGGT cache per frame | 159 MB | **153 MB** |
| Cache growth | 1 frame/step | **1 frame/step, no desync over 14 steps** |
| Latency, 9-image context | 1.0–1.5 s | **0.68 s at 10 frames, +~5 ms per additional cached frame** |

Extrapolating the last five steps to a saturated 56-frame cache gives roughly
**0.9–1.0 s/step and ~19.7 GB allocated / ~21.5 GB peak**. Both land where the
analysis said they would, and both confirm a 24 GB card is enough with nothing
else on it.

Two things the measurement changed:

* Latency rises steeply over the first ~9 steps only because the history is
  still filling to 9 images. After that the slope is gentle — the growth is the
  VGGT cache, and it is much flatter than the early steps suggest. Do not read
  a short run's mean as the steady-state cost.
* `reserved` ran well ahead of `allocated` (16.4 GB reserved at 13.3 GB
  allocated). A cache that grows monotonically for 56 steps is the worst case
  for the caching allocator, so `deploy/server.py` now sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. On a 24 GB card that is
  the difference between fitting and not.

You can run this before the MP3D scenes arrive — it needs only the checkpoint:

```bash
python deploy/smoke_test.py 14
```

The actions it prints are meaningless (synthetic frames, far out of
distribution). It proves the pipeline runs, nothing about navigation quality.

### What a healthy run looks like

Server side, one line per step:

```
[server] step= 12 -> MOVE_FORWARD   1.14s  cache=13f  vram=19.8/21.4GB
```

* `cache=Nf` must equal `min(step+1, 56)`. The agent raises immediately on a
  mismatch rather than reporting a meaningless score.
* `vram=alloc/peak` — if peak approaches 23 GB, restart with
  `--vggt-recent-frames 24`.
* A run of `[FALLBACK]` means the model is emitting something unparseable.

The published R2R val_unseen figures are the bar to compare against. On a
50-episode subset expect several points of sampling noise either way; treat a
subset number as a smoke test, not a replication.

### When it breaks

| Symptom | Cause |
|---|---|
| `AttributeError: module 'numpy' has no attribute 'float'` | numpy ≥ 1.24 in the habitat env. Pin 1.23.5. |
| `ImportError: numpy.core.multiarray failed to import` | opencv built against numpy 2. Pin `opencv-python==4.8.1.78`. |
| `ModuleNotFoundError: habitat_baselines` | habitat-lab installed without `--all`. |
| Sim hangs or segfaults on the first `env.reset()` | Missing `.navmesh`, or scenes downloaded without `--task habitat`. |
| `server rejected /step (409)` | Step counter desync — the agent retried, or a previous episode did not call `/reset`. |
| CUDA OOM part-way through an episode | The cache grows to 56 frames over the first 56 steps, so OOM arrives mid-episode, not at load. Trim it. |
| `size mismatch for bias: ... [2560] ... current model is [5120]` | The released `modeling_qwen3_vl_obshead.py` declares `forward_head`/`inverse_head` at widths the released checkpoint does not use. Fixed in this tree; see the comment in that file. |
| `ImportError: libcudnn.so.9` from a `torch-*.egg` in the habitat env | `setup.py develop --all` easy_installed its own torch. Pre-install the CPU wheel first, as `setup_habitat_env.sh` now does. |
| `Could not find suitable distribution for tensorflow==1.13.1` | habitat-lab's ddppo requirements. Strip the pin; nothing on the eval path imports it. |
| `403` from the gdown episode links | Google Drive quota. Retry later or download in a browser. |

## 4. Bridging the two gaps

### 4.1 Sensor gap — `camera_adapter.py`

| | Habitat (training) | Stock JetRacer |
|---|---|---|
| Resolution | 640 × 480 | 1280 × 720 (IMX219, 8 MP sensor) |
| HFOV | 79° rectilinear | ~160° fisheye |
| Height | 1.25 m | ~0.10–0.15 m |
| Pitch | level | usually tilted down |

`HabitatViewAdapter` rebuilds a virtual pinhole camera with
`fx = fy = (640/2)/tan(79°/2) = 388.2`, `cx = 320`, `cy = 240`, and bakes a
rotation into the same `cv2.fisheye.initUndistortRectifyMap` call so a
physically tilted mount still produces a level image. ~4 ms/frame on a Nano.

**Height cannot be fixed in software.** A 0.12 m camera sees floor, chair legs
and table undersides where the model expects to see room layout, doorways and
counter tops — and VGGT's depth/scale priors are calibrated to the training
viewpoint. Mount the camera on a mast. Every centimetre helps; 0.5–0.7 m is
achievable on a 1/10 chassis with a carbon tube and a counterweight, and gets
you most of the benefit. Below ~0.3 m expect the policy to be substantially
worse at "go through the doorway" style instructions.

### 4.2 Actuator gap — `motion.py`

The action space is defined by `eval/config/vlnce_task_futurenav.yaml`:
`FORWARD_STEP_SIZE: 0.25`, `TURN_ANGLE: 15`, differential-drive turn-in-place.
A JetRacer is Ackermann and **cannot rotate in place**.

Two approximations, selected automatically:

* **ARC** (default) — tightest arc until 15° of yaw. With a measured turn radius
  R = 0.65 m, that also moves the car 0.168 m forward and 0.022 m sideways —
  about ⅔ of a MOVE_FORWARD. Fast, and the closed loop absorbs the drift.
* **KTURN** — forward arc + reverse counter-arc, ~15° of yaw with near-zero net
  translation. Much closer to the training semantics, ~3× slower, needs ~0.4 m
  of clearance. Used automatically when a forward range sensor reports < 0.45 m.

Neither distance nor angle is measured on a stock car, so both are timed against
a calibrated speed model. **Recalibrate for every floor surface and battery
state** — carpet vs hardwood changes the answer by 30–40%, and a 3S pack sagging
from 12.6 V to 9.9 V shrinks your effective step by ~20% over a run. If you can
add hardware, add these two in order:

1. **IMU (BNO055 or MPU6050 on I²C)** — closes the loop on yaw. `motion.py`
   already uses it when `imu.yaw_deg()` is supplied. Biggest single accuracy win
   after the camera mast; angle error compounds far worse than distance error.
2. **Forward ToF (VL53L0X/VL53L1X)** — feeds the range gate and the KTURN
   trigger. Without it, `_blocked()` and `_tight_space()` are no-ops and you
   have no collision protection.

---

## 5. Problems you will hit, ranked

**Blocking — must be solved before anything works**

1. **Model does not fit on the car.** 21 GB vs 4 GB. Solved by the client/server
   split here; there is no quantisation trick that closes a 5× gap on Maxwell.
2. **Software stack.** JetPack 4.6 = Python 3.6, CUDA 10.2, PyTorch ≤1.10.
   `transformers ≥ 4.57`, `torch.amp.autocast`, and Qwen3-VL all need much
   newer. Nothing in this repo will import on a stock Nano.
3. **Ackermann vs turn-in-place.** Every TURN action translates the car. See
   §4.2.

**Severe — the thing will "run" but navigate badly**

4. **Camera height (1.25 m → 0.12 m).** The largest accuracy loss you can
   actually do something about. Build the mast.
5. **Fisheye projection.** Fixed by `camera_adapter.py`; skipping it is worse
   than the height problem.
6. **Sim-to-real appearance gap.** Matterport3D renders have baked lighting,
   seams and reconstruction holes; real rooms have motion blur, rolling-shutter
   skew, auto-exposure hunting and specular glare. Lock exposure and white
   balance on the CSI sensor — auto-exposure swinging as the car turns toward a
   window looks to the model like a scene change.
7. **No collision handling in the policy.** Habitat trains with
   `ALLOW_SLIDING: True`, so the agent slides along walls instead of stopping.
   The policy has never seen a stall and will happily command MOVE_FORWARD into
   a wall forever. `recover_from_collision()` and the range gate exist because
   of this; without a range or current sensor you have no recovery at all.
8. **Odometry drift.** ~5% per-step error compounds to metres over a 60-step
   episode. The 3.0 m success radius is forgiving, but headings are not — add
   the IMU.

**Moderate**

9. **Rotation stalls.** The eval config carries `EARLY_STOP_ROTATION: 50` — the
   authors' own policy gets stuck spinning often enough to need a guard. It is
   mirrored in the client; expect to see it trip.
10. **VGGT cache assumes coherent ego-motion.** It is a geometric memory built
    from a frame sequence. Wheel slip, a KTURN's extra frames, a dropped frame,
    or a network retry all corrupt it. Hence the strict step counter.
11. **Latency and WiFi.** A dropped connection mid-motion leaves the car driving.
    The client zeroes throttle on any exception, on SIGINT/SIGTERM, and via
    `atexit` — keep all three.
12. **Instruction distribution.** R2R/RxR instructions are human-scale house
    tours ("go up the stairs", "wait by the sofa"). Stairs and thresholds are
    impassable for a 1/10 car; phrase instructions for what the car can drive.
13. **Battery sag** changes step size mid-episode. Use the JetRacer Pro's INA219
    and pass `voltage_fn` to `JetRacerMotion`.
14. **Thermals.** Sustained CSI capture + WiFi + servo load will throttle a
    passively-cooled Nano. Use the 5 V/4 A barrel jack, not micro-USB, and add a
    fan.

**Worth knowing**

15. `attn_implementation="eager"` (as in `eval/agent.py`) materialises a
    ~2800×2800 attention matrix per layer. Use `--attn sdpa`. The VGGT tower
    already uses `F.scaled_dot_product_attention`, which is the only reason the
    90k-key global attention does not OOM.
16. Parse fallback: an unparseable generation silently becomes MOVE_FORWARD
    (`eval/agent.py:196`). Preserved here, but logged as `[parse-fallback]`.
    Watch for it — a burst means something upstream is wrong.
17. Only one episode at a time per loaded model; the VGGT cache is a single
    attribute on the model object.

---

## 6. If you must run on-board

In rough order of how much they buy you:

1. **Swap the compute.** AGX Orin 64 GB is the only Jetson that fits the model.
   Orin NX 16 GB gets close if you also do (2) and (4).
2. **Cut the VGGT cache.** `--vggt-recent-frames 8` drops it from ~8.9 GB to
   ~2.5 GB. Costs long-range spatial memory; measure on `val_unseen` before
   trusting it.
3. **Fewer history frames.** `--num-history 4` roughly halves prefill FLOPs and
   image tokens. Also measure this — 8 is what the reported numbers used.
4. **Quantise the LLM** to INT4/AWQ (~4 GB saved). The VGGT tower and
   `merger_vggt` must stay FP16: the fusion is `+ 0.2 *` a geometric feature,
   and quantisation noise there corrupts the geometry signal the model relies on.
5. **Distil.** The honest long-term answer. Log (observation, action) pairs from
   the 4B model driving in Habitat plus your real environment, and train a small
   policy that fits the Nano. This is also the only path that fixes the camera
   height and Ackermann gaps properly, because the student learns *your*
   embodiment.

---

## 7. Files

| File | Runs on | Purpose |
|---|---|---|
| `futurenav_core.py` | server | `FutureNavPolicy` — `eval/agent.py` with Habitat removed, preprocessing bit-identical |
| `server.py` | server | HTTP inference server, monotonic step guard |
| `jetracer_client.py` | car | capture → rectify → request → execute loop, safety gates |
| `camera_adapter.py` | car | fisheye → Habitat-equivalent 79° pinhole remap |
| `motion.py` | car | Ackermann primitives for the 4 discrete actions |
| `calibrate_camera.py` | car | fisheye intrinsics |
| `calibrate_motion.py` | car | speed, turning radius, verification |
