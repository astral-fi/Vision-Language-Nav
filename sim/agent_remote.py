"""
Habitat-side agent that delegates inference to deploy/server.py over HTTP.

Why this exists: habitat-sim 0.1.7 has no build newer than Python 3.8, and
Qwen3-VL needs transformers >= 4.57, which requires Python >= 3.9. The two
cannot share an interpreter. So Habitat runs in a py3.8 environment and calls
the model, which runs in a py3.10 environment, through exactly the protocol the
JetRacer uses.

That is not just a workaround. It means the sim evaluation exercises the same
serving path as the robot, so a number measured here is a number about the code
that will actually drive the car.

This module must stay importable on Python 3.8: stdlib + numpy + PIL only.
"""

import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np

from habitat.core.agent import Agent
from PIL import Image

ACTION_NAME_TO_ENV = {"STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3}


class FutureNavRemoteAgent(Agent):
    def __init__(self, server, result_path=None, timeout=180.0, retries=2,
                 probe_spec=None, want_hidden=False, capture=False,
                 capture_validate_every=0, capture_profile="full", **_ignored):
        self.server = server.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.session = None
        self.step = 0
        self.instruction = None
        self.latencies = []

        # E1.1 probe state (experiments/e1_1). None for an ordinary run.
        self.probe_spec = probe_spec
        self.probe_arms = None
        self.step_arms = None      # E2.3: per-step arms set by run.py
        self.probe_seed = 0
        self.last_probe = None
        self.episode_id = None
        self.probe_steps = 0
        self.probe_arm_errors = 0

        # Hidden-state capture (experiments/state_probes). None for a plain run.
        self.want_hidden = bool(want_hidden)
        self.last_hidden = None

        # experiments/steer_plan: per-step request set by run.py, reply kept here.
        self.step_steer = None
        self.last_steer = None
        self.last_final = None

        # Full internals capture (experiments/capture500). Server-side: the
        # records stay on the server and are written by flush_capture().
        self.capture = bool(capture)
        self.capture_validate_every = int(capture_validate_every or 0)
        self.capture_profile = capture_profile
        self.capture_issues = 0

        health = self._get("/health")
        if not health.get("ready"):
            raise RuntimeError(f"server at {self.server} is not ready: {health}")
        if probe_spec is not None and not health.get("probes_supported"):
            # Fail in the first second rather than the eleventh hour: an
            # unpatched server ignores the "probes" key and returns a perfectly
            # valid 200, so the whole run would come back probe-free and only
            # be caught at analysis time.
            raise RuntimeError(
                "a probe spec was given but the server at %s does not advertise "
                "probes_supported -- it is running an unpatched deploy/server.py "
                "and would silently return probe-free results." % self.server)
        if self.capture and not health.get("capture_supported"):
            raise RuntimeError(
                "capture was requested but the server at %s does not advertise "
                "capture_supported -- restart it with the patched deploy/server.py."
                % self.server)
        if self.want_hidden and not health.get("hidden_supported"):
            raise RuntimeError(
                "hidden-state capture was requested but the server at %s does not "
                "advertise hidden_supported -- it is running an unpatched "
                "deploy/server.py and would return no states at all." % self.server)
        print("=" * 60)
        print("FutureNav remote agent")
        print("  server: %s" % self.server)
        if probe_spec is not None:
            print("  probes: %s" % ", ".join(a["name"] for a in probe_spec["arms"]))
            print("  min_step: %s" % probe_spec.get("min_step", 9))
        if self.want_hidden:
            print("  hidden: capturing final-layer state at the last prompt position")
        if self.capture:
            print("  capture: profile %s, validating every %s steps"
                  % (self.capture_profile, self.capture_validate_every or "no"))
        print("=" * 60)

    # ---------- transport ----------

    def _get(self, path):
        req = urllib.request.Request(self.server + path)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())

    def _post(self, path, payload):
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.server + path, data=body,
            headers={"Content-Type": "application/json"},
        )
        last = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                # 409 = step out of order; retrying would corrupt the VGGT cache.
                # 500 = the step itself raised server-side (usually CUDA OOM).
                # Either way, fail loudly rather than retry.
                detail = exc.read().decode(errors="replace")
                if exc.code == 500 and '"oom": true' in detail.lower():
                    raise RuntimeError(
                        "server hit CUDA OOM at step %s. The VGGT cache saturates at "
                        "56 frames (~8.9 GB); restart the server with "
                        "--vggt-recent-frames 24 and close other GPU users.\n%s"
                        % (payload.get("step"), detail))
                raise RuntimeError("server rejected %s (%s): %s" % (path, exc.code, detail))
            except Exception as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(1.0 + attempt)
        raise RuntimeError("server unreachable at %s%s: %s" % (self.server, path, last))

    # ---------- E1.1 probe (experiments/e1_1) ----------

    def set_episode(self, episode_id):
        """Resolve this episode's probe arms. run.py calls this before reset()
        behind a hasattr() guard, so eval/agent.py stays untouched."""
        self.episode_id = str(episode_id)
        self.step_arms = None
        if self.probe_spec is None:
            return
        if self.probe_spec.get("pointer"):
            # E2.3: arms are rebuilt every step by run.py from the agent's place
            # in the instruction and handed over through step_arms.
            self.probe_arms = []
            self.probe_seed = 0
            return
        swap = self.probe_spec.get("instruction_swap", {}).get(self.episode_id)
        if swap is None:
            raise RuntimeError(
                "no pos_instr donor for episode %s in the probe spec; "
                "re-run experiments/e1_1/select.py" % self.episode_id)
        arms = []
        for a in self.probe_spec["arms"]:
            a = dict(a)
            if a.get("instruction") == "@swap":
                a["instruction"] = swap["text"]
                a["donor_id"] = swap["donor_id"]
                a["donor_traj"] = swap["donor_traj"]
            arms.append(a)
        self.probe_arms = arms
        # sha256, not hash(): hash() of a str is salted per process and would
        # not reproduce across runs.
        digest = hashlib.sha256(("e1.1:" + self.episode_id).encode()).hexdigest()
        self.probe_seed = int(digest[:8], 16)

    def _arms_for_step(self):
        """The arms to send this step. control_repeat is sampled rather than run
        every step: injection is deterministic, so periodic sampling across all
        cache depths finds a fault just as surely at a fraction of the cost."""
        arms = self.step_arms if self.step_arms is not None else self.probe_arms
        if not arms:
            return None
        every = self.probe_spec.get("control_repeat_every", 0)
        out = []
        for a in arms:
            if a["name"] == "control_repeat" and every and self.step % every:
                continue
            out.append(a)
        return out or None

    def flush_capture(self, episode_id, path):
        """Have the server write this episode's capture to <path>/<id>.npz.
        Call before stats_<id>.json so resumability never skips a gap."""
        return self._post("/capture_flush", {"session": self.session,
                                             "episode_id": str(episode_id),
                                             "path": path})

    # ---------- Agent API ----------

    def reset(self):
        self.session = "habitat-%d" % int(time.time() * 1000)
        self.step = 0
        self.instruction = None
        # Count actions directly. Inferring them as path_length / 0.25 only works
        # when turns are pivots; under Ackermann every turn also advances the
        # vehicle ~0.17 m, so that estimate silently under-reports turns by
        # roughly 40%. Recording what was actually commanded keeps turn fraction
        # exact and comparable across every action space.
        self.action_counts = {"STOP": 0, "MOVE_FORWARD": 0,
                              "TURN_LEFT": 0, "TURN_RIGHT": 0}
        self.last_probe = None
        self.last_hidden = None
        self.capture_issues = 0
        self.probe_steps = 0
        self.probe_arm_errors = 0
        payload = {"session": self.session, "instruction": ""}
        # experiments/failure_detection/history_ablation: opt-in via env vars, default off.
        #   FUTURENAV_HISTORY_MODE  full | shuffle | current | repeat
        #   FUTURENAV_VGGT_RESET    1 = VGGT sees the current frame only (cache dropped each step)
        #   FUTURENAV_HISTORY_SEED  int, shuffle seed;  FUTURENAV_HISTORY_LOG  server-side jsonl path
        abl = {k: os.environ.get(v) for k, v in (
            ("history_mode", "FUTURENAV_HISTORY_MODE"), ("vggt_reset", "FUTURENAV_VGGT_RESET"),
            ("history_seed", "FUTURENAV_HISTORY_SEED"), ("history_log", "FUTURENAV_HISTORY_LOG"))}
        if any(abl.values()):
            payload.update(history_mode=abl["history_mode"] or "full",
                           vggt_reset=abl["vggt_reset"] not in (None, "", "0"),
                           history_seed=int(abl["history_seed"] or 0),
                           history_log=abl["history_log"], episode_id=self.episode_id or "")
        self._post("/reset", payload)

    def act(self, observations, info=None, episode_id=None):
        if self.session is None:
            self.reset()

        # Read on every step: run.py --instruction-override can change the text mid-episode.
        # The server stores the instruction at /reset, but this episode's text is only
        # known once the first observation arrives, so it is sent inline.
        self.instruction = observations["instruction"]["text"]

        buf = io.BytesIO()
        Image.fromarray(observations["rgb"]).convert("RGB").save(
            buf, format="PNG"  # lossless: sim frames should not pick up JPEG artifacts
        )
        payload = {
            "session": self.session,
            "step": self.step,
            "instruction": self.instruction,
            "image": base64.b64encode(buf.getvalue()).decode(),
        }
        if self.want_hidden:
            payload["want_hidden"] = True
        if self.capture:
            payload["capture"] = True
            payload["capture_profile"] = self.capture_profile
            every = self.capture_validate_every
            payload["capture_validate"] = bool(every) and self.step % every == 0
        if self.step_steer:
            payload["steer"] = self.step_steer
        # deploy/head_server.py (motion head) needs the agent pose; server.py ignores it.
        if getattr(self, "step_pose", None) is not None:
            payload["pose"] = self.step_pose
        # deploy/head_eval_server.py --exec plan: eval/s1_local.py asks for a System 2 plan only on its triggers
        if getattr(self, "step_query", False):
            payload["query"] = True
        arms = self._arms_for_step()
        if arms:
            payload["probes"] = arms
            payload["probe_min_step"] = self.probe_spec.get("min_step", 9)
            payload["probe_seed"] = self.probe_seed
        t0 = time.time()
        resp = self._post("/step", payload)
        self.latencies.append(time.time() - t0)

        if not resp.get("vggt_cache_ok", True):
            raise RuntimeError(
                "VGGT cache desync at step %d: server holds %s frames. "
                "Results from this point are invalid."
                % (self.step, resp.get("vggt_cache_frames"))
            )

        if self.want_hidden:
            blob = resp.get("hidden")
            if not blob:
                raise RuntimeError(
                    "no hidden state at step %d: %s. Every step must have one or "
                    "the probe training set has silent gaps."
                    % (self.step, resp.get("hidden_error", "server returned none")))
            self.last_hidden = np.frombuffer(
                base64.b64decode(blob), dtype=np.float16).copy()

        if self.capture:
            cap = resp.get("capture")
            if not cap:
                raise RuntimeError("no capture record at step %d; the episode would "
                                   "have a silent gap" % self.step)
            if not cap.get("layout_ok", False):
                self.capture_issues += 1

        self.last_head = resp.get("head")
        self.last_steer = resp.get("steer")
        self.last_final = None
        if self.step_steer:
            if not self.last_steer:
                raise RuntimeError("steer requested at step %d but the server returned none; "
                                   "is deploy/server.py up to date?" % self.step)
            if self.step_steer.get("final"):
                self.last_final = np.frombuffer(base64.b64decode(resp["hidden"]),
                                                dtype=np.float16).copy()
        self.last_probe = resp.get("probe")
        pr = self.last_probe
        if pr and pr.get("ran"):
            if not pr.get("cache_ok", True):
                raise RuntimeError(
                    "VGGT cache advanced during probes at step %d (%s -> %s). "
                    "Feature reuse did not engage, so every later decision in "
                    "this episode is invalid."
                    % (self.step, pr.get("cache_frames_before"),
                       pr.get("cache_frames_after")))
            self.probe_steps += 1
            self.probe_arm_errors += sum(1 for a in pr.get("arms", [])
                                         if "error" in a)

        self.step += 1
        name = resp["action_name"]
        if name in self.action_counts:
            self.action_counts[name] += 1
        out = {"action": ACTION_NAME_TO_ENV.get(name, 1)}
        # deploy/head_eval_server.py --exec continuous: a pose increment for eval/waypoint_exec.py
        if resp.get("motion") is not None:
            out["motion"] = resp["motion"]
        if resp.get("plan") is not None:
            out["plan"] = resp["plan"]
        return out
