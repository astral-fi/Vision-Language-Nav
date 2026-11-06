"""
Habitat-free FutureNav inference core.

This is eval/agent.py with the habitat.core.agent.Agent base class and the
Habitat observation dict removed, so the same model can be driven by a real
robot. The preprocessing must stay bit-identical to eval/agent.py: any drift in
resize/crop changes the token grid and silently degrades the policy.

Preprocessing chain, with the numbers a 640x480 frame actually produces:

  raw 640x480 RGB
    -> load_and_preprocess_images (VGGT loader, target_size=644, /14 grid)
         width 644, height round(480*644/640/14)*14 = 490   -> 3x490x644
    -> this tensor is what VGGT sees (490/14 = 35, 644/14 = 46 -> 1610 patches
       + 1 camera token + 4 register tokens = 1615 tokens per frame)
    -> crop to a multiple of patch_size*merge_size for the Qwen tower
         644//16 = 40 (even) and 490//16 = 30 (even) -> no trim at 640x480 input
    -> Qwen3VL image processor smart_resize (factor 32) -> 480x640
         tokens = (480/16 * 640/16) / 2^2 = 300 per image

  9 images (8 history + current) -> 2700 image tokens + ~80-250 text tokens.

The VGGT tower only ever runs on the CURRENT frame; earlier frames survive as a
KV cache on model.past_key_values_vggt, trimmed by StartRecentKVCache to
8 oldest + 48 most recent frames. That cache is the model's spatial memory, and
it is why this class is stateful and why steps must not be replayed or dropped.
"""

from __future__ import annotations

import copy
import base64
import time
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from qwen_vl_utils import extract_vision_info

from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
from qwen_vl.model.modeling_qwen3_vl_obshead import Qwen3VLForFutureNav

MIN_PIXELS = 28 * 28
MAX_PIXELS = 1605632

ACTION_NAME_TO_ID = {"STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3}
ACTION_ID_TO_NAME = {v: k for k, v in ACTION_NAME_TO_ID.items()}

SYSTEM_PROMPT = (
    "You are a visual language navigation model, and your should go to the locations "
    "to complete the given task. Compare the observation and instruction to infer your "
    "current progress, and then select the correct direction from the candidates to go "
    "to the target location and finish the task."
)


# --------------------------------------------------------------------------
# History-selection helpers for the E1.1 probe (experiments/e1_1).
#
# Pure functions of (history_len, n_images) so they can be tested without a
# GPU. They all pin the LAST index to the current frame, because every arm
# must see the same present -- only the past may differ.
# --------------------------------------------------------------------------

def control_indices(history_len: int, n_images: int):
    """The indices _prepare_images picks, or None when the history is too short
    for linspace to apply. Below that threshold _prepare_images returns every
    frame, so no second equally-valid history exists and probing must skip."""
    if history_len <= n_images - 1:
        return None
    return [int(i) for i in np.linspace(0, history_len, n_images, dtype=int)]


def _repair(out, hi: int):
    """Force strictly increasing, inside [0, hi], with out[-1] == hi.

    Forward max-pass then backward min-pass. Correct whenever hi >= len(out)-1,
    which control_indices returning non-None already guarantees.
    """
    n = len(out)
    out = [int(min(max(v, 0), hi)) for v in out]
    out[-1] = hi
    for i in range(1, n):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1
    out[-1] = hi
    for i in range(n - 2, -1, -1):
        if out[i] >= out[i + 1]:
            out[i] = out[i + 1] - 1
    return out


def jitter_indices(idx, rng):
    """Every history index shifted +-1 step; the current frame is untouched."""
    hi = idx[-1]
    out = [int(v) + int(rng.choice((-1, 1))) for v in idx[:-1]] + [hi]
    return _repair(out, hi)


def offset_indices(history_len: int, n_images: int):
    """linspace shifted up by half a stride -- a different but equally even
    spread over the same history. The current frame stays pinned."""
    stride = history_len / float(n_images - 1)
    out = [int(round(i * stride + stride / 2.0)) for i in range(n_images - 1)]
    return _repair(out + [history_len], history_len)


def parse_action(raw: str):
    """(action_id, fallback). Lifted out of act() so the control arm and every
    probe arm parse identically -- a divergence here would show up as a fake
    disagreement."""
    raw = (raw or "").strip()
    if raw in ACTION_NAME_TO_ID:
        return ACTION_NAME_TO_ID[raw], False
    for name, aid in ACTION_NAME_TO_ID.items():
        if name in raw:
            return aid, False
    return ACTION_NAME_TO_ID["MOVE_FORWARD"], True      # matches eval/agent.py


class FutureNavPolicy:
    """Stateful single-episode policy. Not thread-safe: the VGGT KV cache is a
    single tensor list hanging off the model, so one policy == one episode at a
    time per loaded model."""

    def __init__(self, checkpoint_path: str, device: str | None = None,
                 max_history_frames: int = 8, vggt_recent_frames: int | None = None,
                 attn_implementation: str = "eager", lam: float | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_history_frames = max_history_frames

        config = AutoConfig.from_pretrained(checkpoint_path)
        dtype = torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            # Pre-Ampere (Jetson Nano sm_53, Xavier sm_72, T4 sm_75) has no bf16.
            dtype = torch.float16

        self.model = Qwen3VLForFutureNav.from_pretrained(
            checkpoint_path,
            config=config,
            torch_dtype=dtype,
            device_map={"": self.device},
            attn_implementation=attn_implementation,
            mode="evaluation",
        ).eval()

        if lam is not None:
            # Fusion strength of the geometry tower:
            #   image_embeds = image_embeds + lam * image_embeds_3d
            # lam=0 ablates VGGT's contribution while leaving everything else
            # identical -- the cleanest way to measure what the geometry tower
            # is actually worth, and hence whether its ~14 TFLOP/step and 8.9 GB
            # are justified on constrained hardware.
            print(f"  lam override: {self.model.lam} -> {lam}")
            self.model.lam = lam

        if vggt_recent_frames is not None:
            # Trades spatial memory for VRAM: each cached frame costs
            # 24 layers * 2 (K,V) * 16 heads * 1615 tokens * 64 dim * 2 bytes
            # = ~159 MB. The default 8+48 frames is ~8.9 GB.
            self.model.kv_cache_vggt.recent_size = vggt_recent_frames
            self.model.kv_cache_vggt.cache_size = (
                self.model.kv_cache_vggt.start_size + vggt_recent_frames
            )

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, padding_side="left")
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_path, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS, padding_side="left"
        )
        self.rgb_list: list[Image.Image] = []
        self.step = 0
        self.reset()

    # ---------- diagnostics ----------

    def vggt_cache_frames(self) -> int:
        """Frames currently held in the VGGT KV cache.

        Must equal min(step, start_size + recent_size). If it grows faster than
        one per step, the vision branch is being re-entered during token decode
        and the spatial memory is being corrupted -- that is a hard bug, not a
        tuning issue.
        """
        pkv = getattr(self.model, "past_key_values_vggt", None)
        if not pkv or pkv[0] is None:
            return 0
        return int(pkv[0][0].shape[2])

    def vram(self) -> dict:
        if not torch.cuda.is_available():
            return {}
        dev = torch.device(self.model.device)
        return {"alloc_gb": round(torch.cuda.memory_allocated(dev) / 1e9, 2),
                "peak_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 2),
                "reserved_gb": round(torch.cuda.memory_reserved(dev) / 1e9, 2)}

    # ---------- history ablation (experiments/failure_detection/history_ablation) ----------
    # Defaults reproduce the normal policy exactly; set_ablation() is only called
    # when a client sends ablation keys in /reset.
    history_mode = "full"       # "full" | "shuffle" (8 history frames in random order) | "current" (no history)
                                # | "repeat" (every history slot holds the current frame; added 2026-10-25 10:55)
    vggt_reset = False          # True: drop the VGGT KV cache before every step (current frame only)
    history_seed = 0
    history_log = None          # jsonl path; one line per step with the indices shown
    episode_tag = ""

    def set_ablation(self, history_mode="full", vggt_reset=False, seed=0,
                     log=None, episode="") -> None:
        if history_mode not in ("full", "shuffle", "current", "repeat"):
            raise ValueError("history_mode must be full, shuffle, current or repeat")
        self.history_mode = history_mode
        self.vggt_reset = bool(vggt_reset)
        self.history_seed = int(seed)
        self.history_log = log or None
        self.episode_tag = str(episode or "")

    def _shown_indices(self) -> list:
        """rgb_list indices actually shown, in prompt order (last = current frame)."""
        idx = self._history_indices()
        if self.history_mode == "current":
            return idx[-1:]
        if self.history_mode == "repeat":
            # same number of image slots as "full" (prompt structure and token count unchanged),
            # every slot holding the current frame
            return [idx[-1]] * len(idx)
        if self.history_mode == "shuffle" and len(idx) > 2:
            import hashlib
            h = int(hashlib.sha256(("hist_shuffle:" + self.episode_tag).encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng([self.history_seed, h, len(self.rgb_list) - 1])
            hist = idx[:-1]
            idx = [hist[i] for i in rng.permutation(len(hist))] + idx[-1:]
        return idx

    def reset(self) -> None:
        self.rgb_list = []
        self.step = 0
        self._vggt_feats = None
        self._hidden = None
        if self._capture is not None:
            self._capture.reset()           # drops a partial episode's records
        self.model.past_key_values_vggt = None
        self.model.rope_deltas = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- hidden-state capture (experiments/state_probes) ----------

    _want_hidden = False
    _hidden = None
    _hidden_hook = None

    def enable_hidden_capture(self) -> int:
        """Register a forward hook on the text tower that stashes the final-layer
        hidden state at the LAST prompt position -- the vector the model is about
        to turn into an action word.

        A hook, not a model edit: nothing in modeling_qwen3_vl_obshead.py changes,
        and with _want_hidden False the hook returns immediately.

        Only the prefill is captured. Decode steps run one token at a time, so
        a sequence length of 1 identifies them and they are skipped.
        """
        if self._hidden_hook is not None:
            return self.model.config.text_config.hidden_size
        try:
            text_tower = self.model.model.language_model
        except AttributeError as exc:
            raise RuntimeError("cannot find the text tower to hook: %s" % exc)

        def hook(module, args, output):
            if not self._want_hidden:
                return
            h = output[0] if isinstance(output, (tuple, list)) else getattr(
                output, "last_hidden_state", None)
            if h is None or h.dim() != 3 or h.shape[1] <= 1:
                return                       # decode step, not the prefill
            self._hidden = h[0, -1, :].detach().to(torch.float16).cpu().numpy()

        self._hidden_hook = text_tower.register_forward_hook(hook)
        return int(self.model.config.text_config.hidden_size)

    # ---------- activation steering (experiments/steer_plan) ----------

    # One request per executed call, None otherwise: with it unset the hook
    # returns immediately and the forward is untouched.
    _steer = None
    _steer_hooks = None

    def _enable_steer(self, depth: int) -> None:
        """Forward hook on decoder layer `depth` (capture index: output of layer
        `depth`, so layers[depth - 1]). Prepended so any capture hook on the same
        layer, and every later layer, sees the steered residual.

        At the request's readout layer the last prompt token's pre-steer state is
        kept for the reply. At each layer with a vector, prefill adds delta to the
        last prompt token (positions "last") or to every token (positions "all",
        add only); delta = vec ("add") or vec - h ("set"). Decode tokens (generated
        action words and the forced-TURN forward) get the same delta when decode
        is set, so the second action token is steered too. mode "none" only records.
        """
        if self._steer_hooks is None:
            self._steer_hooks = {}
        if depth in self._steer_hooks:
            return
        layers = self.model.model.language_model.layers

        def hook(module, args, output):
            st = self._steer
            if st is None or (st["layer"] != depth and depth not in st["vecs"]):
                return
            h = output[0] if isinstance(output, (tuple, list)) else output
            if h is None or h.dim() != 3:
                return
            prefill = h.shape[1] > 1
            if prefill and depth == st["layer"]:
                if st["pre"] is not None:
                    raise RuntimeError("steer: a second prefill inside one steered call")
                st["pre"] = h[0, -1].float().cpu().numpy().astype(np.float16)
            if depth not in st["vecs"]:
                return
            if prefill:
                vec = st["vecs"][depth].to(h.device)
                if st["mode"] == "set":
                    st["delta"][depth] = vec - h[0, -1].float()
                else:
                    st["delta"][depth] = vec
                if st["positions"] == "all":
                    h[0] = (h[0].float() + vec).to(h.dtype)
                else:
                    h[0, -1] = (h[0, -1].float() + st["delta"][depth]).to(h.dtype)
            elif st["decode"] and depth in st["delta"]:
                h[0, -1] = (h[0, -1].float() + st["delta"][depth]).to(h.dtype)

        self._steer_hooks[depth] = layers[depth - 1].register_forward_hook(hook, prepend=True)

    def _steer_request(self, steer: dict) -> dict:
        """steer: {"layer": readout depth, "mode", "vec" (at "layer"), or "multi":
        [{"layer", "vec"}, ...] for several depths, "positions", "decode"}."""
        mode = steer.get("mode", "none")
        if mode not in ("none", "add", "set"):
            raise ValueError("steer mode must be none, add or set")
        positions = steer.get("positions", "last")
        if positions not in ("last", "all") or (positions == "all" and mode == "set"):
            raise ValueError("steer positions must be last, or all with mode add")
        depth = int(steer["layer"])
        self._enable_steer(depth)
        hid = self.model.config.text_config.hidden_size
        vecs = {}
        if mode != "none":
            items = steer.get("multi") or [{"layer": depth, "vec": steer["vec"]}]
            for it in items:
                v = torch.from_numpy(np.frombuffer(base64.b64decode(it["vec"]),
                                                   dtype=np.float32).copy())
                if v.numel() != hid:
                    raise ValueError("steer vec has %d values" % v.numel())
                vecs[int(it["layer"])] = v
                self._enable_steer(int(it["layer"]))
        return {"layer": depth, "mode": mode, "vecs": vecs, "positions": positions,
                "decode": bool(steer.get("decode", True)), "pre": None, "delta": {}}

    # ---------- full internals capture (experiments/capture500) ----------

    _capture = None

    def enable_capture(self) -> None:
        """Install the read-only hooks of deploy/capture.py. Idempotent."""
        if self._capture is None:
            from capture import Capture
            self._capture = Capture(self)

    def capture_flush(self, path: str, episode_id: str) -> dict:
        if self._capture is None:
            raise RuntimeError("capture was never enabled")
        return self._capture.flush(path, episode_id)

    def _history_indices(self) -> list:
        """The rgb_list indices _prepare_images shows, in order."""
        history_len = len(self.rgb_list) - 1
        if history_len <= self.max_history_frames:
            return list(range(history_len)) + [history_len]
        return [int(i) for i in np.linspace(0, history_len, self.max_history_frames + 1, dtype=int)]

    # ---------- identical to eval/agent.py ----------

    def _prepare_images(self) -> list[Image.Image]:
        if self.history_mode != "full":
            return [self.rgb_list[i] for i in self._shown_indices()]
        history_len = len(self.rgb_list) - 1
        if history_len <= self.max_history_frames:
            return self.rgb_list[:history_len] + [self.rgb_list[-1]]
        indices = np.linspace(0, history_len, self.max_history_frames + 1, dtype=int)
        return [self.rgb_list[i] for i in indices]

    def _call_model(self, images: list[Image.Image], instruction: str,
                    *, capture: bool = False, inject=None, collect=None) -> str:
        """capture=True stashes the merged pre-interpolate VGGT tensor on
        self._vggt_feats. inject=<tensor> reuses one instead of running the
        aggregator, so the call adds NO frame to the VGGT KV cache."""
        inputs, images_vggt, text = self.build_inputs(images, instruction)
        return self._generate(inputs, images_vggt, text, instruction,
                              capture=capture, inject=inject, collect=collect)

    def build_inputs(self, images: list[Image.Image], instruction: str):
        """Prompt + processor inputs exactly as the policy builds them (also used
        by experiments/s2_lora for training, so train and inference match).
        Returns (inputs on the model device, images_vggt list, prompt text)."""
        context = (
            "These images are your historical observations and your current observation.\n"
            f" Your task is to {instruction} \n"
            " You should take one of the following actions:\n"
            " MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
        )
        message = [{"role": "system", "content": SYSTEM_PROMPT}]
        message.append({"role": "user",
                        "content": [{"type": "image", "image": im} for im in images]
                        + [{"type": "text", "text": context}]})
        messages = [message]

        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)

        images_vggt, image_inputs = [], []
        for msg in messages:
            vision_info = extract_vision_info(msg)
            cur_images_vggt = []
            for i, ele in enumerate(vision_info):
                if "image" not in ele:
                    continue
                image = ele["image"]
                if isinstance(image, str) and "base64," in image:
                    _, b64 = image.split("base64,", 1)
                    with BytesIO(base64.b64decode(b64)) as bio:
                        image = copy.deepcopy(Image.open(bio))
                elif not isinstance(image, Image.Image):
                    raise NotImplementedError("Unsupported image type")

                image = load_and_preprocess_images([image])[0]
                if i == len(vision_info) - 1:
                    cur_images_vggt.append(image)      # VGGT gets the uncropped tensor

                _, height, width = image.shape
                if (width // patch_size) % merge_size > 0:
                    width -= (width // patch_size) % merge_size * patch_size
                if (height // patch_size) % merge_size > 0:
                    height -= (height // patch_size) % merge_size * patch_size
                image_inputs.append(image[:, :height, :width])
            images_vggt.append(torch.stack(cur_images_vggt))

        inputs = self.processor(text=text, images=image_inputs, videos=None,
                                padding=True, return_tensors="pt", do_rescale=False)
        inputs = inputs.to(self.model.device)
        return inputs, images_vggt, text

    def _generate(self, inputs, images_vggt, text, instruction, *, capture=False, inject=None, collect=None):
        self._n_image_tokens = int((inputs.input_ids == self.model.config.image_token_id).sum())
        if inject is None:
            inputs["images_vggt"] = [f.to(self.model.device) for f in images_vggt]
        else:
            # Deliberately None. If the inject branch in the model ever fails to
            # engage, forward() raises TypeError on images_vggt[i] instead of
            # quietly pushing a second frame into the KV cache for this step.
            # A loud crash beats a silently corrupted spatial memory.
            inputs["images_vggt"] = None

        if collect is not None:
            # experiments/capture500: arm the read-only hooks for this prefill.
            # Same generate() call, plus return_dict_in_generate/output_logits,
            # which only return what greedy decoding already computed.
            self._capture.begin(inputs.input_ids, text, instruction, self.tokenizer,
                                self._shown_indices(), collect.get("validate", False),
                                collect.get("profile", "full"))
        gen_kwargs = {}
        if collect is not None:
            gen_kwargs = {"return_dict_in_generate": True, "output_logits": True}
        try:
            with torch.no_grad(), self.model.vggt_feature_reuse(
                    capture=capture, inject=inject) as _reuse:
                out = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False, temperature=0, top_p=None, num_beams=1,
                    max_new_tokens=getattr(self, "max_new_tokens", 24),
                    **gen_kwargs,
                )
        except Exception:
            if collect is not None:
                self._capture.abort_step()
            raise
        if capture:
            self._vggt_feats = _reuse["tensor"]
        if collect is not None:
            forced = self._forced_turn_logits(out, inputs)
            collect["summary"] = self._capture.end(out, forced, self.vggt_cache_frames())
            out = out.sequences
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]

    def _forced_turn_logits(self, gen_out, inputs) -> torch.Tensor:
        """Second-position logits with TURN forced as the first token.

        Greedy decoding only reveals p(LEFT|TURN) on steps where the model chose
        TURN. One extra single-token forward from the prompt's KV cache, cropped
        back to the prompt, gives it on every step. Positions follow the same
        rope_deltas path as generate's own decode step; on steps where the model
        did pick TURN the result must equal generate's logits (turn_check).
        Text-only (no pixel_values), so the VGGT cache is untouched.
        """
        from capture import TOK
        L = inputs.input_ids.shape[1]
        cache = gen_out.past_key_values
        cache.crop(L)
        dev = inputs.input_ids.device
        with torch.no_grad():
            fo = self.model(input_ids=torch.tensor([[TOK["TURN"]]], device=dev),
                            attention_mask=torch.ones(1, L + 1, dtype=inputs.attention_mask.dtype, device=dev),
                            past_key_values=cache,
                            cache_position=torch.tensor([L], device=dev),
                            use_cache=True, logits_to_keep=1)
        logits = fo.logits[0, -1].detach()
        del fo, cache
        gen_out.past_key_values = None
        return logits

    # ---------- robot-facing API ----------

    @torch.no_grad()
    def advance_vggt(self, rgb: np.ndarray) -> None:
        """Take one executed step WITHOUT calling the language model
        (experiments/data_engine, dual system): the frame joins the history and
        the VGGT spatial memory exactly as act() would add it, so the next act()
        sees the same state as if the model had been called on every step.
        Mirrors the evaluation-mode aggregator call in modeling_qwen3_vl_obshead."""
        self.rgb_list.append(Image.fromarray(rgb).convert("RGB"))
        frame = load_and_preprocess_images([self.rgb_list[-1]])[0].to(self.model.device)
        m = self.model
        if m.past_key_values_vggt is None:
            m.past_key_values_vggt = [None] * m.vggt.aggregator.depth
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype):
            out = m.vggt.aggregator(frame.unsqueeze(0).unsqueeze(0), past_key_values=m.past_key_values_vggt,
                                    use_cache=True, past_frame_idx=0)
        m.past_key_values_vggt = m.kv_cache_vggt(out[2])
        self.step += 1

    def act(self, rgb: np.ndarray, instruction: str, probes=None,
            probe_min_step: int = 9, probe_seed: int = 0,
            want_hidden: bool = False, capture: bool = False,
            capture_validate: bool = False, capture_profile: str = "full",
            steer: dict | None = None) -> dict:
        """rgb: HxWx3 uint8 RGB (Habitat-equivalent view). Appends the frame to
        history AND to the VGGT cache, so call it exactly once per executed step.

        probes: optional list of arm specs (experiments/e1_1). They re-query the
        model on this same frozen step with different history selections, reusing
        the geometry features this call computed, so they add NO cache frames.

        steer: optional experiments/steer_plan request {"layer", "mode", "vec",
        "decode"}; applies to this executed call only, never to probe arms.
        """
        st = self._steer_request(steer) if steer else None
        self.rgb_list.append(Image.fromarray(rgb).convert("RGB"))
        # Cleared before the call, never after: a probe must never be able to run
        # on the previous step's geometry if this call fails to capture.
        self._vggt_feats = None
        # Armed only around the control call, so a probe arm can never overwrite
        # the state that belongs to the executed decision.
        self._hidden = None
        self._want_hidden = bool(want_hidden)
        if self.vggt_reset:
            # history ablation: the geometry tower sees the current frame alone
            self.model.past_key_values_vggt = None
        collect = None
        if capture:
            self.enable_capture()
            collect = {"validate": bool(capture_validate), "profile": capture_profile}
        self._steer = st
        try:
            raw = self._call_model(self._prepare_images(), instruction,
                                   capture=bool(probes), collect=collect).strip()
        finally:
            self._want_hidden = False
            self._steer = None
        action_id, fallback = parse_action(raw)

        self.step += 1
        cache_frames = self.vggt_cache_frames()
        expected = min(self.step, self.model.kv_cache_vggt.cache_size)
        if self.vggt_reset:
            expected = 1
        if self.history_log:
            import json as _json
            with open(self.history_log, "a") as fh:
                fh.write(_json.dumps({"episode": self.episode_tag, "step": self.step - 1,
                                      "mode": self.history_mode, "vggt_reset": self.vggt_reset,
                                      "shown": self._shown_indices(),
                                      "n_image_tokens": self._n_image_tokens,
                                      "cache_frames": cache_frames, "action": ACTION_ID_TO_NAME[action_id]}) + "\n")
        result = {"action": action_id, "action_name": ACTION_ID_TO_NAME[action_id],
                  "raw": raw, "fallback": fallback, "step": self.step,
                  "vggt_cache_frames": cache_frames,
                  "vggt_cache_ok": cache_frames == expected,
                  "vram": self.vram()}
        if want_hidden:
            if self._hidden is None:
                result["hidden_error"] = ("no hidden state captured; "
                                          "call enable_hidden_capture() first")
            else:
                result["hidden"] = base64.b64encode(self._hidden.tobytes()).decode()
                result["hidden_dim"] = int(self._hidden.shape[0])
                result["hidden_dtype"] = "float16"
        if collect is not None:
            result["capture"] = collect.get("summary")
        if st is not None:
            if st["pre"] is None:
                raise RuntimeError("steer: layer %d hook never saw the prefill" % st["layer"])
            result["steer"] = {
                "layer": st["layer"], "mode": st["mode"], "decode": st["decode"],
                "pre": base64.b64encode(st["pre"].tobytes()).decode(),
                "norm_pre": round(float(np.linalg.norm(st["pre"].astype(np.float32))), 3),
                "norm_delta": round(float(sum(float(d.norm()) for d in st["delta"].values())), 3),
                "n_layers": len(st["vecs"]), "positions": st["positions"]}
        if probes:
            result["probe"] = self._run_probes(probes, instruction,
                                               probe_min_step, probe_seed)
        return result

    # ---------- E1.1 probe (experiments/e1_1) ----------

    def _run_probes(self, specs, instruction: str, min_step: int, seed: int) -> dict:
        """Re-query the model on the current frozen step, once per arm.

        Every arm injects the geometry tensor captured by the control call, so
        the VGGT aggregator does not run and past_key_values_vggt does not grow.
        cache_frames_before/after bracket the whole block as a hard check.
        """
        t = len(self.rgb_list) - 1
        n_img = self.max_history_frames + 1
        ctl = control_indices(t, n_img)
        if ctl is None or t < min_step:
            return {"ran": False,
                    "reason": "history_len=%d below the probe floor (%d)"
                              % (t, max(min_step, n_img))}
        if self._vggt_feats is None:
            return {"ran": False, "reason": "no captured VGGT features"}

        before = self.vggt_cache_frames()
        rng = np.random.default_rng([int(seed), int(t)])
        out = {"ran": True, "control_indices": list(ctl),
               "vggt_feat_shape": list(self._vggt_feats.shape), "arms": []}

        for spec in specs:
            name = spec.get("name", "?")
            try:
                kind = spec.get("history", "control")
                if kind == "control":
                    idx = list(ctl)
                elif kind == "jitter":
                    idx = jitter_indices(ctl, rng)
                elif kind == "offset":
                    idx = offset_indices(t, n_img)
                else:
                    raise ValueError("unknown history variant %r" % (kind,))

                arm = {"name": name, "history": kind, "indices": idx,
                       "degenerate": bool(kind != "control" and idx == list(ctl)),
                       "keeps_frame0": bool(idx[0] == 0)}
                images = [self.rgb_list[i] for i in idx]

                swap = spec.get("swap_current_from")
                if swap is not None:
                    # Qwen tower only: geometry still comes from the true current
                    # frame, because the injected tensor is not recomputed.
                    j = max(0, t - int(swap))
                    images = images[:-1] + [self.rgb_list[j]]
                    arm["frame_swapped"] = j

                instr = spec.get("instruction") or instruction
                if spec.get("donor_id") is not None:
                    arm["donor_id"] = str(spec["donor_id"])
                if spec.get("donor_traj") is not None:
                    arm["donor_traj"] = spec["donor_traj"]

                t0 = time.time()
                raw = self._call_model(images, instr,
                                       inject=self._vggt_feats).strip()
                arm["latency_s"] = round(time.time() - t0, 3)
                aid, fb = parse_action(raw)
                arm.update(action=aid, action_name=ACTION_ID_TO_NAME[aid],
                           raw=raw, fallback=fb)
            except Exception as exc:
                # Never propagate: by now the real step has already advanced the
                # cache, and killing the episode over a diagnostic loses the
                # whole trajectory. The gap is recorded and gated at analysis.
                arm = {"name": name, "error": "%s: %s" % (type(exc).__name__, exc)}
            out["arms"].append(arm)

        after = self.vggt_cache_frames()
        out["cache_frames_before"] = before
        out["cache_frames_after"] = after
        out["cache_ok"] = bool(before == after)
        return out
