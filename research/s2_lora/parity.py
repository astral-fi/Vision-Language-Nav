"""Gate 3 of PLAN.md: the training input path reproduces the live server.

For cached s2_v1 episodes (actions from the live policy, dense streaming VGGT,
a call every 4 steps), rebuild every call the way training will:
  frames from expert_v1, StreamVGGT for the current frame's merged tensor,
  FutureNavPolicy.build_inputs for the prompt, generation with the tensor injected.
and compare the parsed first action with the cached live action.

--check-capture also runs the real policy path on the first episode and compares
StreamVGGT's tensor with what vggt_feature_reuse(capture=True) captures.

    python parity.py --episodes 20 --check-capture
"""
import argparse
import glob
import os

import numpy as np
import torch

from common import ROOT, Episode, StreamVGGT, episode_path, load_policy

S2 = os.path.join(ROOT, "outputs/data_engine/s2_v1")
R = 4


def captured_tensors(policy, ep, steps):
    """Merged VGGT tensors from the real policy path at the given call steps."""
    from futurenav_core import parse_action  # noqa: F401  (same module as the server)
    policy.reset()
    out = {}
    for t in range(max(steps) + 1):
        img = ep.frame(t)
        if t % R:
            policy.advance_vggt(np.asarray(img))
            continue
        policy.rgb_list.append(img)
        policy._call_model(policy._prepare_images(), ep.meta["instruction"], capture=True)
        policy.step += 1
        out[t] = policy._vggt_feats.float().cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--check-capture", action="store_true")
    a = ap.parse_args()
    from futurenav_core import parse_action
    policy = load_policy()
    stream = StreamVGGT(policy.model)
    files = sorted(glob.glob(os.path.join(S2, "*", "*.npz")))[:a.episodes]
    agree = n = 0
    for k, f in enumerate(files):
        s = np.load(f)
        scene, eid = f.split("/")[-2], os.path.basename(f)[:-4]
        ep = Episode(episode_path(scene, eid))
        live = dict(zip(s["t"].tolist(), s["action"].tolist()))
        stream.reset()
        mine, feats = {}, {}
        for t in range(max(live) + 1):
            x = stream.step(ep.frame(t), want=t in live)
            if x is None:
                continue
            feats[t] = x.float().cpu()
            inputs, _, text = policy.build_inputs(ep.shown(t), ep.meta["instruction"])
            inputs["images_vggt"] = None
            with torch.no_grad(), policy.model.vggt_feature_reuse(inject=x):
                out = policy.model.generate(**inputs, eos_token_id=policy.tokenizer.eos_token_id,
                                            pad_token_id=policy.tokenizer.pad_token_id, do_sample=False,
                                            temperature=0, top_p=None, num_beams=1, max_new_tokens=24)
            raw = policy.processor.batch_decode([out[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0]
            mine[t] = parse_action(raw)[0]
        ag = sum(mine[t] == live[t] for t in live)
        agree += ag
        n += len(live)
        print("%s/%s: %d/%d agree" % (scene, eid, ag, len(live)), flush=True)
        if a.check_capture and k == 0:
            cap = captured_tensors(policy, ep, sorted(live))
            d = max(float((cap[t] - feats[t]).abs().max()) for t in cap)
            print("capture check on %s: max |diff| %.3g over %d calls (scale %.3g)"
                  % (eid, d, len(cap), float(cap[0].abs().mean())), flush=True)
            policy.model.past_key_values_vggt = None
    print("PARITY: %d/%d = %.4f first actions agree with the live server" % (agree, n, agree / n))


if __name__ == "__main__":
    main()
