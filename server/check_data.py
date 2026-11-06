"""
Verify the data layout before spending an hour discovering it is wrong.

  python deploy/check_data.py            # R2R
  python deploy/check_data.py --split rxr

Runs in either environment: stdlib only.
"""

import argparse
import gzip
import json
import os
import sys

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OK, BAD = "  [ok]    ", "  [FAIL]  "


def check(label, path, note=""):
    exists = os.path.exists(path)
    print(("%s%s" % (OK if exists else BAD, label)).ljust(58), path)
    if not exists and note:
        print("         -> %s" % note)
    return exists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="r2r", choices=["r2r", "rxr"])
    ap.add_argument("--checkpoint", default="pretrain_models/FutureNav-4B-Base")
    a = ap.parse_args()
    os.chdir(PROJECT)
    ok = True

    print("\n--- scenes (Matterport3D) ---")
    scenes_dir = "data/scene_datasets/mp3d"
    if check("mp3d scene root", scenes_dir,
             "Gated dataset. Sign the Terms of Use at "
             "https://niessner.github.io/Matterport/ to be emailed download_mp.py, "
             "then: python2 download_mp.py --task habitat -o data/scene_datasets/"):
        ids = sorted(d for d in os.listdir(scenes_dir)
                     if os.path.isdir(os.path.join(scenes_dir, d)))
        glbs = [i for i in ids if os.path.exists(os.path.join(scenes_dir, i, i + ".glb"))]
        print("         %d scene dirs, %d with a .glb" % (len(ids), len(glbs)))
        print("         (R2R val_unseen needs only 11 of the 90 scenes -- see below)")
        if glbs:
            navmesh = os.path.join(scenes_dir, glbs[0], glbs[0] + ".navmesh")
            check("navmesh present (sampled)", navmesh,
                  "the habitat task download includes .navmesh; a raw mp3d "
                  "download does not, and the sim cannot path without it")
    else:
        ok = False

    print("\n--- episodes ---")
    if a.split == "r2r":
        base = "data/datasets/R2R_VLNCE_v1-3_preprocessed"
        files = [("%s/val_unseen/val_unseen.json.gz" % base),
                 ("%s/val_seen/val_seen.json.gz" % base)]
    else:
        base = "data/datasets/RxR_VLNCE_v0"
        files = [("%s/val_unseen/val_unseen_guide.json.gz" % base),
                 ("%s/val_unseen/val_unseen_guide_gt.json.gz" % base),
                 ("%s/val_seen/val_seen_guide.json.gz" % base),
                 ("%s/val_seen/val_seen_guide_gt.json.gz" % base)]
    for f in files:
        if not check(os.path.basename(f), f, "run: bash deploy/fetch_episodes.sh"):
            ok = False

    target = files[0]
    if os.path.exists(target):
        with gzip.open(target, "rt") as fh:
            eps = json.load(fh).get("episodes", [])
        needed = sorted({e["scene_id"].split("/")[-2] for e in eps}) if eps else []
        print("         %d episodes referencing %d scenes" % (len(eps), len(needed)))
        if not os.path.isdir(scenes_dir):
            print("\n         Only these scenes are required for this split. download_mp.py\n"
                  "         takes --id, so you can fetch just them instead of all 90:\n")
            for s_id in needed:
                print("           python2 download_mp.py --task habitat --id %s -o data/scene_datasets/" % s_id)
        if os.path.isdir(scenes_dir):
            have = set(os.listdir(scenes_dir))
            missing = [s for s in needed if s not in have]
            if missing:
                ok = False
                print("%smissing %d scenes needed by this split, e.g. %s"
                      % (BAD, len(missing), ", ".join(missing[:5])))
            else:
                print("%severy scene this split needs is present" % OK)

    print("\n--- checkpoint ---")
    if check("checkpoint dir", a.checkpoint, "run: bash deploy/fetch_checkpoint.sh"):
        for f in ["config.json", "model.safetensors.index.json", "tokenizer_config.json",
                  "preprocessor_config.json", "chat_template.json"]:
            if not check(f, os.path.join(a.checkpoint, f)):
                ok = False
        shards = [f for f in os.listdir(a.checkpoint) if f.endswith(".safetensors")]
        total = sum(os.path.getsize(os.path.join(a.checkpoint, f)) for f in shards)
        print("         %d shards, %.2f GB" % (len(shards), total / 1e9))
        if abs(total - 11.74e9) > 0.5e9:
            ok = False
            print("%sexpected ~11.74 GB; the download looks incomplete" % BAD)
    else:
        ok = False

    print("\n%s" % ("ALL CHECKS PASSED" if ok else "SOMETHING IS MISSING -- see [FAIL] lines above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
