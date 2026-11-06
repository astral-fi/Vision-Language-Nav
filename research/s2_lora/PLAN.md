# LoRA fine-tune of FutureNav-4B to output future steps (plan v2, 2026-11-01)

## Goal

FutureNav (System 2) answers with the next K = 8 actions instead of one, so the route
choice and the number of turns come from the model that reads the instruction.
Motivation (motion-head failure analysis, memory motion-head-bench): the head's losses
were route choice at the first decision (often FutureNav was wrong too) and stopping
turns too early. Camera stays at 1.25 m; all data comes from expert_v1.

Success bar, same 100 holdout episodes: frozen FutureNav 0.70 (clean) / 0.61 (perturbed).

## Decisions so far

| Item | Choice |
|---|---|
| Output | next 8 expert actions as action words, cut at STOP |
| Inputs | identical to inference: same prompt, 8 linspace history frames + current |
| VGGT | training matches inference: precomputed streaming features, injected |
| Trainable | LoRA on the language model only; vision, VGGT, merger frozen |
| Data | expert_v1 train buildings, ~25k sampled steps |
| Camera | existing 1.25 m renders |

## 1. Target format

    TURN_LEFT TURN_LEFT TURN_LEFT MOVE_FORWARD MOVE_FORWARD MOVE_FORWARD MOVE_FORWARD MOVE_FORWARD
    MOVE_FORWARD MOVE_FORWARD STOP

- Labels: expert_v1 `chunk_act[t, :8]` (0 STOP, 1 FORWARD, 2 LEFT, 3 RIGHT), truncated after STOP.
- Same words FutureNav already emits; no new tokens; ~25 answer tokens.
- The first word equals the old single-action answer, so old decoding still works.

## 2. Inputs (must equal inference)

- Prompt: the template in `deploy/futurenav_core.py` (system text line 52,
  "Your task is to {instruction}", `apply_chat_template`), built by importing that code,
  not copying it.
- Qwen images: `control_indices` (8 linspace history frames + current) from the expert_v1
  JPEG blobs; fewer images early in an episode, as at inference.
- VGGT: the current frame's merged tokens from the streaming cache (section 3).

## 3. VGGT features: training matches inference

At inference only the current frame goes through VGGT, attending to every earlier frame
in the streaming cache (first 8 + last 48), advanced every step. Training normally runs
VGGT on the 9 shown frames with a fresh cache. To match inference exactly:

- `precompute_vggt.py`: per chosen episode, stream every frame up to its last sampled
  step through VGGT as `advance_vggt` does; at each sampled step save `merger_vggt`'s output
  for the current frame, the merged PRE-interpolate [391, 2560] fp16 tensor (~2 MB).
- Training injects it with the existing `vggt_feature_reuse(inject=...)` hook
  (`modeling_qwen3_vl_obshead.py:117`). VGGT never runs in training (faster, less memory).
- `merger_vggt` stays frozen so the stored tensors stay valid.

## 4. Data

- Source: expert_v1 (9,744 episodes, ~835k steps). Exclude the 6 holdout buildings
  (split.json); hold out the motion-head val trajectories (`data.py:is_val`) as the
  offline validation set.
- Sample ~25k training steps from ~2,500 episodes (~10 per episode):
  decision steps 2:1 over straight cruising. Decision = a turn in the 8-step chunk,
  inside a perturbation, > 0.5 m off-route, < 2 m from the goal, or the first 15 steps.
- Validation: ~1,000 steps from the val trajectories, same rule.
- Disk: ~26k x 2 MB = ~52 GB of 89 GB free.
- GPU: streaming ~2,600 episodes up to their last sampled step, ~180k frames at ~0.13 s
  = ~6.5 h. Two processes in parallel if memory allows (VGGT + cache only, no LLM).
- Annotation file in the repo's format (`conversations`, `image`, `id = <episode>/step_<t>`)
  plus a pointer to the feature file; frames read from the blobs by a dataset subclass.

## 5. Training

- Install `peft` in the futurenav env (only missing piece; flash-attn absent, use sdpa).
- Changes to `src/qwen_vl/train/`: wrap the language model with LoRA; dataset subclass
  returns frames from blobs + the VGGT tensor; collator passes it; a Trainer subclass
  runs each forward inside `vggt_feature_reuse(inject=...)`.
- LoRA: q,k,v,o,gate,up,down in every LLM layer, r 16, alpha 32, dropout 0.05.
- Loss: LM cross-entropy on answer tokens only (the released model has aux heads off).
- bf16, gradient checkpointing, batch 1 x grad accumulation 16, lr 1e-4 cosine,
  3% warmup, 1 epoch (~1,560 optimizer steps), checkpoint every 250 steps.
- Expected ~1-1.5 s/sample without VGGT: ~7-10 h for 25k. Smoke test first.

## 6. Steps and gates

| # | Step | Time | Gate |
|---|---|---|---|
| 0 | install peft; load model with LoRA; one forward/backward | 0.5 h | runs in 24 GB |
| 1 | `build_samples.py`: sample steps, write annotations | 1-2 h dev | counts and mix as planned |
| 2 | `precompute_vggt.py` on 20 s2_v1 episodes | 0.5 h | feeds the parity test |
| 3 | **parity test**: frozen model, training data path + injected tensor vs live server action (s2_v1) | 1 h | first action agrees ~100% |
| 4 | full precompute (train + val) | ~6.5 h GPU | all files written, sizes right |
| 5 | training code + 20-step smoke test | 2-3 h dev | loss falls, memory and s/sample known |
| 6 | train | ~7-10 h GPU | val loss falls; checkpoints written |
| 7 | offline eval on val | 1 h | beats frozen FutureNav on the metrics below |
| 8 | server: chunk decoding + execution modes | 2 h dev | plumbing test passes |
| 9 | closed loop: 100 holdout clean + perturbed, mode A then B | ~5-8 h GPU | vs 0.70 / 0.61 |
| 10 | val_unseen check (val300) | ~3 h GPU | no drop vs base500 |

Steps 1, 5 and 8 are code and can be written while steps 2-4 use the GPU.

## 7. Evaluation

Offline (val steps, frozen FutureNav as reference):
- first-action accuracy vs the expert
- end-pose error after the 8 steps (position + heading): tolerant to turn timing
- rotation phases: |predicted turns - expert turns| when the expert turns in place
- turn side at route bends

Closed loop (traces on, so `trace_analysis.py` gives departures and turn counts):
- Mode A: query every step, execute the first action (is the decision better?)
- Mode B: query every 4 steps, execute the first 4 actions (4x fewer calls)
- Unparseable answers fall back to the first valid action word; count them.

## 8. Risks

- Expert timing is not fully inferable: judged by end pose, not exact tokens.
- Overfitting to 55 train buildings: the 6 holdout buildings measure it.
- Forgetting general VLN without FutureNav's original data: LoRA, lr 1e-4, val_unseen check.
- Decoding 25 tokens instead of 1 adds ~0.5 s per call; mode B amortises it.
- If the gain is small: K = 16, more samples (disk), or LoRA on the vision side.

## 9. Timeline (~3 days)

- Day 1: steps 0-3, start step 4 overnight.
- Day 2: step 5, then train (step 6) overnight.
- Day 3: steps 7-10.
