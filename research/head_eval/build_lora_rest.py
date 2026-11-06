"""Which subset episodes still need a LoRA run? FutureNav already has all 1,407 stair-free episodes
(base500 + capture_pool); LoRA has lora_v1_C_val300. Writes ids_lora_rest.txt = subset minus val300."""
import json
from pathlib import Path

HERE = Path(__file__).parent
R2R = HERE.parents[1] / "outputs/eval/r2r"
sub = [l.strip() for l in open(HERE / "ids_sf400.txt") if l.strip()]
done = lambda tag: {str(r["id"]) for r in json.load(open(R2R / tag / "val_unseen/summary_split_0.json"))["results"]}
fn = done("base500") | done("capture_pool"); lora = done("lora_v1_C_val300")
assert set(sub) <= fn, "FutureNav is missing subset episodes"
rest = [i for i in sub if i not in lora]
(HERE / "ids_lora_rest.txt").write_text("\n".join(rest) + "\n")
print("subset", len(sub), "| FutureNav reused", len(set(sub) & fn), "| LoRA reused", len(set(sub) & lora), "| LoRA to run", len(rest))
