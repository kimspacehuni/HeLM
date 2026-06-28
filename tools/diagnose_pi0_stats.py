"""Inspect where the pi0 ckpt keeps its normalization stats."""
import json
import os
import sys

ckpt = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/ckpt/pi0_base")
print(f"ckpt: {ckpt}\n")

# 1) policy_preprocessor.json — usually holds per-feature mean/std
prep_path = os.path.join(ckpt, "policy_preprocessor.json")
print(f"=== {prep_path} ===")
if os.path.exists(prep_path):
    with open(prep_path) as f:
        prep = json.load(f)
    s = json.dumps(prep, indent=2)
    print(s if len(s) < 4000 else s[:4000] + "\n... (truncated)")
else:
    print("(missing)")
print()

# 2) policy_postprocessor.json — output un-normalization
post_path = os.path.join(ckpt, "policy_postprocessor.json")
print(f"=== {post_path} ===")
if os.path.exists(post_path):
    with open(post_path) as f:
        post = json.load(f)
    s = json.dumps(post, indent=2)
    print(s if len(s) < 4000 else s[:4000] + "\n... (truncated)")
else:
    print("(missing)")
print()

# 3) safetensors keys — check whether buffers are stored alongside model weights
print("=== model.safetensors keys ===")
try:
    from safetensors import safe_open
    safetensors_path = os.path.join(ckpt, "model.safetensors")
    with safe_open(safetensors_path, framework="pt") as f:
        keys = list(f.keys())
    print(f"total keys: {len(keys)}")
    norm_keys = sorted(
        k for k in keys
        if any(s in k.lower() for s in ("normaliz", "buffer_", "stats"))
    )
    print(f"normalize-related keys ({len(norm_keys)}):")
    for k in norm_keys:
        print(f"  {k}")
    if not norm_keys:
        print("  (none — stats are NOT inside safetensors)")
except Exception as e:
    print(f"failed to read safetensors: {type(e).__name__}: {e}")
