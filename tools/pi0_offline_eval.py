"""
Offline pi0 evaluation: feed in-distribution (training/val) frames to pi0
and compare predicted actions to ground truth.

If MSE is low → model learned the data, deployment/OOD/conditioning is the
issue. If MSE is high even on training data → training failed (silent
bug in data, dataloader, or normalization).

Usage:
PYTHONPATH=/rlwrld2/home/gyeonghun_kim/codes/HeLM \\
/rlwrld2/home/gyeonghun_kim/miniconda3/envs/HeLM/bin/python \\
    tools/pi0_offline_eval.py \\
    --ckpt /rlwrld2/home/gyeonghun_kim/result/stage2_helm_pi0_task01_sub_ep100_relative_0506/checkpoints/030000/pretrained_model \\
    --data-root /rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-30ep/task_01_sub \\
    --repo-id task_01_sub \\
    --n-samples 30 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common.policies.pi0.modeling_pi0 import PI0Policy
from common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to pretrained_model dir")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--repo-id", default=None,
                    help="default: basename(data_root)")
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-jsonl", default=None,
                    help="save per-sample predictions here (default: tmp/pi0_offline_eval.jsonl)")
    args = ap.parse_args()

    repo_id = args.repo_id or Path(args.data_root).name
    out_jsonl = Path(args.out_jsonl or "tmp/pi0_offline_eval.jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"[eval] ckpt:      {args.ckpt}")
    print(f"[eval] data_root: {args.data_root}")
    print(f"[eval] repo_id:   {repo_id}")
    print(f"[eval] device:    {args.device}")

    # Load dataset (no delta_timestamps → returns single-frame items)
    ds = LeRobotDataset(repo_id, root=args.data_root)
    print(f"[eval] dataset frames: {len(ds)}")

    # Load policy
    policy = PI0Policy.from_pretrained(args.ckpt)
    policy = policy.to(args.device).eval()
    print(f"[eval] policy loaded. action_dim={policy.config.action_feature.shape[0]}")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.n_samples, len(ds)), replace=False)

    by_task = defaultdict(list)
    rows = []

    print()
    header = f"{'idx':>5} {'task':<14} {'gt_dxyz':<26} {'pred_dxyz':<26} {'gt_grip':>7} {'pred_grip':>9} {'mse':>10}"
    print(header)
    print("-" * len(header))

    for idx in indices:
        item = ds[int(idx)]
        gt_action_full = item["action"]
        # action may be [action_dim] or [chunk, action_dim] (with delta_timestamps)
        gt_action = gt_action_full[0] if gt_action_full.ndim > 1 else gt_action_full
        gt_action_np = gt_action.cpu().numpy()

        # build inference batch (single frame, batch=1)
        batch = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(args.device)
            elif isinstance(v, str):
                batch[k] = [v]
        # ensure task field present (LeRobotDataset puts language in 'task' field)
        if "task" not in batch:
            batch["task"] = [item.get("task", "") if isinstance(item.get("task"), str) else ""]

        # fresh chunk per call (queue cleared)
        if hasattr(policy, "reset"):
            policy.reset()
        with torch.no_grad():
            pred = policy.select_action(batch).squeeze().detach().cpu().numpy()

        # action dim might exceed gt_action_np.shape[0] (model max_action_dim padding)
        pred_trim = pred[: gt_action_np.shape[0]]
        mse = float(((gt_action_np - pred_trim) ** 2).mean())

        task_str = batch["task"][0] if batch["task"] else "?"
        by_task[task_str].append(mse)

        gt_dxyz = "[" + " ".join(f"{x:+.3f}" for x in gt_action_np[:3]) + "]"
        pred_dxyz = "[" + " ".join(f"{x:+.3f}" for x in pred_trim[:3]) + "]"
        print(f"{int(idx):>5} {task_str[:14]:<14} {gt_dxyz:<26} {pred_dxyz:<26} "
              f"{gt_action_np[6]:+.2f}   {pred_trim[6]:+.2f}     {mse:>10.6f}")

        rows.append({
            "idx": int(idx),
            "task": task_str,
            "gt_action": gt_action_np.tolist(),
            "pred_action": pred_trim.tolist(),
            "mse": mse,
        })

    # summary
    print()
    print("=" * 70)
    print(f"{'task':<20} {'n':>5} {'mean_mse':>12} {'median_mse':>12}")
    print("-" * 70)
    for task, mses in by_task.items():
        print(f"{task[:20]:<20} {len(mses):>5} {np.mean(mses):>12.6f} {np.median(mses):>12.6f}")
    all_mses = [r["mse"] for r in rows]
    print("-" * 70)
    print(f"{'TOTAL':<20} {len(all_mses):>5} {np.mean(all_mses):>12.6f} {np.median(all_mses):>12.6f}")

    out_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"\n[eval] saved per-sample predictions → {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
