"""
Transform a LeRobot v2 dataset's `action` column so dim 0-2 holds the next
absolute EEF xyz (= state[t+1, 0:3]) instead of the original delta xyz.
Other dims (rotation delta, gripper) are passed through unchanged.

Why: LIBERO teleop captures position-delta actions clipped to bang-bang
{-0.4, 0, +0.4} per axis. Flow-matching (pi0) on bang-bang multimodal
targets tends to mode-collapse / attenuate. Absolute EEF xyz at 20 Hz is a
smooth continuous trajectory bounded by the workspace — much friendlier for
flow matching, and reverses cleanly at inference (delta = next_abs - current).

What this does:
  1. Mirror the entire src dataset to dst (copytree)
  2. For each episode parquet, replace action[:, 0:3] with state[t+1, 0:3];
     last frame falls back to state[N-1, 0:3] (no next state).
  3. Recompute action stats (meta/stats.json + meta/episodes_stats.jsonl).
     Other features' stats untouched.

Pair with:
  - PI0Config.action_abs_xyz=true at training (set in slurm script)
  - select_action does abs→delta inverse using current obs anchor.

Usage:
  python tools/transform_dataset_abs_xyz.py \\
      --src /rlwrld2/.../LIBERO-Mem-LeRobot-v2-100ep/task_01_sub \\
      --dst /rlwrld2/.../LIBERO-Mem-LeRobot-v2-100ep/task_01_sub_abs_xyz
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


N_XYZ = 3  # action[0:3] dims that get replaced with absolute xyz
ACTION_DIM = 7
STATE_XYZ = (0, 3)  # observation.state[0:3] is absolute EEF xyz


def transform_episode_parquet(src_pq: Path, dst_pq: Path) -> np.ndarray:
    df = pd.read_parquet(src_pq)
    actions = np.stack(df["action"].values).astype(np.float32)  # (N, 7)
    states = np.stack(df["observation.state"].values).astype(np.float32)  # (N, 8+)
    n = len(actions)
    if n != len(states):
        raise RuntimeError(f"len mismatch in {src_pq.name}: action={n} state={len(states)}")

    abs_xyz = np.empty((n, N_XYZ), dtype=np.float32)
    abs_xyz[:-1] = states[1:, STATE_XYZ[0]:STATE_XYZ[1]]
    abs_xyz[-1] = states[-1, STATE_XYZ[0]:STATE_XYZ[1]]  # last frame: no next, hold
    new_actions = actions.copy()
    new_actions[:, :N_XYZ] = abs_xyz

    df["action"] = list(new_actions)
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_pq, index=False)
    return new_actions


def stats_from_array(arr: np.ndarray) -> dict:
    return {
        "mean": arr.mean(axis=0).astype(np.float64).tolist(),
        "std": arr.std(axis=0).astype(np.float64).tolist(),
        "min": arr.min(axis=0).astype(np.float64).tolist(),
        "max": arr.max(axis=0).astype(np.float64).tolist(),
        "count": int(arr.shape[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source LeRobot dataset root")
    ap.add_argument("--dst", required=True, help="destination root (will be overwritten if exists)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.is_dir():
        print(f"ERROR: src does not exist: {src}", file=sys.stderr)
        return 1
    if src == dst:
        print("ERROR: src and dst must differ", file=sys.stderr)
        return 1

    if dst.exists():
        print(f"[transform] removing existing dst: {dst}")
        shutil.rmtree(dst)
    print(f"[transform] copytree {src} → {dst}")
    shutil.copytree(src, dst)

    # ---- 1) rewrite action[:, 0:3] in each episode parquet ----
    parquets = sorted((dst / "data").rglob("episode_*.parquet"))
    if not parquets:
        print(f"ERROR: no episode parquets under {dst}/data", file=sys.stderr)
        return 2
    print(f"[transform] rewriting {len(parquets)} episode parquet(s)...")
    per_ep_action_stats: dict[int, dict] = {}
    all_actions = []
    for pq_dst in parquets:
        rel = pq_dst.relative_to(dst / "data")
        pq_src = src / "data" / rel
        new_actions = transform_episode_parquet(pq_src, pq_dst)
        # Episode index from path (chunk-XXX/episode_NNNNNN.parquet)
        ep_idx = int(pq_dst.stem.split("_")[-1])
        per_ep_action_stats[ep_idx] = stats_from_array(new_actions)
        all_actions.append(new_actions)
    all_actions_np = np.concatenate(all_actions, axis=0)
    overall_stats = stats_from_array(all_actions_np)

    # ---- 2) update meta/stats.json (action only) ----
    stats_json = dst / "meta" / "stats.json"
    if stats_json.is_file():
        stats = json.loads(stats_json.read_text())
        if "action" in stats:
            for k in ("mean", "std", "min", "max", "count"):
                stats["action"][k] = overall_stats[k]
            stats_json.write_text(json.dumps(stats, indent=2))
            print(f"[transform] updated {stats_json.relative_to(dst)}")
        else:
            print(f"WARN: stats.json has no 'action' key; skipped")
    else:
        print(f"WARN: {stats_json} missing; skipped")

    # ---- 3) update meta/episodes_stats.jsonl (action only, per-episode) ----
    eps_path = dst / "meta" / "episodes_stats.jsonl"
    if eps_path.is_file():
        original = []
        with eps_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    original.append(json.loads(line))
        rewritten = 0
        with eps_path.open("w") as f:
            for entry in original:
                ep_idx = int(entry["episode_index"])
                if ep_idx in per_ep_action_stats and "stats" in entry and "action" in entry["stats"]:
                    new_act_stats = per_ep_action_stats[ep_idx]
                    for k in ("mean", "std", "min", "max"):
                        entry["stats"]["action"][k] = new_act_stats[k]
                    rewritten += 1
                f.write(json.dumps(entry) + "\n")
        print(f"[transform] updated {eps_path.relative_to(dst)} ({rewritten}/{len(original)} entries)")
    else:
        print(f"WARN: {eps_path} missing; skipped")

    # ---- 4) summary ----
    print("\n[transform] DONE")
    print(f"  total frames : {overall_stats['count']}")
    print(f"  action[0:3] (abs xyz)   mean={overall_stats['mean'][:3]}")
    print(f"  action[0:3] (abs xyz)    std={overall_stats['std'][:3]}")
    print(f"  action[0:3] (abs xyz)  range=[{overall_stats['min'][:3]}, {overall_stats['max'][:3]}]")
    print(f"  action[3:7] (drxyz+grip) std={overall_stats['std'][3:]}")
    print(f"\nNext: train pi0 vanilla on {dst} with --policy.action_abs_xyz=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
