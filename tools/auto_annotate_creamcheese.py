#!/usr/bin/env python3
"""Rule-based sub-task annotator for the cream cheese tasks (task_09, task_10).

LIBERO cream cheese task is 2-phase:
  Phase 1: pick the cream cheese (left or right), put it in the NEAREST basket.
  Phase 2: pick a basket (left or right), place it in the center.

task_09 ("that basket"): in phase 2, pick the basket that HAS the cream cheese
  -> same side as cheese
task_10 ("empty basket"): in phase 2, pick the basket WITHOUT cream cheese
  -> opposite side from cheese

Subtask vocabulary (4, matches existing task_09_sub):
  put_left_creamcheese   put_right_creamcheese
  place_left_basket      place_right_basket

Rule per episode:
  1) grasp_intervals(gripper) -> expect exactly 2 grasps (skip ep if not)
  2) cheese_side = sign(grasp[0].start_y) with |y| >= y_thresh; skip if ambiguous
  3) basket_side is DETERMINED by the variant:
        task_10 -> opposite of cheese_side  (empty basket goes to center)
        task_09 -> same as cheese_side      (basket-with-cheese goes to center)
     (We don't read grasp[1].start_y because the empty basket is often grabbed
     near the table centerline so its y is not robustly classifiable.)
  4) labels:
        seg 1 -> put_{cheese_side}_creamcheese
        seg 2 -> place_{basket_side}_basket
  5) boundary at midpoint between grasp[0].open_t and grasp[1].close_t.

Sign convention (matches auto_annotate_subtasks default --sign-y +1):
  y > 0 -> "right", y < 0 -> "left".

Writes <lerobot_root>/meta/subtasks.jsonl (one row per segment).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def grasp_intervals(gripper: np.ndarray, min_len: int = 5) -> List[Tuple[int, int]]:
    closed = gripper > 0.0
    out: List[Tuple[int, int]] = []
    T = len(closed)
    i = 0
    while i < T:
        if closed[i]:
            j = i
            while j < T and closed[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


def classify_side(y: float, thresh: float):
    if y > thresh:
        return "right"
    if y < -thresh:
        return "left"
    return None


def annotate_episode(actions: np.ndarray,
                     states: np.ndarray,
                     variant: str,
                     y_thresh: float) -> Tuple[List[Dict], str]:
    T = len(actions)
    grasps = grasp_intervals(actions[:, -1])
    if len(grasps) != 2:
        return [], f"wrong_grasp_count={len(grasps)}"

    (close1, open1), (close2, open2) = grasps
    cheese_side = classify_side(float(states[close1, 1]), y_thresh)
    if cheese_side is None:
        return [], "ambiguous_cheese_y"

    if variant == "task_10":
        basket_side = "right" if cheese_side == "left" else "left"
    else:  # task_09: basket-with-cheese (same side)
        basket_side = cheese_side

    mid = (open1 + close2) // 2
    seg1 = {"start_frame": 0, "end_frame": int(mid),
            "label": f"put_{cheese_side}_creamcheese"}
    seg2 = {"start_frame": int(mid), "end_frame": int(T - 1),
            "label": f"place_{basket_side}_basket"}
    return [seg1, seg2], "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lerobot-root", required=True)
    ap.add_argument("--variant", choices=("task_09", "task_10"), required=True)
    ap.add_argument("--y-thresh", type=float, default=0.04)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.lerobot_root).expanduser().resolve()
    out = Path(args.out) if args.out else root / "meta" / "subtasks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    parqs = sorted((root / "data").rglob("episode_*.parquet"))
    if not parqs:
        print(f"ERROR: no episode parquets under {root}/data", file=sys.stderr)
        return 1
    print(f"[annotate] scanning {len(parqs)} episodes ({args.variant})")

    status_counter: Counter = Counter()
    label_counter: Counter = Counter()
    rows: List[Dict] = []
    skipped: List[Tuple[int, str]] = []

    for parq in parqs:
        ep_idx = int(parq.stem.replace("episode_", ""))
        df = pd.read_parquet(parq)
        actions = np.stack(df["action"].values).astype(np.float32)
        states = np.stack(df["observation.state"].values).astype(np.float32)
        segs, status = annotate_episode(actions, states, args.variant, args.y_thresh)
        status_counter[status] += 1
        if status != "ok":
            skipped.append((ep_idx, status))
            continue
        for s in segs:
            rows.append({"episode_index": ep_idx, **s})
            label_counter[s["label"]] += 1

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_ok = status_counter["ok"]
    n_skip = sum(v for k, v in status_counter.items() if k != "ok")
    print(f"[annotate] OK      : {n_ok:>3d} episodes ({100.0*n_ok/(n_ok+n_skip):.1f}%)")
    print(f"[annotate] skipped : {n_skip:>3d} episodes")
    for k, v in status_counter.items():
        if k != "ok":
            print(f"             {k:>32s} : {v}")
    if skipped[:10]:
        print(f"[annotate] sample skipped:")
        for ep, reason in skipped[:10]:
            print(f"             ep {ep:>4d}: {reason}")
    print(f"[annotate] label distribution:")
    for k, v in sorted(label_counter.items()):
        print(f"             {k:>26s} : {v}")
    print(f"[annotate] wrote {len(rows)} segments -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
