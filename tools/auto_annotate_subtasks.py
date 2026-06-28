#!/usr/bin/env python3
"""Rule-based auto subtask annotator for LIBERO-Mem swap / rotate tasks.

Operates entirely on a LeRobot v2.x dataset (no raw HDF5 needed). Uses:
  - action[:, -1] (gripper command, +1 close / -1 open) to detect each
    pick-and-place segment (close → open).
  - observation.state[:, :3] (EEF xyz) to locate src plate (at grasp moment) and
    dst plate (at release moment) for each segment.

Per-episode position labeling (no cross-episode clustering):
  For each episode, the 2N positions visited across N moves (N=3 swap / 4 rotate)
  fall onto exactly K unique plate positions (K=3 swap / 4 rotate). We dedupe by
  proximity, compute their centroid in xy, and assign each plate a quadrant label
  {topleft, topright, bottomleft, bottomright} relative to the centroid.

The axis convention (which world axis maps to "top/bottom" in the camera view,
same for left/right) is auto-detected against any existing hand annotations via
--validate-against.

Output: meta/subtasks_auto.jsonl in the same schema the Streamlit annotator writes,
ready to be moved over meta/subtasks.jsonl for the split step.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------- helpers ----------

def grasp_intervals(gripper: np.ndarray, min_len: int = 5) -> List[Tuple[int, int]]:
    """Return [(close_t, open_t), ...] from gripper command (>0 close, <0 open).

    A grasp interval is a contiguous run of gripper>0 lasting at least min_len.
    close_t = first frame of run; open_t = last frame of run (release happens at
    open_t+1).
    """
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


def dedupe_positions(positions: List[Tuple[float, float]],
                     tol: float = 0.03) -> List[Tuple[float, float]]:
    """Greedy dedupe of 2D points by L2 distance. Returns the centroid of each
    cluster, in order of first appearance.

    tol is the merge radius in world units (meters). Plates are ~10cm apart in
    LIBERO so 3cm is a safe threshold.
    """
    clusters: List[List[Tuple[float, float]]] = []
    for p in positions:
        placed = False
        for c in clusters:
            cx = np.mean([q[0] for q in c])
            cy = np.mean([q[1] for q in c])
            if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 < tol ** 2:
                c.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    return [(float(np.mean([q[0] for q in c])),
             float(np.mean([q[1] for q in c]))) for c in clusters]


def quadrant(xy: Tuple[float, float],
             centroid: Tuple[float, float],
             sign_x: int,
             sign_y: int) -> str:
    """LIBERO camera-view convention:
       world x-axis  → camera vertical    (top = smaller x; controlled by sign_x)
       world y-axis  → camera horizontal  (right = larger y; controlled by sign_y)
    """
    dv = (xy[0] - centroid[0]) * sign_x   # vertical (top/bottom)  from x
    dh = (xy[1] - centroid[1]) * sign_y   # horizontal (left/right) from y
    v = "top" if dv > 0 else "bottom"
    h = "right" if dh > 0 else "left"
    return f"{v}{h}"


# 4 corner reference labels and their target sign direction (dv_sign, dh_sign).
_CORNERS = [
    ("topleft",     +1, -1),   # top: dv>0,  left:  dh<0
    ("topright",    +1, +1),   # top: dv>0,  right: dh>0
    ("bottomleft",  -1, -1),   # bottom: dv<0, left:  dh<0
    ("bottomright", -1, +1),   # bottom: dv<0, right: dh>0
]


def assign_unique_corners(plates: List[Tuple[float, float]],
                          centroid: Tuple[float, float],
                          sign_x: int, sign_y: int) -> Dict[int, str]:
    """Bijection: plate_idx -> corner label.

    For K ≤ 4: brute-force assignment maximizing total dot-product score.
    For K > 4: greedy fallback — top-4 plates by confidence get unique corners,
               the rest take their best available corner (may duplicate).
               Indicates over-segmentation in dedupe; ideally don't trigger.
    """
    from itertools import permutations
    K = len(plates)
    if K == 0:
        return {}

    signed: List[Tuple[float, float]] = []
    for (x, y) in plates:
        dv = (x - centroid[0]) * sign_x
        dh = (y - centroid[1]) * sign_y
        signed.append((dv, dh))

    if K <= 4:
        best_score = -float("inf")
        best_perm: Tuple[int, ...] = tuple(range(K))  # safe default
        for perm in permutations(range(len(_CORNERS)), K):
            score = 0.0
            for plate_i, corner_i in enumerate(perm):
                dv, dh = signed[plate_i]
                _, scv, sch = _CORNERS[corner_i]
                score += dv * scv + dh * sch
            if score > best_score:
                best_score = score
                best_perm = perm
        return {i: _CORNERS[best_perm[i]][0] for i in range(K)}

    # K > 4: greedy fallback.
    scores_per_plate: List[List[float]] = []
    for (dv, dh) in signed:
        scores_per_plate.append([dv * scv + dh * sch for (_, scv, sch) in _CORNERS])
    confidence = [max(s) for s in scores_per_plate]
    sorted_idx = sorted(range(K), key=lambda i: -confidence[i])
    used: set = set()
    result: Dict[int, str] = {}
    for plate_i in sorted_idx[:4]:
        ranked = sorted(range(len(_CORNERS)), key=lambda c: -scores_per_plate[plate_i][c])
        for c in ranked:
            if c not in used:
                result[plate_i] = _CORNERS[c][0]
                used.add(c)
                break
    for plate_i in sorted_idx[4:]:
        c = max(range(len(_CORNERS)), key=lambda c: scores_per_plate[plate_i][c])
        result[plate_i] = _CORNERS[c][0]
    return result


# ---------- per-episode annotation ----------

def annotate_episode(actions: np.ndarray,
                     states: np.ndarray,
                     sign_x: int,
                     sign_y: int,
                     boundary_split: str = "release",
                     grasp_min_len: int = 5,
                     pose_margin: int = 2,
                     dedupe_tol: float = 0.04) -> List[Dict]:
    """Annotate one lerobot episode.

    Args:
      actions: (T, 7) — last column is gripper command.
      states:  (T, 8) — first three columns are EEF xyz.
      sign_x, sign_y: axis convention for quadrant labeling.
      boundary_split: 'release' = each subtask ends at the release frame and the
                       next begins there; 'midpoint' = boundary at midpoint
                       between release and next grasp.
      grasp_min_len: ignore grasp blips shorter than this.
      pose_margin: read EEF xy at (close_t - margin) and (open_t + margin) so we
                   sample the still-on-plate moment, not the lift trajectory.
      dedupe_tol: distance threshold to merge grasp/release positions into one
                  plate (meters).

    Returns: list of {start_frame, end_frame, label} dicts in chronological order.
    """
    T = len(states)
    gr = actions[:, -1]
    intervals = grasp_intervals(gr, min_len=grasp_min_len)
    if not intervals:
        return []

    eef_xy = states[:, :2]

    src_pts: List[Tuple[float, float]] = []
    dst_pts: List[Tuple[float, float]] = []
    for (ct, ot) in intervals:
        b = max(0, ct - pose_margin)
        a = min(T - 1, ot + pose_margin)
        src_pts.append((float(eef_xy[b, 0]), float(eef_xy[b, 1])))
        dst_pts.append((float(eef_xy[a, 0]), float(eef_xy[a, 1])))

    all_pts = src_pts + dst_pts
    plates = dedupe_positions(all_pts, tol=dedupe_tol)
    if not plates:
        return []

    cx = float(np.mean([p[0] for p in plates]))
    cy = float(np.mean([p[1] for p in plates]))

    # Unique corner assignment: K plates -> K of the 4 corner labels (no dups).
    plate_idx_to_label = assign_unique_corners(plates, (cx, cy), sign_x, sign_y)

    def nearest_plate_label(xy: Tuple[float, float]) -> str:
        best_i = min(range(len(plates)),
                     key=lambda i: (plates[i][0] - xy[0]) ** 2 + (plates[i][1] - xy[1]) ** 2)
        return plate_idx_to_label[best_i]

    # Build (close_t, open_t, label) for each interval.
    raw = []
    for (ct, ot), s, d in zip(intervals, src_pts, dst_pts):
        s_lbl = nearest_plate_label(s)
        d_lbl = nearest_plate_label(d)
        raw.append((int(ct), int(ot), f"move_{s_lbl}_bowl_to_{d_lbl}"))

    # Segment boundaries.
    if boundary_split == "release":
        segs = []
        prev_end = 0
        for i, (ct, ot, lbl) in enumerate(raw):
            start = prev_end
            end = T - 1 if i == len(raw) - 1 else ot
            segs.append({"start_frame": start, "end_frame": end, "label": lbl})
            prev_end = end
    elif boundary_split == "midpoint":
        segs = []
        for i, (ct, ot, lbl) in enumerate(raw):
            start = 0 if i == 0 else segs[-1]["end_frame"]
            if i == len(raw) - 1:
                end = T - 1
            else:
                end = (ot + raw[i + 1][0]) // 2
            segs.append({"start_frame": int(start), "end_frame": int(end), "label": lbl})
    else:
        raise ValueError(boundary_split)

    return segs


# ---------- driver ----------

def load_episode_parquets(dataset_root: Path) -> Dict[int, pd.DataFrame]:
    """Return episode_index -> DataFrame for every parquet under data/."""
    out: Dict[int, pd.DataFrame] = {}
    for chunk_dir in sorted((dataset_root / "data").iterdir()):
        if not chunk_dir.is_dir():
            continue
        for parq in sorted(chunk_dir.glob("episode_*.parquet")):
            ep_idx = int(parq.stem.replace("episode_", ""))
            out[ep_idx] = pd.read_parquet(parq)
    return out


def auto_detect_signs(hand_subs: Dict[int, List[Dict]],
                       eps: Dict[int, pd.DataFrame],
                       boundary_split: str) -> Tuple[int, int, float]:
    best = (1, 1, -1.0)
    for sx in (+1, -1):
        for sy in (+1, -1):
            total = 0
            matched = 0
            for ep_idx, hand_list in hand_subs.items():
                if ep_idx not in eps:
                    continue
                df = eps[ep_idx]
                actions = np.stack(df["action"].values).astype(np.float32)
                states = np.stack(df["observation.state"].values).astype(np.float32)
                auto = annotate_episode(actions, states, sx, sy, boundary_split=boundary_split)
                ha_sorted = sorted(hand_list, key=lambda s: s["start_frame"])
                for ha, aa in zip(ha_sorted, auto):
                    total += 1
                    if ha["label"] == aa["label"]:
                        matched += 1
            rate = matched / total if total else 0.0
            if rate > best[2]:
                best = (sx, sy, rate)
    return best


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="LeRobot v2.x dataset root")
    p.add_argument("--output", default=None,
                   help="Path to write subtasks_auto.jsonl (default: <dataset>/meta/subtasks_auto.jsonl)")
    p.add_argument("--validate-against", default=None,
                   help="Existing hand-annotated subtasks.jsonl for sign auto-detect + mismatch report")
    p.add_argument("--sign-x", type=int, default=None)
    p.add_argument("--sign-y", type=int, default=None)
    p.add_argument("--boundary-split", choices=("release", "midpoint"), default="release")
    p.add_argument("--episodes", nargs="+", type=int, default=None)
    p.add_argument("--grasp-min-len", type=int, default=5)
    p.add_argument("--pose-margin", type=int, default=2)
    p.add_argument("--dedupe-tol", type=float, default=0.06,
                   help="merge radius for plate dedupe (m). Plates ~10cm apart, "
                        "so 6cm avoids over-segmentation from grasp-position noise.")
    p.add_argument("--annotator", default="auto-rule-v1")
    args = p.parse_args()

    dataset_root = Path(args.dataset).expanduser()
    out_path = Path(args.output).expanduser() if args.output else (
        dataset_root / "meta" / "subtasks_auto.jsonl")

    print(f"[load] {dataset_root}")
    eps = load_episode_parquets(dataset_root)
    print(f"[load] {len(eps)} episodes")

    if args.episodes is not None:
        eps = {e: v for e, v in eps.items() if e in set(args.episodes)}

    hand_subs: Dict[int, List[Dict]] = {}
    if args.validate_against:
        with Path(args.validate_against).expanduser().open() as f:
            for ln in f:
                if not ln.strip():
                    continue
                d = json.loads(ln)
                hand_subs.setdefault(int(d["episode_index"]), []).append(d)
        print(f"[validate] loaded {sum(len(v) for v in hand_subs.values())} hand subtasks "
              f"across {len(hand_subs)} eps")

    if args.sign_x is not None and args.sign_y is not None:
        sx, sy = args.sign_x, args.sign_y
        print(f"[signs] forced: sign_x={sx} sign_y={sy}")
    elif hand_subs:
        sx, sy, rate = auto_detect_signs(hand_subs, eps, args.boundary_split)
        print(f"[signs] auto-detected: sign_x={sx} sign_y={sy} (match {rate:.1%} on hand-labeled eps)")
    else:
        sx, sy = +1, +1
        print("[signs] no hint; defaulting to sign_x=+1 sign_y=+1")

    out_rows: List[Dict] = []
    now_iso = dt.datetime.now().isoformat() + "+00:00"
    per_ep_segs: Dict[int, List[Dict]] = {}
    skipped_eps: List[int] = []
    for ep_idx in sorted(eps.keys()):
        df = eps[ep_idx]
        actions = np.stack(df["action"].values).astype(np.float32)
        states = np.stack(df["observation.state"].values).astype(np.float32)
        segs = annotate_episode(actions, states, sx, sy,
                                  boundary_split=args.boundary_split,
                                  grasp_min_len=args.grasp_min_len,
                                  pose_margin=args.pose_margin,
                                  dedupe_tol=args.dedupe_tol)
        if not segs:
            skipped_eps.append(ep_idx)
            continue
        per_ep_segs[ep_idx] = segs
        for s in segs:
            out_rows.append({
                "id": str(uuid.uuid4()),
                "episode_index": ep_idx,
                "start_frame": int(s["start_frame"]),
                "end_frame": int(s["end_frame"]),
                "label": s["label"],
                "annotator": args.annotator,
                "created_at": now_iso,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {len(out_rows)} subtasks across {len(per_ep_segs)} eps "
          f"(skipped={len(skipped_eps)}) -> {out_path}")
    if skipped_eps:
        print(f"[skip] episodes with no grasp detected: {skipped_eps[:10]}"
              f"{' ...' if len(skipped_eps) > 10 else ''}")

    lbl_counter = Counter(r["label"] for r in out_rows)
    print(f"[dist] {len(lbl_counter)} unique labels:")
    for lbl, n in lbl_counter.most_common():
        print(f"    {n:>4}  {lbl}")

    if hand_subs:
        total = 0
        matched = 0
        mismatch: List[str] = []
        for ep_idx, hand_list in sorted(hand_subs.items()):
            hand_sorted = sorted(hand_list, key=lambda s: s["start_frame"])
            auto_sorted = sorted(per_ep_segs.get(ep_idx, []), key=lambda s: s["start_frame"])
            for i, ha in enumerate(hand_sorted):
                total += 1
                if i >= len(auto_sorted):
                    mismatch.append(f"  ep{ep_idx:3d}#{i}  MISSING (hand={ha['label']})")
                    continue
                aa = auto_sorted[i]
                if aa["label"] == ha["label"]:
                    matched += 1
                else:
                    mismatch.append(
                        f"  ep{ep_idx:3d}#{i}  hand={ha['label']:50s}  auto={aa['label']}")
        print(f"\n[match] {matched}/{total} ({matched/total:.1%}) hand-label agreement")
        if mismatch:
            print(f"[match] {len(mismatch)} mismatches (showing first 30):")
            for ml in mismatch[:30]:
                print(ml)

    return 0


if __name__ == "__main__":
    sys.exit(main())
