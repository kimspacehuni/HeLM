#!/usr/bin/env python3
"""Scan a v3.0 LeRobot dataset for action/state anomalies that could cause
training loss spikes.

Reports:
  - NaN/Inf counts per column
  - Per-dim global stats + indices of >K-sigma outliers
  - Per-episode max action magnitude (flag top-N episodes)
  - Per-frame discontinuities: |action[t] - action[t-1]| (within-episode)
  - Gripper action histogram (should be bimodal around ±1)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_all_parquets(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise SystemExit(f"no parquets under {root}/data")
    print(f"[load] {len(files)} parquet file(s)")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def stack(col: pd.Series) -> np.ndarray:
    return np.stack(col.values).astype(np.float32)


def scan(arr: np.ndarray, name: str, sigma_k: float = 5.0, top_n: int = 20):
    print(f"\n=== {name}  shape={arr.shape}  dtype={arr.dtype} ===")
    nan_per = np.isnan(arr).sum(axis=0)
    inf_per = np.isinf(arr).sum(axis=0)
    if nan_per.any():
        print(f"  NaN per dim: {nan_per.tolist()}")
    if inf_per.any():
        print(f"  Inf per dim: {inf_per.tolist()}")
    if not (nan_per.any() or inf_per.any()):
        print("  no NaN/Inf")

    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    amin = np.nanmin(arr, axis=0)
    amax = np.nanmax(arr, axis=0)
    print(f"  {'dim':>4} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'>{:.0f}σ'.format(sigma_k):>8}")
    over_sigma_counts = []
    for d in range(arr.shape[1]):
        if std[d] == 0:
            n_out = 0
        else:
            n_out = int(np.sum(np.abs(arr[:, d] - mean[d]) > sigma_k * std[d]))
        over_sigma_counts.append(n_out)
        print(f"  {d:>4} {amin[d]:>10.4f} {amax[d]:>10.4f} {mean[d]:>10.4f} {std[d]:>10.4f} {n_out:>8d}")
    return over_sigma_counts


def per_episode_max(df: pd.DataFrame, action: np.ndarray, top_n: int = 15):
    print("\n=== top per-episode max(|action|) ===")
    ep_idx = df["episode_index"].values
    abs_act = np.abs(action).max(axis=1)
    ep_max = pd.Series(abs_act).groupby(ep_idx).max().sort_values(ascending=False)
    print(ep_max.head(top_n).to_string())


def jump_scan(df: pd.DataFrame, action: np.ndarray, top_n: int = 30, jump_thresh: float = 1.0):
    """Find large within-episode action discontinuities."""
    print(f"\n=== top within-episode |Δaction| (jumps > {jump_thresh}) ===")
    ep_idx = df["episode_index"].values
    fr_idx = df["frame_index"].values
    same_ep = ep_idx[1:] == ep_idx[:-1]
    da = np.abs(action[1:] - action[:-1]).max(axis=1)
    da_in_ep = np.where(same_ep, da, 0.0)
    big = np.argsort(da_in_ep)[::-1][:top_n]
    print(f"  {'rank':>4} {'ep':>5} {'frame':>6} {'maxΔ':>10}  per-dim Δ")
    for r, i in enumerate(big):
        if da_in_ep[i] <= jump_thresh:
            break
        diffs = action[i + 1] - action[i]
        per_dim = " ".join(f"{v:+.3f}" for v in diffs)
        print(f"  {r:>4} {ep_idx[i+1]:>5} {fr_idx[i+1]:>6} {da_in_ep[i]:>10.4f}  {per_dim}")


def gripper_hist(action: np.ndarray):
    g = action[:, -1]
    bins = np.linspace(-1.05, 1.05, 22)
    h, edges = np.histogram(g, bins=bins)
    print("\n=== gripper action histogram (last dim) ===")
    for cnt, lo, hi in zip(h, edges[:-1], edges[1:]):
        bar = "#" * int(50 * cnt / max(h))
        print(f"  [{lo:+.2f}, {hi:+.2f}) {cnt:>7d}  {bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sigma-k", type=float, default=5.0)
    ap.add_argument("--jump-thresh", type=float, default=1.0,
                    help="report |Δaction| jumps above this between consecutive frames in the SAME episode")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    print(f"=== {root.name} ===")
    print(f"  codebase_version: {info.get('codebase_version')}")
    print(f"  total_episodes:   {info.get('total_episodes')}")
    print(f"  total_frames:     {info.get('total_frames')}")

    df = load_all_parquets(root)
    print(f"[load] {len(df)} rows, {df['episode_index'].nunique()} episodes")

    action = stack(df["action"])
    state = stack(df["observation.state"])

    scan(action, "action", sigma_k=args.sigma_k)
    scan(state, "observation.state", sigma_k=args.sigma_k)

    per_episode_max(df, action)
    jump_scan(df, action, jump_thresh=args.jump_thresh)
    gripper_hist(action)


if __name__ == "__main__":
    main()
