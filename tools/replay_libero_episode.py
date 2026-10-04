"""
Replay a LIBERO-Mem demonstration in the simulator and compare against the
LeRobot dataset's stored mp4 frame-by-frame. Used to verify that:
  1) actions and init_state from the dataset reproduce the original demo
  2) image orientations match between training data and what env produces

Output: a side-by-side mp4 (left = simulated replay, right = recorded mp4).

Usage:
    python tools/replay_libero_episode.py \\
        --raw /rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-Raw \\
        --lerobot /rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot \\
        --task-id 0 \\
        --demo-idx 0 \\
        --out /tmp/replay_compare.mp4

Run inside helm_libero env (libero + mujoco). Requires h5py, numpy, imageio.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def natural_key(s: str):
    nums = re.findall(r"\d+", s)
    return int(nums[-1]) if nums else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True,
                    help="Path to LIBERO-Mem-Raw dir (with *_demo.hdf5 per task)")
    ap.add_argument("--lerobot", required=True,
                    help="Path to LeRobot dataset (with videos/data/meta)")
    ap.add_argument("--task-suite", default="libero_mem")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--demo-idx", type=int, default=0,
                    help="Demo index within the task's HDF5 (0..N-1)")
    ap.add_argument("--lerobot-ep", type=int, default=None,
                    help="LeRobot episode index. Default: same as --demo-idx if "
                         "the dataset preserves order (per-task split usually does).")
    ap.add_argument("--out", default="/tmp/replay_compare.mp4")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    if args.lerobot_ep is None:
        args.lerobot_ep = args.demo_idx

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[replay] task[{args.task_id}]: {task.language!r}")

    # ---- 1) Load actions + init_state from raw HDF5 ----
    raw_path = os.path.join(args.raw, f"{task.name}_demo.hdf5")
    if not os.path.isfile(raw_path):
        print(f"ERROR: raw HDF5 not found: {raw_path}", file=sys.stderr)
        return 1
    with h5py.File(raw_path, "r") as f:
        keys = sorted(f["data"].keys(), key=natural_key)
        if args.demo_idx >= len(keys):
            print(f"demo-idx {args.demo_idx} out of range (have {len(keys)})", file=sys.stderr)
            return 2
        grp = f["data"][keys[args.demo_idx]]
        init_state = (grp["init_state"][()] if "init_state" in grp
                      else grp["states"][()][0])
        actions = grp["actions"][()]
    print(f"[replay] raw: {keys[args.demo_idx]}, actions={actions.shape}")

    # ---- 2) Locate recorded mp4 via meta/tasks.jsonl + meta/episodes.jsonl ----
    # The LIBERO suite task_id and the LeRobot task_index can differ (different
    # sort order). We match on task language string instead.
    import json
    rec_frames = None
    lerobot_root = Path(args.lerobot)
    tasks_jsonl = lerobot_root / "meta" / "tasks.jsonl"
    episodes_jsonl = lerobot_root / "meta" / "episodes.jsonl"
    info_json = lerobot_root / "meta" / "info.json"

    if not tasks_jsonl.is_file() or not episodes_jsonl.is_file():
        print(f"WARNING: meta files missing under {lerobot_root}; skipping recorded mp4")
    else:
        # Load task_index → task language map
        task_idx_to_lang = {}
        with tasks_jsonl.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    task_idx_to_lang[int(obj["task_index"])] = obj["task"]

        # Find target task_index by matching language. Compare loosely (libero
        # suite text often has a numeric prefix like '1 pick up...').
        def _norm(s):
            return s.strip().lower()
        suite_lang = _norm(task.language).lstrip("0123456789 _-.")
        match_idx = None
        for ti, lang in task_idx_to_lang.items():
            if _norm(lang).lstrip("0123456789 _-.") == suite_lang or suite_lang in _norm(lang):
                match_idx = ti
                break
        if match_idx is None:
            print(f"WARNING: could not match suite task '{task.language}' in tasks.jsonl. "
                  f"Available: {list(task_idx_to_lang.values())[:5]}...; skipping recorded mp4")
        else:
            # Get the demo_idx-th episode whose task_index matches.
            matching_eps = []
            with episodes_jsonl.open() as f:
                for line in f:
                    if line.strip():
                        ep = json.loads(line)
                        # episodes.jsonl 'tasks' field carries language; we still
                        # confirm via reading parquet's task_index column for safety.
                        matching_eps.append(int(ep["episode_index"]))
            # Filter by task_index from the parquet (more reliable than tasks.jsonl).
            try:
                import pandas as pd
                chunk_size = int(json.loads(info_json.read_text()).get("chunks_size", 50))
                filtered = []
                for ep_idx in matching_eps:
                    chunk = ep_idx // chunk_size
                    pq = (lerobot_root / "data" / f"chunk-{chunk:03d}"
                          / f"episode_{ep_idx:06d}.parquet")
                    if not pq.is_file():
                        continue
                    df = pd.read_parquet(pq, columns=["task_index"])
                    if int(df["task_index"].iloc[0]) == match_idx:
                        filtered.append(ep_idx)
                matching_eps = filtered
            except Exception as e:
                print(f"WARNING: could not filter episodes by task_index ({e})")

            if args.demo_idx >= len(matching_eps):
                print(f"WARNING: demo_idx {args.demo_idx} out of range "
                      f"({len(matching_eps)} matching episodes); using 0")
                args.demo_idx = 0
            if matching_eps:
                lerobot_ep = matching_eps[args.demo_idx]
                chunk_size = int(json.loads(info_json.read_text()).get("chunks_size", 50))
                chunk = lerobot_ep // chunk_size
                rec_mp4 = (lerobot_root / "videos" / f"chunk-{chunk:03d}"
                           / "observation.images.base_0_rgb"
                           / f"episode_{lerobot_ep:06d}.mp4")
                if rec_mp4.is_file():
                    rdr = imageio.get_reader(str(rec_mp4))
                    rec_frames = [np.asarray(f) for f in rdr]
                    rdr.close()
                    print(f"[replay] recorded mp4: ep={lerobot_ep}, "
                          f"{len(rec_frames)} frames at {rec_frames[0].shape}")
                else:
                    print(f"WARNING: mp4 not found: {rec_mp4}")
            else:
                print(f"WARNING: no matching episodes found for task '{task.language}'")

    # ---- 3) Build env and replay ----
    print("[replay] building env ...")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(0)
    env.reset()
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()

    sim_frames = []
    success = False
    for step, action in enumerate(actions):
        # Capture exterior camera as the env "saw" it (RAW mujoco — we'll show
        # both raw and rotated for diagnostic).
        obs, _, done, info = env.step(action.tolist())
        sim_frames.append(np.asarray(obs["agentview_image"]))
        if env.check_success():
            success = True
            print(f"[replay] sim SUCCESS at step {step}")
            break
    print(f"[replay] sim total steps={len(sim_frames)} success={success}")

    # ---- 4) Side-by-side compare ----
    # Layout:
    #   [sim_raw | sim_180rot | recorded] (3 columns, equal width)
    H = args.resolution
    W = args.resolution
    n = min(len(sim_frames), len(rec_frames) if rec_frames is not None else len(sim_frames))
    if rec_frames is None:
        cols = 2  # raw + 180rot
    else:
        cols = 3

    out_w = W * cols + 2 * (cols - 1)  # 2px separator
    # pad to multiple of 16 for libx264
    pad_w = (16 - out_w % 16) % 16
    pad_h = (16 - H % 16) % 16

    print(f"[replay] writing comparison mp4 → {args.out} ({n} frames, {out_w + pad_w}x{H + pad_h})")
    writer = imageio.get_writer(
        args.out, fps=args.fps, codec="libx264",
        pixelformat="yuv420p", quality=8,
    )
    for i in range(n):
        sim_raw = sim_frames[i]
        sim_rot = sim_raw[::-1, ::-1, :]
        canvas = np.zeros((H + pad_h, out_w + pad_w, 3), dtype=np.uint8)
        x = 0
        canvas[:H, x:x + W] = sim_raw
        x += W + 2
        canvas[:H, x:x + W] = sim_rot
        x += W + 2
        if rec_frames is not None:
            canvas[:H, x:x + W] = rec_frames[i]
        writer.append_data(canvas)
    writer.close()

    print(f"\n[replay] OK. Layout left→right: sim_raw | sim_180rot"
          + (" | recorded" if rec_frames is not None else ""))
    print(f"  - if sim_raw matches recorded → recorded was raw mujoco; eval should NOT rotate")
    print(f"  - if sim_180rot matches recorded → recorded was rotated; eval SHOULD rotate")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
