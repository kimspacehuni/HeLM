"""
Diagnostic: compare sim wrist cam (and agentview) at demo[0] init_state
against the v2 LeRobot training distribution frame 0.

Output: ~/tmp/wrist_check/{wrist,agent}_compare.png — side-by-side
[v2_truth | sim_raw | sim_vflip | sim_hflip | sim_rot180]. Whichever
sim_* panel matches v2_truth tells you the correct orient transform
for that camera at inference time.

Run:
    PYTHONPATH=/rlwrld2/home/gyeonghun_kim/codes/HeLM/libero-mem \
    /rlwrld2/home/gyeonghun_kim/miniconda3/envs/helm_libero/bin/python \
        tools/check_wrist_orient.py [--task-id 1] [--demo-idx 0]
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np

LIBERO_SRC = "/rlwrld2/home/gyeonghun_kim/codes/HeLM/libero-mem"
if LIBERO_SRC not in sys.path:
    sys.path.insert(0, LIBERO_SRC)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ.setdefault("MUJOCO_GL", "egl")


def label(img: np.ndarray, text: str) -> np.ndarray:
    h, w, c = img.shape
    pad = np.full((30, w, c), 30, dtype=img.dtype)
    out = np.vstack([img, pad])
    cv2.putText(out, text, (5, h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=1)
    ap.add_argument("--demo-idx", type=int, default=0)
    ap.add_argument("--task-suite", default="libero_mem")
    ap.add_argument("--raw-data-dir", default="/rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-Raw")
    ap.add_argument("--v2-pertask",
                    default="/rlwrld2/home/gyeonghun_kim/data/LIBERO-Mem/LIBERO-Mem-LeRobot-v2-PerTask")
    ap.add_argument("--out-dir", default="tmp/wrist_check",
                    help="output dir (relative to cwd, default: <repo>/tmp/wrist_check)")
    ap.add_argument("--resolution", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve task & bddl
    task = benchmark.get_benchmark_dict()[args.task_suite]().get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[check] task[{args.task_id}]: {task.language}")

    # Init state from raw HDF5
    raw_h5 = Path(args.raw_data_dir) / f"{task.name}_demo.hdf5"
    with h5py.File(raw_h5, "r") as f:
        keys = sorted(f["data"].keys(),
                      key=lambda s: int("".join(c for c in s if c.isdigit()) or 0))
        g = f["data"][keys[args.demo_idx]]
        init_state = g["init_state"][()] if "init_state" in g else g["states"][()][0]

    # Sim
    env = OffScreenRenderEnv(bddl_file_name=bddl,
                             camera_heights=args.resolution,
                             camera_widths=args.resolution)
    env.reset()
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()
    obs = env._get_observations() if hasattr(env, "_get_observations") else env.env._get_observations()
    agent_raw = obs["agentview_image"]
    wrist_raw = obs["robot0_eye_in_hand_image"]
    print(f"[check] sim wrist={wrist_raw.shape} agent={agent_raw.shape}")

    # v2 frame 0 — match by task language since LIBERO benchmark task_id and
    # v2 PerTask folder index use different orderings (LIBERO uses the "1 …",
    # "2 …" numeric prefix in task names; PerTask is alphabetic on language).
    lang_norm = task.language.lstrip("0123456789 _-.").strip().lower()
    candidates = sorted(Path(args.v2_pertask).glob("task_*"))
    v2_dir = None
    for d in candidates:
        # strip "task_NN_" prefix from folder name and compare prefix-of-prefix
        rest = d.name.split("_", 2)[-1].lower().replace("_", " ")
        if rest.startswith(lang_norm[:30]) or lang_norm.startswith(rest[:30]):
            v2_dir = d
            break
    if v2_dir is None:
        raise RuntimeError(f"v2 PerTask dir not found for language: {lang_norm!r}")
    print(f"[check] v2: {v2_dir}")

    chunk = args.demo_idx // 50
    ep_mp4 = f"episode_{args.demo_idx:06d}.mp4"
    v2_wrist_mp4 = v2_dir / "videos" / f"chunk-{chunk:03d}" / "observation.images.left_wrist_0_rgb" / ep_mp4
    v2_agent_mp4 = v2_dir / "videos" / f"chunk-{chunk:03d}" / "observation.images.base_0_rgb" / ep_mp4

    r = imageio.get_reader(str(v2_wrist_mp4)); v2_wrist = r.get_data(0); r.close()
    r = imageio.get_reader(str(v2_agent_mp4)); v2_agent = r.get_data(0); r.close()

    H, W = wrist_raw.shape[:2]
    v2_wrist = cv2.resize(v2_wrist, (W, H))
    v2_agent = cv2.resize(v2_agent, (W, H))

    def panels(img_raw, v2_truth):
        return [
            ("v2_truth", v2_truth),
            ("sim_raw", img_raw),
            ("sim_vflip", img_raw[::-1, :, :]),
            ("sim_hflip", img_raw[:, ::-1, :]),
            ("sim_rot180", img_raw[::-1, ::-1, :]),
        ]

    def compose(ps, path):
        out = np.hstack([label(img.copy(), name) for name, img in ps])
        imageio.imwrite(str(path), out)
        print(f"[check] wrote {path}")

    compose(panels(wrist_raw, v2_wrist), out_dir / "wrist_compare.png")
    compose(panels(agent_raw, v2_agent), out_dir / "agent_compare.png")

    print()
    print("Visually inspect — whichever sim_* panel matches v2_truth is the correct transform:")
    print(f"  Wrist:  {out_dir/'wrist_compare.png'}")
    print(f"  Agent:  {out_dir/'agent_compare.png'}   (expected: sim_vflip ≈ v2_truth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
