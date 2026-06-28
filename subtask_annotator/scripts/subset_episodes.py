"""Create a smaller lerobot v2.1 dataset by keeping only the first N (or specified) episodes.

Useful for quickly testing the end-to-end pipeline (annotate → split → mem_generate)
without waiting to annotate all episodes.

Preserves original episode_index numbering (0..N-1 for --num-episodes N).
Updates info.json totals and filters episodes.jsonl / episodes_stats.jsonl.
Other meta files (tasks.jsonl, modality.json, stats.json, relative_stats_40.json)
are copied as-is — note that stats.json will no longer reflect the subset accurately;
for MVP testing this is fine, regenerate if you need exact stats.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="path to source lerobot v2.1 dataset")
    p.add_argument("--dest", default=None,
                   help="output path (default: <source>_<N>ep)")
    p.add_argument("--num-episodes", type=int, default=5,
                   help="how many episodes to keep (default: 5)")
    p.add_argument("--episodes", default=None,
                   help="comma-separated explicit episode indices (overrides --num-episodes)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    src = Path(args.source).resolve()
    info = json.loads((src / "meta" / "info.json").read_text())
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    if args.episodes:
        keep = sorted({int(x) for x in args.episodes.split(",")})
    else:
        total = info["total_episodes"]
        N = min(args.num_episodes, total)
        keep = list(range(N))
    keep_set = set(keep)

    if args.dest:
        dst = Path(args.dest).resolve()
    else:
        dst = src.parent / f"{src.name}_{len(keep)}ep"

    if dst.exists():
        if not args.force:
            print(f"ERROR: {dst} already exists (use --force)")
            return 2
        shutil.rmtree(dst)

    (dst / "meta").mkdir(parents=True)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    for vk in video_keys:
        (dst / "videos" / "chunk-000" / vk).mkdir(parents=True)

    print(f"[subset] source: {src}")
    print(f"[subset] dest:   {dst}")
    print(f"[subset] keeping episodes: {keep}")

    chunks_size = int(info.get("chunks_size", 1000))
    for ep in keep:
        chunk = ep // chunks_size
        pq_name = f"episode_{ep:06d}.parquet"
        src_pq = src / f"data/chunk-{chunk:03d}/{pq_name}"
        dst_pq = dst / f"data/chunk-{chunk:03d}/{pq_name}"
        dst_pq.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_pq, dst_pq)
        for vk in video_keys:
            mp4_name = f"episode_{ep:06d}.mp4"
            src_mp4 = src / f"videos/chunk-{chunk:03d}/{vk}/{mp4_name}"
            dst_mp4 = dst / f"videos/chunk-{chunk:03d}/{vk}/{mp4_name}"
            dst_mp4.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_mp4, dst_mp4)
        print(f"  ep {ep:06d}: parquet + {len(video_keys)} videos copied")

    total_frames = 0
    with (dst / "meta" / "episodes.jsonl").open("w") as out:
        for row in load_jsonl(src / "meta" / "episodes.jsonl"):
            if int(row["episode_index"]) in keep_set:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_frames += int(row["length"])

    ep_stats_src = src / "meta" / "episodes_stats.jsonl"
    if ep_stats_src.exists():
        with (dst / "meta" / "episodes_stats.jsonl").open("w") as out:
            for row in load_jsonl(ep_stats_src):
                if int(row.get("episode_index")) in keep_set:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")

    for name in ("tasks.jsonl", "stats.json", "modality.json", "relative_stats_40.json"):
        s = src / "meta" / name
        if s.exists():
            shutil.copy2(s, dst / "meta" / name)
            print(f"  meta/{name} copied as-is")

    new_info = dict(info)
    new_info["total_episodes"] = len(keep)
    new_info["total_frames"] = total_frames
    new_info["total_videos"] = len(keep) * len(video_keys)
    new_info["total_chunks"] = 1
    new_info["splits"] = {"train": f"0:{len(keep)}"}
    (dst / "meta" / "info.json").write_text(
        json.dumps(new_info, ensure_ascii=False, indent=2)
    )

    print(f"[done] {dst}")
    print(f"[done] {len(keep)} episodes, {total_frames} frames")
    print(f"[note] stats.json copied as-is (reflects original full dataset, "
          f"regenerate if you need exact stats)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
