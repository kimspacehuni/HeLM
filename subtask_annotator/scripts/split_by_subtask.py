"""Split a lerobot v2.1 dataset into per-subtask episodes based on meta/subtasks.jsonl.

Each annotated subtask becomes one new episode. Outputs a new v2.1 dataset at --dest.
Videos are re-encoded via ffmpeg (codec preserved by default). Stats are regenerated.
`relative_stats_40.json` is intentionally NOT regenerated (custom VLA artifact).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.dataset import LerobotV21Dataset


VECTOR_FEATURES = ("action", "observation.state", "observation.effort", "observation.velocity")


@dataclass
class SegmentSpec:
    new_episode_index: int
    source_episode_index: int
    start_frame: int
    end_frame: int
    label: str
    task_index: int

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1


def log(stage: str, msg: str = "") -> None:
    print(f"[{stage}] {msg}", flush=True)


def load_subtasks(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def validate(subtasks: list[dict], ds: LerobotV21Dataset) -> tuple[list[str], list[str]]:
    warns: list[str] = []
    errors: list[str] = []
    by_ep: dict[int, list[dict]] = {}
    for s in subtasks:
        by_ep.setdefault(s["episode_index"], []).append(s)

    for ep_idx, subs in by_ep.items():
        if ep_idx not in ds.episodes:
            errors.append(f"episode {ep_idx} referenced in subtasks.jsonl is not in the dataset")
            continue
        ep_length = ds.episodes[ep_idx].length
        for s in subs:
            if s["start_frame"] < 0 or s["end_frame"] >= ep_length:
                errors.append(
                    f"ep {ep_idx}: [{s['start_frame']},{s['end_frame']}] out of range [0,{ep_length})"
                )
            if s["start_frame"] > s["end_frame"]:
                errors.append(
                    f"ep {ep_idx}: start_frame > end_frame for '{s['label']}'"
                )
        subs_sorted = sorted(subs, key=lambda x: x["start_frame"])
        for i, a in enumerate(subs_sorted):
            for b in subs_sorted[i + 1:]:
                if b["start_frame"] > a["end_frame"]:
                    break
                warns.append(
                    f"ep {ep_idx}: overlap '{a['label']}' [{a['start_frame']},{a['end_frame']}] "
                    f"vs '{b['label']}' [{b['start_frame']},{b['end_frame']}]"
                )
        covered = np.zeros(ep_length, dtype=bool)
        for s in subs:
            s0 = max(0, s["start_frame"])
            s1 = min(ep_length - 1, s["end_frame"])
            covered[s0:s1 + 1] = True
        gap_start = None
        for i in range(ep_length):
            if not covered[i] and gap_start is None:
                gap_start = i
            elif covered[i] and gap_start is not None:
                warns.append(
                    f"ep {ep_idx}: unlabeled gap [{gap_start},{i - 1}] ({i - gap_start} frames)"
                )
                gap_start = None
        if gap_start is not None:
            warns.append(
                f"ep {ep_idx}: unlabeled gap [{gap_start},{ep_length - 1}] ({ep_length - gap_start} frames)"
            )

    for ep_idx in ds.episodes:
        if ep_idx not in by_ep:
            warns.append(f"episode {ep_idx} has no subtasks (entire episode uncovered)")

    return warns, errors


def build_segments(subtasks: list[dict]) -> tuple[list[SegmentSpec], list[str]]:
    labels: list[str] = []
    label_to_idx: dict[str, int] = {}
    for s in subtasks:
        if s["label"] not in label_to_idx:
            label_to_idx[s["label"]] = len(labels)
            labels.append(s["label"])
    sorted_subs = sorted(subtasks, key=lambda s: (s["episode_index"], s["start_frame"]))
    segs = [
        SegmentSpec(
            new_episode_index=i,
            source_episode_index=s["episode_index"],
            start_frame=s["start_frame"],
            end_frame=s["end_frame"],
            label=s["label"],
            task_index=label_to_idx[s["label"]],
        )
        for i, s in enumerate(sorted_subs)
    ]
    return segs, labels


def slice_parquet(src: Path, seg: SegmentSpec, global_offset: int, fps: int) -> pa.Table:
    t = pq.read_table(src)
    fi = t.column("frame_index").to_numpy()
    mask = (fi >= seg.start_frame) & (fi <= seg.end_frame)
    t = t.filter(pa.array(mask))
    n = t.num_rows
    if n != seg.length:
        raise RuntimeError(
            f"ep {seg.source_episode_index}: expected {seg.length} rows, got {n} after slicing"
        )

    new_cols = []
    for field in t.schema:
        name = field.name
        if name == "frame_index":
            arr = pa.array(np.arange(n, dtype=np.int64))
        elif name == "episode_index":
            arr = pa.array(np.full(n, seg.new_episode_index, dtype=np.int64))
        elif name == "index":
            arr = pa.array(np.arange(global_offset, global_offset + n, dtype=np.int64))
        elif name == "timestamp":
            arr = pa.array((np.arange(n) / fps).astype(np.float32))
        elif name == "task_index":
            arr = pa.array(np.full(n, seg.task_index, dtype=np.int64))
        elif name == "language_instruction":
            arr = pa.array([seg.label] * n, type=field.type)
        elif name == "next.done":
            done = np.zeros(n, dtype=bool)
            if n > 0:
                done[-1] = True
            arr = pa.array(done)
        else:
            arr = t.column(name).combine_chunks()
        new_cols.append(arr)
    return pa.Table.from_arrays(new_cols, schema=t.schema)


def detect_codec(video_path: Path) -> str:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def ffmpeg_codec_args(src_codec: str, enc_threads: int) -> list[str]:
    thr = max(1, enc_threads)
    if src_codec == "av1":
        return [
            "-c:v", "libsvtav1", "-preset", "8", "-crf", "30", "-pix_fmt", "yuv420p",
            "-svtav1-params", f"lp={thr}",
        ]
    if src_codec == "h264":
        return [
            "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
            "-threads", str(thr),
        ]
    if src_codec == "hevc":
        return [
            "-c:v", "libx265", "-preset", "medium", "-crf", "28", "-pix_fmt", "yuv420p",
            "-threads", str(thr),
        ]
    return [
        "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
        "-threads", str(thr),
    ]


def trim_video(
    src: Path, dst: Path, start_frame: int, end_frame: int, fps: int, codec_args: list[str]
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = end_frame - start_frame + 1
    start_time = start_frame / fps
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_time:.6f}",
        "-i", str(src),
        "-frames:v", str(count),
        *codec_args,
        "-an",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {dst.name}: {r.stderr.strip()}")


def _trim_worker(args: dict) -> tuple[str, float, str | None]:
    key = args["key"]
    t0 = time.perf_counter()
    try:
        trim_video(
            Path(args["src"]), Path(args["dst"]),
            args["start_frame"], args["end_frame"], args["fps"], args["codec_args"],
        )
        return key, time.perf_counter() - t0, None
    except Exception as e:
        return key, time.perf_counter() - t0, str(e)


def compute_vector_stats(table: pa.Table, cols: tuple[str, ...]) -> dict:
    out = {}
    for col in cols:
        if col not in table.schema.names:
            continue
        values = table.column(col).to_pylist()
        arr = np.asarray(values, dtype=np.float32)
        out[col] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "count": int(arr.shape[0]),
        }
    return out


def aggregate_stats(per_ep: list[dict]) -> dict:
    if not per_ep:
        return {}
    total = {}
    keys = set()
    for s in per_ep:
        keys.update(s.keys())
    for key in keys:
        means, m2s, counts, mins, maxs = [], [], [], [], []
        for s in per_ep:
            if key not in s:
                continue
            m = np.asarray(s[key]["mean"])
            sd = np.asarray(s[key]["std"])
            c = int(s[key]["count"])
            means.append(m * c)
            m2s.append((sd ** 2 + m ** 2) * c)
            counts.append(c)
            mins.append(np.asarray(s[key]["min"]))
            maxs.append(np.asarray(s[key]["max"]))
        C = sum(counts)
        if C == 0:
            continue
        mean = np.sum(means, axis=0) / C
        second = np.sum(m2s, axis=0) / C
        var = np.clip(second - mean ** 2, 0.0, None)
        std = np.sqrt(var)
        total[key] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": np.min(mins, axis=0).tolist(),
            "max": np.max(maxs, axis=0).tolist(),
            "count": C,
        }
    return total


def default_dest(src: Path) -> Path:
    name = src.name
    if name.endswith("_subtask"):
        name = name[: -len("_subtask")]
    return src.parent / f"{name}_per_subtask"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="path to source lerobot v2.1 dataset")
    parser.add_argument("--dest", default=None, help="output dataset path (default: auto)")
    parser.add_argument("--force", action="store_true", help="overwrite --dest if exists")
    parser.add_argument("--workers", type=int, default=2,
                        help="parallel ffmpeg processes (default: 2; AV1 re-encode is very CPU-heavy)")
    parser.add_argument("--enc-threads", type=int, default=8,
                        help="threads per ffmpeg process (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="validate and print plan; no writes")
    args = parser.parse_args()

    src_root = Path(args.source).resolve()
    dst_root = Path(args.dest).resolve() if args.dest else default_dest(src_root)

    log("1/6", f"Loading source dataset: {src_root}")
    ds = LerobotV21Dataset(src_root)
    log("1/6", f"  {ds.info.codebase_version}, {ds.info.total_episodes} episodes, fps={ds.info.fps}")
    log("1/6", f"  video_keys: {ds.info.video_keys}")

    subtasks_path = src_root / "meta" / "subtasks.jsonl"
    log("2/6", f"Loading subtasks: {subtasks_path}")
    if not subtasks_path.exists():
        log("2/6", "  ERROR: subtasks.jsonl not found")
        return 2
    subtasks = load_subtasks(subtasks_path)
    log("2/6", f"  found {len(subtasks)} subtasks across {len({s['episode_index'] for s in subtasks})} episodes")
    if not subtasks:
        log("2/6", "  ERROR: subtasks.jsonl is empty")
        return 2

    log("3/6", "Validating…")
    warns, errors = validate(subtasks, ds)
    for w in warns:
        log("3/6", f"  WARN  {w}")
    for e in errors:
        log("3/6", f"  ERROR {e}")
    log("3/6", f"  {len(warns)} warnings, {len(errors)} errors")
    if errors:
        return 2

    segments, labels = build_segments(subtasks)
    log("4/6", f"Plan: {len(segments)} output episodes, {len(labels)} unique labels")
    for i, lbl in enumerate(labels):
        count = sum(1 for s in segments if s.label == lbl)
        log("4/6", f"  [{i:>2}] '{lbl}' ({count} segments)")

    if args.dry_run:
        log("4/6", "Dry run — stopping before writes.")
        return 0

    if dst_root.exists():
        if not args.force:
            log("4/6", f"  ERROR: {dst_root} already exists (use --force to overwrite)")
            return 2
        log("4/6", f"  --force: removing existing {dst_root}")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)
    (dst_root / "data" / "chunk-000").mkdir(parents=True)
    for vk in ds.info.video_keys:
        (dst_root / "videos" / "chunk-000" / vk).mkdir(parents=True)
    (dst_root / "meta").mkdir()

    log("5/6", f"Writing {len(segments)} episodes to {dst_root}")
    per_ep_stats: list[dict] = []
    episodes_stats_rows: list[dict] = []
    global_offset = 0
    total_frames = 0
    fps = ds.info.fps

    trim_jobs: list[dict] = []
    src_codecs: dict[str, str] = {}
    for vk in ds.info.video_keys:
        sample = ds.video_path(next(iter(ds.episodes)), vk)
        src_codecs[vk] = detect_codec(sample) or "av1"
    for vk, codec in src_codecs.items():
        log("5/6", f"  codec[{vk}] = {codec} → {ffmpeg_codec_args(codec, args.enc_threads)[:3]}")

    for idx, seg in enumerate(segments, 1):
        src_parq = ds.data_path(seg.source_episode_index)
        dst_parq = dst_root / f"data/chunk-000/episode_{seg.new_episode_index:06d}.parquet"
        log(
            "5/6",
            f"  [{idx:>3}/{len(segments)}] new_ep={seg.new_episode_index:06d} "
            f"src_ep={seg.source_episode_index} [{seg.start_frame},{seg.end_frame}] "
            f"len={seg.length} label={seg.label!r}",
        )

        t = slice_parquet(src_parq, seg, global_offset, fps)
        pq.write_table(t, dst_parq)
        global_offset += seg.length
        total_frames += seg.length

        ep_stats = compute_vector_stats(t, VECTOR_FEATURES)
        per_ep_stats.append(ep_stats)
        episodes_stats_rows.append({
            "episode_index": seg.new_episode_index,
            "stats": ep_stats,
        })

        for vk in ds.info.video_keys:
            src_v = ds.video_path(seg.source_episode_index, vk)
            dst_v = dst_root / f"videos/chunk-000/{vk}/episode_{seg.new_episode_index:06d}.mp4"
            trim_jobs.append({
                "key": f"ep{seg.new_episode_index:06d}/{vk}",
                "src": str(src_v),
                "dst": str(dst_v),
                "start_frame": seg.start_frame,
                "end_frame": seg.end_frame,
                "fps": fps,
                "codec_args": ffmpeg_codec_args(src_codecs[vk], args.enc_threads),
            })

    log("5/6", f"Trimming {len(trim_jobs)} videos with {args.workers} workers…")
    done = 0
    t0 = time.perf_counter()
    failed: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_trim_worker, j) for j in trim_jobs]
        for fut in as_completed(futures):
            key, dt, err = fut.result()
            done += 1
            if err:
                failed.append((key, err))
                log("5/6", f"    [{done:>4}/{len(trim_jobs)}] FAIL {key} ({dt:.2f}s): {err}")
            else:
                log("5/6", f"    [{done:>4}/{len(trim_jobs)}] ok   {key} ({dt:.2f}s)")
    log("5/6", f"  video trimming done in {time.perf_counter() - t0:.1f}s, failures={len(failed)}")
    if failed:
        log("5/6", "  ERROR: video trimming had failures, aborting meta write")
        return 3

    log("6/6", "Writing meta files")
    tasks_path = dst_root / "meta" / "tasks.jsonl"
    with tasks_path.open("w") as f:
        for i, lbl in enumerate(labels):
            f.write(json.dumps({"task_index": i, "task": lbl}, ensure_ascii=False) + "\n")
    log("6/6", f"  tasks.jsonl ({len(labels)} tasks)")

    episodes_path = dst_root / "meta" / "episodes.jsonl"
    with episodes_path.open("w") as f:
        for seg in segments:
            row = {
                "episode_index": seg.new_episode_index,
                "tasks": [seg.label],
                "length": seg.length,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log("6/6", f"  episodes.jsonl ({len(segments)} episodes)")

    ep_stats_path = dst_root / "meta" / "episodes_stats.jsonl"
    with ep_stats_path.open("w") as f:
        for row in episodes_stats_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log("6/6", f"  episodes_stats.jsonl")

    dataset_stats = aggregate_stats(per_ep_stats)
    # Image stats live only in the source dataset's meta/stats.json (not in
    # episodes_stats.jsonl), so aggregate_stats can't recover them. Try to
    # propagate from src/meta/stats.json; if missing (e.g., a subset that
    # didn't carry stats.json), fall back to placeholder zero stats from
    # info.json's video features. ImageNet stats override mean/std at train
    # time, so the values themselves don't matter — only the keys must
    # exist (factory.py: `dataset.meta.stats[<image_key>][stats_type] = ...`).
    src_stats_path = src_root / "meta" / "stats.json"
    src_stats = {}
    if src_stats_path.is_file():
        try:
            src_stats = json.loads(src_stats_path.read_text())
        except Exception as e:
            print(f"[warn] could not read {src_stats_path}: {e}")

    src_info = json.loads((src_root / "meta" / "info.json").read_text())
    image_keys = [k for k, v in src_info.get("features", {}).items() if v.get("dtype") == "video"]
    placeholder = {
        "mean": [[[0.0]], [[0.0]], [[0.0]]],
        "std":  [[[1.0]], [[1.0]], [[1.0]]],
        "min":  [[[0.0]], [[0.0]], [[0.0]]],
        "max":  [[[1.0]], [[1.0]], [[1.0]]],
        "count": [0],
    }
    for k in image_keys:
        if k in dataset_stats:
            continue
        dataset_stats[k] = src_stats.get(k, placeholder)

    # Action / observation.state: per-segment stats from a SHORT subtask trajectory
    # (e.g. only lift_bowl frames) are highly biased — pi0 learns trivial
    # mean-prediction and the bias dominates. If src_stats has these keys,
    # prefer them (broader distribution = less harmful trivial output).
    for k in ("action", "observation.state"):
        if k in src_stats:
            dataset_stats[k] = src_stats[k]
            print(f"[split] propagated {k} stats from src (broader distribution)")
    (dst_root / "meta" / "stats.json").write_text(
        json.dumps(dataset_stats, ensure_ascii=False, indent=2)
    )
    log("6/6", f"  stats.json ({len(dataset_stats)} features)")

    src_info = json.loads((src_root / "meta" / "info.json").read_text())
    new_info = dict(src_info)
    new_info["total_episodes"] = len(segments)
    new_info["total_frames"] = total_frames
    new_info["total_tasks"] = len(labels)
    new_info["total_videos"] = len(segments) * len(ds.info.video_keys)
    new_info["total_chunks"] = 1
    # All split episodes go into chunk-000, so chunks_size must be ≥
    # total_episodes — otherwise LeRobot's `idx // chunks_size` lookup
    # routes episodes 50+ to a non-existent chunk-001 (assertion fail).
    new_info["chunks_size"] = max(int(src_info.get("chunks_size", 1)), len(segments))
    new_info["splits"] = {"train": f"0:{len(segments)}"}
    (dst_root / "meta" / "info.json").write_text(
        json.dumps(new_info, ensure_ascii=False, indent=2)
    )
    log("6/6", f"  info.json (total_episodes={len(segments)}, total_frames={total_frames})")

    src_modality = src_root / "meta" / "modality.json"
    if src_modality.exists():
        shutil.copy2(src_modality, dst_root / "meta" / "modality.json")
        log("6/6", "  modality.json (copied)")

    src_rel = src_root / "meta" / "relative_stats_40.json"
    if src_rel.exists():
        log("6/6", "  relative_stats_40.json SKIPPED (project-specific; regenerate via VLA pipeline)")

    log("done", f"Output: {dst_root}")
    log("done", f"Summary: {len(segments)} episodes, {total_frames} frames, {len(labels)} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
