"""FastAPI server for subtask annotation."""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.dataset import LerobotV21Dataset
from backend.frame_cache import FrameCache


class SubtaskIn(BaseModel):
    episode_index: int
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    label: str
    annotator: str | None = None


class SubtaskOut(SubtaskIn):
    id: str
    created_at: str


class LabelsIn(BaseModel):
    labels: list[str]


def _load_subtasks(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return list(data.get("labels", []))


def _save_labels(path: Path, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"labels": labels}, ensure_ascii=False, indent=2))
    tmp.replace(path)


def create_app(data_root: Path, frontend_dir: Path) -> FastAPI:
    ds = LerobotV21Dataset(data_root)
    cache = FrameCache(fps=ds.info.fps)
    subtasks_path = data_root / "meta" / "subtasks.jsonl"
    labels_path = data_root / "meta" / "subtask_labels.json"
    subtasks_lock = threading.Lock()
    labels_lock = threading.Lock()

    app = FastAPI(title="Subtask Annotator")

    @app.get("/api/info")
    def get_info():
        return {
            "codebase_version": ds.info.codebase_version,
            "fps": ds.info.fps,
            "total_episodes": ds.info.total_episodes,
            "video_keys": ds.info.video_keys,
            "data_root": str(data_root),
        }

    @app.get("/api/episodes")
    def list_episodes():
        eps = ds.list_episodes()
        saved = _load_subtasks(subtasks_path)
        counts: dict[int, int] = {}
        for s in saved:
            counts[s["episode_index"]] = counts.get(s["episode_index"], 0) + 1
        return [
            {
                "episode_index": e.episode_index,
                "length": e.length,
                "tasks": e.tasks,
                "subtask_count": counts.get(e.episode_index, 0),
            }
            for e in eps
        ]

    @app.get("/api/episodes/{ep}")
    def get_episode(ep: int):
        if ep not in ds.episodes:
            raise HTTPException(404, f"episode {ep} not found")
        e = ds.get_episode(ep)
        return {
            "episode_index": e.episode_index,
            "length": e.length,
            "tasks": e.tasks,
            "video_keys": ds.info.video_keys,
            "fps": ds.info.fps,
            "subtasks": [s for s in _load_subtasks(subtasks_path) if s["episode_index"] == ep],
        }

    @app.get("/api/video")
    def get_video(ep: int, cam: str):
        if cam not in ds.info.video_keys:
            raise HTTPException(400, f"unknown cam '{cam}'")
        if ep not in ds.episodes:
            raise HTTPException(404, f"episode {ep} not found")
        path = ds.video_path(ep, cam)
        if not path.exists():
            raise HTTPException(404, f"video missing: {path}")
        return FileResponse(str(path), media_type="video/mp4")

    @app.get("/api/frame")
    def get_frame(ep: int, cam: str, frame: int):
        if cam not in ds.info.video_keys:
            raise HTTPException(400, f"unknown cam '{cam}'; choices: {ds.info.video_keys}")
        if ep not in ds.episodes:
            raise HTTPException(404, f"episode {ep} not found")
        length = ds.get_episode(ep).length
        if not (0 <= frame < length):
            raise HTTPException(400, f"frame {frame} out of range [0,{length})")
        video = ds.video_path(ep, cam)
        if not video.exists():
            raise HTTPException(404, f"video missing: {video}")
        try:
            data = cache.get(video, frame)
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"})

    @app.post("/api/subtasks", response_model=SubtaskOut)
    def add_subtask(s: SubtaskIn):
        if s.episode_index not in ds.episodes:
            raise HTTPException(404, f"episode {s.episode_index} not found")
        length = ds.get_episode(s.episode_index).length
        if not (0 <= s.start_frame < length):
            raise HTTPException(400, f"start_frame out of range [0,{length})")
        if not (0 <= s.end_frame < length):
            raise HTTPException(400, f"end_frame out of range [0,{length})")
        if s.end_frame < s.start_frame:
            raise HTTPException(400, "end_frame must be >= start_frame")
        if not s.label.strip():
            raise HTTPException(400, "label must be non-empty")

        entry = {
            "id": f"{int(time.time() * 1000)}-{os.getpid()}",
            "episode_index": s.episode_index,
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
            "label": s.label.strip(),
            "annotator": s.annotator,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with subtasks_lock:
            subtasks_path.parent.mkdir(parents=True, exist_ok=True)
            with subtasks_path.open("a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return JSONResponse(entry)

    @app.get("/api/labels")
    def get_labels():
        with labels_lock:
            return {"labels": _load_labels(labels_path)}

    @app.put("/api/labels")
    def put_labels(payload: LabelsIn):
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in payload.labels:
            s = raw.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            cleaned.append(s)
        with labels_lock:
            _save_labels(labels_path, cleaned)
        return {"labels": cleaned}

    @app.delete("/api/subtasks/{subtask_id}")
    def delete_subtask(subtask_id: str):
        with subtasks_lock:
            rows = _load_subtasks(subtasks_path)
            kept = [r for r in rows if r.get("id") != subtask_id]
            if len(kept) == len(rows):
                raise HTTPException(404, "subtask id not found")
            tmp = subtasks_path.with_suffix(".jsonl.tmp")
            with tmp.open("w") as f:
                for r in kept:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(subtasks_path)
        return {"deleted": subtask_id}

    @app.get("/")
    def root():
        return FileResponse(frontend_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True,
                        help="path to lerobot v2.1 dataset root")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    app = create_app(Path(args.data_root).resolve(), frontend_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
