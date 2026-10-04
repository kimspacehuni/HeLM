from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image
"""
export PYTHONPATH=$(pwd)
streamlit run helm_datasets/validate_helm_data.py
"""

# ---------- IO ----------
def collect_jsonl_files(p: Path) -> List[Path]:
    if p.is_file() and p.suffix == ".jsonl":
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.jsonl"))
    return []


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                st.warning(f"JSON parse error {path}:{ln}: {e}")
    return rows


def open_image(path: str) -> Optional[Image.Image]:
    try:
        p = Path(path)
        if not p.exists():
            return None
        return Image.open(p)
    except Exception:
        return None


# ---------- Index helpers ----------
def row_episode_key(r: Dict[str, Any]) -> Tuple[Any, Any]:
    return (r.get("chunk"), r.get("episode"))


def build_episode_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[Any, Any], List[int]]:
    idx: Dict[Tuple[Any, Any], List[int]] = {}
    for i, r in enumerate(rows):
        key = row_episode_key(r)
        if key[0] is None or key[1] is None:
            continue
        idx.setdefault(key, []).append(i)
    # keep stable order
    for k in idx:
        idx[k].sort()
    return idx


def current_episode_rows(ep_index: Dict[Tuple[Any, Any], List[int]], rows: List[Dict[str, Any]], cur_i: int) -> List[int]:
    key = row_episode_key(rows[cur_i])
    return ep_index.get(key, [])


def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def show_value(title: str, value: Any):
    """Render a value in Streamlit without JSON parse errors.

    - dict/list -> st.json
    - str/int/float/bool/None -> st.code/st.write
    """
    st.markdown(f"**{title}**")
    if isinstance(value, (dict, list)):
        st.json(value)
    elif value is None:
        st.code("None")
    elif isinstance(value, bool):
        st.code("true" if value else "false")
    else:
        # use code for compactness / copy-paste
        st.code(str(value))


# ---------- UI ----------
def main():
    st.set_page_config(layout="wide", page_title="HeLM v2 Simple Viewer")
    st.title("HeLM v2 Simple Viewer (detect/update)")

    st.sidebar.header("Load JSONL")
    path_str = st.sidebar.text_input("JSONL file or folder", value="")
    if st.sidebar.button("Load"):
        p = Path(path_str)
        files = collect_jsonl_files(p)
        if not files:
            st.sidebar.error("No .jsonl found.")
        else:
            st.session_state.files = [str(x) for x in files]
            st.session_state.file_idx = 0
            st.session_state.rows = read_jsonl(files[0])
            st.session_state.row_idx = 0

    if "files" not in st.session_state:
        st.info("왼쪽에서 jsonl 파일/폴더를 입력하고 Load를 누르세요.")
        return

    files: List[str] = st.session_state.files
    if not files:
        st.warning("No files loaded.")
        return

    # choose file
    file_idx = st.sidebar.selectbox(
        "File",
        options=list(range(len(files))),
        index=int(st.session_state.get("file_idx", 0)),
        format_func=lambda i: Path(files[i]).name,
    )

    if file_idx != st.session_state.get("file_idx", 0):
        st.session_state.file_idx = file_idx
        st.session_state.rows = read_jsonl(Path(files[file_idx]))
        st.session_state.row_idx = 0

    rows: List[Dict[str, Any]] = st.session_state.rows
    if not rows:
        st.warning("Empty JSONL.")
        return

    # build episode index
    ep_index = build_episode_index(rows)
    ep_keys = sorted(ep_index.keys(), key=lambda x: (str(x[0]), str(x[1])))

    st.sidebar.header("Navigate")
    # row slider
    row_idx = st.sidebar.slider("Row", 0, len(rows) - 1, int(st.session_state.get("row_idx", 0)))
    st.session_state.row_idx = int(row_idx)

    # prev/next row
    c1, c2 = st.sidebar.columns(2)
    if c1.button("◀ Prev row"):
        st.session_state.row_idx = max(0, st.session_state.row_idx - 1)
        rerun()
    if c2.button("Next row ▶"):
        st.session_state.row_idx = min(len(rows) - 1, st.session_state.row_idx + 1)
        rerun()

    # episode jump
    st.sidebar.subheader("Episode jump")
    cur_key = row_episode_key(rows[st.session_state.row_idx])
    cur_ep_label = f"{cur_key[0]}/{cur_key[1]}"
    # find current episode index
    cur_ep_pos = 0
    if cur_key in ep_index:
        try:
            cur_ep_pos = ep_keys.index(cur_key)
        except ValueError:
            cur_ep_pos = 0

    ep_pos = st.sidebar.selectbox(
        "Episode",
        options=list(range(len(ep_keys))),
        index=cur_ep_pos,
        format_func=lambda i: f"{ep_keys[i][0]}/{ep_keys[i][1]} ({len(ep_index[ep_keys[i]])} rows)",
    )

    if st.sidebar.button("Go to episode"):
        # jump to first row of that episode
        st.session_state.row_idx = ep_index[ep_keys[ep_pos]][0]
        rerun()

    # show current
    r = rows[st.session_state.row_idx]
    mode = r.get("mode")

    left, right = st.columns([0.5, 1])

    with left:
        st.subheader("Image (if exists)")
        images = r.get("images", {})
        if isinstance(images, dict) and images:
            # show all cameras sorted
            cams = sorted(images.keys())
            for cam in cams:
                st.caption(f"camera: {cam}")
                im = open_image(str(images[cam]))
                if im is None:
                    st.warning(f"Missing image: {images[cam]}")
                else:
                    st.image(im, use_container_width=True)
        else:
            st.info("No images in this row (likely update row).")



        # show same-episode row list quick jump
        st.subheader("Rows in this episode")
        ep_rows = current_episode_rows(ep_index, rows, st.session_state.row_idx)
        if ep_rows:
            # show small selector
            pick = st.selectbox(
                "Jump within episode",
                options=ep_rows,
                index=max(0, ep_rows.index(st.session_state.row_idx)) if st.session_state.row_idx in ep_rows else 0,
                format_func=lambda i: f"row {i} | mode={rows[i].get('mode')} | t={rows[i].get('t')} | t_event={rows[i].get('t_event')}",
            )
            if st.button("Go to selected row"):
                st.session_state.row_idx = int(pick)
                rerun()

    with right:
        # === Mode-specific TOP fields (the data you actually need to verify) ===
        # Long fields (system prompt, full row JSON) are pushed below so they
        # don't bury the answer.
        mode_str = str(mode).upper() if mode is not None else "UNKNOWN"
        st.markdown(f"### Mode: `{mode_str}`")

        if mode_str == "DETECT":
            ed = r.get("event_detected")
            ev = r.get("event")
            badge = "✅ TRUE" if ed is True else ("❌ FALSE" if ed is False else f"`{ed}`")
            st.markdown(f"**event_detected:** {badge}  &nbsp;&nbsp;|&nbsp;&nbsp; **event:** `{ev}`")
            show_value("gt_text", r.get("gt_text"))
        elif mode_str == "UPDATE":
            st.markdown("**memory_in (previous):**")
            show_value("memory_in", r.get("memory_in", {}))
            st.markdown("**memory_out (updated):**")
            show_value("memory_out", r.get("memory_out"))
            show_value("gt_text", r.get("gt_text"))
        else:
            show_value("gt_text", r.get("gt_text"))

        # === Compact metadata strip ===
        meta_cols = st.columns(4)
        meta_cols[0].caption(f"uid: `{r.get('uid')}`")
        meta_cols[1].caption(f"task_id: `{r.get('task_id')}`")
        meta_cols[2].caption(f"label: `{r.get('label')}`")
        meta_cols[3].caption(f"step: `{r.get('inter')}/{r.get('step')} f={r.get('frame_id')}`")

        # === Context fields (long) — moved below ===
        with st.expander("global_instruction (task)", expanded=False):
            show_value("global_instruction", r.get("task", ""))

        with st.expander("user_prompt (system prompt)", expanded=False):
            show_value("user_prompt", r.get("user_prompt"))

        with st.expander("gt_yaml", expanded=False):
            show_value("gt_yaml", r.get("gt_yaml"))

        # For DETECT also surface memory fields if present (some flows include them)
        if mode_str == "DETECT" and (r.get("memory_in") or r.get("memory_out")):
            with st.expander("memory_in / memory_out", expanded=False):
                show_value("memory_in", r.get("memory_in", {}))
                show_value("memory_out", r.get("memory_out"))

        with st.expander("Full row JSON", expanded=False):
            st.json(r)

        pc = r.get("prompt_context", {})
        if isinstance(pc, dict) and pc:
            with st.expander("prompt_context", expanded=False):
                st.json(pc)


if __name__ == "__main__":
    main()