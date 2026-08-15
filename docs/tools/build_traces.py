#!/usr/bin/env python3
"""Build the project page's memory-trace data from the TaskSpec sources.

The page's memory inspector replays real HeLM memory states rather than mock-ups,
so this reads helm_datasets/taskspecs/ and emits docs/static/js/traces.js.

Re-run after editing a taskspec:

    python docs/tools/build_traces.py

TaskSpec shape (see the paper appendix, Fig. 13-14):
  intra[b]           number of subtask steps in branch b
  event_grid[b][i]   the event that fires when step i completes
  memory_grid[b][i]  the memory state *before* step i runs
                     (length intra[b] + 1; the extra tail entry is the done state)

When inter > 0 the branches are consecutive episodes of one task, which is what the
inspector needs in order to show working memory resetting while episodic context is
promoted across the boundary.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPECS = ROOT / "helm_datasets" / "taskspecs"
OUT = ROOT / "docs" / "static" / "js" / "traces.js"

# TaskSpec directory -> the paper's task id, display name, and how to label variants.
# Descriptions stay in main.js; this file only carries what the inspector replays.
TASKS = [
    ("A1", "intra", "Press N Times",         "press_button_N_times",               "count"),
    ("A2", "intra", "Press in Order",        "press_button_in_order",              "seq"),
    ("A3", "intra", "Wipe the Window",       "wipe_the_window",                    "wipe_all"),
    ("A4", "intra", "Find the Object",       "find_object_in_drawer",              "drawer"),
    ("A5", "intra", "Pick, Place and Press", "pick_place_press",                   "seq"),
    ("E1", "inter", "Cumulative Press",      "press_button_N_times_M_times_total", "total"),
    ("E2", "inter", "Human-Order Press",     "press_button_in_human_order",        "seq"),
    ("E3", "inter", "Remaining Wipe",        "wipe_the_remaining_window",          "wipe_rest"),
    ("E4", "inter", "Direct Retrieval",      "open_drawer_with_object",            "drawer"),
    ("E5", "inter", "Original Restore",      "pick_place_original",                "seq"),
]

# Insertion order matters: it is the window's physical bottom-to-top order.
REGION = {"bottom": "B", "middle": "M", "top": "T"}


def variant_label(stem: str, kind: str) -> str:
    """Turn a taskspec filename into something readable in the variant picker."""
    if kind == "seq":
        # press_BGR / human_press_BGR / pickplace_BRW / pickplace_original_BRW
        return " → ".join(stem.rsplit("_", 1)[-1])

    if kind == "count":                       # press_blue_button_3
        return f"N = {stem.rsplit('_', 1)[-1]}"

    if kind == "total":                       # press_blue_button_1+2
        n, m = stem.rsplit("_", 1)[-1].split("+")
        return f"{n} + {m} = {int(n) + int(m)} total"

    if kind == "drawer":                      # find_top_left / open_top_left
        return stem.split("_", 1)[1].replace("_", "-")

    if kind == "wipe_all":                    # wipe_the_window
        return " → ".join(REGION)

    if kind == "wipe_rest":                   # wipe_bottom_middle_remain
        first = stem.split("_")[1:-1]
        rest = [r for r in REGION if r not in first]
        fmt = lambda rs: ", ".join(REGION[r] for r in rs)
        return f"{fmt(first)} → {fmt(rest)}"

    raise ValueError(f"unknown label kind: {kind}")


def build_episodes(spec: dict) -> list:
    """Flatten a taskspec into consecutive episodes of timeline states."""
    intra = spec["intra"]
    grid = spec["memory_grid"]
    events = spec["event_grid"]
    texts = spec["task_text"]

    episodes = []
    for b, n in enumerate(intra):
        states = grid[b]
        if len(states) != n + 1:
            raise ValueError(f"branch {b}: memory_grid is {len(states)}, expected {n + 1}")

        steps = []
        for i, m in enumerate(states):
            steps.append({
                "cmd": m["Action_Command"],
                "work": m["Working_Memory"],
                "epi": m["Episodic_Context"],
                # The event that advances *out* of this state; the tail state has none.
                "event": events[b][i] if i < len(events[b]) else None,
            })
        episodes.append({
            "instruction": texts[b] if b < len(texts) else texts[-1],
            "steps": steps,
        })
    return episodes


def main() -> int:
    if not SPECS.is_dir():
        print(f"taskspecs not found at {SPECS}", file=sys.stderr)
        return 1

    out, total = [], 0
    for tid, group, name, dirname, kind in TASKS:
        d = SPECS / dirname
        files = sorted(d.glob("*.json"))
        if not files:
            print(f"  ! {tid}: no taskspecs in {dirname}", file=sys.stderr)
            continue

        variants = []
        for f in files:
            spec = json.loads(f.read_text(encoding="utf-8"))
            if group == "inter" and spec.get("inter", 0) < 1:
                print(f"  ! {f.name}: expected an inter-episode spec", file=sys.stderr)
            variants.append({
                "id": f.stem,
                "label": variant_label(f.stem, kind),
                "episodes": build_episodes(spec),
            })

        total += len(variants)
        out.append({"id": tid, "group": group, "name": name, "variants": variants})
        print(f"  {tid}  {name:<24} {len(variants):>2} variants")

    body = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(
        "/* GENERATED by docs/tools/build_traces.py — do not edit by hand.\n"
        " * Real HeLM memory states, replayed by the inspector in main.js.\n"
        " * Regenerate after changing helm_datasets/taskspecs/. */\n"
        f"window.HELM_TRACES = {body};\n",
        encoding="utf-8",
    )
    print(f"\n  {total} variants -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
