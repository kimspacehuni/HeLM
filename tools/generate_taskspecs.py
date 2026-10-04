#!/usr/bin/env python3
"""Generate HeLM taskspec JSON files for swap / rotate tasks based on
auto-annotated unique sequences.

Reads:
  <task_07_sub>/meta/subtasks.jsonl   (per-episode 3-move sequences for swap)
  <task_08_sub>/meta/subtasks.jsonl   (per-episode 4-move sequences for rotate)
  (or the *_sub LeRobot meta — but it's easier to read the source task_07/task_08
  meta because that's where subtasks.jsonl lives)

Writes (one file per unique observed sequence):
  helm_datasets/libero_taskspecs/swap_<task_id>.json   (intra = [2])
  helm_datasets/libero_taskspecs/rotate_<task_id>.json (intra = [3])

Follows the existing taskspec convention strictly:
  - Working_Memory: state/progress BEFORE the current action_command
  - Episodic_Context: "None" for ALL intermediate steps; summary at "done"

For swap, the canonical plan is:
  bowl_A at posA, bowl_B at posB, empty at posC
  → A→C   (clear A's spot)
  → B→A   (B to A's spot)
  → C→B   (A from C to B's spot)
Whether the algorithm picked bowl_A or bowl_B first is reflected in the actual
sequence and recorded as-is.

For rotate, the rotation order varies across episodes; each unique sequence
becomes its own taskspec.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


POSITIONS = ("topleft", "topright", "bottomleft", "bottomright")
POS_SHORT = {"topleft": "TL", "topright": "TR", "bottomleft": "BL", "bottomright": "BR"}

# Full 12-move vocabulary for swap/rotate. Every taskspec advertises ALL 12
# moves in llp_commands / event_list so the HLP can pick whichever move
# matches the actual env init state (16 swap permutations / 8 rotate
# permutations all share the same vocabulary). The memory_grid still encodes
# the canonical sequence for the specific init this taskspec describes —
# build_helm uses memory_grid to construct ground-truth transitions, so
# training stays aligned to the per-episode pattern, while inference gets
# the full vocab to choose from.
FULL_VOCAB_MOVES = [f"move_{s}_bowl_to_{d}"
                    for s in POSITIONS for d in POSITIONS if s != d]
LLP_COMMANDS_FULL = "\n".join(FULL_VOCAB_MOVES) + "\ndone"
EVENT_LIST_FULL = "\n".join(FULL_VOCAB_MOVES) + "\nnone\ndone"


def parse_move(label: str) -> Tuple[str, str]:
    """move_<src>_bowl_to_<dst> → (src, dst)"""
    assert label.startswith("move_") and "_bowl_to_" in label, label
    body = label[len("move_"):]
    src, dst = body.split("_bowl_to_")
    return src, dst


def derive_initial_state(seq: List[Tuple[str, str]]) -> Tuple[List[str], str]:
    """Given the full sequence of (src, dst) moves, derive the initial bowl
    positions and the empty plate position.

    Approach: simulate the moves backwards from the final state. For swap and
    rotate, the FIRST move's src is always an initial-bowl position, and its
    dst is the initial-empty position (the unused plate). Each subsequent move
    either picks up from a non-empty plate that has a bowl or from a previously
    used empty buffer.

    Simpler & equivalent: track positions used over time. The set of all unique
    positions that ever appeared as src or dst gives the full set of plate
    positions in this episode. The position appearing FIRST as dst (but not yet
    as src in any prior move) is the initial empty.
    """
    # All plates touched
    touched = []
    for s, d in seq:
        if s not in touched:
            touched.append(s)
        if d not in touched:
            touched.append(d)
    # Initial empty = the first dst that wasn't already touched as src
    # In a valid swap/rotate, the first move's dst IS the initial empty.
    empty0 = seq[0][1]
    # Initial bowls = all touched plates EXCEPT empty0
    bowls0 = [p for p in touched if p != empty0]
    return bowls0, empty0


def simulate(seq: List[Tuple[str, str]]) -> List[Tuple[List[str], str]]:
    """Run the move sequence and return state (bowls, empty) BEFORE each move
    and after the final move.

    Returns: list of (bowls, empty) — length len(seq) + 1.
             First entry = initial, last entry = final state.
    """
    bowls, empty = derive_initial_state(seq)
    states = [(list(bowls), empty)]
    for s, d in seq:
        # bowl moves from s to d. After: s is empty, d has bowl, others unchanged.
        if s in bowls:
            bowls.remove(s)
        bowls.append(d)
        empty = s
        states.append((sorted(bowls), empty))
    return states


def wm_for_step(bowls: List[str], empty: str,
                step_idx: int,
                total_steps: int,
                task_kind: str,
                last_placed: str | None) -> str:
    """Build Working_Memory string for the state BEFORE step (step_idx).

    Format (counter + last_placed, rule-friendly):
      step 0     : "Bowls: TL, BL; Empty: TR; Done: 0/3; Goal: swap"
      step i>0   : "Bowls: TR, BL; Empty: TL; Done: 1/3; Last_Placed: TR"
    """
    bowls_str = ", ".join(bowls)
    goal = "swap" if task_kind == "swap" else "rotate"
    if step_idx == 0:
        return (f"Bowls: {bowls_str}; Empty: {empty}; "
                f"Done: 0/{total_steps}; Goal: {goal}")
    return (f"Bowls: {bowls_str}; Empty: {empty}; "
            f"Done: {step_idx}/{total_steps}; Last_Placed: {last_placed}")


def ec_done(initial_bowls: List[str], initial_empty: str,
            task_kind: str) -> str:
    """Episodic context to set ONLY at the 'done' step.

    Pattern follows cream cheese: short, factual record of what happened.
    """
    verb = "swapped" if task_kind == "swap" else "rotated"
    return (f"Bowls were at {', '.join(initial_bowls)}; "
            f"{verb} via {initial_empty}")


def short_name(seq: List[Tuple[str, str]]) -> str:
    """Compact filename suffix from the move sequence."""
    return "_".join(f"{POS_SHORT[s]}2{POS_SHORT[d]}" for s, d in seq)


def build_taskspec(seq: List[Tuple[str, str]],
                   task_kind: str,
                   task_text: str) -> Dict:
    """Construct one taskspec dict for a given move sequence."""
    moves = [f"move_{s}_bowl_to_{d}" for s, d in seq]
    n = len(moves)
    intra = [2 * (n - 1)]  # Match existing convention? bowl_1 has intra=[2] for 2 moves.
    # Actually existing convention: intra = [<len(commands) - 1>] for one full cycle?
    # bowl_1: 2 moves (lift, place) → intra=[2]; bowl_3: 6 moves → intra=[6]
    # So intra = [number of commands] basically. Use n.
    intra = [n]

    states = simulate(seq)
    initial_bowls, initial_empty = states[0]
    memory_grid_inner = []
    for i, m in enumerate(moves):
        bowls_i, empty_i = states[i]
        # last_placed = dst of the previous move (None at step 0)
        last_placed = seq[i - 1][1] if i > 0 else None
        memory_grid_inner.append({
            "Action_Command": m,
            "Working_Memory": wm_for_step(bowls_i, empty_i, i, n,
                                           task_kind, last_placed),
            "Episodic_Context": "None",
        })
    memory_grid_inner.append({
        "Action_Command": "done",
        "Working_Memory": "task done (None)",
        "Episodic_Context": ec_done(initial_bowls, initial_empty, task_kind),
    })

    return {
        "task_id": f"{task_kind}_{short_name(seq)}",
        "inter": 0,
        "intra": intra,
        "task_text": [task_text],
        "episode_filters": [[{"tasks": f"['{m}']"} for m in moves]],
        "llp_commands": LLP_COMMANDS_FULL,
        "event_list": EVENT_LIST_FULL,
        "event_grid": [list(moves)],
        "memory_grid": [memory_grid_inner],
    }


def load_sequences(subtasks_path: Path) -> Counter:
    """Return Counter of unique tuple-of-labels sequences across all episodes."""
    by_ep: Dict[int, List[Dict]] = defaultdict(list)
    with subtasks_path.open() as f:
        for ln in f:
            if not ln.strip():
                continue
            d = json.loads(ln)
            by_ep[int(d["episode_index"])].append(d)
    seqs: Counter = Counter()
    for ep_idx, lst in by_ep.items():
        seq = tuple(s["label"] for s in sorted(lst, key=lambda x: x["start_frame"]))
        seqs[seq] += 1
    return seqs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap-subtasks", required=True,
                    help="task_07/meta/subtasks.jsonl")
    ap.add_argument("--rotate-subtasks", required=True,
                    help="task_08/meta/subtasks.jsonl")
    ap.add_argument("--out-dir", default=str(
        Path(__file__).resolve().parents[1] / "helm_datasets" / "libero_taskspecs"))
    ap.add_argument("--swap-task-text",
                    default="swap the 2 bowls on their plates using the empty plate")
    ap.add_argument("--rotate-task-text",
                    default="rotate the 3 bowls on their plates from left to right using the empty plate")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Tuple[str, str, int]] = []

    for kind, sub_path, task_text in (
        ("swap",   Path(args.swap_subtasks),   args.swap_task_text),
        ("rotate", Path(args.rotate_subtasks), args.rotate_task_text),
    ):
        seqs = load_sequences(sub_path)
        print(f"[{kind}] {sum(seqs.values())} eps, {len(seqs)} unique sequences")
        for seq_tuple, n in seqs.most_common():
            seq = [parse_move(lbl) for lbl in seq_tuple]
            spec = build_taskspec(seq, kind, task_text)
            fname = f"{spec['task_id']}.json"
            (out_dir / fname).write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
            summary.append((kind, fname, n))

    print("\n=== Generated taskspecs ===")
    for kind, fname, n in summary:
        print(f"  [{kind}] {fname}  (covers {n} eps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
