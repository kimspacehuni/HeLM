#!/usr/bin/env python3
"""Filter subtasks.jsonl to keep only episodes whose move sequence matches a
canonical rule.

Rules:
  swap   (3-move): position priority "TL < BL < TR < BR"; the
                   lower-priority bowl moves to the empty first.
                   Canonical sequence: [A->C, B->A, C->B] where {A,B} are the
                   two bowl positions (A is lower priority), C is the empty.
  rotate (4-move): one step clockwise per move.
                   Clockwise order: TL -> TR -> BR -> BL -> TL.
                   Canonical sequence repeats: bowl at CW_PREV(empty) -> empty.

Reads:  <task_meta_dir>/subtasks.jsonl
Writes: <task_meta_dir>/subtasks_canonical.jsonl  (same schema; only kept eps)
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

POS_PRIORITY = {"topleft": 0, "bottomleft": 1, "topright": 2, "bottomright": 3}
CW_NEXT = {"topleft": "topright", "topright": "bottomright",
           "bottomright": "bottomleft", "bottomleft": "topleft"}
CW_PREV = {v: k for k, v in CW_NEXT.items()}


def parse_move(label: str) -> Tuple[str, str]:
    body = label[len("move_"):]
    src, dst = body.split("_bowl_to_")
    return src, dst


def derive_init_state(seq: List[Tuple[str, str]], n_bowls: int) -> Tuple[List[str], str]:
    touched: List[str] = []
    for s, d in seq:
        if s not in touched: touched.append(s)
        if d not in touched: touched.append(d)
    empty0 = seq[0][1]
    bowls0 = [p for p in touched if p != empty0]
    if len(bowls0) != n_bowls:
        raise ValueError(f"expected {n_bowls} bowls, derived {bowls0} (touched={touched})")
    return bowls0, empty0


def canonical_swap(seq: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    bowls, empty = derive_init_state(seq, 2)
    sorted_bowls = sorted(bowls, key=lambda p: POS_PRIORITY[p])
    A, B = sorted_bowls[0], sorted_bowls[1]
    C = empty
    return [(A, C), (B, A), (C, B)]


def canonical_rotate_cw(seq: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    bowls, empty = derive_init_state(seq, 3)
    result: List[Tuple[str, str]] = []
    cur_empty = empty
    for _ in range(4):
        src = CW_PREV[cur_empty]
        dst = cur_empty
        result.append((src, dst))
        cur_empty = src
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=["swap", "rotate"], required=True)
    args = ap.parse_args()

    in_path = Path(args.subtasks)
    out_path = Path(args.out)

    by_ep: Dict[int, List[Dict]] = defaultdict(list)
    with in_path.open() as f:
        for ln in f:
            if not ln.strip():
                continue
            d = json.loads(ln)
            by_ep[int(d["episode_index"])].append(d)

    canon_fn = canonical_swap if args.kind == "swap" else canonical_rotate_cw
    n_bowls = 2 if args.kind == "swap" else 3
    kept_eps = set()
    init_count: Dict[Tuple[str, Tuple[str, ...]], int] = defaultdict(int)

    for ep, lst in by_ep.items():
        seq_dicts = sorted(lst, key=lambda x: x["start_frame"])
        seq = [parse_move(d["label"]) for d in seq_dicts]
        try:
            canon = canon_fn(seq)
        except Exception:
            continue
        if seq == canon:
            kept_eps.add(ep)
            bowls, empty = derive_init_state(seq, n_bowls)
            init_count[(empty, tuple(sorted(bowls)))] += 1

    with in_path.open() as f, out_path.open("w") as out:
        for ln in f:
            if not ln.strip():
                continue
            d = json.loads(ln)
            if int(d["episode_index"]) in kept_eps:
                out.write(ln if ln.endswith("\n") else ln + "\n")

    total_eps = len(by_ep)
    print(f"[{args.kind}] kept {len(kept_eps)}/{total_eps} episodes "
          f"({100.0*len(kept_eps)/total_eps:.1f}%)")
    print(f"[{args.kind}] {len(init_count)} unique init configurations covered:")
    for (empty, bowls), n in sorted(init_count.items(), key=lambda x: -x[1]):
        print(f"    empty={empty:>11s}  bowls={','.join(bowls):<28s}  {n} eps")
    print(f"[{args.kind}] wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
