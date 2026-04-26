#!/usr/bin/env python3
"""Summarize smoke-test runs by extracting `KL&PPL on <dataset>` lines from logs.

Usage:
    python scripts/collect_smoke_results.py [log_dir]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

LOG_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/smoke_20260426")
DATASETS = ["wikitext2", "ultrachat_2k", "numinamath"]
PAT = re.compile(r"KL&PPL on (\S+): ([-+0-9.eE]+), ([-+0-9.eE]+)")
LAYER_PAT = re.compile(r"(\d+)/28")


def parse_log(path: Path) -> dict:
    result = {"kl": {}, "ppl": {}, "layers": 0, "done": False}
    text = path.read_text(errors="ignore")
    # layer progress (from any tqdm line)
    layer_matches = LAYER_PAT.findall(text)
    if layer_matches:
        result["layers"] = max(int(x) for x in layer_matches)
    # KL/PPL results
    for name, kl, ppl in PAT.findall(text):
        result["kl"][name] = float(kl)
        result["ppl"][name] = float(ppl)
    result["done"] = all(d in result["ppl"] for d in DATASETS)
    return result


def main() -> None:
    files = sorted(LOG_DIR.glob("*.log"))
    if not files:
        print(f"[WARN] no logs under {LOG_DIR}")
        return
    rows = []
    for f in files:
        r = parse_log(f)
        rows.append((f.stem, r))

    # header
    header = f"{'Task':<26} {'Lyr':>5} {'done':>5} " + "  ".join(f"{d[:12]:>10}" for d in DATASETS)
    print(header)
    print("-" * len(header))
    for name, r in rows:
        cells = []
        for d in DATASETS:
            v = r["ppl"].get(d)
            cells.append(f"{v:>10.2f}" if v is not None else f"{'-':>10}")
        print(f"{name:<26} {r['layers']:>5} {str(r['done']):>5} " + "  ".join(cells))

    all_done = all(r["done"] for _, r in rows)
    unfinished = [n for n, r in rows if not r["done"]]
    print()
    print(f"[status] all_done={all_done}  unfinished={len(unfinished)}")
    if unfinished:
        print("   " + ", ".join(unfinished))


if __name__ == "__main__":
    main()
