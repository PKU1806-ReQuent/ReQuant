#!/usr/bin/env python3
"""Collect lm-eval results from logs_eval/*.log into a markdown table.

Prefer the final markdown row already printed by pretty_print_results, and fall
back to per-task 'acc: xx.xx%' lines when the run is still in progress.
"""
import re
import os
import glob

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs_eval"))

TASKS = [
    "arc_challenge", "arc_easy", "boolq", "ceval-valid", "hellaswag",
    "lambada_openai", "openbookqa", "piqa", "social_iqa", "winogrande",
]
TASK_SHORT = {
    "arc_challenge": "arc_c", "arc_easy": "arc_e", "boolq": "boolq",
    "ceval-valid": "ceval", "hellaswag": "hella", "lambada_openai": "lambada",
    "openbookqa": "obqa", "piqa": "piqa", "social_iqa": "siqa",
    "winogrande": "wino",
}


def parse_partial(lines):
    """Map task name -> acc(%) using the per-task log trace."""
    result = {}
    cur = None
    for line in lines:
        m = re.search(r"Evaluating (\S+?)\.\.\.", line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r"acc: ([0-9]+(?:\.[0-9]+)?)%", line)
        if m and cur is not None:
            result[cur] = float(m.group(1))
            cur = None
    return result


def parse_final_row(lines):
    """Find the final markdown row produced by pretty_print_results."""
    header_idx = None
    for i, line in enumerate(lines):
        if "arc_challenge" in line and "acc_avg" in line and "|" in line:
            header_idx = i
    if header_idx is None:
        return None
    # data row is typically 2 lines after header (separator in between)
    for j in range(header_idx + 1, min(header_idx + 5, len(lines))):
        cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
        if len(cells) >= 11 and all(re.fullmatch(r"[0-9]+(\.[0-9]+)?", c) for c in cells[:10]):
            return cells
    return None


def main():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "0*.log")))
    hdr = ["Tag"] + [TASK_SHORT[t] for t in TASKS] + ["avg"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    done = 0
    for f in files:
        tag = os.path.splitext(os.path.basename(f))[0]
        with open(f, "r", errors="ignore") as fh:
            lines = fh.readlines()
        row = parse_final_row(lines)
        if row is not None:
            values = row[:10]
            avg = row[10] if len(row) > 10 else f"{sum(float(v) for v in values)/len(values):.2f}"
            done += 1
            print(f"| {tag} | " + " | ".join(values) + f" | {avg} |")
        else:
            res = parse_partial(lines)
            values = [f"{res[t]:.2f}" if t in res else "-" for t in TASKS]
            nums = [res[t] for t in TASKS if t in res]
            avg = f"{sum(nums)/len(nums):.2f}*" if nums else "-"
            print(f"| {tag} | " + " | ".join(values) + f" | {avg} |")
    print(f"\n[status] fully_done={done}/{len(files)}  (* = partial avg)")


if __name__ == "__main__":
    main()
