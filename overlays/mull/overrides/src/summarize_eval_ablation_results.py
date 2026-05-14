#!/usr/bin/env python3
"""Summarize Mull eval/ablation outputs produced by the Aurora overlay."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "r1-v" / "eval_results"
OUT_PATH = RESULTS_DIR / "overall_eval_ablation_summary.txt"


def parse_result(path: Path):
    match = re.match(r"^eval_(?P<dataset>[^_]+)_(?P<run_id>.+)_greedy_output\.json$", path.name)
    if not match:
        return None

    note = ""
    final_acc = None
    ablation_mode = "baseline"
    num_samples = None

    try:
        payload = json.loads(path.read_text())
        final_acc = payload.get("final_acc")
        ablation_mode = payload.get("ablation_mode", ablation_mode)
        results = payload.get("results", [])
        num_samples = len(results) if isinstance(results, list) else None
    except Exception as exc:  # pragma: no cover - summary should survive malformed logs
        note = str(exc)

    return {
        "dataset": match.group("dataset"),
        "run_id": match.group("run_id"),
        "ablation_mode": ablation_mode,
        "final_acc": final_acc,
        "num_samples": num_samples,
        "note": note,
        "path": str(path),
    }


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(RESULTS_DIR.glob("eval_*_greedy_output.json")):
        parsed = parse_result(path)
        if parsed:
            records.append(parsed)

    lines = []
    lines.append("=" * 140)
    lines.append("MULL EVAL + ABLATION SUMMARY")
    lines.append("=" * 140)
    lines.append(f"Eval results dir: {RESULTS_DIR}")
    lines.append(f"Total eval JSON files parsed: {len(records)}")
    lines.append("")

    if records:
        header = (
            f"{'Dataset':<10} | {'Ablation':<24} | {'Final Acc':>10} | "
            f"{'Samples':>8} | {'Run ID':<64}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for record in records:
            lines.append(
                f"{record['dataset']:<10} | {record['ablation_mode']:<24} | "
                f"{fmt(record['final_acc']):>10} | {fmt(record['num_samples']):>8} | "
                f"{record['run_id']:<64}"
            )
            if record["note"]:
                lines.append(f"  note: {record['note']}")
    else:
        lines.append("No eval JSON files found yet.")

    lines.append("")
    lines.append("Output path:")
    lines.append(f"  {OUT_PATH}")

    text = "\n".join(lines) + "\n"
    OUT_PATH.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
