#!/usr/bin/env python3
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "VLMEvalKit" / "outputs"
OUT_PATH = OUTPUTS_DIR / "overall_eval_ablation_summary.txt"


def parse_acc_csv(path: Path):
    name = path.name
    m = re.match(r"^(.+?)_(.+?)_acc\.csv$", name)
    if not m:
        return None

    model_name = m.group(1)
    benchmark = m.group(2)
    run_id = path.parent.name

    overall = None
    split = ""
    note = ""

    try:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            row = rows[0]
            split = row.get("split", "")
            value = row.get("Overall")
            overall = float(value) if value not in (None, "") else None
    except Exception as exc:
        note = str(exc)

    return {
        "model": model_name,
        "benchmark": benchmark,
        "run_id": run_id,
        "split": split,
        "overall": overall,
        "note": note,
        "path": str(path),
    }


def f(v):
    if v is None:
        return "NA"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for p in sorted(OUTPUTS_DIR.glob("*/*/*_acc.csv")):
        parsed = parse_acc_csv(p)
        if parsed:
            records.append(parsed)

    lines = []
    lines.append("=" * 130)
    lines.append("COVT EVAL + ABLATION SUMMARY")
    lines.append("=" * 130)
    lines.append(f"Output directory: {OUTPUTS_DIR}")
    lines.append(f"Total *_acc.csv parsed: {len(records)}")
    lines.append("")

    if records:
        header = f"{'Model':<32} | {'Benchmark':<16} | {'Run ID':<20} | {'Split':<8} | {'Overall':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        for rec in records:
            lines.append(
                f"{rec['model']:<32} | {rec['benchmark']:<16} | {rec['run_id']:<20} | {rec['split']:<8} | {f(rec['overall']):>10}"
            )
            if rec["note"]:
                lines.append(f"  note: {rec['note']}")

    lines.append("")
    lines.append("Output path:")
    lines.append(f"  {OUT_PATH}")

    text = "\n".join(lines) + "\n"
    OUT_PATH.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
