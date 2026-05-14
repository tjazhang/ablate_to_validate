#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
HARDBLINK_DIR = Path(
    os.environ.get(
        "MIRAGE_HARDBLINK_ANSWERS_DIR",
        os.environ.get("MIRAGE_OUTPUT_ROOT", ROOT / "data" / "hardblink" / "answers_mirage"),
    )
)
OUT_PATH = RESULTS_DIR / "overall_eval_ablation_summary.txt"

MIRAGE_MODES = [
    "gt_latent_count_matched",
    "random_latent_model_dist",
    "random_latent_gt_dist",
    "random_latent",
    "zero_latent",
    "model_latent",
    "first_latent_repeat",
    "gt_latent",
]


def infer_mode_from_name(name: str) -> str:
    for mode in MIRAGE_MODES:
        if name == mode or name.endswith(f"_{mode}") or f"_{mode}_" in name:
            return mode
    return "baseline"


def parse_mirage_json_logs():
    records = []
    for p in sorted(RESULTS_DIR.glob("log_*.json")):
        m = re.match(r"^log_(train|test)_(.+)\.json$", p.name)
        if not m:
            continue
        split = m.group(1)
        run_name = m.group(2)

        note = ""
        acc = invalid = total = correct = None
        try:
            data = json.loads(p.read_text())
            summary = data.get("summary", {})
            acc = summary.get("accuracy")
            invalid = summary.get("invalid")
            total = summary.get("total")
            correct = summary.get("correct")
        except Exception as exc:
            note = str(exc)

        records.append(
            {
                "split": split,
                "run_name": run_name,
                "mode": infer_mode_from_name(run_name),
                "accuracy": acc,
                "invalid": invalid,
                "total": total,
                "correct": correct,
                "note": note,
                "path": str(p),
            }
        )
    return records


def parse_hardblink_summaries():
    records = []
    for p in sorted(HARDBLINK_DIR.glob("*/evaluation_summary_long.json")):
        run_name = p.parent.name
        note = ""
        overall = {}
        per_file = {}
        try:
            data = json.loads(p.read_text())
            overall = data.get("overall", {})
            per_file = data.get("per_file", {})
        except Exception as exc:
            note = str(exc)

        def file_acc(name):
            item = per_file.get(name, {})
            return item.get("accuracy")

        records.append(
            {
                "run_name": run_name,
                "mode": infer_mode_from_name(run_name),
                "overall_acc": overall.get("accuracy"),
                "total": overall.get("total"),
                "correct": overall.get("correct"),
                "acc_3pt": file_acc("blink_3pointscenter_answers_long.jsonl"),
                "acc_4pt": file_acc("blink_4pointscenter_answers_long.jsonl"),
                "acc_5pt": file_acc("blink_5pointscenter_answers_long.jsonl"),
                "note": note,
                "path": str(p),
            }
        )
    return records


def f(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mirage_logs = parse_mirage_json_logs()
    hardblink_logs = parse_hardblink_summaries()

    lines = []
    lines.append("=" * 140)
    lines.append("MIRAGE EVAL + ABLATION SUMMARY")
    lines.append("=" * 140)
    lines.append(f"Mirage result JSON dir: {RESULTS_DIR}")
    lines.append(f"HardBlink summary dir:  {HARDBLINK_DIR}")
    lines.append("")

    lines.append(f"[Section 1] Mirage JSON logs ({len(mirage_logs)} files)")
    lines.append("-" * 140)
    if mirage_logs:
        header = f"{'Split':<8} | {'Mode':<28} | {'Run Name':<34} | {'Acc':>8} | {'Invalid':>8} | {'Correct':>8} | {'Total':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for rec in mirage_logs:
            lines.append(
                f"{rec['split']:<8} | {rec['mode']:<28} | {rec['run_name']:<34} | {f(rec['accuracy']):>8} | "
                f"{f(rec['invalid']):>8} | {f(rec['correct']):>8} | {f(rec['total']):>8}"
            )
            if rec["note"]:
                lines.append(f"  note: {rec['note']}")
    lines.append("")

    lines.append(f"[Section 2] HardBlink summaries ({len(hardblink_logs)} files)")
    lines.append("-" * 140)
    if hardblink_logs:
        header = f"{'Mode':<28} | {'Run Name':<44} | {'Overall':>8} | {'3pt':>8} | {'4pt':>8} | {'5pt':>8} | {'Correct':>8} | {'Total':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for rec in hardblink_logs:
            lines.append(
                f"{rec['mode']:<28} | {rec['run_name']:<44} | {f(rec['overall_acc']):>8} | "
                f"{f(rec['acc_3pt']):>8} | {f(rec['acc_4pt']):>8} | {f(rec['acc_5pt']):>8} | "
                f"{f(rec['correct']):>8} | {f(rec['total']):>8}"
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
