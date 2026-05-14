#!/usr/bin/env python3

import argparse
import filecmp
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set


ROOT = Path(__file__).resolve().parents[1]
CODING_RE = re.compile(r"^#.*coding[:=]\s*[-\w.]+")


CONFIGS = {
    "llava": {
        "method": "llava",
        "method_dir": ROOT / "methods" / "llava",
        "baseline_dir": ROOT / "external" / "llava_upstream",
        "baseline_repo": "https://github.com/haotian-liu/LLaVA.git",
        "baseline_commit": "24fa1d065bbeac8a145a796ab7218c6945a2536e",
        "baseline_ref": "v1.2.2.post1",
        "ignore_missing_roots": (Path("data"),),
        "ignore_parts": (".git", "__pycache__"),
    },
    "qwen": {
        "method": "qwen",
        "method_dir": ROOT / "methods" / "qwen",
        "baseline_dir": ROOT / "external" / "qwen_upstream",
        "baseline_repo": "https://github.com/QwenLM/Qwen2.5-VL.git",
        "baseline_commit": "96588727e44c78b25ba03ea03b8e12f7e64fd0da",
        "baseline_ref": "HEAD",
        "ignore_missing_roots": (Path("cookbooks"), Path("sample_data")),
        "ignore_parts": (".git", ".ipynb_checkpoints"),
    },
}


def should_track_file(path, ignore_parts):
    return path.is_file() and not any(part in path.parts for part in ignore_parts)


def list_files(root, ignore_parts):
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if should_track_file(path, ignore_parts)
    }


def is_ignored_missing(path, ignored_roots):
    return any(path == ignored_root or ignored_root in path.parents for ignored_root in ignored_roots)


def is_markable(path):
    name = path.name
    suffix = path.suffix.lower()
    if path.name == "UPSTREAM_DIFF.md":
        return False
    if suffix in {".py", ".sh", ".md", ".yml", ".yaml", ".toml", ".sbatch"}:
        return True
    if name in {".gitignore"}:
        return True
    if name.startswith("Dockerfile"):
        return True
    if name == "requirements.txt" or name.startswith("requirements_"):
        return True
    return False


def gather_diffs(config):
    if not config["method_dir"].exists():
        raise FileNotFoundError("missing method directory: {}".format(config["method_dir"]))
    if not config["baseline_dir"].exists():
        raise FileNotFoundError("missing baseline directory: {}".format(config["baseline_dir"]))

    baseline_files = list_files(config["baseline_dir"], config["ignore_parts"])
    method_files = list_files(config["method_dir"], config["ignore_parts"])

    modified = []
    for relative_path in sorted(baseline_files & method_files):
        try:
            identical = filecmp.cmp(
                config["baseline_dir"] / relative_path,
                config["method_dir"] / relative_path,
                shallow=False,
            )
        except OSError:
            identical = False
        if not identical:
            modified.append(relative_path)

    extra = sorted(method_files - baseline_files)
    missing = sorted(
        relative_path
        for relative_path in (baseline_files - method_files)
        if not is_ignored_missing(relative_path, config["ignore_missing_roots"])
    )

    marked_modified = [path for path in modified if is_markable(path)]
    marked_extra = [path for path in extra if is_markable(path)]
    skipped = [path for path in (modified + extra) if not is_markable(path)]

    return {
        "modified": modified,
        "extra": extra,
        "missing": missing,
        "marked_modified": marked_modified,
        "marked_extra": marked_extra,
        "skipped": skipped,
    }


def build_banner(config, relative_path, status):
    baseline = "{} @ {} ({})".format(
        config["baseline_repo"], config["baseline_ref"], config["baseline_commit"]
    )
    if status == "modified":
        lines = [
            "NEW: Aurora modification relative to upstream {}.".format(config["method"].upper()),
            "NEW: Baseline {}.".format(baseline),
            "NEW: Upstream-tracked path: {}".format(relative_path.as_posix()),
        ]
    else:
        lines = [
            "NEW: Aurora-only file; not present in upstream {}.".format(config["method"].upper()),
            "NEW: Baseline {}.".format(baseline),
            "NEW: Aurora path: {}".format(relative_path.as_posix()),
        ]

    if relative_path.suffix.lower() == ".md":
        body = "\n".join(lines)
        return "<!--\n{}\n-->\n\n".format(body)

    return "".join(f"# {line}\n" for line in lines) + "\n"


def already_marked(text):
    head = "\n".join(text.splitlines()[:8])
    return "NEW: Aurora" in head


def insertion_index(lines):
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1
    if index < len(lines) and CODING_RE.match(lines[index]):
        index += 1
    return index


def apply_banner(file_path, banner):
    text = file_path.read_text(encoding="utf-8")
    if already_marked(text):
        return False

    lines = text.splitlines(keepends=True)
    index = insertion_index(lines)
    updated = "".join(lines[:index]) + banner + "".join(lines[index:])
    file_path.write_text(updated, encoding="utf-8")
    return True


def write_manifest(config, diffs):
    manifest_path = config["method_dir"] / "UPSTREAM_DIFF.md"
    lines = [
        "# {} Upstream Diff".format(config["method"].upper()),
        "",
        "Baseline repo: `{}`".format(config["baseline_repo"]),
        "Baseline ref: `{}`".format(config["baseline_ref"]),
        "Baseline commit: `{}`".format(config["baseline_commit"]),
        "Method directory: `{}`".format(config["method_dir"].relative_to(ROOT)),
        "",
        "Modified upstream files marked with `NEW:`: {}".format(len(diffs["marked_modified"])),
        "Aurora-only files marked with `NEW:`: {}".format(len(diffs["marked_extra"])),
        "Data or artifact files left unmarked: {}".format(len(diffs["skipped"])),
        "Upstream files missing from the copied snapshot: {}".format(len(diffs["missing"])),
        "",
        "## Modified Upstream Files",
    ]
    lines.extend(f"- `{path.as_posix()}`" for path in diffs["modified"])
    lines.extend(["", "## Aurora-Only Files"])
    lines.extend(f"- `{path.as_posix()}`" for path in diffs["extra"])
    lines.extend(["", "## Unmarked Data And Artifacts"])
    lines.extend(f"- `{path.as_posix()}`" for path in diffs["skipped"])
    lines.extend(["", "## Missing From Copied Snapshot"])
    lines.extend(f"- `{path.as_posix()}`" for path in diffs["missing"])
    lines.append("")
    manifest_path.write_text("\n".join(lines), encoding="utf-8")


def process_method(config, dry_run=False):
    diffs = gather_diffs(config)
    changed = 0
    considered = diffs["marked_modified"] + diffs["marked_extra"]

    if dry_run:
        print("[{}] would mark {} files".format(config["method"], len(considered)))
        print("[{}] would write {}".format(config["method"], config["method_dir"] / "UPSTREAM_DIFF.md"))
        return 0, len(considered)

    for relative_path in diffs["marked_modified"]:
        if apply_banner(config["method_dir"] / relative_path, build_banner(config, relative_path, "modified")):
            changed += 1
    for relative_path in diffs["marked_extra"]:
        if apply_banner(config["method_dir"] / relative_path, build_banner(config, relative_path, "extra")):
            changed += 1

    write_manifest(config, diffs)
    return changed, len(considered)


def main():
    parser = argparse.ArgumentParser(description="Mark files that diverge from clean upstream baselines.")
    parser.add_argument(
        "methods",
        nargs="*",
        help="Methods to process. Defaults to llava and qwen.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report files that would be marked without editing them.")
    args = parser.parse_args()

    methods = args.methods or sorted(CONFIGS)
    for method in methods:
        if method not in CONFIGS:
            parser.error("unknown method: {}".format(method))
        changed, total = process_method(CONFIGS[method], dry_run=args.dry_run)
        action = "would inspect" if args.dry_run else "inspected"
        print("[{}] {} {} markable divergent files; updated {} headers".format(
            method, action, total, changed
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
