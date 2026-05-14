#!/usr/bin/env python3

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OVERLAYS = ROOT / "overlays"


class OverlayError(RuntimeError):
    pass


def shell_join(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def run(command, cwd=None, dry_run=False):
    location = f" (cwd={cwd})" if cwd else ""
    print(f"+ {shell_join(command)}{location}")
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def load_lock(method):
    lock_path = OVERLAYS / method / "source.lock"
    if not lock_path.exists():
        raise OverlayError(f"missing source.lock for {method}: {lock_path}")
    return json.loads(lock_path.read_text())


def apply_patches(method, checkout_dir, dry_run=False):
    patch_dir = OVERLAYS / method / "patches"
    patch_files = sorted(path for path in patch_dir.glob("*.patch") if path.is_file())
    for patch_file in patch_files:
        run(["git", "apply", "--check", str(patch_file)], cwd=checkout_dir, dry_run=dry_run)
        run(["git", "apply", str(patch_file)], cwd=checkout_dir, dry_run=dry_run)


def apply_overrides(method, checkout_dir, dry_run=False):
    override_root = OVERLAYS / method / "overrides"
    for path in sorted(override_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "README.md":
            continue
        relative = path.relative_to(override_root)
        target = checkout_dir / relative
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
        run(["cp", str(path), str(target)], dry_run=dry_run)


def bootstrap(method, refresh=False, dry_run=False):
    data = load_lock(method)
    repo_url = data["repo_url"]
    commit = data["commit"]
    checkout_dir = (ROOT / data["checkout_dir"]).resolve()
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)

    if refresh and checkout_dir.exists():
        run(["rm", "-rf", str(checkout_dir)], dry_run=dry_run)

    if not (checkout_dir / ".git").exists():
        if checkout_dir.exists():
            run(["rm", "-rf", str(checkout_dir)], dry_run=dry_run)
        run(["git", "clone", repo_url, str(checkout_dir)], dry_run=dry_run)

    run(["git", "fetch", "--all", "--tags"], cwd=checkout_dir, dry_run=dry_run)
    run(["git", "checkout", "--force", commit], cwd=checkout_dir, dry_run=dry_run)
    run(["git", "clean", "-fd"], cwd=checkout_dir, dry_run=dry_run)
    run(["git", "reset", "--hard", commit], cwd=checkout_dir, dry_run=dry_run)
    apply_patches(method, checkout_dir, dry_run=dry_run)
    apply_overrides(method, checkout_dir, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description="Clone and materialize an overlay-backed external repo.")
    parser.add_argument("method", choices=sorted(path.name for path in OVERLAYS.iterdir() if path.is_dir()))
    parser.add_argument("--refresh", action="store_true", help="Delete and re-clone the external checkout before applying local changes.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    try:
        bootstrap(args.method, refresh=args.refresh, dry_run=args.dry_run)
        return 0
    except OverlayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

