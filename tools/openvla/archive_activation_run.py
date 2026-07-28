#!/usr/bin/env python3
"""Optional cold archive of a completed activation run to /home/HDD.

`/home/HDD` is NTFS over FUSE. It is fine as a cold archive but must never be the
collector's hot path: FUSE adds per-syscall overhead and NTFS does not preserve
POSIX ownership or permissions. Measured sustained write there is well below what
live collection needs.

This tool is manual and off by default. It never deletes the ext4 original --
the archive is a second copy, not a move.

    python3 tools/openvla/archive_activation_run.py \
        --run_dir /home/user/4TB/hwkim/openvla_activation_collection/<run_id> \
        --archive_root /home/HDD/hwkim/openvla_activation_archive \
        --mode tar --dry_run

Modes:
  tar   one .tar per episode (far friendlier to NTFS/FUSE than 3k small files)
  copy  plain directory copy
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from activation_integrity import FAIL, build_integrity_report  # noqa: E402
from activation_storage import build_manifest, free_gb, verify_against_manifest  # noqa: E402

DEFAULT_ARCHIVE_ROOT = "/home/HDD/hwkim/openvla_activation_archive"


def episode_dirs(run_dir: str) -> List[str]:
    root = os.path.join(run_dir, "episodes")
    if not os.path.isdir(root):
        return []
    return sorted(
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and not d.startswith(".") and not d.endswith(".partial")
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive a completed activation run to /home/HDD.")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--archive_root", default=DEFAULT_ARCHIVE_ROOT)
    ap.add_argument("--mode", choices=["tar", "copy"], default="tar")
    ap.add_argument("--dry_run", action="store_true", default=False)
    ap.add_argument("--skip_integrity", action="store_true", default=False,
                    help="skip re-verifying the source (not recommended)")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"[FATAL] run_dir not found: {run_dir}")
        return 1

    # Verify the ext4 original BEFORE copying anything anywhere.
    if not args.skip_integrity:
        print("Verifying source run integrity...")
        report = build_integrity_report(run_dir)
        print(f"  overall: {report['overall']}")
        if report["overall"] == FAIL:
            print("[FATAL] source run failed integrity checks; refusing to archive.")
            return 1

    episodes = episode_dirs(run_dir)
    total_bytes = sum(build_manifest(d)["total_bytes"] for d in episodes)
    archive_dir = os.path.join(args.archive_root, os.path.basename(run_dir))
    available = free_gb(args.archive_root)

    print(f"Run          : {run_dir}")
    print(f"Episodes     : {len(episodes)}")
    print(f"Source size  : {total_bytes / 1e9:.2f} GB")
    print(f"Archive to   : {archive_dir}  (mode={args.mode})")
    print(f"Archive free : {available:.1f} GB")
    if available < total_bytes / (1024 ** 3) + 5:
        print("[FATAL] not enough free space on the archive filesystem.")
        return 1
    if args.dry_run:
        print("\n[DRY RUN] nothing was written. Re-run without --dry_run to archive.")
        return 0

    os.makedirs(archive_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for source in episodes:
        name = os.path.basename(source)
        manifest = build_manifest(source)
        started = time.time()
        if args.mode == "tar":
            target = os.path.join(archive_dir, f"{name}.tar")
            if os.path.exists(target):
                print(f"  {name}: already archived, skipping")
                continue
            tmp = target + ".partial"
            with tarfile.open(tmp, "w") as tar:
                tar.add(source, arcname=name)
            os.rename(tmp, target)
            verified = os.path.getsize(target) > 0
            problems: List[str] = [] if verified else ["empty tar"]
        else:
            target = os.path.join(archive_dir, name)
            if os.path.isdir(target):
                print(f"  {name}: already archived, skipping")
                continue
            tmp = target + ".partial"
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(source, tmp)
            problems = verify_against_manifest(tmp, manifest)
            if problems:
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                os.rename(tmp, target)
            verified = not problems

        elapsed = time.time() - started
        results.append({
            "episode": name, "target": target, "verified": verified,
            "problems": problems, "seconds": elapsed,
            "mbps": manifest["total_bytes"] / 1e6 / elapsed if elapsed else None,
        })
        print(f"  {name}: {'OK' if verified else 'FAILED'} "
              f"({manifest['total_bytes']/1e6:.0f} MB, {elapsed:.1f}s)")

    for extra in ("run_config.json", "collection_summary.json", "integrity_report.json", "dashboard.html"):
        src = os.path.join(run_dir, extra)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(archive_dir, extra))

    with open(os.path.join(archive_dir, "archive_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"source_run_dir": run_dir, "mode": args.mode, "episodes": results},
                  handle, indent=2, ensure_ascii=False)

    failed = [r for r in results if not r["verified"]]
    print(f"\nArchived {len(results) - len(failed)}/{len(results)} episodes to {archive_dir}")
    print("The ext4 original was NOT deleted. Remove it manually only after you have "
          "independently confirmed the archive.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
