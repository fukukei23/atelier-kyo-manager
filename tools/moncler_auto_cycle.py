#!/usr/bin/env python3
# ==============================================================================
# File: tools/moncler_auto_cycle.py
# Date: 2025-11-15
# Purpose: Moncler PLP self-heal orchestrator wrapper.
# Usage:
#   python tools/moncler_auto_cycle.py \
#       --site MONCLER_OFFICIAL \
#       --query "down jacket" \
#       --headful \
#       --auto-heal \
#       --apply-proposal \
#       --git-branch moncler-auto-$(date +%Y%m%d)
# ==============================================================================

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


def run_orchestrator(site: str, query: str, *, headful: bool, auto_heal: bool, extra_args: Optional[list[str]] = None) -> int:
    cmd = [sys.executable, "run_orchestrator.py", "--preset", "moncler_headful", "--site", site, "--query", query]
    if headful:
        cmd.append("--headful")
    if auto_heal:
        cmd.append("--auto-heal")
    if extra_args:
        cmd.extend(extra_args)
    print(f"[moncler_auto_cycle] executing: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def _latest_run_dir(base: Path) -> Optional[Path]:
    runs = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _load_auto_heal(run_dir: Path) -> Optional[dict]:
    path = run_dir / "auto_heal_log.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[moncler_auto_cycle] auto_heal_log.json parse error: {exc}")
        return None


def _apply_proposal(site: str, proposal_path: Path) -> bool:
    cmd = [sys.executable, "tools/apply_llm_proposal.py", "--site", site, "--proposal", str(proposal_path)]
    print(f"[moncler_auto_cycle] applying proposal: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode == 0


def _git_commit(branch: Optional[str], message: str) -> None:
    if not branch:
        return
    subprocess.run(["git", "checkout", "-B", branch], cwd=ROOT, check=False)
    subprocess.run(["git", "add", "app/config/sites/overrides.local.json"], cwd=ROOT, check=False)
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Moncler PLP auto-run + proposal apply helper.")
    parser.add_argument("--site", default="MONCLER_OFFICIAL")
    parser.add_argument("--query", default="down jacket")
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--auto-heal", action="store_true")
    parser.add_argument("--apply-proposal", action="store_true")
    parser.add_argument("--git-branch", help="Optional branch to create/fast-forward for applied proposals.")
    parser.add_argument("--extra-args", nargs="*", default=[])
    args = parser.parse_args()

    ret = run_orchestrator(
        site=args.site,
        query=args.query,
        headful=args.headful,
        auto_heal=args.auto_heal,
        extra_args=args.extra_args,
    )
    print(f"[moncler_auto_cycle] orchestrator exit code: {ret}")

    runs_dir = ROOT / "instance" / "runs"
    latest = _latest_run_dir(runs_dir)
    if not latest:
        print("[moncler_auto_cycle] no run directories found.")
        return
    print(f"[moncler_auto_cycle] latest run: {latest.name}")

    auto_heal = _load_auto_heal(latest)
    if not auto_heal:
        print("[moncler_auto_cycle] auto_heal_log.json not found or unreadable; skip proposal apply.")
        return

    if args.apply_proposal:
        proposal_path_str = auto_heal.get("llm_proposal_path")
        if not proposal_path_str:
            print("[moncler_auto_cycle] llm_proposal_path not present; skip apply.")
        else:
            proposal_path = ROOT / Path(proposal_path_str)
            if not proposal_path.exists():
                print(f"[moncler_auto_cycle] proposal file missing: {proposal_path}")
            else:
                ok = _apply_proposal(args.site, proposal_path)
                if ok and args.git_branch:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    _git_commit(args.git_branch, f"[auto] Apply Moncler proposal ({timestamp})")


if __name__ == "__main__":
    main()
