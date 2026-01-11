#!/usr/bin/env python3
# ==============================================================================
# File: tools/run_with_summary.py
# Date: 2025-11-17
# Purpose: Execute a command (default: run_orchestrator.py) and save its stdout/stderr
#          to logs/last_run_summary.txt for quick sharing without copy/paste.
# Usage:
#   # デフォルトの Moncler 実行
#   python tools/run_with_summary.py --preset moncler_headful --site MONCLER_OFFICIAL --query "down jacket" --auto-heal --headful
#
#   # 任意のコマンドを叩きたい場合
#   python tools/run_with_summary.py --cmd "python run_orchestrator.py --help"
# ==============================================================================

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_FILE = ROOT / "logs" / "last_run_summary.txt"


def run_command(cmd: List[str]) -> int:
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_FILE.open("w", encoding="utf-8") as fh:
        fh.write(f"[run_with_summary] cmd: {' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        fh.write(f"\n[run_with_summary] exit code: {proc.returncode}\n")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command and write stdout/stderr to logs/last_run_summary.txt",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--cmd",
        help="Custom command to run. If omitted, defaults to run_orchestrator.py with provided args.",
    )
    # orchestrator_args は REMAINDER だと先頭の -- が必須になるため、parse_known_args で受ける
    args, remainder = parser.parse_known_args()

    if args.cmd:
        # Execute the custom command string through the shell-style split.
        cmd = shlex.split(args.cmd)
    else:
        # Default: python run_orchestrator.py <args...>
        python_exe = ROOT / ".venv" / "Scripts" / "python.exe"
        cmd = [str(python_exe) if python_exe.exists() else "python", "run_orchestrator.py"]
        cmd.extend([str(a) for a in remainder])
    code = run_command([str(c) for c in cmd])
    print(f"[run_with_summary] Written to {SUMMARY_FILE} (exit code={code})")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
