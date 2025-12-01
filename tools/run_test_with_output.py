#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: tools/run_test_with_output.py
Purpose: テストを実行し、結果を表示するヘルパースクリプト

Usage:
    python tools/run_test_with_output.py tests/test_product_extractor.py
    python tools/run_test_with_output.py tests/test_product_extractor.py::test_product_extractor_title
    python tools/run_test_with_output.py tests/ -v
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def run_tests(test_path: str, verbose: bool = False) -> int:
    """テストを実行し、結果を表示する"""
    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
    ]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-v", "--tb=short"])
    
    print(f"[run_test_with_output] 実行コマンド: {' '.join(cmd)}")
    print(f"[run_test_with_output] 実行開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # テストを実行し、出力をリアルタイムで表示
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # 行バッファリング
        universal_newlines=True,
    )
    
    # 出力をリアルタイムで表示
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()
    
    process.wait()
    exit_code = process.returncode
    
    print()
    print("=" * 80)
    print(f"[run_test_with_output] 実行終了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[run_test_with_output] 終了コード: {exit_code}")
    
    # 最新のテスト結果ファイルを確認
    reports_dir = ROOT / "docs" / "reports"
    if reports_dir.exists():
        test_result_files = sorted(reports_dir.glob("TEST_RESULTS_*.txt"), reverse=True)
        if test_result_files:
            latest_result = test_result_files[0]
            print(f"\n[run_test_with_output] 最新のテスト結果ファイル: {latest_result}")
            print(f"[run_test_with_output] 生成時刻: {datetime.fromtimestamp(latest_result.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="テストを実行し、結果を表示する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "test_path",
        help="テストファイルまたはテストパス（例: tests/test_product_extractor.py または tests/test_product_extractor.py::test_product_extractor_title）",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="詳細な出力を表示",
    )
    
    args = parser.parse_args()
    
    exit_code = run_tests(args.test_path, args.verbose)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

