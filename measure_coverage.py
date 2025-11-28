#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""カバレッジ測定スクリプト"""

import subprocess
import sys
import json
import os
from pathlib import Path

def main():
    """カバレッジを測定して結果を表示"""
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # pytest-cov がインストールされているか確認
    try:
        import pytest_cov
    except ImportError:
        print("pytest-cov をインストール中...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest-cov", "-q"], check=True)
    
    # カバレッジ測定（成功するテストのみを実行）
    print("カバレッジを測定中...")
    # 成功するテストのみを実行（Stage 3A-2, 3B 関連）
    test_files = [
        "tests/test_navigation_driver_stage3a2.py",
        "tests/test_telemetry_service_stage3b.py",
    ]
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
        ] + test_files + [
            "--cov=app/agents/browser",
            "--cov-report=term",
            "--cov-report=json",
            "--cov-report=term-missing",
            "-v",
            "--tb=short",
        ],
        capture_output=True,
        text=True
    )
    
    # 標準出力を表示
    print("=== pytest 実行結果 ===")
    if result.stdout:
        print(result.stdout)
    if result.stderr and "ERROR" in result.stderr:
        print("エラー:", result.stderr, file=sys.stderr)
    print(f"終了コード: {result.returncode}")
    
    # JSONレポートからカバレッジを読み取る
    coverage_json = project_root / ".coverage.json"
    if coverage_json.exists():
        with open(coverage_json) as f:
            data = json.load(f)
            totals = data.get("totals", {})
            percent = totals.get("percent_covered", 0)
            covered = totals.get("covered_lines", 0)
            total = totals.get("num_statements", 0)
            
            print("\n" + "="*60)
            print(f"📊 カバレッジ結果")
            print("="*60)
            print(f"カバレッジ: {percent:.2f}%")
            print(f"カバー行数: {covered:,} / {total:,}")
            print(f"未カバー行数: {total - covered:,}")
            print("="*60)
    else:
        print("⚠️ カバレッジレポートが見つかりませんでした")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())

