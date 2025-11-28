#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
診断スクリプトを実行して結果をファイルに保存
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent

# 結果ファイル
results_dir = project_root / "docs" / "test_results"
results_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = results_dir / f"moncler_diagnostics_execution_{timestamp}.txt"

print("=" * 80)
print("MONCLER Drission 診断スクリプト実行")
print("=" * 80)
print(f"結果ファイル: {output_file.relative_to(project_root)}")
print()

# 診断スクリプトを実行
cmd = [
    sys.executable,
    "scripts/run_moncler_drission_diagnostics.py",
    "--query", "down jacket",
    "--headless",
]

print(f"実行コマンド: {' '.join(cmd)}")
print()

try:
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=300,  # 5分のタイムアウト
    )
    
    # 結果をファイルに保存
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MONCLER Drission 診断スクリプト実行結果\n")
        f.write("=" * 80 + "\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"コマンド: {' '.join(cmd)}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("STDOUT:\n")
        f.write("=" * 80 + "\n")
        f.write(result.stdout)
        f.write("\n")
        
        if result.stderr:
            f.write("\nSTDERR:\n")
            f.write("=" * 80 + "\n")
            f.write(result.stderr)
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"終了コード: {result.returncode}\n")
        f.write("=" * 80 + "\n")
    
    # コンソールにも出力
    print("STDOUT:")
    print("=" * 80)
    print(result.stdout)
    
    if result.stderr:
        print("\nSTDERR:")
        print("=" * 80)
        print(result.stderr)
    
    print()
    print("=" * 80)
    print(f"終了コード: {result.returncode}")
    print(f"結果ファイル: {output_file.relative_to(project_root)}")
    print("=" * 80)
    
    # 生成された診断ディレクトリを確認
    artifacts_dir = project_root / "artifacts" / "moncler_drission"
    if artifacts_dir.exists():
        diag_dirs = sorted(artifacts_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if diag_dirs:
            latest_dir = diag_dirs[0]
            print(f"\n最新の診断ディレクトリ: {latest_dir.relative_to(project_root)}")
            files = list(latest_dir.glob("*"))
            print(f"ファイル数: {len(files)}")
            for f in files[:10]:  # 最初の10個を表示
                print(f"  - {f.name}")
    
    sys.exit(result.returncode)
    
except subprocess.TimeoutExpired:
    print("❌ タイムアウト: 5分以内に完了しませんでした")
    sys.exit(1)
except Exception as e:
    print(f"❌ 実行エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

