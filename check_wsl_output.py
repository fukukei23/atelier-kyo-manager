#!/usr/bin/env python3
"""
WSL環境でのコマンド出力確認スクリプト
ファイルシステムから直接情報を取得
"""
import os
import json
from pathlib import Path
from datetime import datetime

def check_output():
    """WSL環境の状態を確認"""
    project_root = Path(__file__).parent
    
    print("=" * 80)
    print("WSL環境のコマンド出力確認")
    print("=" * 80)
    print()
    
    # 1. プロジェクトルートの確認
    print(f"プロジェクトルート: {project_root}")
    print(f"存在確認: {project_root.exists()}")
    print()
    
    # 2. ログファイルの確認
    print("--- ログファイル ---")
    log_files = list(project_root.glob("*.log"))
    if log_files:
        for log_file in sorted(log_files, key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            size = log_file.stat().st_size
            print(f"  {log_file.name} ({size} bytes, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        print("  ログファイルが見つかりません")
    print()
    
    # 3. 実行結果ディレクトリの確認
    print("--- 実行結果ディレクトリ ---")
    runs_dir = project_root / "instance" / "runs"
    if runs_dir.exists():
        run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], 
                         key=lambda p: p.stat().st_mtime, reverse=True)[:5]
        if run_dirs:
            for run_dir in run_dirs:
                mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
                result_file = run_dir / "result.json"
                if result_file.exists():
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                            ok = result.get('ok', 'N/A')
                            print(f"  {run_dir.name}")
                            print(f"    時刻: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"    結果: ok={ok}")
                            if 'message' in result:
                                print(f"    メッセージ: {result['message'][:100]}")
                    except Exception as e:
                        print(f"  {run_dir.name} (JSON読み込みエラー: {e})")
                else:
                    print(f"  {run_dir.name} (result.jsonなし)")
        else:
            print("  実行結果ディレクトリが空です")
    else:
        print("  instance/runs ディレクトリが存在しません")
    print()
    
    # 4. テストスクリプトの確認
    print("--- テストスクリプト ---")
    test_scripts = [
        "check_browser_test.sh",
        "test_browser_with_logging.py",
        "run_test_with_immediate_output.py",
        "tools/run_browser_use.py"
    ]
    for script in test_scripts:
        script_path = project_root / script
        if script_path.exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} (見つかりません)")
    print()
    
    # 5. 最近のログファイルの内容（最初の50行）
    print("--- 最新のログファイルの内容（最初の50行） ---")
    if log_files:
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        print(f"ファイル: {latest_log.name}")
        print()
        try:
            with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:50], 1):
                    print(f"{i:3d}: {line.rstrip()}")
                if len(lines) > 50:
                    print(f"\n... (残り {len(lines) - 50} 行)")
        except Exception as e:
            print(f"  読み込みエラー: {e}")
    else:
        print("  ログファイルが見つかりません")
    print()
    
    print("=" * 80)
    print("確認完了")
    print("=" * 80)

if __name__ == "__main__":
    check_output()

