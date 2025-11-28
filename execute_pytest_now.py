#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest を即座に実行するスクリプト（ターミナル問題回避版）
"""

import sys
import os
from pathlib import Path
import subprocess

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 仮想環境の Python を探す
venv_python = None
venv_paths = [
    project_root / "venv" / "bin" / "python3",
    project_root / "venv" / "bin" / "python",
    project_root / ".venv" / "bin" / "python3",
    project_root / ".venv" / "bin" / "python",
    project_root / "myenv" / "Scripts" / "python.exe",
]

for venv_python_path in venv_paths:
    if venv_python_path.exists():
        venv_python = str(venv_python_path)
        print(f"✅ 仮想環境の Python を使用: {venv_python}")
        break

if not venv_python:
    print("⚠️  仮想環境の Python が見つかりません。システムの Python を使用します。")
    venv_python = sys.executable

def run_pytest(test_path=None):
    """pytest を実行する"""
    # pytest コマンドを構築
    cmd = [venv_python, "-m", "pytest", "-v"]
    
    if test_path:
        cmd.append(str(test_path))
    else:
        # デフォルトで tests/ ディレクトリを指定
        tests_dir = project_root / "tests"
        if tests_dir.exists():
            cmd.append(str(tests_dir))
        else:
            print(f"❌ テストディレクトリが見つかりません: {tests_dir}")
            return 1
    
    print("=" * 80)
    print("pytest 実行")
    print("=" * 80)
    print(f"Python: {venv_python}")
    print(f"コマンド: {' '.join(cmd)}")
    print("=" * 80)
    print()
    
    try:
        # pytest を実行（stdout/stderr をリアルタイムで表示）
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        
        # 出力をリアルタイムで表示
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        print()
        print("=" * 80)
        if process.returncode == 0:
            print("✅ すべてのテストが成功しました")
        else:
            print(f"❌ テストが失敗しました (終了コード: {process.returncode})")
        print("=" * 80)
        
        return process.returncode
        
    except Exception as e:
        print(f"❌ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="pytest を即座に実行")
    parser.add_argument(
        "test_path",
        nargs="?",
        default=None,
        help="テストファイルまたはディレクトリ（デフォルト: tests/）"
    )
    
    args = parser.parse_args()
    
    return run_pytest(test_path=args.test_path)

if __name__ == "__main__":
    sys.exit(main())

