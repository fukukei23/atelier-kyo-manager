#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1.2 テスト実行スクリプト（即座に実行版）
ターミナルの問題を回避するため、Pythonコードを直接実行します。
"""

import sys
import os
from pathlib import Path
import subprocess
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 仮想環境を有効化（あれば）
venv_paths = [
    project_root / "venv" / "bin" / "activate",
    project_root / ".venv" / "bin" / "activate",
    project_root / "myenv" / "Scripts" / "activate",
]

venv_python = None
for venv_activate in venv_paths:
    if venv_activate.exists():
        if venv_activate.parent.parent.name in ["venv", ".venv"]:
            venv_python = venv_activate.parent.parent / "bin" / "python3"
        elif venv_activate.parent.parent.name == "myenv":
            venv_python = venv_activate.parent.parent / "Scripts" / "python.exe"
        if venv_python and venv_python.exists():
            break

# 仮想環境のPythonを使用（あれば）
if venv_python:
    python_executable = str(venv_python)
else:
    python_executable = sys.executable

def run_test():
    """テストを実行"""
    print("=" * 80)
    print("Stage 1.2 テスト実行")
    print("=" * 80)
    print()
    
    test_file = project_root / "tests" / "test_session_manager.py"
    
    if not test_file.exists():
        print(f"❌ テストファイルが見つかりません: {test_file}")
        return 1
    
    print(f"テストファイル: {test_file}")
    print(f"Python実行ファイル: {python_executable}")
    print()
    
    try:
        # pytest を実行
        result = subprocess.run(
            [python_executable, "-m", "pytest", str(test_file), "-v"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print("=" * 80)
        print("テスト実行結果")
        print("=" * 80)
        print()
        print(result.stdout)
        if result.stderr:
            print("--- エラー出力 ---")
            print(result.stderr)
        print()
        print("=" * 80)
        if result.returncode == 0:
            print("✅ すべてのテストが成功しました")
        else:
            print(f"❌ テストが失敗しました (終了コード: {result.returncode})")
        print("=" * 80)
        
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("❌ テストがタイムアウトしました（60秒）")
        return 1
    except Exception as e:
        print(f"❌ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_test())

