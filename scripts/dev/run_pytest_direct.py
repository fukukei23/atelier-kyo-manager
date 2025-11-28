#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest を直接実行するスクリプト（ターミナル問題回避版）
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
        break

if not venv_python:
    print("⚠️  仮想環境の Python が見つかりません。システムの Python を使用します。")
    venv_python = sys.executable

def run_pytest(test_path=None, verbose=True, extra_args=None):
    """
    pytest を実行する
    
    Args:
        test_path: テストファイルまたはディレクトリ（None の場合は tests/）
        verbose: -v フラグを追加するか
        extra_args: 追加の pytest 引数
    """
    # pytest コマンドを構築
    cmd = [venv_python, "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
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
    
    if extra_args:
        cmd.extend(extra_args)
    
    print("=" * 80)
    print("pytest 実行")
    print("=" * 80)
    print(f"Python: {venv_python}")
    print(f"コマンド: {' '.join(cmd)}")
    print("=" * 80)
    print()
    
    try:
        # pytest を実行
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            timeout=300,  # 5分タイムアウト
        )
        
        print()
        print("=" * 80)
        if result.returncode == 0:
            print("✅ すべてのテストが成功しました")
        else:
            print(f"❌ テストが失敗しました (終了コード: {result.returncode})")
        print("=" * 80)
        
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("❌ テストがタイムアウトしました（5分）")
        return 1
    except Exception as e:
        print(f"❌ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="pytest を実行するスクリプト")
    parser.add_argument(
        "test_path",
        nargs="?",
        default=None,
        help="テストファイルまたはディレクトリ（デフォルト: tests/）"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="詳細出力（デフォルト: True）"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="簡潔出力（-v を無効化）"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="-v フラグを付けない"
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="追加の pytest 引数"
    )
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet and not args.no_verbose
    
    return run_pytest(
        test_path=args.test_path,
        verbose=verbose,
        extra_args=args.extra if args.extra else None,
    )

if __name__ == "__main__":
    sys.exit(main())

