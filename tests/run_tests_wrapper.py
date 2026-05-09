#!/usr/bin/env python3
"""
テスト実行ラッパースクリプト
仮想環境を自動で有効化してテストを実行します
"""

import os
import subprocess
import sys


def find_venv():
    """仮想環境のパスを探す"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    venv_paths = [
        os.path.join(project_root, "venv"),
        os.path.join(project_root, "myenv"),
    ]

    for venv_path in venv_paths:
        if os.path.isdir(venv_path):
            if os.path.exists(os.path.join(venv_path, "bin", "activate")):
                return os.path.join(venv_path, "bin", "activate")
            elif os.path.exists(os.path.join(venv_path, "Scripts", "activate")):
                return os.path.join(venv_path, "Scripts", "activate")

    return None


def install_pytest():
    """pytestをインストール"""
    print("pytest をインストール中...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest", "pytest-asyncio"], check=False)


def run_tests():
    """テストを実行"""
    venv_activate = find_venv()

    if venv_activate:
        print(f"仮想環境を検出: {venv_activate}")
    else:
        print("警告: 仮想環境が見つかりません。システムのPythonを使用します。")

    # pytestがインストールされているか確認
    try:
        import pytest

        print(f"pytest version: {pytest.__version__}")
    except ImportError:
        print("pytest が見つかりません。インストールします...")
        install_pytest()

    # テストファイルのパス
    test_files = [
        "tests/test_plp_driver.py",
        "tests/test_product_extractor.py",
        "tests/test_browser_use_agent_plp_integration.py",
    ]

    # テストを実行
    cmd = [sys.executable, "-m", "pytest"] + test_files + ["-v", "--tb=short"]

    print("\n" + "=" * 60)
    print("テストを実行中...")
    print("=" * 60 + "\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())
