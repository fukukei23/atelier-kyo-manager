#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""テスト実行スクリプト - 結果をファイルに保存"""

import sys
import os
import subprocess

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def run_tests():
    """テストを実行して結果をファイルに保存"""
    
    # 仮想環境を探す
    venv_path = None
    if os.path.exists(os.path.join(project_root, "venv", "bin", "python")):
        venv_path = os.path.join(project_root, "venv", "bin", "python")
    elif os.path.exists(os.path.join(project_root, "myenv", "Scripts", "python.exe")):
        venv_path = os.path.join(project_root, "myenv", "Scripts", "python.exe")
    
    python_cmd = venv_path if venv_path else sys.executable
    
    # pytestをインストール
    print("pytest をチェック中...")
    try:
        subprocess.run([python_cmd, "-m", "pip", "install", "-q", "pytest", "pytest-asyncio"], 
                      check=False, capture_output=True)
    except Exception as e:
        print(f"Warning: {e}")
    
    # テストファイル
    test_files = [
        "tests/test_plp_driver.py",
        "tests/test_product_extractor.py",
        "tests/test_browser_use_agent_plp_integration.py",
    ]
    
    # 結果ファイル
    result_file = os.path.join(project_root, "test_results.txt")
    
    print(f"\nテストを実行中... (結果は {result_file} に保存されます)\n")
    print("=" * 70)
    
    # テストを実行
    cmd = [python_cmd, "-m", "pytest"] + test_files + ["-v", "--tb=short"]
    
    try:
        with open(result_file, "w", encoding="utf-8") as f:
            result = subprocess.run(cmd, cwd=project_root, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT,
                                  text=True)
            output = result.stdout
            f.write(output)
            print(output)
        
        print("=" * 70)
        print(f"\nテスト結果を {result_file} に保存しました。")
        return result.returncode
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())

