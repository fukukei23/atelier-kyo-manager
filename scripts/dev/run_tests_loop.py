#!/usr/bin/env python3
"""
テストを継続的に実行するループ
失敗した場合は再実行（最大3回まで）
"""
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

def run_tests():
    """テストを実行して結果を返す"""
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--maxfail=5"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=600  # 10分タイムアウト
        )
        return result
    except subprocess.TimeoutExpired:
        print("[エラー] テストがタイムアウトしました（10分）")
        return None
    except Exception as e:
        print(f"[エラー] テスト実行エラー: {e}")
        return None

def main():
    max_attempts = 3
    attempt = 0
    
    print("=" * 80)
    print("テスト継続実行ループ")
    print("=" * 80)
    print(f"最大試行回数: {max_attempts}")
    print()
    
    while attempt < max_attempts:
        attempt += 1
        print(f"[試行 {attempt}/{max_attempts}]")
        print("-" * 80)
        
        result = run_tests()
        
        if result is None:
            print("[停止] テスト実行に失敗しました")
            break
        
        # 結果を表示
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        # 成功した場合
        if result.returncode == 0:
            print()
            print("=" * 80)
            print("✓ すべてのテストが成功しました！")
            print("=" * 80)
            return 0
        
        # 失敗した場合
        print()
        print("=" * 80)
        print(f"✗ テストが失敗しました（Return code: {result.returncode}）")
        print("=" * 80)
        
        if attempt < max_attempts:
            print()
            print(f"[再実行] {attempt + 1}回目の試行を開始します...")
            print()
            time.sleep(2)  # 少し待機
        else:
            print()
            print("[停止] 最大試行回数に達しました")
            print("=" * 80)
    
    return 1

if __name__ == "__main__":
    sys.exit(main())

