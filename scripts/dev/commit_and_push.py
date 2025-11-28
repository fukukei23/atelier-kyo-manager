#!/usr/bin/env python3
"""コミットとプッシュを実行"""
import subprocess
import sys
from pathlib import Path

cwd = Path('/home/yn441611/atelier-kyo-manager')

def run_cmd(cmd, description):
    print(f"\n{'='*80}")
    print(f"{description}")
    print('='*80)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(cwd)
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        print(f"Return code: {result.returncode}")
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        return False, "", str(e)

# 現在の状態確認
run_cmd("git status --short", "現在の状態")

# コミット
success, stdout, stderr = run_cmd(
    "git commit -m 'Add browser agent improvements, test tools, and documentation updates'",
    "コミット実行"
)

if not success:
    print("\nコミットに失敗しました。エラーを確認してください。")
    sys.exit(1)

# プッシュ
success, stdout, stderr = run_cmd(
    "git push origin main",
    "プッシュ実行"
)

if success:
    print("\n✓ プッシュが成功しました！")
else:
    print("\n✗ プッシュに失敗しました。エラーを確認してください。")
    if stderr:
        print(f"エラー内容: {stderr}")

# 最終確認
run_cmd("git log --oneline -3", "最新のコミット")
run_cmd("git status -sb", "最終状態")

