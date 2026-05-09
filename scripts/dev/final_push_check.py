#!/usr/bin/env python3
"""最終的なプッシュ状態確認"""

import subprocess
from datetime import datetime


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/home/yn441611/atelier-kyo-manager")
    return result.stdout.strip(), result.stderr.strip(), result.returncode


output_lines = []
output_lines.append(f"確認日時: {datetime.now()}")
output_lines.append("=" * 80)

# Git設定確認
stdout, stderr, code = run_cmd("git config user.name && git config user.email")
output_lines.append("\n【Git設定】")
output_lines.append(stdout if stdout else f"エラー: {stderr}")

# 最新のコミット
stdout, stderr, code = run_cmd("git log --oneline -5")
output_lines.append("\n【ローカルの最新コミット（5件）】")
output_lines.append(stdout if stdout else f"エラー: {stderr}")

# リモートの最新コミット
stdout, stderr, code = run_cmd("git log origin/main --oneline -3")
output_lines.append("\n【リモート（origin/main）の最新コミット（3件）】")
output_lines.append(stdout if stdout else f"エラー: {stderr}")

# プッシュ待ち
stdout, stderr, code = run_cmd("git log origin/main..HEAD --oneline")
output_lines.append("\n【プッシュ待ちのコミット】")
if stdout:
    output_lines.append(stdout)
    output_lines.append("⚠️ プッシュ待ちのコミットがあります！")
else:
    output_lines.append("なし（すべてプッシュ済み）" if code == 0 else f"エラー: {stderr}")

# 状態
stdout, stderr, code = run_cmd("git status -sb")
output_lines.append("\n【現在の状態】")
output_lines.append(stdout if stdout else f"エラー: {stderr}")

result = "\n".join(output_lines)
print(result)

# ファイルにも保存
with open("final_push_check_result.txt", "w", encoding="utf-8") as f:
    f.write(result)
