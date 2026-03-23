#!/usr/bin/env python3
import subprocess
import os

os.chdir("/home/yn441611/atelier-kyo-manager")

# Add all remaining changes
result = subprocess.run(
    ["git", "add", "-A"],
    capture_output=True, text=True
)
print("git add -A:", result.returncode)

# Commit
commit_msg = """既存ファイル変更・削除: クリーンアップ・仕様更新

- README.md削除 (docs/README.mdへ統合)
- app/utils/ai_supplier_scout_legacy.py 削除 (archiveへ移動)
- docs/codex_notes/ 削除 (archiveへ移動)
- run_tests_now.py 削除
- SPECファイル更新 (CR-ATELIER-002/003)
- 各種設定・ルート更新
"""
result = subprocess.run(
    ["git", "commit", "-m", commit_msg],
    capture_output=True, text=True
)
print("git commit:", result.returncode)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
