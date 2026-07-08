#!/usr/bin/env python
"""テストを実行して結果をファイルに保存"""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent
os.chdir(project_root)

# テストを実行
result = subprocess.run(
    [sys.executable, "test_site_config_connection.py"], capture_output=True, text=True, cwd=str(project_root)
)

# 結果をファイルに保存
output_file = project_root / "test_latest_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
    f.write(f"\n=== RETURN CODE ===\n{result.returncode}\n")

# 結果を表示
print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)

sys.exit(result.returncode)
