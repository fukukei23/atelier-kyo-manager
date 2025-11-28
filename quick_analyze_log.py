#!/usr/bin/env python3
"""簡易ログ解析スクリプト（出力を確実に表示）"""
import re
from pathlib import Path

project_root = Path(__file__).parent
log_files = list(project_root.glob("browser_test_*.log"))

if not log_files:
    print("ログファイルが見つかりません")
    exit(1)

latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
print(f"解析対象: {latest_log.name}")
print("=" * 80)

with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

issues = []
if re.search(r'Collected 0 PDP links', content):
    issues.append("✗ PDP リンクが0件")
if re.search(r'PLP materialization failed', content):
    issues.append("✗ PLP materialization が失敗")
if re.search(r'Timeout after \d+s', content):
    issues.append("✗ タイムアウトが発生")

selector_errors = re.findall(r"waiting for locator\(['\"]([^'\"]+)['\"]\)", content)
if selector_errors:
    issues.append(f"✗ セレクタが見つからない: {selector_errors[0]}")

if issues:
    print("検出された問題:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("問題は検出されませんでした")

print("=" * 80)

