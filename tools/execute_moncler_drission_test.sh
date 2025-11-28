#!/bin/bash
# MonclerDrissionHandler を実行して結果を確認するスクリプト

cd /home/yn441611/atelier-kyo-manager

echo "========================================"
echo "MonclerDrissionHandler 実行"
echo "========================================"
echo ""

# 実行
python3 tools/run_moncler_drission_with_report.py

echo ""
echo "========================================"
echo "結果ファイル確認"
echo "========================================"
echo ""

# 結果ファイルを確認
echo "最新のログファイル:"
ls -lt docs/test_results/moncler_drission_test_*.log 2>/dev/null | head -1

echo ""
echo "最新のJSON結果:"
ls -lt docs/test_results/moncler_drission_test_*.json 2>/dev/null | head -1

echo ""
echo "最新のレポート:"
ls -lt docs/test_results/moncler_drission_test_*.md 2>/dev/null | head -1

echo ""
echo "========================================"
echo "詳細なログ（最新）"
echo "========================================"
echo ""

LATEST_LOG=$(ls -t docs/test_results/moncler_drission_test_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "ファイル: $LATEST_LOG"
    echo ""
    tail -100 "$LATEST_LOG"
else
    echo "ログファイルが見つかりませんでした"
fi

