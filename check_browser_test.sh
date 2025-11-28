#!/bin/bash
# 実ブラウザテスト実行と結果確認スクリプト

set -e

cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

echo "=========================================="
echo "実ブラウザテスト開始: MONCLER_OFFICIAL"
echo "=========================================="
echo ""

# テスト実行（バックグラウンドで実行し、ログをファイルに保存）
LOG_FILE="browser_test_$(date +%Y%m%d_%H%M%S).log"
python tools/run_browser_use.py \
  --site "MONCLER_OFFICIAL" \
  --url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \
  --query "down jacket" \
  --headful \
  --timeout 120 \
  2>&1 | tee "$LOG_FILE" || true

echo ""
echo "=========================================="
echo "テスト完了"
echo "=========================================="
echo "ログファイル: $LOG_FILE"
echo ""

# ログから重要な情報を抽出
if [ -f "$LOG_FILE" ]; then
    echo "=== 重要なログメッセージ ==="
    grep -E "(NavigationDriver|PLP→PDP|collected.*PDP|ERROR|FAILED|成功|Success|result\.ok)" "$LOG_FILE" | tail -30 || echo "該当するログが見つかりませんでした"
    echo ""
    
    # 最新の実行結果ディレクトリを確認
    LATEST_RUN=$(ls -td instance/runs/* 2>/dev/null | head -1)
    if [ -n "$LATEST_RUN" ]; then
        echo "=== 実行結果ディレクトリ ==="
        echo "$LATEST_RUN"
        if [ -f "$LATEST_RUN/result.json" ]; then
            echo ""
            echo "=== result.json の内容 ==="
            cat "$LATEST_RUN/result.json" | python -m json.tool 2>/dev/null || cat "$LATEST_RUN/result.json"
        fi
    fi
fi

