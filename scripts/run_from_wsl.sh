#!/bin/bash
# WSL環境から直接実行するスクリプト
# 注意: WSL環境では実際のブラウザ操作はできません（環境確認のみ）

set -e

# プロジェクトルートに移動
cd "$(dirname "$0")/.."

# 仮想環境を有効化
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# 引数の解析
QUERY="${1:-down jacket}"
TARGET_URL="${2:-}"
HEADLESS="${3:-}"

echo "================================================================================"
echo "MONCLER Drission 診断スクリプト実行 (WSL環境)"
echo "================================================================================"
echo ""
echo "⚠️  注意: WSL環境では実際のブラウザ操作はできません"
echo "   このスクリプトは環境確認のみを行います"
echo "   実際のブラウザ操作を実行するには、Windows環境で実行してください"
echo ""
echo "================================================================================"
echo ""

# 環境確認スクリプトを実行
echo "環境確認中..."
python3 scripts/check_windows_environment.py

if [ $? -ne 0 ]; then
    echo ""
    echo "環境確認に失敗しました。"
    exit 1
fi

echo ""
echo "================================================================================"
echo "診断スクリプトを実行します（環境確認のみ）"
echo "================================================================================"
echo ""

# 診断スクリプトを実行（実際のブラウザ操作はできないため、エラーになる可能性があります）
CMD="python3 scripts/run_moncler_drission_diagnostics.py --query \"$QUERY\""

if [ -n "$TARGET_URL" ]; then
    CMD="$CMD --target_url \"$TARGET_URL\""
fi

if [ "$HEADLESS" = "--headless" ] || [ "$HEADLESS" = "headless" ]; then
    CMD="$CMD --headless"
fi

echo "実行コマンド: $CMD"
echo ""

eval $CMD

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "実行が正常に完了しました"
else
    echo "実行中にエラーが発生しました (終了コード: $EXIT_CODE)"
    echo ""
    echo "注意: WSL環境では実際のブラウザ操作はできません。"
    echo "      Windows環境で実行する必要があります。"
fi
echo "================================================================================"

exit $EXIT_CODE

