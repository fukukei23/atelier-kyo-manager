#!/bin/bash
# WSL環境でテスト結果を確認するスクリプト

cd /home/yn441611/atelier-kyo-manager

echo "=== テスト結果の確認 ==="
echo ""

# ログファイルを確認
if [ -f "test_site_config_connection.log" ]; then
    echo "--- ログファイルの内容 ---"
    cat test_site_config_connection.log
    echo ""
else
    echo "ログファイルが見つかりません"
fi

# 出力ファイルを確認
if [ -f "test_output.log" ]; then
    echo "--- 出力ファイルの内容 ---"
    cat test_output.log
    echo ""
fi

# エラーファイルを確認
if [ -f "test_error.log" ]; then
    echo "--- エラーファイルの内容 ---"
    cat test_error.log
    echo ""
fi

echo "=== 確認完了 ==="

