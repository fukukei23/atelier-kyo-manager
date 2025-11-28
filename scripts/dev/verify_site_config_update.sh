#!/bin/bash
# site_config の更新を確認するスクリプト

cd /home/yn441611/atelier-kyo-manager

echo "=== JSON構文チェック ==="
python3 -m json.tool app/config/sites/overrides.local.json > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ JSON構文チェック: OK"
else
    echo "✗ JSON構文エラー"
    exit 1
fi

echo ""
echo "=== site_config 接続テスト ==="
source venv/bin/activate
python test_site_config_connection.py > test_update_output.log 2>&1

echo ""
echo "=== テスト結果 ==="
if [ -f test_update_output.log ]; then
    # 重要なメッセージを抽出
    echo "--- selectors.plp の確認 ---"
    grep -E "selectors\.plp|✓ selectors\.plp|⚠ selectors\.plp" test_update_output.log || echo "見つかりません"
    
    echo ""
    echo "--- navigation.header_search の確認 ---"
    grep -E "navigation\.header_search|✓.*header_search|⚠.*header_search" test_update_output.log || echo "見つかりません"
    
    echo ""
    echo "--- navigation.overlays の確認 ---"
    grep -E "navigation\.overlays|✓.*overlays|⚠.*overlays" test_update_output.log || echo "見つかりません"
    
    echo ""
    echo "=== 完全なログ ==="
    cat test_update_output.log
else
    echo "ログファイルが見つかりません"
fi

