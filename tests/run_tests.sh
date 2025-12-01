#!/bin/bash
# テスト実行スクリプト

cd /home/yn441611/atelier-kyo-manager

# 仮想環境を有効化
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "myenv" ]; then
    source myenv/Scripts/activate
else
    echo "仮想環境が見つかりません。venv または myenv を作成してください。"
    exit 1
fi

# pytest をインストール（まだインストールされていない場合）
pip install -q pytest pytest-asyncio

# テストを実行
echo "=========================================="
echo "PlpDriver のテストを実行中..."
echo "=========================================="
python -m pytest tests/test_plp_driver.py -v --tb=short

echo ""
echo "=========================================="
echo "ProductExtractor のテストを実行中..."
echo "=========================================="
python -m pytest tests/test_product_extractor.py -v --tb=short

echo ""
echo "=========================================="
echo "BrowserUseAgent 統合テストを実行中..."
echo "=========================================="
python -m pytest tests/test_browser_use_agent_plp_integration.py -v --tb=short

echo ""
echo "=========================================="
echo "すべてのテストを実行中..."
echo "=========================================="
python -m pytest tests/test_plp_driver.py tests/test_product_extractor.py tests/test_browser_use_agent_plp_integration.py -v --tb=short

