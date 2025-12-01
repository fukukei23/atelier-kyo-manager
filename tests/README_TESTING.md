# テスト実行ガイド

## クイックスタート

```bash
cd /home/yn441611/atelier-kyo-manager
bash tests/run_tests.sh
```

## 前提条件

- Python 3.8+
- pytest
- pytest-asyncio

## インストール

```bash
# 仮想環境を有効化
source venv/bin/activate  # または source myenv/Scripts/activate

# pytest をインストール
pip install pytest pytest-asyncio
```

または、requirements.txtからインストール：

```bash
pip install -r requirements.txt
```

## テスト実行

### すべてのテストを実行

```bash
python -m pytest tests/test_plp_driver.py tests/test_product_extractor.py tests/test_browser_use_agent_plp_integration.py -v
```

### 個別のテストファイルを実行

```bash
# PlpDriver のテスト
python -m pytest tests/test_plp_driver.py -v

# ProductExtractor のテスト
python -m pytest tests/test_product_extractor.py -v

# BrowserUseAgent 統合テスト
python -m pytest tests/test_browser_use_agent_plp_integration.py -v
```

### 特定のテスト関数を実行

```bash
# PlpDriver の Happy path テスト
python -m pytest tests/test_plp_driver.py::test_plp_driver_navigate_to_pdp_happy_path -v

# ProductExtractor の価格正規化テスト
python -m pytest tests/test_product_extractor.py::test_product_extractor_normalize_price -v
```

## トラブルシューティング

### pytest が見つからない

```bash
pip install pytest pytest-asyncio
```

### 仮想環境が見つからない

```bash
# venv を作成
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### インポートエラー

プロジェクトルートにいることを確認：

```bash
cd /home/yn441611/atelier-kyo-manager
python -m pytest tests/test_plp_driver.py -v
```

