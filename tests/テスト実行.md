# テスト実行手順

## 実行前の準備

### 1. 仮想環境を有効化

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
```

### 2. pytest をインストール

```bash
pip install pytest pytest-asyncio
```

または、requirements.txtからすべてインストール：

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

### インポートエラーが発生する場合

プロジェクトルートで実行していることを確認：

```bash
cd /home/yn441611/atelier-kyo-manager
pwd  # /home/yn441611/atelier-kyo-manager であることを確認
```

### 仮想環境が見つからない

```bash
# venv を作成
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 期待されるテスト結果

以下のテストが含まれています：

- **test_plp_driver.py**: 7つのテスト
  - `test_plp_driver_materialize_tiles`
  - `test_plp_driver_trap_detection`
  - `test_plp_driver_trap_detection_no_recovery`
  - `test_plp_driver_click_tile`
  - `test_plp_driver_navigate_to_pdp_happy_path`
  - `test_plp_driver_navigate_to_pdp_same_tab`
  - `test_plp_driver_handle_overlays`

- **test_product_extractor.py**: 8つのテスト
  - `test_product_extractor_title`
  - `test_product_extractor_price`
  - `test_product_extractor_currency`
  - `test_product_extractor_images`
  - `test_product_extractor_extract_full`
  - `test_product_extractor_partial_selectors`
  - `test_product_extractor_normalize_price`
  - `test_product_extractor_json_ld_fallback`

- **test_browser_use_agent_plp_integration.py**: 2つのテスト
  - `test_browser_use_agent_delegates_to_plp_driver`
  - `test_browser_use_agent_uses_plp_driver_result`

## 実行例

```bash
$ python -m pytest tests/test_plp_driver.py -v
============================= test session starts ==============================
platform linux -- Python 3.x.x, pytest-x.x.x, pluggy-x.x.x
collected 7 items

tests/test_plp_driver.py::test_plp_driver_materialize_tiles PASSED    [ 14%]
tests/test_plp_driver.py::test_plp_driver_trap_detection PASSED       [ 28%]
...
============================== 17 passed in X.XXs ===============================
```

