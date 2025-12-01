# テスト実行方法

## テストファイル一覧

以下のテストファイルが作成されています：

1. `tests/test_plp_driver.py` - PlpDriverのユニットテスト
2. `tests/test_product_extractor.py` - ProductExtractorのユニットテスト
3. `tests/test_browser_use_agent_plp_integration.py` - BrowserUseAgentとPlpDriverの統合テスト

## 実行方法

### 方法1: 自動実行スクリプト（推奨）

```bash
cd /home/yn441611/atelier-kyo-manager
bash tests/run_tests.sh
```

または

```bash
cd /home/yn441611/atelier-kyo-manager
python3 tests/run_tests_wrapper.py
```

### 方法2: 手動実行

まず、pytestをインストール：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate  # または source myenv/Scripts/activate
pip install pytest pytest-asyncio
```

その後、テストを実行：

```bash
python -m pytest tests/test_plp_driver.py tests/test_product_extractor.py tests/test_browser_use_agent_plp_integration.py -v
```

### 個別にテストを実行

```bash
# PlpDriverのテスト
python -m pytest tests/test_plp_driver.py -v

# ProductExtractorのテスト
python -m pytest tests/test_product_extractor.py -v

# BrowserUseAgentの統合テスト
python -m pytest tests/test_browser_use_agent_plp_integration.py -v
```

### 特定のテスト関数を実行

```bash
# PlpDriverのHappy pathテスト
python -m pytest tests/test_plp_driver.py::test_plp_driver_navigate_to_pdp_happy_path -v

# ProductExtractorの価格正規化テスト
python -m pytest tests/test_product_extractor.py::test_product_extractor_normalize_price -v
```

## テスト内容

### test_plp_driver.py

- `test_plp_driver_materialize_tiles` - PLPタイルのマテリアライズ
- `test_plp_driver_trap_detection` - Trapページ検出
- `test_plp_driver_trap_detection_no_recovery` - リカバリ失敗ケース
- `test_plp_driver_click_tile` - タイルクリック → PDP遷移
- `test_plp_driver_navigate_to_pdp_happy_path` - Happy path: PLP → PDP success
- `test_plp_driver_navigate_to_pdp_same_tab` - 同タブでのPDP遷移
- `test_plp_driver_handle_overlays` - オーバーレイ処理

### test_product_extractor.py

- `test_product_extractor_title` - タイトル抽出
- `test_product_extractor_price` - 価格抽出（float変換）
- `test_product_extractor_currency` - 通貨抽出
- `test_product_extractor_images` - 画像抽出
- `test_product_extractor_extract_full` - Full PDP extraction
- `test_product_extractor_partial_selectors` - Partial selectors / missing elements
- `test_product_extractor_normalize_price` - 価格正規化
- `test_product_extractor_json_ld_fallback` - JSON-LDフォールバック抽出

### test_browser_use_agent_plp_integration.py

- `test_browser_use_agent_delegates_to_plp_driver` - BrowserUseAgentがPlpDriverを正しく使用
- `test_browser_use_agent_uses_plp_driver_result` - BrowserUseAgentがPlpDriverの結果を正しく使用

## 注意事項

- テストはモックを使用しているため、実際のブラウザは不要です
- pytestが必要です（`pip install pytest pytest-asyncio`）
- すべてのテストは非同期（async）で実行されます

