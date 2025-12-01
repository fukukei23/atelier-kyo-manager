# Stage 5 テスト実行ガイド

## テストファイル

- **対象ファイル**: `tests/test_product_extractor.py`
- **テスト関数数**: 16個

## テスト実行方法

以下のコマンドでテストを実行できます：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate 2>/dev/null || source myenv/Scripts/activate 2>/dev/null || true
python -m pytest tests/test_product_extractor.py -v
```

## テスト一覧

### 既存テスト（8個）

1. `test_product_extractor_title` - タイトル抽出
2. `test_product_extractor_price` - 価格抽出
3. `test_product_extractor_currency` - 通貨抽出
4. `test_product_extractor_images` - 画像抽出
5. `test_product_extractor_extract_full` - フル抽出（既存）
6. `test_product_extractor_partial_selectors` - 部分セレクタ
7. `test_product_extractor_normalize_price` - 価格正規化
8. `test_product_extractor_json_ld_fallback` - JSON-LD フォールバック

### 新規テスト（8個）

9. `test_product_extractor_full_extraction` - Full extraction
10. `test_product_extractor_price_normalization_various_formats` - 様々な価格表記の正規化
11. `test_product_extractor_missing_config_graceful_degradation` - Missing config graceful degradation
12. `test_product_extractor_metadata_counts` - Metadata counts
13. `test_get_pdp_config_with_new_schema` - Config getter (新スキーマ)
14. `test_get_pdp_config_fallback_to_defaults` - Config getter (デフォルト)
15. `test_get_price_rules_with_new_schema` - Price rules (新スキーマ)
16. `test_get_price_rules_fallback_to_legacy` - Price rules (旧スキーマ)

## テスト結果の確認

テスト結果は自動的に `docs/reports/TEST_RESULTS_YYYYMMDD_HHMMSS.txt` に保存されます。

最新のテスト結果ファイルを確認するには：

```bash
ls -lt docs/reports/TEST_RESULTS_*.txt | head -1
```

## 個別テストの実行

特定のテストだけを実行する場合：

```bash
# 1つのテストを実行
python -m pytest tests/test_product_extractor.py::test_product_extractor_title -v

# 複数のテストを実行
python -m pytest tests/test_product_extractor.py::test_product_extractor_title tests/test_product_extractor.py::test_product_extractor_price -v

# パターンマッチで実行
python -m pytest tests/test_product_extractor.py -k "price" -v
```

## テストカバレッジ

すべてのテストは以下の観点をカバーしています：

- ✅ Full extraction（すべてのフィールドを抽出）
- ✅ Partial selectors（部分的なセレクタのみ）
- ✅ Price normalization（様々な価格表記の正規化）
- ✅ Missing config graceful degradation（設定欠損時の動作）
- ✅ Metadata counts（メタデータのカウント）
- ✅ Config getter（新スキーマ）
- ✅ Config getter（デフォルトフォールバック）
- ✅ Price rules（新スキーマ）
- ✅ Price rules（旧スキーマフォールバック）

## 注意事項

- すべてのテストはモックを使用しており、実際のブラウザを起動しません
- `Playwright` の `Page` や `Locator` オブジェクトは `AsyncMock` でモック化されています
- テストは非同期 (`async def`) で定義されています

