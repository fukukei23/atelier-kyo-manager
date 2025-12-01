# Stage 5 Task 4: テスト追加 - 差分形式

## 概要

Stage 5 の実装に対応した新しいテストケースを追加しました。

## 追加されたテストケース

### 1. test_product_extractor_full_extraction

**目的**: Full extraction（すべてのフィールドを抽出）

**テスト内容**:
- site_config にすべてのセレクタが定義されている
- モック Page で各セレクタに対応する要素が存在する
- ProductInfo のすべてのフィールド（title, price, list_price, currency, images, sizes, colors, description, brand）が抽出される
- metadata が正しく設定されている（has_title, has_price, image_count, size_count, color_count）

### 2. test_product_extractor_price_normalization_various_formats

**目的**: Price normalization（様々な価格表記の正規化）

**テスト内容**:
- 日本円用の設定で複数の価格表記をテスト（¥ 12,345, ¥123,456, 1,234円, 12,345.67）
- ユーロ用の設定で複数の価格表記をテスト（€ 99,99, 1.234,56）
- 各価格表記が正しく float に変換されることを確認

### 3. test_product_extractor_missing_config_graceful_degradation

**目的**: Missing config graceful degradation（設定欠損時の動作）

**テスト内容**:
- site_config に `selectors.pdp` が存在しない場合でも動作する
- デフォルトセレクタが使用されることを確認
- 例外が発生しないことを確認
- metadata が正しく設定されることを確認

### 4. test_product_extractor_metadata_counts

**目的**: Metadata counts（メタデータのカウント）

**テスト内容**:
- 3枚の画像、5つのサイズ、2つのカラーを抽出
- metadata の image_count, size_count, color_count が正しい値になることを確認

### 5. test_get_pdp_config_with_new_schema

**目的**: Config getter（新スキーマ）

**テスト内容**:
- 新スキーマ（`selectors.pdp.*`）から設定を取得できることを確認

### 6. test_get_pdp_config_fallback_to_defaults

**目的**: Config getter（デフォルトフォールバック）

**テスト内容**:
- site_config が空の場合でもデフォルトセレクタが使用されることを確認

### 7. test_get_price_rules_with_new_schema

**目的**: Price rules（新スキーマ）

**テスト内容**:
- 新スキーマ（`selectors.pdp.price.normalize_rules`）から価格正規化ルールを取得できることを確認

### 8. test_get_price_rules_fallback_to_legacy

**目的**: Price rules（旧スキーマフォールバック）

**テスト内容**:
- 旧スキーマ（トップレベルの `price_rules`）から価格正規化ルールを取得できることを確認

## 既存テストの更新

### 既存のテストを Stage 5 の変更に合わせて更新

以下のテストを更新しました：

- `test_product_extractor_title` - `pdp_config` を引数に追加
- `test_product_extractor_price` - `pdp_config` と `price_rules` を引数に追加
- `test_product_extractor_currency` - `pdp_config` を引数に追加
- `test_product_extractor_images` - `pdp_config` を引数に追加
- `test_product_extractor_normalize_price` - `price_rules` を引数に追加
- `test_product_extractor_json_ld_fallback` - `pdp_config` を引数に追加

## テストファイルの変更

### tests/test_product_extractor.py

**追加されたテスト関数**:
- `test_product_extractor_full_extraction()` (376行目～)
- `test_product_extractor_price_normalization_various_formats()` (約500行目～)
- `test_product_extractor_missing_config_graceful_degradation()` (約560行目～)
- `test_product_extractor_metadata_counts()` (約620行目～)
- `test_get_pdp_config_with_new_schema()` (約700行目～)
- `test_get_pdp_config_fallback_to_defaults()` (約730行目～)
- `test_get_price_rules_with_new_schema()` (約750行目～)
- `test_get_price_rules_fallback_to_legacy()` (約780行目～)

## テスト実行

すべてのテストは pytest で実行可能です：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate 2>/dev/null || source myenv/Scripts/activate 2>/dev/null || true
python -m pytest tests/test_product_extractor.py -v
```

## テストカバレッジ

### カバーされている観点

- ✅ Full extraction（すべてのフィールドを抽出）
- ✅ Partial selectors（部分的なセレクタのみ）
- ✅ Price normalization（様々な価格表記の正規化）
- ✅ Missing config graceful degradation（設定欠損時の動作）
- ✅ Metadata counts（メタデータのカウント）
- ✅ Config getter（新スキーマ）
- ✅ Config getter（デフォルトフォールバック）
- ✅ Price rules（新スキーマ）
- ✅ Price rules（旧スキーマフォールバック）

## 既存テストとの関係

既存のテスト（`test_product_extractor_extract_full`, `test_product_extractor_partial_selectors` など）も引き続き動作します。新しく追加したテストは、Stage 5 の変更に特化したより詳細なテストケースを提供します。

