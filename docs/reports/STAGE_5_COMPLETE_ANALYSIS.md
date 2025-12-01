# Stage 5 完全分析レポート

## Task A: ハードコード箇所の一覧

詳細は `docs/reports/STAGE_5_TASK_A_HARDCODE_AUDIT.md` を参照。

### 主要なハードコード箇所

1. **デフォルトセレクタ** (32-54行目)
   - `DEFAULT_TITLE_SELECTORS`
   - `DEFAULT_PRICE_SELECTORS`
   - `DEFAULT_SIZE_BUTTON_SELECTORS`

2. **価格正規化ロジック** (279-323行目)
   - 正規表現パターン: `r"[\d.,]+"`
   - デフォルトの `strip_chars`: `["¥", ",", " "]`（日本円向け）

3. **画像 URL 正規化** (376-381行目)
   - プロトコル強制: `https:` を強制付与
   - 相対 URL 処理が簡易的

4. **在庫チェックテキスト** (611-612行目)
   - `"out of stock"` と `"在庫なし"` が固定

5. **JSON-LD パス** (650-665行目)
   - `offers.price`, `offers.priceCurrency` が固定

6. **メタデータキー** (174-183行目)
   - メタデータのキー名が固定

7. **HTML 保存パス** (189行目)
   - ファイル名 `"pdp_raw.html"` が固定

8. **Moncler 専用分岐** (extractor.py 260-264行目)
   - `if site.upper() == "MONCLER_OFFICIAL":` が固定

## Task B: PDP 用 site_config スキーマ案

詳細は `docs/reports/STAGE_5_TASK_B_SCHEMA_PROPOSAL.md` を参照。

### 主要なスキーマ設計

1. **selectors.pdp.* のキー設計**
   - 基本情報: `title`, `price`, `list_price`, `currency`
   - メディア: `images` (image_attr, image_base_url 設定可能)
   - 属性: `colors`, `sizes`, `brand`, `sku`, `availability`
   - サイズ選択: `size_button`, `size_select_policy`
   - HTML 保存: `raw_html_capture`

2. **価格正規化ルール**
   - `price_rules` または `selectors.pdp.price.normalize_rules`
   - `strip_chars`, `thousands_separator`, `decimal_separator`, `currency_fallback`, `price_pattern`, `currency_symbols`

3. **JSON-LD / Meta タグフォールバック**
   - `selectors.pdp.json_ld.paths` で JSON-LD のパスを設定可能
   - `selectors.pdp.meta_fallback.selectors` で Meta タグセレクタを設定可能

4. **Moncler 用サンプル**
   - 完全な site_config 断片を提供

## Task C: リファクタリング方針と具体案

詳細は `docs/reports/STAGE_5_TASK_C_REFACTORING_PROPOSAL.md` を参照。

### 主要な変更点

1. **Config Getter の追加**
   - `_get_pdp_config()`: PDP 設定を取得（新スキーマ優先、旧スキーマフォールバック）
   - `_get_price_rules()`: 価格正規化ルールを取得

2. **抽出メソッドのシグネチャ変更**
   - すべての `_extract_*()` メソッドが `pdp_config` を引数として受け取る
   - `_extract_price()`, `_extract_list_price_and_discount()` が `price_rules` も受け取る

3. **価格正規化の改善**
   - `_normalize_price_to_float()` が `price_rules` を引数として受け取る
   - 正規表現パターンが設定可能に

4. **画像 URL 正規化の改善**
   - `_normalize_image_url()` メソッドを追加
   - `image_base_url` 設定に対応

5. **BrowserExtractionService 側の調整**
   - `price` が None でも返す（graceful degradation）
   - すべてのフィールド（`raw_html_path`, `metadata` を含む）を dict に変換

## Task D: テスト戦略と具体的テストケース案

詳細は `docs/reports/STAGE_5_TASK_D_TEST_STRATEGY.md` を参照。

### 必須のテスト観点

1. **Full extraction** - すべてのフィールドを抽出
2. **Partial selectors** - 部分的なセレクタのみ
3. **Price normalization** - 様々な価格表記の正規化
4. **Missing config graceful degradation** - 設定欠損時の動作
5. **Metadata counts** - メタデータのカウント

### 代表的なテスト関数

- `test_product_extractor_full_extraction`
- `test_product_extractor_partial_selectors`
- `test_product_extractor_price_normalization_various_formats`
- `test_product_extractor_missing_config_graceful_degradation`
- `test_product_extractor_metadata_counts`
- `test_get_pdp_config_with_new_schema`
- `test_get_pdp_config_fallback_to_defaults`
- `test_get_price_rules_with_new_schema`
- `test_get_price_rules_fallback_to_legacy`

## Stage 4 追加タスク: 薄い統合テスト

詳細は `docs/reports/STAGE_4_ADDITIONAL_INTEGRATION_TEST_DIFF.md` を参照。

### 追加テスト

- `test_run_plp_flow_saves_plp_navigation_result`
  - BrowserUseAgent._run_plp_flow() を実際に通す
  - PlpDriver.navigate_to_pdp() が新シグネチャで呼ばれることを確認
  - RunContext.save_json() に新フィールドが保存されることを確認

## 次のステップ

1. **Stage 5 の実装開始**
   - Task A の洗い出し結果を基に、ProductExtractor をリファクタリング
   - Task B のスキーマ案を実装
   - Task C のリファクタリング方針に従ってコード変更

2. **テストの実装**
   - Task D のテストケースを実装
   - 既存のテストを更新

3. **Moncler 用 site_config の更新**
   - Task B のサンプルを基に、実際の Moncler 用 site_config を更新

4. **段階的移行**
   - 既存のコードとの互換性を保ちながら、段階的に移行

