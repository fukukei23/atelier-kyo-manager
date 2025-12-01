# Stage 5 実装 - 差分形式

## 概要

ProductExtractor を site_config 駆動に完全移行しました。

## Task 1: ProductExtractor に config getter を実装

### app/agents/browser/product_extractor.py の変更

#### 1. __init__ メソッドの変更

```diff
  class ProductExtractor:
      def __init__(
          self,
          site_config: Dict[str, Any],
          run_context: Optional["RunContext"] = None,
          logger: Optional[logging.Logger] = None,
      ) -> None:
          self.site_config = site_config
          self.run_context = run_context
          self.logger = logger or logging.getLogger(__name__)
          
-         # price_rules を取得
-         price_rules_cfg = site_config.get("price_rules", {}) or {}
-         self.price_rules = PriceRules(
-             strip_chars=price_rules_cfg.get("strip_chars", ["¥", ",", " "]),
-             decimal_separator=price_rules_cfg.get("decimal_separator", "."),
-             thousands_separator=price_rules_cfg.get("thousands_separator", ","),
-         )
+         # Stage 5: config getter のキャッシュ用
+         self._pdp_config: Optional[Dict[str, Any]] = None
+         self._price_rules: Optional[Dict[str, Any]] = None
```

#### 2. _get_pdp_config() メソッドの追加

```python
def _get_pdp_config(self) -> Dict[str, Any]:
    """
    site_config から PDP 設定を取得（後方互換性維持）
    
    Stage 5: 新しい selectors.pdp.* スキーマを優先しつつ、
    既存の selectors.pdp.* やデフォルト値からもフォールバック
    """
    # キャッシュがあれば返す
    if self._pdp_config is not None:
        return self._pdp_config
    
    # 新スキーマ: selectors.pdp.* から取得
    # 旧スキーマからのフォールバック
    # デフォルト値のフォールバック
    # ... (実装詳細はコード参照)
    
    return self._pdp_config
```

#### 3. _get_price_rules() メソッドの追加

```python
def _get_price_rules(self) -> Dict[str, Any]:
    """
    site_config から価格正規化ルールを取得
    
    Stage 5: selectors.pdp.price.normalize_rules を優先しつつ、
    トップレベルの price_rules からもフォールバック
    """
    # 新スキーマ: selectors.pdp.price.normalize_rules
    # 旧スキーマ: トップレベルの price_rules
    # デフォルト値
    # ... (実装詳細はコード参照)
    
    return self._price_rules
```

#### 4. extract() メソッドの変更

```diff
  async def extract(
      self,
      page: Page,
      *,
      context: Optional[BrowserContext] = None,
      prepare_page: Optional[Any] = None,
  ) -> ProductInfo:
      # ...
+     # Stage 5: config getter を使用
+     pdp_config = self._get_pdp_config()
+     price_rules = self._get_price_rules()
      
      product_info = ProductInfo(url=page.url)
      
-     # 1. タイトル抽出
-     product_info.title = await self._extract_title(page)
+     # 1. タイトル抽出
+     product_info.title = await self._extract_title(page, pdp_config)
      
      # 2. 価格抽出（サイズ選択を試行）
-     product_info.price = await self._extract_price_with_size_option(page)
+     product_info.price = await self._extract_price_with_size_option(page, pdp_config, price_rules)
      
      # ... (他の抽出メソッドも同様に変更)
```

#### 5. 抽出メソッドのシグネチャ統一

すべての `_extract_*()` メソッドが `pdp_config` (必要に応じて `price_rules`) を引数として受け取るように変更:

- `_extract_title(page, pdp_config)`
- `_extract_price(page, pdp_config, price_rules)`
- `_extract_price_with_size_option(page, pdp_config, price_rules)`
- `_extract_currency(page, pdp_config)`
- `_extract_images(page, pdp_config)`
- `_extract_sizes(page, pdp_config)`
- `_extract_colors(page, pdp_config)`
- `_extract_description(page, pdp_config)`
- `_extract_brand(page, pdp_config)`
- `_extract_list_price_and_discount(page, pdp_config, price_rules)`
- `_extract_from_json_ld_or_meta(page, pdp_config)`
- `_click_size_to_reveal_price(page, pdp_config, size_select_policy)`

#### 6. 価格正規化メソッドの変更

```diff
- def _normalize_price_to_float(self, price_text: str) -> Optional[float]:
+ def _normalize_price_to_float(self, price_text: str, price_rules: Dict[str, Any]) -> Optional[float]:
      """
      価格テキストを正規化して float に変換する。
-     Task 2: price_rules に基づいて正規化し、float に変換
+     Stage 5: price_rules を引数として受け取る
      """
      # price_rules から設定を取得して正規化
      # ...
```

#### 7. 画像 URL 正規化メソッドの追加

```python
def _normalize_image_url(
    self,
    src: str,
    page_url: str,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """画像 URL を正規化する（Stage 5: site_config から base_url を取得）"""
    # プロトコル相対 URL / 絶対 URL / 相対 URL の処理
    # ...
```

#### 8. JSON-LD フォールバックの改善

```diff
- async def _extract_from_json_ld_or_meta(self, page: Page) -> Optional[Dict[str, Any]]:
+ async def _extract_from_json_ld_or_meta(self, page: Page, pdp_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
      """JSON-LD または Meta タグから価格情報を抽出する（フォールバック）。Stage 5: pdp_config を引数として受け取る"""
+     json_ld_cfg = pdp_config.get("json_ld", {})
+     meta_fallback_cfg = pdp_config.get("meta_fallback", {})
      
      # JSON-LD から抽出（設定されたパスを使用）
      # Meta タグから抽出（設定されたセレクタを使用）
      # ...
```

#### 9. 在庫チェックテキストの site_config 化

```diff
  # _click_size_to_reveal_price 内
- if "out of stock" in text or "在庫なし" in text:
+ availability_patterns = pdp_config.get("availability_patterns", [
+     "out of stock",
+     "在庫なし",
+ ])
+ if any(pattern.lower() in text for pattern in availability_patterns):
```

#### 10. HTML 保存パスの設定化

```diff
  # extract() メソッド内
- if self.run_context:
+ html_capture = pdp_config.get("raw_html_capture", {
+     "enabled": True,
+     "filename": "pdp_raw.html",
+ })
+ if html_capture.get("enabled", True) and self.run_context:
      try:
          html_content = await page.content()
-         self.run_context.save_content("pdp_raw.html", html_content)
-         product_info.raw_html_path = str(self.run_context.get_path("pdp_raw.html"))
+         filename = html_capture.get("filename", "pdp_raw.html")
+         self.run_context.save_content(filename, html_content)
+         product_info.raw_html_path = str(self.run_context.get_path(filename))
```

---

## Task 2: ハードコードの site_config 移行

### 主な変更点

1. **デフォルトセレクタの扱い**
   - `DEFAULT_*_SELECTORS` は `_get_pdp_config()` 内で「最終フォールバック」として使用
   - 抽出メソッド内からは直接参照しない

2. **価格正規化ロジック**
   - 正規表現パターン `r"[\d.,]+"` は `price_rules.get("price_pattern", r"[\d.,]+")` から取得
   - `strip_chars` は `price_rules.get("strip_chars", ["¥", ",", " "])` から取得

3. **画像 URL 正規化**
   - `_normalize_image_url()` メソッドに切り出し
   - `image_base_url` 設定に対応

4. **在庫チェックテキスト**
   - `availability_patterns` を site_config から取得

5. **JSON-LD パス**
   - `selectors.pdp.json_ld.paths` から取得

---

## Task 3: BrowserExtractionService の調整

### app/agents/browser/extractor.py の変更

```diff
  # ProductInfo を Dict に変換
- if product_info.price is not None:  # 価格が見つかった場合のみ返す
      data = {
          "title": product_info.title,
          "price": product_info.price,  # float or None
          "currency": product_info.currency,
          "url": product_info.url or page.url,
          "images": product_info.images,
          "sizes": product_info.sizes,
          "colors": product_info.colors,
          "description": product_info.description,
          "brand": product_info.brand,
          "list_price": product_info.list_price,  # float or None
          "discount_pct": product_info.discount_pct,
          "raw_html_path": product_info.raw_html_path,  # Stage 5: HTML パス
          "metadata": product_info.metadata,  # Stage 5: metadata
      }
-     self.logger.debug(f"[Extractor] ProductExtractor succeeded for {url}")
-     return data
+     self.logger.debug(f"[Extractor] ProductExtractor succeeded for {url} (price: {product_info.price})")
+     return data  # price が None でも返す（graceful degradation）
```

**変更点**:
- `price` が `None` でも例外を出さずに返す（graceful degradation）
- すべてのフィールド（`raw_html_path`, `metadata` を含む）を dict に変換

---

## Task 4: テスト実装

### tests/test_product_extractor.py の拡張

以下のテスト関数を追加・更新（詳細は次のセクションで説明）:

1. `test_product_extractor_full_extraction` - Full extraction
2. `test_product_extractor_partial_selectors` - Partial selectors
3. `test_product_extractor_price_normalization_various_formats` - Price normalization
4. `test_product_extractor_missing_config_graceful_degradation` - Missing config
5. `test_product_extractor_metadata_counts` - Metadata counts
6. `test_get_pdp_config_with_new_schema` - Config getter (新スキーマ)
7. `test_get_pdp_config_fallback_to_defaults` - Config getter (デフォルト)
8. `test_get_price_rules_with_new_schema` - Price rules (新スキーマ)
9. `test_get_price_rules_fallback_to_legacy` - Price rules (旧スキーマ)

---

## 後方互換性

- 既存の `selectors.pdp.*` スキーマは引き続き動作
- トップレベルの `price_rules` も引き続き動作
- デフォルトセレクタは引き続き使用される（site_config に定義がない場合）
- `price` が `None` でも ProductInfo が返される（graceful degradation）

