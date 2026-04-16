# Stage 5 Task A: ハードコード箇所の洗い出し

## 対象ファイル分析結果

### 1. app/agents/browser/product_extractor.py

#### 1.1 タイトル抽出のハードコード

**箇所**: `_extract_title()` メソッド（196-222行目）

- **デフォルトセレクタ**（32-46行目）:
  ```python
  DEFAULT_TITLE_SELECTORS = [
      "h1[itemprop='name']",
      "h1.product-title",
      "h1",
      "meta[property='og:title']",
  ]
  ```
- **処理ロジック**: `selector.startswith("meta[")` で meta タグかどうかを判定し、`get_attribute("content")` または `inner_text()` を選択

#### 1.2 価格抽出のハードコード

**箇所**: `_extract_price()` メソッド（248-277行目）

- **デフォルトセレクタ**（32-39行目）:
  ```python
  DEFAULT_PRICE_SELECTORS = [
      "meta[property='product:price:amount']",
      "meta[itemprop='price']",
      "[itemprop=price]",
      "span:has(meta[itemprop='price'])",
      "[data-testid*=price]",
      "[class*=price]",
  ]
  ```
- **価格正規化ロジック**（279-323行目）:
  - `_normalize_price()`: `strip_chars` を削除し、`re.search(r"[\d.,]+", normalized)` で数値部分を抽出
  - `_normalize_price_to_float()`: `thousands_separator` と `decimal_separator` を処理して float に変換
  - **ハードコードされた正規表現**: `r"[\d.,]+"` で数値パターンを抽出

#### 1.3 通貨抽出のハードコード

**箇所**: `_extract_currency()` メソッド（325-351行目）

- **デフォルトセレクタ**（331-335行目）:
  ```python
  currency_selectors = [
      "meta[property='product:price:currency']",
      "meta[itemprop='priceCurrency']",
  ]
  ```
- **処理ロジック**: `get_attribute("content")` のみを使用（meta タグ専用）

#### 1.4 画像抽出のハードコード

**箇所**: `_extract_images()` メソッド（353-388行目）

- **デフォルトセレクタ**（359-364行目）:
  ```python
  image_selectors = [
      ".product-images img",
      ".product-gallery img",
      "[data-testid*='image'] img",
      "img[itemprop='image']",
  ]
  ```
- **URL 正規化ロジック**（376-381行目）:
  ```python
  if src.startswith("//"):
      src = f"https:{src}"
  elif src.startswith("/"):
      src = f"{page.url.split('/')[0]}//{page.url.split('/')[2]}{src}"
  ```
  - **ハードコード**: `https:` プロトコルを強制、URL パースロジックが簡易的

#### 1.5 サイズ抽出のハードコード

**箇所**: `_extract_sizes()` メソッド（390-419行目）

- **デフォルトセレクタ**（396-400行目）:
  ```python
  size_selectors = [
      ".size-selector option",
      "button[data-size]",
      "[role='radiogroup'] [role='radio']",
  ]
  ```
- **処理ロジック**: `inner_text()` のみを使用

#### 1.6 カラー抽出のハードコード

**箇所**: `_extract_colors()` メソッド（421-450行目）

- **デフォルトセレクタ**（427-431行目）:
  ```python
  color_selectors = [
      ".color-selector .swatch",
      "button[data-color]",
      "[data-testid*='color']",
  ]
  ```
- **処理ロジック**: `get_attribute("aria-label")` または `inner_text()` を使用

#### 1.7 説明抽出のハードコード

**箇所**: `_extract_description()` メソッド（452-482行目）

- **デフォルトセレクタ**（458-462行目）:
  ```python
  description_selectors = [
      ".product-description",
      "[itemprop='description']",
      "meta[property='og:description']",
  ]
  ```

#### 1.8 ブランド抽出のハードコード

**箇所**: `_extract_brand()` メソッド（484-514行目）

- **デフォルトセレクタ**（490-494行目）:
  ```python
  brand_selectors = [
      "meta[property='og:site_name']",
      "[itemprop='brand']",
      ".product-brand",
  ]
  ```

#### 1.9 定価・割引率抽出のハードコード

**箇所**: `_extract_list_price_and_discount()` メソッド（516-551行目）

- **セレクタ**: `pdp_selectors.get("list_price", [])` から取得（デフォルトなし）
- **割引率計算ロジック**（546-549行目）:
  ```python
  if list_price > current_price:
      discount_pct = ((list_price - current_price) / list_price) * 100
  ```
  - **ハードコード**: 割引率の計算式が固定

#### 1.10 サイズ選択ロジックのハードコード

**箇所**: `_click_size_to_reveal_price()` メソッド（553-636行目）

- **デフォルトセレクタ**（48-54行目）:
  ```python
  DEFAULT_SIZE_BUTTON_SELECTORS = [
      "button[aria-disabled='false'][data-size]",
      "button[aria-disabled='false'][data-testid*='size']",
      "button:not([disabled])[data-size]",
      "li[data-size] button:not([disabled])",
      "button[class*='size']:not([disabled])",
      "[role='radiogroup'] [role='radio'][aria-checked='false']",
      "[aria-pressed='false'][class*='size']",
      "[aria-selected='false'][class*='size']",
  ]
  ```
- **在庫チェックロジック**（611-612行目）:
  ```python
  if "out of stock" in text or "在庫なし" in text:
      continue
  ```
  - **ハードコード**: 英語と日本語の在庫なしテキストが固定

#### 1.11 JSON-LD / Meta タグフォールバックのハードコード

**箇所**: `_extract_from_json_ld_or_meta()` メソッド（638-681行目）

- **JSON-LD パス**（650-665行目）:
  ```python
  offers = data.get("offers")
  price = offers.get("price")
  currency = offers.get("priceCurrency")
  ```
  - **ハードコード**: JSON-LD の構造パスが固定（`offers.price`, `offers.priceCurrency`）
- **Meta タグセレクタ**（670行目）:
  ```python
  for selector in ("meta[property='og:price:amount']", "meta[name='twitter:data1']"):
  ```
  - **ハードコード**: 特定の meta タグセレクタが固定

#### 1.12 価格正規化ルールのハードコード

**箇所**: `PriceRules` クラス（76-81行目）、`__init__()` メソッド（99-105行目）

- **デフォルト値**（102-104行目）:
  ```python
  strip_chars=price_rules_cfg.get("strip_chars", ["¥", ",", " "]),
  decimal_separator=price_rules_cfg.get("decimal_separator", "."),
  thousands_separator=price_rules_cfg.get("thousands_separator", ","),
  ```
  - **ハードコード**: デフォルトの `strip_chars` が `["¥", ",", " "]` に固定（日本円向け）

### 2. app/agents/browser/extractor.py

#### 2.1 グローバル定数のハードコード

**箇所**: ファイル先頭（29-48行目）

- **PRICE_SELECTORS**（29-36行目）: ProductExtractor と重複
- **SIZE_BUTTON_SELECTORS**（39-48行目）: ProductExtractor と重複

#### 2.2 Moncler 専用抽出のハードコード

**箇所**: `_extract_from_pdp()` メソッド（260-264行目）

```python
if site.upper() == "MONCLER_OFFICIAL":
    enriched = await self.moncler_extractor.extract(page=page, context=context)
    if enriched:
        return enriched
```
- **ハードコード**: サイト名による分岐が固定

### 3. app/extractors/moncler_extractor.py（存在する場合）

- Moncler 専用の抽出ロジックが別ファイルに存在する可能性
- 詳細はファイル内容を確認する必要がある

### 4. 価格正規化の正規表現パターン

**箇所**: `product_extractor.py` の `_normalize_price()` メソッド（290行目）

```python
match = re.search(r"[\d.,]+", normalized)
```
- **ハードコード**: 数値パターンが `[\d.,]+` に固定（カンマ、ピリオド、数字のみ）

### 5. 通貨記号・区切り文字のハードコード

- **円記号**: `"¥"` がデフォルトの `strip_chars` に含まれる
- **カンマ**: `","` が thousands_separator のデフォルト
- **ピリオド**: `"."` が decimal_separator のデフォルト
- **スペース**: `" "` がデフォルトの `strip_chars` に含まれる

### 6. 画像 URL 正規化のハードコード

**箇所**: `_extract_images()` メソッド（376-381行目）

- **プロトコル強制**: `//` で始まる URL に `https:` を強制付与
- **相対 URL 処理**: `/` で始まる URL の処理ロジックが簡易的（`page.url.split('/')` を使用）

### 7. メタデータ収集のハードコード

**箇所**: `extract()` メソッド（174-183行目）

```python
product_info.metadata = {
    "extraction_timestamp": time.time(),
    "url": product_info.url,
    "has_title": product_info.title is not None,
    "has_price": product_info.price is not None,
    "has_currency": product_info.currency is not None,
    "image_count": len(product_info.images),
    "size_count": len(product_info.sizes),
    "color_count": len(product_info.colors),
}
```
- **ハードコード**: メタデータのキー名と計算ロジックが固定

### 8. HTML 保存パスのハードコード

**箇所**: `extract()` メソッド（189行目）

```python
self.run_context.save_content("pdp_raw.html", html_content)
product_info.raw_html_path = str(self.run_context.get_path("pdp_raw.html"))
```
- **ハードコード**: ファイル名 `"pdp_raw.html"` が固定

## まとめ

### ハードコード箇所の分類

1. **デフォルトセレクタ**: 各抽出メソッドにデフォルトセレクタが定義されている
2. **価格正規化ロジック**: 正規表現パターン、区切り文字処理が固定
3. **URL 正規化ロジック**: 画像 URL の正規化ロジックが簡易的
4. **在庫チェックテキスト**: 英語・日本語の在庫なしテキストが固定
5. **JSON-LD パス**: JSON-LD の構造パスが固定
6. **メタデータキー**: メタデータのキー名が固定
7. **HTML 保存パス**: ファイル名が固定
8. **サイト固有分岐**: Moncler 専用の分岐が存在

### 優先度

- **高**: 価格正規化ロジック、デフォルトセレクタ、サイト固有分岐
- **中**: URL 正規化ロジック、JSON-LD パス、在庫チェックテキスト
- **低**: メタデータキー、HTML 保存パス（設定可能にする価値はあるが影響は小さい）

