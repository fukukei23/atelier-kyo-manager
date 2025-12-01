# Stage 5 Task D: テスト戦略と具体的テストケース案

## 1. 必須のテスト観点

### 1.1 Full extraction（すべてのフィールドを抽出）

**テスト名**: `test_product_extractor_full_extraction`

**目的**: title, price, images, colors, sizes, description, brand, sku などすべてのフィールドが正しく抽出されることを確認

**テストケース**:
- site_config にすべてのセレクタが定義されている
- モック Page で各セレクタに対応する要素が存在する
- ProductInfo のすべてのフィールドが None でない / 空でない

### 1.2 Partial selectors（部分的なセレクタのみ）

**テスト名**: `test_product_extractor_partial_selectors`

**目的**: site_config に title と price しかない場合でも例外を出さずに ProductInfo を返すことを確認

**テストケース**:
- site_config に `title` と `price` のみが定義されている
- 他のフィールド（images, colors, sizes 等）は None または空リスト
- 例外は発生しない

### 1.3 Price normalization（価格正規化）

**テスト名**: `test_product_extractor_price_normalization_various_formats`

**目的**: 様々な価格表記（¥12,345 / $1,234.56 / 1,234円 / € 99,99 など）が正しく float に変換されることを確認

**テストケース**:
- 日本円: `"¥12,345"` → `12345.0`
- 米ドル: `"$1,234.56"` → `1234.56`
- 円記号なし: `"1,234円"` → `1234.0`
- ユーロ（カンマ区切り）: `"€ 99,99"` → `99.99`（decimal_separator が "," の場合）
- カンマ区切り（千の位）: `"12,345.67"` → `12345.67`

### 1.4 site_config が一部欠損している場合

**テスト名**: `test_product_extractor_missing_config_graceful_degradation`

**目的**: site_config に `selectors.pdp` が無い、または一部のキーが欠損していても例外を出さずに動作することを確認

**テストケース**:
- `selectors.pdp` が存在しない → デフォルトセレクタを使用
- `selectors.pdp.title` が存在しない → デフォルトタイトルセレクタを使用
- `price_rules` が存在しない → デフォルト価格正規化ルールを使用

### 1.5 metadata のカウント

**テスト名**: `test_product_extractor_metadata_counts`

**目的**: metadata の has_title, has_price, image_count, size_count, color_count が正しくカウントされることを確認

**テストケース**:
- title が抽出された → `has_title: true`
- price が抽出された → `has_price: true`
- 画像が3枚抽出された → `image_count: 3`
- サイズが5つ抽出された → `size_count: 5`
- カラーが2つ抽出された → `color_count: 2`

## 2. 代表的なテスト関数のコード例

### 2.1 Full extraction テスト

```python
@pytest.mark.asyncio
async def test_product_extractor_full_extraction(mock_page, site_config, run_context):
    """
    Stage 5: Full extraction（すべてのフィールドを抽出）
    
    Given:
    - site_config にすべてのセレクタが定義されている
    - モック Page で各セレクタに対応する要素が存在する
    
    When:
    - ProductExtractor.extract() が呼ばれる
    
    Then:
    - ProductInfo のすべてのフィールドが None でない / 空でない
    - metadata が正しく設定されている
    """
    # 完全な site_config
    full_site_config = {
        "selectors": {
            "pdp": {
                "title": ["h1.product-title"],
                "price": [".product-price"],
                "list_price": [".price-original"],
                "currency": ["meta[property='product:price:currency']"],
                "images": [".product-images img"],
                "sizes": [".size-selector option"],
                "colors": [".color-selector .swatch"],
                "description": [".product-description"],
                "brand": ["meta[property='og:site_name']"],
                "sku": [".product-sku"],
            },
        },
        "price_rules": {
            "strip_chars": ["¥", ",", " "],
            "thousands_separator": ",",
            "decimal_separator": ".",
        },
    }
    
    extractor = ProductExtractor(
        site_config=full_site_config,
        run_context=run_context,
    )
    
    # 各フィールドのモック設定
    title_locator = AsyncMock()
    title_locator.count = AsyncMock(return_value=1)
    title_locator.inner_text = AsyncMock(return_value="Test Product Title")
    
    price_locator = AsyncMock()
    price_locator.count = AsyncMock(return_value=1)
    price_locator.inner_text = AsyncMock(return_value="¥ 12,345")
    
    list_price_locator = AsyncMock()
    list_price_locator.count = AsyncMock(return_value=1)
    list_price_locator.inner_text = AsyncMock(return_value="¥ 15,000")
    
    currency_locator = AsyncMock()
    currency_locator.count = AsyncMock(return_value=1)
    currency_locator.get_attribute = AsyncMock(return_value="JPY")
    
    image_locator = AsyncMock()
    image_locator.count = AsyncMock(return_value=3)
    image_locator.nth = AsyncMock(return_value=image_locator)
    image_locator.get_attribute = AsyncMock(side_effect=[
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
        "https://example.com/image3.jpg",
    ])
    
    size_locator = AsyncMock()
    size_locator.count = AsyncMock(return_value=5)
    size_locator.nth = AsyncMock(return_value=size_locator)
    size_locator.inner_text = AsyncMock(side_effect=["S", "M", "L", "XL", "XXL"])
    
    color_locator = AsyncMock()
    color_locator.count = AsyncMock(return_value=2)
    color_locator.nth = AsyncMock(return_value=color_locator)
    color_locator.get_attribute = AsyncMock(side_effect=["Red", "Blue"])
    
    description_locator = AsyncMock()
    description_locator.count = AsyncMock(return_value=1)
    description_locator.inner_text = AsyncMock(return_value="Product description text")
    
    brand_locator = AsyncMock()
    brand_locator.count = AsyncMock(return_value=1)
    brand_locator.get_attribute = AsyncMock(return_value="Test Brand")
    
    sku_locator = AsyncMock()
    sku_locator.count = AsyncMock(return_value=1)
    sku_locator.inner_text = AsyncMock(return_value="SKU-12345")
    
    def locator_side_effect(selector):
        selector_lower = str(selector).lower()
        if "title" in selector_lower or "h1" in selector_lower:
            return title_locator
        elif "price" in selector_lower and "original" not in selector_lower:
            return price_locator
        elif "original" in selector_lower or "list_price" in selector_lower:
            return list_price_locator
        elif "currency" in selector_lower:
            return currency_locator
        elif "images" in selector_lower or "img" in selector_lower:
            return image_locator
        elif "size" in selector_lower:
            return size_locator
        elif "color" in selector_lower:
            return color_locator
        elif "description" in selector_lower:
            return description_locator
        elif "brand" in selector_lower or "site_name" in selector_lower:
            return brand_locator
        elif "sku" in selector_lower:
            return sku_locator
        else:
            default = AsyncMock()
            default.count = AsyncMock(return_value=0)
            return default
    
    mock_page.locator.side_effect = locator_side_effect
    mock_page.content = AsyncMock(return_value="<html><body>Test HTML</body></html>")
    
    product_info = await extractor.extract(mock_page)
    
    # アサーション: すべてのフィールドが設定されている
    assert product_info.title == "Test Product Title"
    assert product_info.price == 12345.0
    assert product_info.list_price == 15000.0
    assert product_info.currency == "JPY"
    assert len(product_info.images) == 3
    assert len(product_info.sizes) == 5
    assert len(product_info.colors) == 2
    assert product_info.description == "Product description text"
    assert product_info.brand == "Test Brand"
    assert product_info.raw_html_path is not None
    
    # metadata の確認
    assert product_info.metadata.get("has_title") is True
    assert product_info.metadata.get("has_price") is True
    assert product_info.metadata.get("image_count") == 3
    assert product_info.metadata.get("size_count") == 5
    assert product_info.metadata.get("color_count") == 2
```

### 2.2 Partial selectors テスト

```python
@pytest.mark.asyncio
async def test_product_extractor_partial_selectors(mock_page, run_context):
    """
    Stage 5: Partial selectors（部分的なセレクタのみ）
    
    Given:
    - site_config に title と price しか定義されていない
    
    When:
    - ProductExtractor.extract() が呼ばれる
    
    Then:
    - title と price のみが設定される
    - 他のフィールドは None または空リスト
    - 例外は発生しない
    """
    partial_site_config = {
        "selectors": {
            "pdp": {
                "title": ["h1.product-title"],
                "price": [".product-price"],
            },
        },
        "price_rules": {
            "strip_chars": ["¥", ",", " "],
        },
    }
    
    extractor = ProductExtractor(
        site_config=partial_site_config,
        run_context=run_context,
    )
    
    title_locator = AsyncMock()
    title_locator.count = AsyncMock(return_value=1)
    title_locator.inner_text = AsyncMock(return_value="Test Product")
    
    price_locator = AsyncMock()
    price_locator.count = AsyncMock(return_value=1)
    price_locator.inner_text = AsyncMock(return_value="¥ 5,000")
    
    def locator_side_effect(selector):
        selector_lower = str(selector).lower()
        if "title" in selector_lower or "h1" in selector_lower:
            return title_locator
        elif "price" in selector_lower:
            return price_locator
        else:
            default = AsyncMock()
            default.count = AsyncMock(return_value=0)
            return default
    
    mock_page.locator.side_effect = locator_side_effect
    mock_page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    
    product_info = await extractor.extract(mock_page)
    
    # アサーション: title と price のみが設定されている
    assert product_info.title == "Test Product"
    assert product_info.price == 5000.0
    
    # 他のフィールドは None または空リスト
    assert product_info.currency is None
    assert product_info.images == []
    assert product_info.sizes == []
    assert product_info.colors == []
    assert product_info.description is None
    assert product_info.brand is None
    
    # 例外は発生しない（既に extract() が正常に完了している）
```

### 2.3 Price normalization テスト

```python
@pytest.mark.asyncio
async def test_product_extractor_price_normalization_various_formats(mock_page, run_context):
    """
    Stage 5: Price normalization（様々な価格表記の正規化）
    
    Given:
    - 様々な価格表記（¥12,345 / $1,234.56 / 1,234円 / € 99,99 など）
    - 対応する price_rules 設定
    
    When:
    - ProductExtractor._normalize_price_to_float() が呼ばれる
    
    Then:
    - 正しく float に変換される
    """
    # 日本円用の設定
    jpy_config = {
        "price_rules": {
            "strip_chars": ["¥", ",", " ", "円"],
            "thousands_separator": ",",
            "decimal_separator": ".",
        },
    }
    
    extractor_jpy = ProductExtractor(
        site_config=jpy_config,
        run_context=run_context,
    )
    
    price_rules_jpy = extractor_jpy._get_price_rules()
    
    test_cases_jpy = [
        ("¥ 12,345", 12345.0),
        ("¥123,456", 123456.0),
        ("1,234円", 1234.0),
        ("12,345.67", 12345.67),
    ]
    
    for input_price, expected_float in test_cases_jpy:
        normalized = extractor_jpy._normalize_price_to_float(input_price, price_rules_jpy)
        assert normalized is not None
        assert isinstance(normalized, float)
        assert abs(normalized - expected_float) < 0.01
    
    # ユーロ用の設定（カンマが小数点）
    eur_config = {
        "price_rules": {
            "strip_chars": ["€", " ", "."],
            "thousands_separator": ".",
            "decimal_separator": ",",
        },
    }
    
    extractor_eur = ProductExtractor(
        site_config=eur_config,
        run_context=run_context,
    )
    
    price_rules_eur = extractor_eur._get_price_rules()
    
    test_cases_eur = [
        ("€ 99,99", 99.99),
        ("1.234,56", 1234.56),
    ]
    
    for input_price, expected_float in test_cases_eur:
        normalized = extractor_eur._normalize_price_to_float(input_price, price_rules_eur)
        assert normalized is not None
        assert isinstance(normalized, float)
        assert abs(normalized - expected_float) < 0.01
```

### 2.4 Missing config graceful degradation テスト

```python
@pytest.mark.asyncio
async def test_product_extractor_missing_config_graceful_degradation(mock_page, run_context):
    """
    Stage 5: site_config が一部欠損している場合でも例外を出さずに動作する
    
    Given:
    - site_config に selectors.pdp が存在しない
    - または一部のキーが欠損している
    
    When:
    - ProductExtractor.extract() が呼ばれる
    
    Then:
    - デフォルトセレクタが使用される
    - 例外は発生しない
    """
    # selectors.pdp が存在しない site_config
    minimal_site_config = {
        "price_rules": {
            "strip_chars": ["¥", ",", " "],
        },
    }
    
    extractor = ProductExtractor(
        site_config=minimal_site_config,
        run_context=run_context,
    )
    
    # デフォルトセレクタで見つかる要素をモック
    title_locator = AsyncMock()
    title_locator.count = AsyncMock(return_value=1)
    title_locator.inner_text = AsyncMock(return_value="Default Title")
    
    price_locator = AsyncMock()
    price_locator.count = AsyncMock(return_value=1)
    price_locator.inner_text = AsyncMock(return_value="¥ 10,000")
    
    def locator_side_effect(selector):
        selector_lower = str(selector).lower()
        if "h1" in selector_lower or "title" in selector_lower:
            return title_locator
        elif "price" in selector_lower:
            return price_locator
        else:
            default = AsyncMock()
            default.count = AsyncMock(return_value=0)
            return default
    
    mock_page.locator.side_effect = locator_side_effect
    mock_page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    
    # 例外が発生しないことを確認
    product_info = await extractor.extract(mock_page)
    
    # デフォルトセレクタで抽出できていることを確認
    assert product_info.title == "Default Title"
    assert product_info.price == 10000.0
```

### 2.5 Metadata counts テスト

```python
@pytest.mark.asyncio
async def test_product_extractor_metadata_counts(mock_page, site_config, run_context):
    """
    Stage 5: metadata の has_title, has_price, image_count, size_count, color_count が正しくカウントされる
    
    Given:
    - 各フィールドが特定の数だけ抽出される
    
    When:
    - ProductExtractor.extract() が呼ばれる
    
    Then:
    - metadata の各カウントが正しい
    """
    extractor = ProductExtractor(
        site_config=site_config,
        run_context=run_context,
    )
    
    # 3枚の画像、5つのサイズ、2つのカラーをモック
    image_locator = AsyncMock()
    image_locator.count = AsyncMock(return_value=3)
    image_locator.nth = AsyncMock(return_value=image_locator)
    image_locator.get_attribute = AsyncMock(side_effect=[
        "https://example.com/img1.jpg",
        "https://example.com/img2.jpg",
        "https://example.com/img3.jpg",
    ])
    
    size_locator = AsyncMock()
    size_locator.count = AsyncMock(return_value=5)
    size_locator.nth = AsyncMock(return_value=size_locator)
    size_locator.inner_text = AsyncMock(side_effect=["S", "M", "L", "XL", "XXL"])
    
    color_locator = AsyncMock()
    color_locator.count = AsyncMock(return_value=2)
    color_locator.nth = AsyncMock(return_value=color_locator)
    color_locator.get_attribute = AsyncMock(side_effect=["Red", "Blue"])
    
    title_locator = AsyncMock()
    title_locator.count = AsyncMock(return_value=1)
    title_locator.inner_text = AsyncMock(return_value="Test Product")
    
    price_locator = AsyncMock()
    price_locator.count = AsyncMock(return_value=1)
    price_locator.inner_text = AsyncMock(return_value="¥ 5,000")
    
    def locator_side_effect(selector):
        selector_lower = str(selector).lower()
        if "title" in selector_lower:
            return title_locator
        elif "price" in selector_lower:
            return price_locator
        elif "images" in selector_lower or "img" in selector_lower:
            return image_locator
        elif "size" in selector_lower:
            return size_locator
        elif "color" in selector_lower:
            return color_locator
        else:
            default = AsyncMock()
            default.count = AsyncMock(return_value=0)
            return default
    
    mock_page.locator.side_effect = locator_side_effect
    mock_page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    
    product_info = await extractor.extract(mock_page)
    
    # metadata の確認
    assert product_info.metadata.get("has_title") is True
    assert product_info.metadata.get("has_price") is True
    assert product_info.metadata.get("image_count") == 3
    assert product_info.metadata.get("size_count") == 5
    assert product_info.metadata.get("color_count") == 2
```

### 2.6 Config getter テスト（Stage 5 追加）

```python
def test_get_pdp_config_with_new_schema(mock_page, mock_context, run_context):
    """Stage 5: 新スキーマから PDP 設定を取得するテスト"""
    site_config = {
        "selectors": {
            "pdp": {
                "title": ["h1.custom-title"],
                "price": [".custom-price"],
                "images": [".custom-images img"],
            },
        },
    }
    
    extractor = ProductExtractor(
        site_config=site_config,
        run_context=run_context,
    )
    
    config = extractor._get_pdp_config()
    
    assert config.get("title") == ["h1.custom-title"]
    assert config.get("price") == [".custom-price"]
    assert config.get("images") == [".custom-images img"]


def test_get_pdp_config_fallback_to_defaults(mock_page, mock_context, run_context):
    """Stage 5: デフォルトセレクタにフォールバックするテスト"""
    site_config = {}  # 空の site_config
    
    extractor = ProductExtractor(
        site_config=site_config,
        run_context=run_context,
    )
    
    config = extractor._get_pdp_config()
    
    # デフォルトセレクタが使用されていることを確認
    assert len(config.get("title", [])) > 0
    assert len(config.get("price", [])) > 0
    assert len(config.get("images", [])) > 0


def test_get_price_rules_with_new_schema(mock_page, mock_context, run_context):
    """Stage 5: 新スキーマから価格正規化ルールを取得するテスト"""
    site_config = {
        "selectors": {
            "pdp": {
                "price": {
                    "selectors": [".price"],
                    "normalize_rules": {
                        "strip_chars": ["$", "€"],
                        "thousands_separator": ".",
                        "decimal_separator": ",",
                    },
                },
            },
        },
    }
    
    extractor = ProductExtractor(
        site_config=site_config,
        run_context=run_context,
    )
    
    price_rules = extractor._get_price_rules()
    
    assert price_rules.get("strip_chars") == ["$", "€"]
    assert price_rules.get("thousands_separator") == "."
    assert price_rules.get("decimal_separator") == ","


def test_get_price_rules_fallback_to_legacy(mock_page, mock_context, run_context):
    """Stage 5: 旧スキーマ（トップレベルの price_rules）からフォールバックするテスト"""
    site_config = {
        "price_rules": {
            "strip_chars": ["¥", "円"],
            "thousands_separator": ",",
            "decimal_separator": ".",
        },
    }
    
    extractor = ProductExtractor(
        site_config=site_config,
        run_context=run_context,
    )
    
    price_rules = extractor._get_price_rules()
    
    assert price_rules.get("strip_chars") == ["¥", "円"]
    assert price_rules.get("thousands_separator") == ","
    assert price_rules.get("decimal_separator") == "."
```

## 3. テストファイル構成

### tests/test_product_extractor.py の拡張

既存のテストに加えて、以下のテストを追加：

1. `test_product_extractor_full_extraction` - Full extraction
2. `test_product_extractor_partial_selectors` - Partial selectors
3. `test_product_extractor_price_normalization_various_formats` - Price normalization
4. `test_product_extractor_missing_config_graceful_degradation` - Missing config
5. `test_product_extractor_metadata_counts` - Metadata counts
6. `test_get_pdp_config_with_new_schema` - Config getter (新スキーマ)
7. `test_get_pdp_config_fallback_to_defaults` - Config getter (デフォルト)
8. `test_get_price_rules_with_new_schema` - Price rules (新スキーマ)
9. `test_get_price_rules_fallback_to_legacy` - Price rules (旧スキーマ)

## 4. テスト実行の優先順位

1. **高優先度**: Full extraction, Partial selectors, Price normalization
2. **中優先度**: Missing config graceful degradation, Metadata counts
3. **低優先度**: Config getter テスト（既存のテストでカバーされている可能性がある）

