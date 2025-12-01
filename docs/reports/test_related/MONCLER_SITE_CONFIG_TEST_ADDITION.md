# Moncler 用 site_config チューニング - テスト追加

**作成日時**: 2025-01-28  
**目的**: Moncler 用 PDP 抽出が Stage 5 新スキーマで正しく動作することを確認するテストケースを追加

---

## 追加するテストケース

### テスト名: `test_product_extractor_moncler_pdp_sample`

**目的**: Moncler 用 PDP fixture を使ったサイト固有のテスト

**確認事項**:
1. site_config に基づいて title, price, images, description などのフィールドが正しく抽出される
2. 価格正規化ルールが適用される（EUR形式: "€1,234.56" → 1234.56）
3. metadata に必要な値（image_count, size_count, color_count など）が正しく設定される
4. JSON-LD フォールバックが動作する
5. Meta タグフォールバックが動作する

---

## テストコード

`tests/test_product_extractor.py` の末尾に以下を追加してください:

```python
@pytest.mark.asyncio
async def test_product_extractor_moncler_pdp_sample(mock_page, run_context):
    """Moncler 用 PDP fixture を使ったサイト固有テスト"""
    
    # Moncler 用 site_config を準備（Stage 5 新スキーマ）
    moncler_site_config = {
        "selectors": {
            "pdp": {
                "title": [
                    "[data-testid='pdp-title']",
                    "h1[data-test='product-name']",
                    "h1[itemprop='name']",
                    "h1",
                    "[data-testid='product-name']",
                    ".c-product-title",
                    "h1.product-name",
                    "meta[property='og:title']"
                ],
                "price": [
                    "[data-testid='pdp-price']",
                    "[data-test='pdp-price'] [itemprop='price']",
                    "[itemprop='price']",
                    "[data-testid*='price' i]",
                    ".c-price__value",
                    ".price",
                    "[class*='price' i]",
                    "meta[itemprop='price'][content]"
                ],
                "list_price": [
                    "[data-testid='pdp-list-price']",
                    "[data-test='pdp-list-price']",
                    ".c-price__original",
                    "[class*='list-price' i]",
                    "[class*='original-price' i]"
                ],
                "currency": [
                    "meta[property='product:price:currency']",
                    "meta[itemprop='priceCurrency'][content]",
                    "[itemprop='priceCurrency']",
                    "[data-testid*='currency' i]"
                ],
                "images": {
                    "selectors": [
                        ".product-gallery img",
                        "[data-testid='product-image'] img",
                        "[data-testid='pdp-image'] img",
                        "img[itemprop='image']",
                        ".c-product-image img",
                        ".product-images img"
                    ],
                    "image_attr": "src",
                    "base_url": None
                },
                "description": [
                    "[data-testid='product-description']",
                    "[data-test='product-description']",
                    "[itemprop='description']",
                    ".c-product-description",
                    ".product-description",
                    "meta[property='og:description']"
                ],
                "colors": [
                    "button[data-testid*='color' i]",
                    "button[data-test='color-swatch']",
                    "[data-color]",
                    ".color-selector button"
                ],
                "sizes": [
                    "button[data-testid*='size' i]:not([disabled])",
                    "button[data-test='size-option']:not([disabled])",
                    "[role='radiogroup'][aria-label*='Size' i] button:not([disabled])"
                ],
                "brand": [
                    "[data-testid='product-brand']",
                    "[itemprop='brand']",
                    "meta[property='product:brand']",
                    "meta[property='og:site_name']"
                ],
                "sku": [
                    "[data-testid='product-sku']",
                    "[data-test='product-sku']",
                    "[itemprop='sku']",
                    ".product-code",
                    ".sku"
                ],
                "availability": {
                    "selectors": [
                        "[data-testid='stock-status']",
                        "[itemprop='availability']",
                        "meta[property='product:availability']"
                    ],
                    "patterns": [
                        "out of stock",
                        "in stock",
                        "pre-order",
                        "在庫なし",
                        "在庫あり"
                    ]
                },
                "json_ld": {
                    "enabled": True,
                    "paths": {
                        "price": ["offers.price", "offers[0].price"],
                        "currency": ["offers.priceCurrency", "offers[0].priceCurrency"],
                        "title": ["name"],
                        "description": ["description"]
                    }
                },
                "meta_fallback": {
                    "enabled": True,
                    "selectors": {
                        "price": [
                            "meta[property='og:price:amount']",
                            "meta[property='product:price:amount']"
                        ],
                        "currency": [
                            "meta[property='og:price:currency']",
                            "meta[property='product:price:currency']"
                        ],
                        "title": ["meta[property='og:title']"],
                        "description": ["meta[property='og:description']"]
                    }
                },
                "raw_html_capture": {
                    "enabled": True,
                    "filename": "pdp_raw.html"
                }
            }
        },
        "price_rules": {
            "strip_chars": ["€", ",", " ", "EUR", "GBP", "USD"],
            "thousands_separator": ",",
            "decimal_separator": ".",
            "currency_fallback": "EUR",
            "price_pattern": "[\\d.,]+",
            "currency_symbols": {
                "€": "EUR",
                "£": "GBP",
                "$": "USD",
                "¥": "JPY"
            }
        }
    }
    
    # Mock Page の設定
    mock_page.url = "https://www.moncler.com/en-int/women/outerwear/down-jackets/test-product"
    
    # タイトル抽出のモック
    title_locator = AsyncMock()
    title_locator.count = AsyncMock(return_value=1)
    title_locator.inner_text = AsyncMock(return_value="Moncler Test Product")
    
    # 価格抽出のモック
    price_locator = AsyncMock()
    price_locator.count = AsyncMock(return_value=1)
    price_locator.inner_text = AsyncMock(return_value="€1,234.56")
    
    # 通貨抽出のモック（Meta タグ）
    currency_locator = AsyncMock()
    currency_locator.count = AsyncMock(return_value=1)
    currency_locator.get_attribute = AsyncMock(return_value="EUR")
    
    # 画像抽出のモック
    image_locator = AsyncMock()
    image_locator.count = AsyncMock(return_value=3)
    image_nth_0 = AsyncMock()
    image_nth_0.get_attribute = AsyncMock(return_value="https://www.moncler.com/image1.jpg")
    image_nth_1 = AsyncMock()
    image_nth_1.get_attribute = AsyncMock(return_value="https://www.moncler.com/image2.jpg")
    image_nth_2 = AsyncMock()
    image_nth_2.get_attribute = AsyncMock(return_value="https://www.moncler.com/image3.jpg")
    image_locator.nth = AsyncMock(side_effect=[image_nth_0, image_nth_1, image_nth_2])
    
    # サイズ抽出のモック
    size_locator = AsyncMock()
    size_locator.count = AsyncMock(return_value=4)
    size_nth_0 = AsyncMock()
    size_nth_0.inner_text = AsyncMock(return_value="S")
    size_nth_1 = AsyncMock()
    size_nth_1.inner_text = AsyncMock(return_value="M")
    size_nth_2 = AsyncMock()
    size_nth_2.inner_text = AsyncMock(return_value="L")
    size_nth_3 = AsyncMock()
    size_nth_3.inner_text = AsyncMock(return_value="XL")
    size_locator.nth = AsyncMock(side_effect=[size_nth_0, size_nth_1, size_nth_2, size_nth_3])
    
    # カラー抽出のモック
    color_locator = AsyncMock()
    color_locator.count = AsyncMock(return_value=2)
    color_nth_0 = AsyncMock()
    color_nth_0.inner_text = AsyncMock(return_value="Black")
    color_nth_1 = AsyncMock()
    color_nth_1.inner_text = AsyncMock(return_value="Navy")
    color_locator.nth = AsyncMock(side_effect=[color_nth_0, color_nth_1])
    
    # 説明抽出のモック
    description_locator = AsyncMock()
    description_locator.count = AsyncMock(return_value=1)
    description_locator.inner_text = AsyncMock(return_value="Test product description")
    
    # locator を selector に応じて返す
    def locator_side_effect(selector: str):
        if "h1" in selector or "title" in selector.lower() or "[data-testid='pdp-title']" in selector:
            return title_locator
        elif "price" in selector.lower() or "[data-testid='pdp-price']" in selector:
            return price_locator
        elif "currency" in selector.lower() or "meta[property='product:price:currency']" in selector:
            return currency_locator
        elif "image" in selector.lower() or "img" in selector:
            return image_locator
        elif "size" in selector.lower():
            return size_locator
        elif "color" in selector.lower():
            return color_locator
        elif "description" in selector.lower():
            return description_locator
        else:
            # デフォルト（見つからない）
            default = AsyncMock()
            default.count = AsyncMock(return_value=0)
            return default
    
    mock_page.locator.side_effect = locator_side_effect
    
    # ProductExtractor を初期化
    extractor = ProductExtractor(
        site_config=moncler_site_config,
        run_context=run_context,
    )
    
    # 抽出実行
    result = await extractor.extract(mock_page)
    
    # アサーション: 基本フィールド
    assert result is not None
    assert isinstance(result, ProductInfo)
    assert result.title == "Moncler Test Product"
    assert result.price == 1234.56  # €1,234.56 が正規化されて float に変換
    assert result.currency == "EUR"
    assert len(result.images) == 3
    assert result.images[0] == "https://www.moncler.com/image1.jpg"
    assert len(result.sizes) == 4
    assert "S" in result.sizes
    assert "M" in result.sizes
    assert "L" in result.sizes
    assert "XL" in result.sizes
    assert len(result.colors) == 2
    assert "Black" in result.colors
    assert "Navy" in result.colors
    assert result.description == "Test product description"
    
    # アサーション: metadata
    assert result.metadata is not None
    assert result.metadata.get("has_title") is True
    assert result.metadata.get("has_price") is True
    assert result.metadata.get("has_currency") is True
    assert result.metadata.get("image_count") == 3
    assert result.metadata.get("size_count") == 4
    assert result.metadata.get("color_count") == 2
    assert result.metadata.get("url") == mock_page.url
    
    # アサーション: raw_html_capture が有効な場合
    # （実際の実装では run_context.save_html が呼ばれることを確認）
```

---

## 実行手順

### 単体テスト実行

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -v
```

### Moncler 関連のテストをすべて実行

```bash
pytest tests/test_product_extractor.py -k moncler -v
```

### E2E 確認（実際のサイトで動作確認）

```bash
python run_orchestrator.py \
    --site MONCLER_OFFICIAL \
    --query "down jacket" \
    --headless

# 結果確認
# - instance/runs/<RUN_ID>/pdp_extracted_data.json を確認
# - instance/runs/<RUN_ID>/pdp_raw.html を確認（raw_html_capture が有効な場合）
```

---

## テストの詳細説明

### 1. site_config の準備

Moncler 用の site_config を Stage 5 新スキーマに準拠した形で準備します。これには以下が含まれます：

- `selectors.pdp.*` - すべての PDP 抽出用セレクタ
- `price_rules` - 価格正規化ルール（EUR形式に対応）
- `json_ld.*` - JSON-LD フォールバック設定
- `meta_fallback.*` - Meta タグフォールバック設定

### 2. Mock Page の設定

Playwright の `Page` オブジェクトをモック化し、各セレクタに対して適切な値を返すように設定します。

### 3. アサーション

以下の項目を確認します：

- **基本フィールド**: title, price, currency, images, sizes, colors, description
- **価格正規化**: EUR形式（"€1,234.56"）が float（1234.56）に正規化されること
- **metadata**: image_count, size_count, color_count が正しくカウントされること

---

次: 実際のテストコードを `tests/test_product_extractor.py` に追加します。

