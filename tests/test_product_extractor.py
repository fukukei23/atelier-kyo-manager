# -*- coding: utf-8 -*-
"""
Task 2-4: ProductExtractor のユニットテスト（PDP HTML fixture）

ProductExtractor の主要な機能をテストする：
- (a) Full PDP extraction
- (b) Partial selectors / missing elements
- (c) Price normalization
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Dict, Any

from app.agents.browser.product_extractor import ProductExtractor, ProductInfo
from app.core.run_context import RunContext


@pytest.fixture
def mock_page():
    """モック Page オブジェクト"""
    page = AsyncMock()
    page.url = "https://example.com/product/123"
    page.is_closed.return_value = False
    page.content = AsyncMock(return_value="<html><body>Test</body></html>")
    
    # デフォルトの locator モック
    default_locator = AsyncMock()
    default_locator.count = AsyncMock(return_value=0)
    default_locator.first = default_locator
    default_locator.nth = default_locator
    default_locator.inner_text = AsyncMock(return_value="")
    default_locator.get_attribute = AsyncMock(return_value=None)
    
    page.locator.return_value = default_locator
    return page


@pytest.fixture
def site_config():
    """テスト用 site_config"""
    return {
        "selectors": {
            "pdp": {
                "title": ["h1.product-title"],
                "price": [".product-price"],
                "currency": ["meta[property='product:price:currency']"],
                "images": [".product-images img"],
                "size": [".size-selector option"],
                "color": [".color-selector .swatch"],
                "description": [".product-description"],
                "brand": ["meta[property='og:site_name']"],
            },
        },
        "price_rules": {
            "strip_chars": ["¥", ",", " "],
            "decimal_separator": ".",
            "thousands_separator": ",",
        },
    }


@pytest.fixture
def run_context():
    """テスト用 RunContext"""
    return RunContext()


@pytest.mark.asyncio
async def test_product_extractor_title(mock_page, site_config, run_context):
    """タイトル抽出をテスト"""
    extractor = ProductExtractor(
        site_config=site_config,
        run_context=run_context,
    )
    
    # タイトルが見つかる場合
    title_locator = AsyncMock()
    title_locator.count = AsyncMock(return_value=1)
    title_locator.inner_text = AsyncMock(return_value="Test Product Title")
    mock_page.locator.return_value = title_locator
    
    # Stage 5: pdp_config を取得
    pdp_config = extractor._get_pdp_config()
    
    title = await extractor._extract_title(mock_page, pdp_config)
    assert title == "Test Product Title"


@pytest.mark.asyncio
async def test_product_extractor_price(mock_page, site_config, run_context):
    """価格抽出をテスト"""
    extractor = ProductExtractor(
        site_config=site_config,
        run_context=run_context,
    )
    
    # 価格が見つかる場合
    price_locator = AsyncMock()
    price_locator.count = AsyncMock(return_value=1)
    price_locator.inner_text = AsyncMock(return_value="¥ 12,345")
    mock_page.locator.return_value = price_locator
    
    # Stage 5: pdp_config と price_rules を取得
    pdp_config = extractor._get_pdp_config()
    price_rules = extractor._get_price_rules()
    
    price = await extractor._extract_price(mock_page, pdp_config, price_rules)
    assert price is not None
    # Stage 5: float に変換されている
    assert isinstance(price, float)
    assert price == 12345.0


@pytest.mark.asyncio
async def test_product_extractor_currency(mock_page, site_config, run_context):
    """通貨抽出をテスト"""
    extractor = ProductExtractor(
        site_config=site_config,
        run_context=run_context,
    )
    
    # 通貨が見つかる場合
    currency_locator = AsyncMock()
    currency_locator.count = AsyncMock(return_value=1)
    currency_locator.get_attribute = AsyncMock(return_value="JPY")
    mock_page.locator.return_value = currency_locator
    
    # Stage 5: pdp_config を取得
    pdp_config = extractor._get_pdp_config()
    
    currency = await extractor._extract_currency(mock_page, pdp_config)
    assert currency == "JPY"


@pytest.mark.asyncio
async def test_product_extractor_images(mock_page, site_config, run_context):
    """画像抽出をテスト"""
    extractor = ProductExtractor(
        site_config=site_config,
        run_context=run_context,
    )
    
    # 画像が見つかる場合
    image_locator = AsyncMock()
    image_locator.count = AsyncMock(return_value=2)
    image_locator.nth = AsyncMock(return_value=image_locator)
    image_locator.get_attribute = AsyncMock(side_effect=["https://example.com/image1.jpg", "https://example.com/image2.jpg"])
    mock_page.locator.return_value = image_locator
    
    # Stage 5: pdp_config を取得
    pdp_config = extractor._get_pdp_config()
    
    images = await extractor._extract_images(mock_page, pdp_config)
    assert len(images) == 2
    assert "image1.jpg" in images[0]
    assert "image2.jpg" in images[1]


@pytest.mark.asyncio
async def test_product_extractor_extract_full(mock_page, site_config, run_context):
    """
    (a) Full PDP extraction
    
    Given:
    - A PDP HTML with all fields present
    - site_config["selectors"]["pdp"] specifying selectors for all fields
    
    When:
    - ProductExtractor.extract is called
    
    Then:
    - ProductInfo fields are populated correctly
    - raw_html_path is non-empty (snapshot was saved)
    """
    extractor = ProductExtractor(
        site_config=site_config,
        run_context=run_context,
    )
    
    # 各フィールドのモック設定
    title_locator = AsyncMock()
    title_locator.count = AsyncMock(return_value=1)
    title_locator.inner_text = AsyncMock(return_value="Test Product")
    
    price_locator = AsyncMock()
    price_locator.count = AsyncMock(return_value=1)
    price_locator.inner_text = AsyncMock(return_value="¥ 12,345")
    
    currency_locator = AsyncMock()
    currency_locator.count = AsyncMock(return_value=1)
    currency_locator.get_attribute = AsyncMock(return_value="JPY")
    
    image_locator = AsyncMock()
    image_locator.count = AsyncMock(return_value=2)
    image_locator.nth = AsyncMock(return_value=image_locator)
    image_locator.get_attribute = AsyncMock(side_effect=["https://example.com/image1.jpg", "https://example.com/image2.jpg"])
    
    description_locator = AsyncMock()
    description_locator.count = AsyncMock(return_value=1)
    description_locator.inner_text = AsyncMock(return_value="Product description")
    
    # locator の呼び出しに応じて異なる locator を返す
    def locator_side_effect(selector):
        if "title" in selector or "h1" in selector:
            return title_locator
        elif "price" in selector:
            return price_locator
        elif "currency" in selector:
            return currency_locator
        elif "images" in selector or "img" in selector:
            return image_locator
        elif "description" in selector:
            return description_locator
        else:
            default = AsyncMock()
            default.count = AsyncMock(return_value=0)
            return default
    
    mock_page.locator.side_effect = locator_side_effect
    mock_page.content = AsyncMock(return_value="<html><body>Test HTML</body></html>")
    
    product_info = await extractor.extract(mock_page)
    
    # アサーション
    assert isinstance(product_info, ProductInfo)
    assert product_info.title == "Test Product"
    assert product_info.price == 12345.0  # Task 2: float に変換されている
    assert product_info.currency == "JPY"
    assert len(product_info.images) == 2
    assert product_info.description == "Product description"
    assert product_info.url == "https://example.com/product/123"
    assert product_info.raw_html_path is not None  # HTML が保存された
    assert len(product_info.raw_html_path) > 0
    assert "metadata" in product_info.__dict__  # metadata フィールドが存在
    assert product_info.metadata.get("has_title") is True
    assert product_info.metadata.get("has_price") is True


@pytest.mark.asyncio
async def test_product_extractor_partial_selectors(mock_page, site_config, run_context):
    """
    (b) Partial selectors / missing elements
    
    Given:
    - PDP HTML missing some optional elements
    - site_config that only defines title and price
    
    Then:
    - ProductExtractor returns ProductInfo with title / price set, others None or empty list
    - No exception is raised
    """
    # 部分的な site_config（title と price のみ）
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
    
    # タイトルと価格のみが見つかる
    title_locator = AsyncMock()
    title_locator.count = AsyncMock(return_value=1)
    title_locator.inner_text = AsyncMock(return_value="Test Product")
    
    price_locator = AsyncMock()
    price_locator.count = AsyncMock(return_value=1)
    price_locator.inner_text = AsyncMock(return_value="¥ 5,000")
    
    def locator_side_effect(selector):
        if "title" in selector or "h1" in selector:
            return title_locator
        elif "price" in selector:
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
    # 例外は発生しない


@pytest.mark.asyncio
async def test_product_extractor_normalize_price(site_config, run_context):
    """
    (c) Price normalization
    
    Given:
    - Raw price text like "¥123,456" and price_rules specifying strip_chars and separators
    
    Then:
    - ProductInfo.price == 123456.0 (float)
    - On failure to parse, price should be None and a warning logged
    """
    extractor = ProductExtractor(
        site_config=site_config,
        run_context=run_context,
    )
    
    # 様々な価格表記をテスト（float に変換されることを確認）
    test_cases = [
        ("¥ 12,345", 12345.0),
        ("¥123,456", 123456.0),
        ("$ 1,234.56", 1234.56),
        ("1,234円", 1234.0),
        ("€ 99.99", 99.99),
        ("12,345.67", 12345.67),
    ]
    
    # Stage 5: price_rules を取得
    price_rules = extractor._get_price_rules()
    
    for input_price, expected_float in test_cases:
        normalized = extractor._normalize_price_to_float(input_price, price_rules)
        assert normalized is not None
        assert isinstance(normalized, float)
        assert abs(normalized - expected_float) < 0.01  # 浮動小数点の誤差を考慮
    
    # パース失敗のケース
    invalid_cases = [
        "",
        "invalid",
        "no numbers",
    ]
    
    for invalid_price in invalid_cases:
        normalized = extractor._normalize_price_to_float(invalid_price, price_rules)
        assert normalized is None  # パース失敗時は None


@pytest.mark.asyncio
async def test_product_extractor_json_ld_fallback(mock_page, site_config, run_context):
    """JSON-LD フォールバック抽出をテスト"""
    extractor = ProductExtractor(
        site_config=site_config,
        run_context=run_context,
    )
    
    # JSON-LD スクリプトをモック
    script = AsyncMock()
    script.inner_text = AsyncMock(return_value='{"offers": {"price": "12345", "priceCurrency": "JPY"}}')
    mock_page.query_selector_all = AsyncMock(return_value=[script])
    
    # Stage 5: pdp_config を取得
    pdp_config = extractor._get_pdp_config()
    
    fallback_data = await extractor._extract_from_json_ld_or_meta(mock_page, pdp_config)
    assert fallback_data is not None
    assert fallback_data["price"] == "12345"
    assert fallback_data["currency"] == "JPY"


# === Stage 5: 新しいテストケース ===

@pytest.mark.asyncio
async def test_product_extractor_full_extraction(mock_page, run_context):
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
    sku_locator.count = AsyncMock(return_value=0)  # sku はオプションなので見つからなくてもOK
    
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


@pytest.mark.asyncio
async def test_product_extractor_price_normalization_various_formats(run_context):
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
    
    # 他のフィールドは None または空リスト
    assert product_info.currency is None
    assert product_info.images == []
    assert product_info.sizes == []
    assert product_info.colors == []
    
    # metadata の確認
    assert product_info.metadata.get("has_title") is True
    assert product_info.metadata.get("has_price") is True
    assert product_info.metadata.get("image_count") == 0
    assert product_info.metadata.get("size_count") == 0
    assert product_info.metadata.get("color_count") == 0


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


def test_get_pdp_config_with_new_schema(run_context):
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


def test_get_pdp_config_fallback_to_defaults(run_context):
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


def test_get_price_rules_with_new_schema(run_context):
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


def test_get_price_rules_fallback_to_legacy(run_context):
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
                    "[class*='list-price' i]"
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
    # _extract_title は page.locator(selector).first を使用するため、2段階の Locator が必要
    title_locator_first = AsyncMock()
    title_locator_first.count = AsyncMock(return_value=1)
    title_locator_first.inner_text = AsyncMock(return_value="Moncler Test Product")
    title_locator_base = AsyncMock()
    title_locator_base.first = title_locator_first
    title_locator = title_locator_base
    
    # 価格抽出のモック
    # _extract_price は page.locator(selector).first を使用するため、2段階の Locator が必要
    price_locator_first = AsyncMock()
    price_locator_first.count = AsyncMock(return_value=1)
    price_locator_first.inner_text = AsyncMock(return_value="€1,234.56")
    price_locator_base = AsyncMock()
    price_locator_base.first = price_locator_first
    price_locator = price_locator_base
    
    # 通貨抽出のモック（Meta タグ）
    # _extract_currency は page.locator(selector).first を使用するため、2段階の Locator が必要
    # test_product_extractor_currency では .first を使わないため、直接 currency_locator を返している
    # しかし、実際のコードでは .first を使うため、2段階の構造が必要
    # 問題: AsyncMock の .first プロパティが正しく動作していない可能性があるため、
    # MagicMock を使用して .first プロパティを設定し、その上で get_attribute を AsyncMock で設定
    currency_locator_first = MagicMock()  # .first が返す Locator（MagicMock を使用）
    currency_locator_first.count = AsyncMock(return_value=1)
    currency_locator_first.get_attribute = AsyncMock(return_value="EUR")  # AsyncMock で get_attribute を設定
    
    currency_locator_base = MagicMock()  # page.locator(selector) が返す Locator（MagicMock を使用）
    currency_locator_base.first = currency_locator_first  # .first プロパティとして設定
    currency_locator = currency_locator_base  # locator_side_effect で返す用
    
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
    # _extract_description も .first を使用する可能性があるため、同様の構造にする
    description_locator_first = AsyncMock()
    description_locator_first.count = AsyncMock(return_value=1)
    description_locator_first.inner_text = AsyncMock(return_value="Test product description for Moncler")
    description_locator_base = AsyncMock()
    description_locator_base.first = description_locator_first
    description_locator = description_locator_base
    
    # locator を selector に応じて返す
    def locator_side_effect(selector: str):
        selector_str = str(selector).lower()
        selector_full = str(selector)
        
        # タイトルセレクタのマッチング（より確実に）
        if any([
            "h1" in selector,
            "title" in selector_str,
            "[data-testid='pdp-title']" in selector_full,
            "[data-testid=\"pdp-title\"]" in selector_full,
            "pdp-title" in selector_str,
            "product-name" in selector_str,
            "itemprop='name'" in selector_full,
            "itemprop=\"name\"" in selector_full,
            "og:title" in selector_str
        ]):
            return title_locator
        # 価格セレクタのマッチング
        elif any([
            "price" in selector_str,
            "[data-testid='pdp-price']" in selector_full,
            "[data-testid=\"pdp-price\"]" in selector_full,
            "pdp-price" in selector_str,
            "itemprop='price'" in selector_full,
            "itemprop=\"price\"" in selector_full
        ]):
            return price_locator
        # 通貨セレクタのマッチング
        elif any([
            "currency" in selector_str,
            "meta[property='product:price:currency']" in selector_full,
            "meta[property=\"product:price:currency\"]" in selector_full,
            "meta[itemprop='priceCurrency']" in selector_full,
            "meta[itemprop=\"priceCurrency\"]" in selector_full,
            "priceCurrency" in selector_str
        ]):
            return currency_locator
        # 画像セレクタのマッチング
        elif "image" in selector_str or "img" in selector_str:
            return image_locator
        # サイズセレクタのマッチング
        elif "size" in selector_str and "button" in selector_str:
            return size_locator
        # カラーセレクタのマッチング
        elif "color" in selector_str:
            return color_locator
        # 説明セレクタのマッチング
        elif "description" in selector_str:
            return description_locator
        else:
            # デフォルト（見つからない）
            default_base = AsyncMock()
            default_first = AsyncMock()
            default_first.count = AsyncMock(return_value=0)
            default_first.inner_text = AsyncMock(return_value="")
            default_first.get_attribute = AsyncMock(return_value=None)
            default_base.first = default_first
            default_base.count = AsyncMock(return_value=0)
            return default_base
    
    # side_effect を設定（return_value を上書きする）
    mock_page.locator = MagicMock(side_effect=locator_side_effect)
    
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
    assert result.description == "Test product description for Moncler"
    
    # アサーション: metadata
    assert result.metadata is not None
    assert result.metadata.get("has_title") is True
    assert result.metadata.get("has_price") is True
    assert result.metadata.get("has_currency") is True
    assert result.metadata.get("image_count") == 3
    assert result.metadata.get("size_count") == 4
    assert result.metadata.get("color_count") == 2
    assert result.metadata.get("url") == mock_page.url

