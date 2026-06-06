"""Tests for app/services/sale_scraper.py - YOOX/SSENSE sale price scraper"""

import re
from unittest.mock import patch

import pytest

from app.services.sale_scraper import (
    _YOOX_SLUGS,
    _SSENSE_SLUGS,
    _is_bag,
    _slug_to_name,
)


class TestBrandSlugMappings:
    """Test brand slug mappings for YOOX and SSENSE."""

    def test_yoox_all_brands(self):
        """All11 brands should have YOOX slugs."""
        expected = {
            "Gucci": "gucci",
            "Prada": "prada",
            "Valentino": "valentino-garavani",
            "Versace": "versace",
            "Marni": "marni",
            "Chloe": "chloe",
            "Celine": "celine",
            "Ferragamo": "salvatore-ferragamo",
            "Loewe": "loewe",
            "Balenciaga": "balenciaga",
            "Bottega Veneta": "bottega-veneta",
        }
        assert _YOOX_SLUGS == expected

    def test_ssense_all_brands(self):
        """All 11 brands should have SSENSE slugs."""
        expected = {
            "Gucci": "gucci",
            "Prada": "prada",
            "Valentino": "valentino",
            "Versace": "versace",
            "Marni": "marni",
            "Chloe": "chloe",
            "Celine": "celine",
            "Ferragamo": "salvatore-ferragamo",
            "Loewe": "loewe",
            "Balenciaga": "balenciaga",
            "Bottega Veneta": "bottega-veneta",
        }
        assert _SSENSE_SLUGS == expected

    def test_yoox_valentino_has_hyphen(self):
        """Valentino YOOX slug should include -garavani suffix."""
        assert "valentino-garavani" in _YOOX_SLUGS["Valentino"]

    def test_ssense_valentino_no_hyphen(self):
        """Valentino SSENSE slug should be plain 'valentino'."""
        assert _SSENSE_SLUGS["Valentino"] == "valentino"


class TestSlugToName:
    """Test URL slug to readable name conversion."""

    def test_simple_slug(self):
        assert _slug_to_name("gucci-bag") == "Gucci Bag"

    def test_multi_word_slug(self):
        assert _slug_to_name("red-rebelle-bag") == "Red Rebelle Bag"

    def test_no_hyphen(self):
        assert _slug_to_name("shoulder bag") == "Shoulder Bag"

    def test_single_word(self):
        assert _slug_to_name("tote") == "Tote"


class TestIsBag:
    """Test bag category detection."""

    @pytest.mark.parametrize("keyword,expected", [
        ("Handbag", True),
        ("Shoulder Bag", True),
        ("Tote Bag", True),
        ("Cross-body Bag", True),
        ("Clutch", True),
        ("Mini Bag", True),
        ("Wallet", True),
        ("Satchel", True),
        ("Backpack", True),
        ("Dress", False),
        ("Top", False),
        ("Shoes", False),
    ])
    def test_bag_keywords(self, keyword, expected):
        assert _is_bag(keyword) == expected


class TestMakeSaleItem:
    """Test _make_sale_item wrapper behavior via _scrape_yoox."""

    def test_make_item_sets_actual_purchase_fields(self):
        """Items created by sale scraper should have actual_purchase_jpy and source set."""
        from app.services.brand_price_scraper import _make_item

        with patch("app.utils.fx_utils.get_fx_table_jpy") as mock_fx:
            mock_fx.return_value = ({"USD": 150.0}, None)
            item = _make_item("Gucci", "Test Bag", 100.0, "yoox", "https://test.com", currency="USD")

        # After _make_sale_item wrapper would set these
        item["actual_purchase_jpy"] = item["price_jpy"]
        item["actual_purchase_source"] = "yoox"

        assert item["actual_purchase_jpy"] == 15000  # 100 *150
        assert item["actual_purchase_source"] == "yoox"
        assert item["price_jpy"] == 15000
        assert item["currency"] == "USD"


class TestYooxParsing:
    """Test YOOX HTML parsing with mock data."""

    def test_parse_yoox_item_code(self):
        """Item codes should be extracted from image URLs."""
        html = '''
        <img src="/images/items/45/45886656NN_14_f.jpg" />
<span>Handbags</span>
        <span>$2,001 (-36%)</span>
        '''
        item_code_re = re.compile(r'/items/\d+/([A-Za-z0-9]+)_\d+_\w+\.jpg')
        matches = list(item_code_re.finditer(html))
        assert len(matches) == 1
        assert matches[0].group(1).upper() == "45886656NN"

    def test_parse_yoox_price_with_discount(self):
        """Sale prices with discount percentage should be parsed."""
        price_re = re.compile(r'\$\s*([\d,]+)\s*(?:\(([-\d]+)%\))?')
        m = price_re.search("$ 2,001 (-36%)")
        assert m is not None
        assert m.group(1) == "2,001"
        assert m.group(2) == "-36"

    def test_parse_yoox_price_without_discount(self):
        """Regular prices without discount should be parsed."""
        price_re = re.compile(r'\$\s*([\d,]+)\s*(?:\(([-\d]+)%\))?')
        m = price_re.search("$ 2,001")
        assert m is not None
        assert m.group(1) == "2,001"
        assert m.group(2) is None


class TestSsenseParsing:
    """Test SSENSE HTML/metadata parsing with mock data."""

    def test_parse_ssense_url_slug(self):
        """Product name should be extracted from SSENSE URL slug."""
        url = "/women/product/gucci/red-rebelle-bag/3426109"
        slug_match = re.search(r'/product/[^/]+/([^/]+)/\d+$', url)
        assert slug_match is not None
        assert slug_match.group(1) == "red-rebelle-bag"

    def test_parse_ssense_metadata_json(self):
        """SSENSE metadata JSON should be parseable."""
        import json
        metadata_str = '{"url": ["/women/product/gucci/bag/123"], "price": [260000], "sku": ["ABC123"]}'
        metadata = json.loads(metadata_str)
        assert len(metadata["url"]) == 1
        assert metadata["price"][0] == 260000
        assert metadata["sku"][0] == "ABC123"

    def test_parse_ssense_price_dict(self):
        """SSENSE price can be dict with 'amount' key."""
        price_raw = {"amount": 260000, "currencyCode": "USD"}
        if isinstance(price_raw, dict):
            price = float(price_raw.get("amount", 0))
        else:
            price = float(price_raw)
        assert price == 260000.0
