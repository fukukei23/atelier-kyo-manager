"""Additional tests for app/services/brand_price_scraper.py - unit tests"""

import re
from unittest.mock import patch

from app.services.brand_price_scraper import (
    _CHROME_HEADERS,
    _CURRENCY_MAP,
    _FARFETCH_BRAND_SLUGS,
    _make_item,
)


class TestBrandPriceConstants:
    def test_chrome_headers_defined(self):
        assert "User-Agent" in _CHROME_HEADERS
        assert "Accept-Language" in _CHROME_HEADERS
        assert len(_CHROME_HEADERS) > 0

    def test_currency_map_gucci_eur(self):
        assert _CURRENCY_MAP["Gucci"] == "EUR"

    def test_currency_map_balenciaga_jpy(self):
        assert _CURRENCY_MAP["Balenciaga"] == "JPY"

    def test_farfetch_brand_slugs_gucci(self):
        assert _FARFETCH_BRAND_SLUGS["Gucci"] == "gucci"


class TestMakeItem:
    def test_make_item_jpy(self):
        # JPY doesn't need FX conversion, so no patch needed
        result = _make_item("Gucci", "Bag", 100000, "test", "https://test.com", currency="JPY")
        assert result["price_jpy"] == 100000
        assert result["currency"] == "JPY"

    def test_make_item_eur_converts(self):
        with patch("app.utils.fx_utils.get_fx_table_jpy") as mock_fx:
            mock_fx.return_value = ({"EUR": 160.0}, None)
            result = _make_item("Gucci", "Bag", 1000.0, "test", "https://test.com", currency="EUR")
            assert result["price_jpy"] == 160000
            assert result["exchange_rate"] == 160.0

    def test_make_item_unknown_currency(self):
        with patch("app.utils.fx_utils.get_fx_table_jpy") as mock_fx:
            mock_fx.return_value = ({}, None)
            result = _make_item("Gucci", "Bag", 1000, "test", "https://test.com", currency="UNKNOWN")
            assert result["price_jpy"] == 1000
            assert result["exchange_rate"] == 1.0

    def test_make_item_fields(self):
        result = _make_item("Prada", "Re-edition", 258000, "prada", "https://prada.com", currency="EUR")
        assert result["brand"] == "Prada"
        assert result["product_name"] == "Re-edition"
        assert result["source_site"] == "prada"
        assert result["price_original"] == 258000
        assert result["in_stock"] is True
        assert "scraped_at" in result


class TestBrandPriceScraper:
    def test_scraper_headless_default(self):
        from app.services.brand_price_scraper import BrandPriceScraper
        scraper = BrandPriceScraper()
        assert scraper.headless is True

    def test_scraper_headless_false(self):
        from app.services.brand_price_scraper import BrandPriceScraper
        scraper = BrandPriceScraper(headless=False)
        assert scraper.headless is False

    def test_scraper_has_scrape_method(self):
        from app.services.brand_price_scraper import BrandPriceScraper
        scraper = BrandPriceScraper()
        assert callable(scraper.scrape)

    def test_scraper_private_methods(self):
        from app.services.brand_price_scraper import BrandPriceScraper
        scraper = BrandPriceScraper()
        assert hasattr(scraper, "_init_browser")
        assert hasattr(scraper, "_close_browser")
        assert hasattr(scraper, "_scrape_farfetch")
        assert hasattr(scraper, "_scrape_cffi_official")


class TestScraperPatterns:
    def test_gucci_pattern(self):
        pattern = r'aria-label="([^"]+?),\s*€\s*([\d.]+)"'
        assert re.compile(pattern) is not None

    def test_prada_pattern(self):
        pattern = r'aria-label="\s*([^"]+?)\s*€\s*([\d.]+)'
        assert re.compile(pattern) is not None

    def test_valentino_pattern(self):
        pattern = r'data-canonical-url="([^"]+)"[^>]*?data-price="([\d.]+)"'
        assert re.compile(pattern) is not None

    def test_loewe_pattern(self):
        pattern = r'>([^<]{5,60}?)<.*?¥([\d,]+)'
        assert re.compile(pattern) is not None

    def test_balenciaga_pattern(self):
        pattern = r'itemprop="price"\s+content="(\d+)"'
        assert re.compile(pattern) is not None
