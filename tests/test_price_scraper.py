"""
price_scraper のテスト — 価格抽出・在庫判定・リトライ・キャッシュの検証
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from app.services.price_scraper import PriceScraper


# ==============================
# classify_error のテスト
# ==============================


class TestClassifyError:
    @pytest.mark.parametrize(
        "error_string,expected",
        [
            ("HTTP 403 Forbidden", "blocked"),
            ("HTTP 429 Too Many Requests", "blocked"),
            ("Cloudflare block", "blocked"),
            ("HTTP 404 Not Found", "not_found"),
            ("タイムアウト", "timeout"),
            ("Timeout after 10s", "timeout"),
            ("接続失敗", "connection"),
            ("ConnectionError: refused", "connection"),
            ("HTTP 500 Internal Server Error", "server_error"),
            ("HTTP 502 Bad Gateway", "server_error"),
            ("something weird happened", "unknown"),
        ],
    )
    def test_error_classification(self, error_string, expected):
        assert PriceScraper.classify_error(error_string) == expected


# ==============================
# _extract_price のテスト
# ==============================


class TestExtractPrice:
    def _make_soup(self, html):
        from bs4 import BeautifulSoup

        return BeautifulSoup(html, "html.parser")

    def test_og_price_meta_tag(self):
        scraper = PriceScraper()
        html = '<html><meta property="og:price:amount" content="29800"></html>'
        soup = self._make_soup(html)
        result = scraper._extract_price(soup, html)
        assert result["price"] == 29800

    def test_json_ld_offers_price(self):
        scraper = PriceScraper()
        ld = json.dumps({"offers": {"price": "45000"}})
        html = f'<html><script type="application/ld+json">{ld}</script></html>'
        soup = self._make_soup(html)
        result = scraper._extract_price(soup, html)
        assert result["price"] == 45000

    def test_json_ld_top_level_price(self):
        scraper = PriceScraper()
        ld = json.dumps({"price": "12800"})
        html = f'<html><script type="application/ld+json">{ld}</script></html>'
        soup = self._make_soup(html)
        result = scraper._extract_price(soup, html)
        assert result["price"] == 12800

    def test_regex_yen_symbol(self):
        scraper = PriceScraper()
        html = "<html><body>Price: ¥29,800</body></html>"
        soup = self._make_soup(html)
        result = scraper._extract_price(soup, html)
        assert result["price"] == 29800
        assert result["raw"] == "¥29,800"

    def test_regex_dollar(self):
        scraper = PriceScraper()
        html = "<html><body>Price: $199.99</body></html>"
        soup = self._make_soup(html)
        result = scraper._extract_price(soup, html)
        assert result["price"] == 199

    def test_regex_euro(self):
        scraper = PriceScraper()
        html = "<html><body>Preis: €350.00</body></html>"
        soup = self._make_soup(html)
        result = scraper._extract_price(soup, html)
        assert result["price"] == 350

    def test_regex_japanese_yen(self):
        scraper = PriceScraper()
        html = "<html><body>価格 15,000円（税込）</body></html>"
        soup = self._make_soup(html)
        result = scraper._extract_price(soup, html)
        assert result["price"] == 15000

    def test_no_price_found(self):
        scraper = PriceScraper()
        html = "<html><body>No price here</body></html>"
        soup = self._make_soup(html)
        result = scraper._extract_price(soup, html)
        assert result["price"] is None

    def test_og_price_takes_priority_over_regex(self):
        scraper = PriceScraper()
        html = '<html><meta property="og:price:amount" content="5000"><body>¥10,000</body></html>'
        soup = self._make_soup(html)
        result = scraper._extract_price(soup, html)
        assert result["price"] == 5000


# ==============================
# _check_stock のテスト
# ==============================


class TestCheckStock:
    def _make_soup(self, html):
        from bs4 import BeautifulSoup

        return BeautifulSoup(html, "html.parser")

    @pytest.mark.parametrize(
        "keyword",
        ["売切れ", "在庫切れ", "sold out", "out of stock", "在庫なし", "品切れ", "取り扱い終了"],
    )
    def test_out_of_stock_keywords(self, keyword):
        scraper = PriceScraper()
        html = f"<html><body>{keyword}</body></html>"
        soup = self._make_soup(html)
        assert scraper._check_stock(soup, html) is False

    def test_in_stock_no_keywords(self):
        scraper = PriceScraper()
        html = "<html><body>Available for purchase</body></html>"
        soup = self._make_soup(html)
        assert scraper._check_stock(soup, html) is True

    def test_disabled_cart_button(self):
        scraper = PriceScraper()
        html = '<html><button class="add-to-cart" disabled>Add to Cart</button></html>'
        soup = self._make_soup(html)
        assert scraper._check_stock(soup, html) is False

    def test_cart_button_sold_out_text(self):
        scraper = PriceScraper()
        html = '<html><button class="add-to-cart">SOLD OUT</button></html>'
        soup = self._make_soup(html)
        assert scraper._check_stock(soup, html) is False

    def test_enabled_cart_button_is_in_stock(self):
        scraper = PriceScraper()
        html = '<html><button class="add-to-cart">カートに追加</button></html>'
        soup = self._make_soup(html)
        assert scraper._check_stock(soup, html) is True


# ==============================
# _extract_title のテスト
# ==============================


class TestExtractTitle:
    def _make_soup(self, html):
        from bs4 import BeautifulSoup

        return BeautifulSoup(html, "html.parser")

    def test_title_tag(self):
        scraper = PriceScraper()
        html = "<html><title>Product Name - Shop</title></html>"
        soup = self._make_soup(html)
        assert scraper._extract_title(soup) == "Product Name - Shop"

    def test_og_title_fallback(self):
        scraper = PriceScraper()
        html = '<html><meta property="og:title" content="OG Product Name"></html>'
        soup = self._make_soup(html)
        assert scraper._extract_title(soup) == "OG Product Name"

    def test_no_title(self):
        scraper = PriceScraper()
        html = "<html><body></body></html>"
        soup = self._make_soup(html)
        assert scraper._extract_title(soup) is None


# ==============================
# fetch_with_retry のテスト
# ==============================


class TestFetchWithRetry:
    @patch.object(PriceScraper, "fetch")
    def test_success_first_try(self, mock_fetch):
        mock_fetch.return_value = {"success": True, "price": 10000}
        scraper = PriceScraper()
        result = scraper.fetch_with_retry("https://example.com", max_retries=3)
        assert result["success"] is True
        assert result["retry_count"] == 0

    @patch.object(PriceScraper, "fetch")
    def test_retries_on_blocked(self, mock_fetch):
        mock_fetch.side_effect = [
            {"success": False, "error": "HTTP 403"},
            {"success": False, "error": "HTTP 403"},
            {"success": True, "price": 5000},
        ]
        scraper = PriceScraper()
        result = scraper.fetch_with_retry("https://example.com", max_retries=3)
        assert result["success"] is True
        assert result["retry_count"] == 2
        assert len(result["attempts"]) == 2  # 2 failures before success

    @patch.object(PriceScraper, "fetch")
    def test_no_retry_on_404(self, mock_fetch):
        mock_fetch.return_value = {"success": False, "error": "HTTP 404"}
        scraper = PriceScraper()
        result = scraper.fetch_with_retry("https://example.com", max_retries=3)
        assert result["success"] is False
        assert result["retry_count"] == 0
        assert result["error_category"] == "not_found"

    @patch.object(PriceScraper, "fetch")
    def test_exhausts_retries(self, mock_fetch):
        mock_fetch.return_value = {"success": False, "error": "HTTP 503"}
        scraper = PriceScraper()
        result = scraper.fetch_with_retry("https://example.com", max_retries=3)
        assert result["success"] is False
        assert result["retry_count"] == 2
        assert len(result["attempts"]) == 3


# ==============================
# fetch_cached のテスト
# ==============================


class TestFetchCached:
    @patch.object(PriceScraper, "fetch_with_retry")
    def test_cache_hit(self, mock_fetch):
        scraper = PriceScraper()
        scraper._cache["https://example.com"] = {
            "result": {"success": True, "price": 1000},
            "timestamp": time.time(),
        }
        result = scraper.fetch_cached("https://example.com")
        assert result["price"] == 1000
        mock_fetch.assert_not_called()

    @patch.object(PriceScraper, "fetch_with_retry")
    def test_cache_miss_fetches(self, mock_fetch):
        mock_fetch.return_value = {"success": True, "price": 2000}
        scraper = PriceScraper()
        result = scraper.fetch_cached("https://example.com")
        assert result["price"] == 2000
        mock_fetch.assert_called_once()

    @patch.object(PriceScraper, "fetch_with_retry")
    def test_expired_cache_refetches(self, mock_fetch):
        mock_fetch.return_value = {"success": True, "price": 3000}
        scraper = PriceScraper()
        scraper._cache["https://example.com"] = {
            "result": {"success": True, "price": 1000},
            "timestamp": time.time() - 25 * 3600,  # 25時間前
        }
        result = scraper.fetch_cached("https://example.com")
        assert result["price"] == 3000
        mock_fetch.assert_called_once()

    @patch.object(PriceScraper, "fetch_with_retry")
    def test_custom_cache_hours(self, mock_fetch):
        mock_fetch.return_value = {"success": True, "price": 4000}
        scraper = PriceScraper()
        scraper._cache["https://example.com"] = {
            "result": {"success": True, "price": 1000},
            "timestamp": time.time() - 2 * 3600,  # 2時間前
        }
        # cache_hours=1 → 期限切れ
        result = scraper.fetch_cached("https://example.com", cache_hours=1)
        assert result["price"] == 4000
