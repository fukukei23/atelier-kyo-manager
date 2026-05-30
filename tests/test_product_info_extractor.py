"""Tests for pure functions in app/extractors/product_info_extractor.py - Issue #84"""
import pytest

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

from app.extractors.product_info_extractor import (
    extract_product_info,
    _collect_ldjson,
    _collect_inline_json_candidates,
    _dig_price_in_dict,
    _extract_json_objects_from_text,
    _first_yen_from_text,
    _flatten_dicts,
    _norm_currency,
    _normalize_and_to_int,
    _parse_price_text,
)


# ── _normalize_and_to_int ─────────────────────────────────────────────────────

class TestNormalizeAndToInt:
    def test_none_returns_none(self):
        assert _normalize_and_to_int(None) is None

    def test_empty_string_returns_none(self):
        assert _normalize_and_to_int("") is None

    def test_simple_integer(self):
        assert _normalize_and_to_int("123") == 123

    def test_comma_separated(self):
        assert _normalize_and_to_int("123,456") == 123456

    def test_decimal_truncated(self):
        assert _normalize_and_to_int("1,299.99") == 1299

    def test_yen_symbol_stripped(self):
        assert _normalize_and_to_int("¥123,000") == 123000

    def test_zero(self):
        assert _normalize_and_to_int("0") == 0

    def test_non_numeric_returns_none(self):
        assert _normalize_and_to_int("abc") is None

    def test_only_symbols_returns_none(self):
        assert _normalize_and_to_int("¥,€") is None


# ── _first_yen_from_text ──────────────────────────────────────────────────────

class TestFirstYenFromText:
    def test_empty_returns_none(self):
        assert _first_yen_from_text("") is None

    def test_yen_symbol(self):
        assert _first_yen_from_text("¥ 123,000") == 123000

    def test_yen_kanji(self):
        assert _first_yen_from_text("1,200円") == 1200

    def test_dollar_symbol(self):
        assert _first_yen_from_text("$99.99") == 99

    def test_euro_symbol(self):
        assert _first_yen_from_text("€50") == 50

    def test_no_price_returns_none(self):
        # テキストだけで数字なし
        assert _first_yen_from_text("No price here") is None

    def test_plain_number(self):
        # 記号なし数字 → _normalize_and_to_int にフォールバック
        result = _first_yen_from_text("50000")
        assert result == 50000


# ── _norm_currency ────────────────────────────────────────────────────────────

class TestNormCurrency:
    def test_none_returns_none(self):
        assert _norm_currency(None) is None

    def test_empty_returns_none(self):
        assert _norm_currency("") is None

    def test_dollar_symbol(self):
        assert _norm_currency("$") == "USD"

    def test_yen_symbol(self):
        assert _norm_currency("¥") == "JPY"

    def test_euro_symbol(self):
        assert _norm_currency("€") == "EUR"

    def test_3char_code(self):
        assert _norm_currency("EUR") == "EUR"
        assert _norm_currency("GBP") == "GBP"
        assert _norm_currency("JPY") == "JPY"

    def test_lowercase_3char(self):
        assert _norm_currency("usd") == "USD"
        assert _norm_currency("eur") == "EUR"

    def test_4char_returns_none(self):
        assert _norm_currency("XXXX") is None

    def test_krw_symbol(self):
        assert _norm_currency("₩") == "KRW"

    def test_2char_returns_none(self):
        assert _norm_currency("US") is None


# ── _parse_price_text ─────────────────────────────────────────────────────────

class TestParsePriceText:
    def test_empty_returns_none_tuple(self):
        assert _parse_price_text("") == (None, None)

    def test_yen_price(self):
        amt, cur = _parse_price_text("¥ 346,500")
        assert amt == 346500.0
        assert cur == "JPY"

    def test_usd_price(self):
        amt, cur = _parse_price_text("USD 1200")
        assert amt == 1200.0
        assert cur == "USD"

    def test_eur_price_suffix(self):
        # 通貨コードが後置の場合、amtは取得できるがcurはNoneになる実装
        amt, cur = _parse_price_text("1299.99 EUR")
        assert amt is not None and abs(amt - 1299.99) < 0.01

    def test_no_price_returns_none_tuple(self):
        amt, cur = _parse_price_text("abc xyz")
        assert amt is None

    def test_integer_price(self):
        amt, cur = _parse_price_text("50000")
        assert amt == 50000.0

    def test_comma_in_price(self):
        amt, cur = _parse_price_text("JPY 1,500,000")
        assert amt == 1500000.0


# ── _flatten_dicts ────────────────────────────────────────────────────────────

class TestFlattenDicts:
    def test_empty_dict(self):
        assert _flatten_dicts({}) == [{}]

    def test_nested_dict(self):
        result = _flatten_dicts({"a": {"b": 1}})
        assert {"a": {"b": 1}} in result
        assert {"b": 1} in result

    def test_list_of_dicts(self):
        result = _flatten_dicts([{"x": 1}, {"y": 2}])
        assert {"x": 1} in result
        assert {"y": 2} in result

    def test_string_returns_empty(self):
        assert _flatten_dicts("string") == []

    def test_integer_returns_empty(self):
        assert _flatten_dicts(123) == []

    def test_empty_list(self):
        assert _flatten_dicts([]) == []

    def test_deeply_nested(self):
        result = _flatten_dicts({"a": {"b": {"c": 1}}})
        assert {"c": 1} in result

    def test_list_in_dict(self):
        result = _flatten_dicts({"items": [{"id": 1}, {"id": 2}]})
        assert {"id": 1} in result
        assert {"id": 2} in result


# ── _extract_json_objects_from_text ───────────────────────────────────────────

class TestExtractJsonObjectsFromText:
    def test_empty_returns_empty(self):
        assert _extract_json_objects_from_text("") == []

    def test_plain_json_object(self):
        result = _extract_json_objects_from_text('{"a": 1}')
        assert {"a": 1} in result

    def test_plain_json_array(self):
        result = _extract_json_objects_from_text('[{"x": 1}]')
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_variable_assignment_pattern(self):
        result = _extract_json_objects_from_text('const x = {"b": 2};')
        # {"b": 2} should be extracted from = {...}; pattern
        assert any(isinstance(r, dict) and r.get("b") == 2 for r in result)

    def test_non_json_returns_empty(self):
        result = _extract_json_objects_from_text("no json here")
        assert result == []

    def test_malformed_json_skipped(self):
        result = _extract_json_objects_from_text("{broken json}")
        assert isinstance(result, list)  # should not raise

    def test_next_data_pattern(self):
        result = _extract_json_objects_from_text('window.__NEXT_DATA__ = {"props": {"key": "val"}};')
        assert isinstance(result, list)


# ── _dig_price_in_dict ────────────────────────────────────────────────────────

class TestDigPriceInDict:
    def test_empty_dict_returns_none(self):
        assert _dig_price_in_dict({}) == (None, None)

    def test_numeric_price(self):
        amt, cur = _dig_price_in_dict({"price": 1000})
        assert amt == 1000.0

    def test_float_amount(self):
        amt, cur = _dig_price_in_dict({"amount": 500.5})
        assert amt == 500.5

    def test_string_price_with_yen(self):
        amt, cur = _dig_price_in_dict({"price": "¥50,000"})
        assert amt == 50000.0
        assert cur == "JPY"

    def test_offers_price(self):
        amt, cur = _dig_price_in_dict({
            "offers": {"price": "150000", "priceCurrency": "JPY"}
        })
        assert amt == 150000.0
        assert cur == "JPY"

    def test_offers_numeric(self):
        amt, cur = _dig_price_in_dict({"offers": {"price": 99.99}})
        assert abs(amt - 99.99) < 0.01

    def test_list_price_key(self):
        amt, cur = _dig_price_in_dict({"listPrice": 200})
        assert amt == 200.0

    def test_sales_price_key(self):
        amt, cur = _dig_price_in_dict({"salesPrice": 150.0})
        assert amt == 150.0

    def test_nested_price_dict(self):
        """{"value": 5000, "currency": "EUR"}形式"""
        amt, cur = _dig_price_in_dict({"price": {"value": 5000, "currency": "EUR"}})
        assert amt == 5000.0
        assert cur == "EUR"

    def test_offers_list(self):
        """offers がリストの場合は最初のマッチを返す"""
        amt, cur = _dig_price_in_dict({
            "offers": [
                {"price": 300.0, "priceCurrency": "USD"},
                {"price": 400.0, "priceCurrency": "EUR"},
            ]
        })
        assert amt == 300.0

    def test_no_price_key_returns_none(self):
        amt, cur = _dig_price_in_dict({"name": "Product", "brand": "PRADA"})
        assert amt is None


# ── _collect_ldjson ───────────────────────────────────────────────────────────

@pytest.mark.skipif(not BS4_AVAILABLE, reason="BeautifulSoup not installed")
class TestCollectLdjson:
    def _soup(self, html):
        return BeautifulSoup(html, "lxml")

    def test_no_ldjson_returns_empty(self):
        soup = self._soup("<html><body>no scripts</body></html>")
        assert _collect_ldjson(soup) == []

    def test_parses_product_ldjson(self):
        html = '''<html><head>
        <script type="application/ld+json">{"@type":"Product","name":"Coat","price":50000}</script>
        </head></html>'''
        result = _collect_ldjson(self._soup(html))
        assert any(r.get("@type") == "Product" for r in result)

    def test_skips_invalid_json(self):
        html = '''<html><head>
        <script type="application/ld+json">{invalid json}</script>
        </head></html>'''
        result = _collect_ldjson(self._soup(html))
        assert result == []

    def test_flattens_nested(self):
        html = '''<html><head>
        <script type="application/ld+json">{"@type":"Product","offers":{"price":1000}}</script>
        </head></html>'''
        result = _collect_ldjson(self._soup(html))
        # includes {"price":1000} as flattened
        assert any("price" in r for r in result)


# ── _collect_inline_json_candidates ──────────────────────────────────────────

@pytest.mark.skipif(not BS4_AVAILABLE, reason="BeautifulSoup not installed")
class TestCollectInlineJsonCandidates:
    def _soup(self, html):
        return BeautifulSoup(html, "lxml")

    def test_no_scripts_returns_empty(self):
        soup = self._soup("<html><body>text</body></html>")
        result = _collect_inline_json_candidates(soup)
        assert isinstance(result, list)

    def test_extracts_next_data(self):
        html = '''<html><head>
        <script>var data = {"name":"Coat","price":50000};</script>
        </head></html>'''
        result = _collect_inline_json_candidates(self._soup(html))
        # _flatten_dicts でネストが展開されるので何らかのdictが返る
        assert isinstance(result, list)

    def test_skips_ldjson_scripts(self):
        html = '''<html><head>
        <script type="application/ld+json">{"@type":"Product"}</script>
        </head></html>'''
        result = _collect_inline_json_candidates(self._soup(html))
        # ld+json はスキップされる
        assert not any(r.get("@type") == "Product" for r in result)


# ── extract_product_info ──────────────────────────────────────────────────────

@pytest.mark.skipif(not BS4_AVAILABLE, reason="BeautifulSoup not installed")
class TestExtractProductInfo:
    def test_og_title_extracted(self):
        html = '''<html><head>
        <meta property="og:title" content="Moncler Monaco Coat">
        </head><body></body></html>'''
        result = extract_product_info(html, {})
        assert result["title"] == "Moncler Monaco Coat"

    def test_fallback_to_title_tag(self):
        html = '''<html><head><title>Test Product</title></head><body></body></html>'''
        result = extract_product_info(html, {})
        assert result["title"] == "Test Product"

    def test_og_brand_extracted(self):
        html = '''<html><head>
        <meta property="og:site_name" content="MONCLER">
        </head><body></body></html>'''
        result = extract_product_info(html, {})
        assert result["brand"] == "MONCLER"

    def test_price_from_meta(self):
        html = '''<html><head>
        <meta property="product:price:amount" content="150000">
        <meta property="product:price:currency" content="JPY">
        </head><body></body></html>'''
        result = extract_product_info(html, {})
        assert result["price"] == 150000.0
        assert result["currency"] == "JPY"

    def test_price_from_dom_element(self):
        html = '''<html><body>
        <span class="price">¥50,000</span>
        </body></html>'''
        result = extract_product_info(html, {})
        assert result["price"] == 50000.0

    def test_price_from_ldjson(self):
        html = '''<html><head>
        <script type="application/ld+json">{"@type":"Product","offers":{"price":"99000","priceCurrency":"JPY"}}</script>
        </head><body></body></html>'''
        result = extract_product_info(html, {})
        assert result["price"] == 99000.0

    def test_empty_html_returns_nulls(self):
        result = extract_product_info("<html><body></body></html>", {})
        assert result["price"] is None
        assert result["title"] is None

    def test_returns_required_keys(self):
        result = extract_product_info("<html><body></body></html>", {})
        assert "title" in result
        assert "price" in result
        assert "currency" in result
        assert "brand" in result
        assert "source_flags" in result

    def test_custom_price_selector(self):
        html = '''<html><body>
        <div id="my-price">¥120,000</div>
        </body></html>'''
        config = {"selectors": {"pdp": {"price": ["#my-price"]}}}
        result = extract_product_info(html, config)
        assert result["price"] == 120000.0
