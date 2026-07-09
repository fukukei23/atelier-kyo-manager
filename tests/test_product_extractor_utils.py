"""Tests for app/extractors/product_info_extractor.py — utility functions"""

from app.extractors.product_info_extractor import (
    CURRENCY_SYMBOLS,
    VALID_PARSING_KEYS,
    _first_yen_from_text,
    _norm_currency,
    _normalize_and_to_int,
)


class TestNormalizeAndToInt:
    def test_plain_integer(self):
        assert _normalize_and_to_int("1500") == 1500

    def test_with_comma(self):
        assert _normalize_and_to_int("1,500") == 1500

    def test_with_decimal(self):
        assert _normalize_and_to_int("1500.99") == 1500

    def test_none_input(self):
        assert _normalize_and_to_int(None) is None

    def test_empty_string(self):
        assert _normalize_and_to_int("") is None

    def test_non_numeric(self):
        assert _normalize_and_to_int("abc") is None

    def test_large_number(self):
        assert _normalize_and_to_int("1,234,567") == 1234567


class TestFirstYenFromText:
    def test_yen_prefix(self):
        result = _first_yen_from_text("¥1,500")
        assert result == 1500

    def test_fullwidth_yen(self):
        result = _first_yen_from_text("￥2000")
        assert result == 2000

    def test_dollar_prefix(self):
        result = _first_yen_from_text("$99")
        assert result == 99

    def test_euro_prefix(self):
        result = _first_yen_from_text("€500")
        assert result == 500

    def test_none_input(self):
        assert _first_yen_from_text(None) is None

    def test_empty_string(self):
        assert _first_yen_from_text("") is None

    def test_plain_number_falls_back(self):
        result = _first_yen_from_text("5000")
        assert result == 5000


class TestNormCurrency:
    def test_jpy_uppercase(self):
        assert _norm_currency("JPY") == "JPY"

    def test_jpy_lowercase(self):
        assert _norm_currency("jpy") == "JPY"

    def test_none_returns_none(self):
        assert _norm_currency(None) is None

    def test_yen_symbol(self):
        result = _norm_currency("¥")
        assert result == "JPY"

    def test_dollar_symbol(self):
        result = _norm_currency("$")
        assert result == "USD"

    def test_euro_symbol(self):
        result = _norm_currency("€")
        assert result == "EUR"

    def test_unknown_returns_uppercase(self):
        result = _norm_currency("xyz")
        assert result == "XYZ"


class TestConstants:
    def test_valid_parsing_keys(self):
        assert "title" in VALID_PARSING_KEYS
        assert "price" in VALID_PARSING_KEYS
        assert "brand" in VALID_PARSING_KEYS

    def test_currency_symbols_includes_yen(self):
        assert "¥" in CURRENCY_SYMBOLS
        assert CURRENCY_SYMBOLS["¥"] == "JPY"

    def test_currency_symbols_includes_dollar(self):
        assert "$" in CURRENCY_SYMBOLS
        assert CURRENCY_SYMBOLS["$"] == "USD"
