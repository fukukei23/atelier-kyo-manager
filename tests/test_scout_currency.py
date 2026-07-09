"""Tests for app/utils/scout_currency.py"""

from app.utils.scout_currency import convert_price, detect_currency, to_number


class TestDetectCurrency:
    def test_hint_overrides_all(self):
        assert detect_currency("¥1000", currency_hint="USD") == "USD"

    def test_yen_sign(self):
        assert detect_currency("¥1,500") == "JPY"

    def test_fullwidth_yen(self):
        assert detect_currency("￥1500") == "JPY"

    def test_dollar_sign(self):
        assert detect_currency("$99.99") == "USD"

    def test_euro_sign(self):
        assert detect_currency("€500") == "EUR"

    def test_pound_sign(self):
        assert detect_currency("£200") == "GBP"

    def test_won_sign(self):
        assert detect_currency("₩50000") == "KRW"

    def test_word_jpy(self):
        assert detect_currency("Price: 1000 JPY") == "JPY"

    def test_word_eur(self):
        assert detect_currency("Total: 100 EUR") == "EUR"

    def test_unknown_currency(self):
        assert detect_currency("1234 xyz") == "UNKNOWN"

    def test_hint_uppercase_normalized(self):
        assert detect_currency("text", currency_hint="eur") == "EUR"


class TestToNumber:
    def test_plain_integer(self):
        assert to_number("1500") == 1500

    def test_with_comma(self):
        assert to_number("1,500") == 1500

    def test_with_yen_prefix(self):
        assert to_number("¥1,000") == 1000

    def test_with_spaces(self):
        assert to_number("1 500") == 1500

    def test_decimal_drops_fraction(self):
        # digits only, no decimal point handling in regex
        result = to_number("99.99")
        assert result == 9999  # all digits concatenated

    def test_no_digits_returns_none(self):
        assert to_number("abc") is None

    def test_empty_string_returns_none(self):
        assert to_number("") is None


class TestConvertPrice:
    def test_jpy_to_jpy(self):
        result = convert_price(1000, "JPY", "JPY", {})
        assert result == 1000.0

    def test_usd_to_jpy(self):
        fx = {"USD": 150.0}
        result = convert_price(100, "USD", "JPY", fx)
        assert result == 15000.0

    def test_eur_to_jpy(self):
        fx = {"EUR": 160.0}
        result = convert_price(50, "EUR", "JPY", fx)
        assert result == 8000.0

    def test_missing_rate_returns_none(self):
        result = convert_price(100, "GBP", "JPY", {})
        assert result is None

    def test_no_fx_to_returns_none(self):
        result = convert_price(100, "USD", None, {"USD": 150.0})
        assert result is None

    def test_non_jpy_target_returns_none(self):
        # Only JPY target is implemented
        result = convert_price(100, "USD", "EUR", {"USD": 150.0})
        assert result is None
