"""Tests for sourcing_csv_adapter module."""

import pytest

from app.utils.sourcing_csv_adapter import (
    _normalize_unknown_value,
    _parse_numeric_value,
    parse_csv_row_dict,
    parse_csv_row_to_sourcing_input,
)


class TestNormalizeUnknownValue:
    def test_none_returns_unknown(self):
        assert _normalize_unknown_value(None) == "unknown"

    def test_empty_string_returns_unknown(self):
        assert _normalize_unknown_value("") == "unknown"

    def test_whitespace_returns_unknown(self):
        assert _normalize_unknown_value("   ") == "unknown"

    @pytest.mark.parametrize("val", ["unknown", "-", "n/a", "na", "n.a.", "null", "none", "?", "??"])
    def test_unknown_patterns(self, val):
        assert _normalize_unknown_value(val) == "unknown"

    def test_case_insensitive(self):
        assert _normalize_unknown_value("Unknown") == "unknown"
        assert _normalize_unknown_value("UNKNOWN") == "unknown"

    def test_numeric_string_preserved(self):
        assert _normalize_unknown_value("100") == "100"

    def test_regular_string_preserved(self):
        assert _normalize_unknown_value("hello") == "hello"


class TestParseNumericValue:
    def test_none_returns_failure(self):
        val, ok = _parse_numeric_value(None)
        assert val is None
        assert ok is False

    def test_unknown_returns_failure(self):
        val, ok = _parse_numeric_value("unknown")
        assert val is None
        assert ok is False

    def test_integer_string(self):
        val, ok = _parse_numeric_value("100")
        assert val == 100.0
        assert ok is True

    def test_float_string(self):
        val, ok = _parse_numeric_value("99.5")
        assert val == 99.5
        assert ok is True

    def test_comma_separated(self):
        val, ok = _parse_numeric_value("1,000")
        assert val == 1000.0
        assert ok is True

    def test_invalid_string_returns_failure(self):
        val, ok = _parse_numeric_value("abc")
        assert val is None
        assert ok is False

    def test_whitespace_stripped(self):
        val, ok = _parse_numeric_value("  50  ")
        assert val == 50.0
        assert ok is True


class TestParseCsvRowDict:
    def test_valid_row(self):
        row = {"purchase_price": "100", "selling_price": "200"}
        result = parse_csv_row_dict(row)
        assert result["status"] == "valid"
        assert result["data"]["purchase_price"] == 100.0
        assert result["data"]["selling_price"] == 200.0

    def test_missing_required_column(self):
        row = {"purchase_price": "100"}
        result = parse_csv_row_dict(row)
        assert result["status"] == "invalid"
        assert "selling_price" in " ".join(result["errors"])

    def test_unknown_price(self):
        row = {"purchase_price": "unknown", "selling_price": "200"}
        result = parse_csv_row_dict(row)
        assert result["status"] == "valid"
        assert result["data"]["purchase_price"] == "unknown"

    def test_invalid_price_format(self):
        row = {"purchase_price": "abc", "selling_price": "200"}
        result = parse_csv_row_dict(row)
        assert result["status"] == "invalid"

    def test_optional_fields_parsed(self):
        row = {
            "purchase_price": "100",
            "selling_price": "200",
            "shipping_cost": "10",
            "customs_duty": "5",
        }
        result = parse_csv_row_dict(row)
        assert result["status"] == "valid"
        assert result["data"]["shipping_cost"] == 10.0
        assert result["data"]["customs_duty"] == 5.0

    def test_optional_fields_unknown(self):
        row = {
            "purchase_price": "100",
            "selling_price": "200",
            "shipping_cost": "-",
        }
        result = parse_csv_row_dict(row)
        assert result["status"] == "valid"
        assert result["data"]["shipping_cost"] == "unknown"

    def test_comma_in_prices(self):
        row = {"purchase_price": "1,000", "selling_price": "2,500"}
        result = parse_csv_row_dict(row)
        assert result["status"] == "valid"
        assert result["data"]["purchase_price"] == 1000.0
        assert result["data"]["selling_price"] == 2500.0


class TestParseCsvRowToSourcingInput:
    def test_valid_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "purchase_price,selling_price\n100,200\n",
            encoding="utf-8",
        )
        result = parse_csv_row_to_sourcing_input(csv_file, row_index=0)
        assert result["status"] == "valid"
        assert result["data"]["purchase_price"] == 100.0

    def test_file_not_found(self):
        result = parse_csv_row_to_sourcing_input("/nonexistent/file.csv")
        assert result["status"] == "invalid"
        assert "見つかりません" in result["errors"][0]

    def test_row_index_out_of_range(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "purchase_price,selling_price\n100,200\n",
            encoding="utf-8",
        )
        result = parse_csv_row_to_sourcing_input(csv_file, row_index=99)
        assert result["status"] == "invalid"
        assert "範囲外" in result["errors"][0]

    def test_negative_row_index(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "purchase_price,selling_price\n100,200\n",
            encoding="utf-8",
        )
        result = parse_csv_row_to_sourcing_input(csv_file, row_index=-1)
        assert result["status"] == "invalid"

    def test_bom_encoded_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_bytes(b"\xef\xbb\xbf" + b"purchase_price,selling_price\n100,200\n")
        result = parse_csv_row_to_sourcing_input(csv_file, row_index=0)
        assert result["status"] == "valid"

    def test_missing_columns_in_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,brand\nitem,test\n", encoding="utf-8")
        result = parse_csv_row_to_sourcing_input(csv_file, row_index=0)
        assert result["status"] == "invalid"
