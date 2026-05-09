"""fx_utils テスト"""

import pytest

from app.utils.fx_utils import (
    _ecb_xml_to_eur_table,
    build_jpy_table_from_eur,
    parse_fx_rates_str,
)


# --- parse_fx_rates_str ---
class TestParseFxRatesStr:
    def test_valid_input(self):
        result = parse_fx_rates_str("USD=155.0,EUR=170.5,GBP=1.5")
        assert result == {"USD": 155.0, "EUR": 170.5, "GBP": 1.5}

    def test_valid_with_spaces(self):
        result = parse_fx_rates_str(" USD = 155.0 , EUR = 170.5 ")
        assert result == {"USD": 155.0, "EUR": 170.5}

    def test_empty_string(self):
        assert parse_fx_rates_str("") == {}

    def test_no_equals_ignored(self):
        result = parse_fx_rates_str("USD155.0,EUR=170.5,INVALID")
        assert result == {"EUR": 170.5}

    def test_invalid_value_ignored(self):
        result = parse_fx_rates_str("USD=ABC,EUR=170.5")
        assert result == {"EUR": 170.5}

    def test_case_insensitive(self):
        result = parse_fx_rates_str("usd=155.0")
        assert "USD" in result


# --- build_jpy_table_from_eur ---
class TestBuildJpyTableFromEur:
    def test_valid_table(self):
        eur_table = {"EUR": 1.0, "JPY": 150.0, "USD": 1.5}
        result = build_jpy_table_from_eur(eur_table)
        assert result["JPY"] == 1.0
        assert result["USD"] == pytest.approx(100.0)

    def test_missing_jpy(self):
        assert build_jpy_table_from_eur({"EUR": 1.0, "USD": 1.5}) == {}

    def test_only_jpy(self):
        result = build_jpy_table_from_eur({"EUR": 1.0, "JPY": 150.0})
        assert result["JPY"] == 1.0
        assert result["EUR"] == 150.0

    def test_zero_rate_ignored(self):
        eur_table = {"EUR": 1.0, "JPY": 150.0, "USD": 0.0}
        result = build_jpy_table_from_eur(eur_table)
        assert "JPY" in result
        assert "USD" not in result


# --- _ecb_xml_to_eur_table ---
class TestEcbXmlToEurTable:
    def test_valid_xml(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <gesmes:Envelope xmlns:gesmes="http://www.gesmes.org/xml/2002-08-01"
                         xmlns="http://www.ecb.int/vocabulary/2002-08-01/eurofxref">
            <Cube>
                <Cube time="2023-09-15">
                    <Cube currency="USD" rate="1.0650"/>
                    <Cube currency="JPY" rate="157.30"/>
                </Cube>
            </Cube>
        </gesmes:Envelope>"""
        result = _ecb_xml_to_eur_table(xml)
        assert result.get("EUR") == 1.0
        assert result.get("USD") == 1.0650
        assert result.get("JPY") == 157.30

    def test_malformed_xml(self):
        result = _ecb_xml_to_eur_table("<root><Cube currency='USD' rate='1.0'></root>")
        assert result == {"EUR": 1.0}

    def test_empty_xml(self):
        result = _ecb_xml_to_eur_table("")
        assert result == {"EUR": 1.0}

    def test_missing_rate_attribute(self):
        xml = """<root>
            <Cube currency="USD" rate="1.5"/>
            <Cube currency="JPY" />
        </root>"""
        result = _ecb_xml_to_eur_table(xml)
        assert result == {"EUR": 1.0, "USD": 1.5}

    def test_invalid_rate_value(self):
        xml = """<root>
            <Cube currency="USD" rate="INVALID"/>
            <Cube currency="JPY" rate="150"/>
        </root>"""
        result = _ecb_xml_to_eur_table(xml)
        # float parse error causes early return with just EUR
        assert result.get("EUR") == 1.0
