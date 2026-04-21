"""
包括テスト: sourcing_tier, fx_utils, pricing_calculator
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from app.utils.fx_utils import (
    _cache_paths,
    _ecb_xml_to_eur_table,
    _load_json,
    _now_iso,
    _save_json,
    build_jpy_table_from_eur,
    get_fx_table_jpy,
    parse_fx_rates_str,
)
from app.utils.pricing_calculator import (
    BUYMA_COMMISSION_RATE,
    BUYMA_SYSTEM_FEE,
    DEFAULT_EXCHANGE_RATE,
    calculate_profit,
    get_exchange_rate,
)
from app.utils.sourcing_tier import judge_tier


# ==========================================
# sourcing_tier
# ==========================================
class TestJudgeTier:
    def test_invalid_status_tier_d(self):
        assert judge_tier({"status": "invalid"})["tier"] == "D"

    def test_partial_status_tier_c_even_high_profit(self):
        assert judge_tier({"status": "partial", "profit_rate": 0.99})["tier"] == "C"

    def test_complete_profit_rate_none_tier_d(self):
        assert judge_tier({"status": "complete", "profit_rate": None})["tier"] == "D"

    def test_complete_tier_a_boundary(self):
        assert judge_tier({"status": "complete", "profit_rate": 0.30})["tier"] == "A"

    def test_complete_tier_a_above(self):
        assert judge_tier({"status": "complete", "profit_rate": 0.50})["tier"] == "A"

    def test_complete_tier_b_boundary(self):
        assert judge_tier({"status": "complete", "profit_rate": 0.20})["tier"] == "B"

    def test_complete_tier_b_range(self):
        assert judge_tier({"status": "complete", "profit_rate": 0.25})["tier"] == "B"

    def test_complete_tier_c_below_20(self):
        assert judge_tier({"status": "complete", "profit_rate": 0.19})["tier"] == "C"

    def test_complete_tier_c_zero(self):
        assert judge_tier({"status": "complete", "profit_rate": 0.0})["tier"] == "C"

    def test_no_status_key_uses_profit_rate(self):
        assert judge_tier({"profit_rate": 0.40})["tier"] == "A"

    def test_empty_dict_tier_d(self):
        assert judge_tier({})["tier"] == "D"

    def test_reason_contains_profit_rate_for_tier_a(self):
        result = judge_tier({"status": "complete", "profit_rate": 0.35})
        assert "35.00%" in result["reason"]

    def test_invalid_reason(self):
        result = judge_tier({"status": "invalid"})
        assert "無効" in result["reason"]


# ==========================================
# fx_utils
# ==========================================
class TestParseFxRatesStr:
    def test_basic(self):
        assert parse_fx_rates_str("USD=155,EUR=170") == {"USD": 155.0, "EUR": 170.0}

    def test_single(self):
        assert parse_fx_rates_str("GBP=130.5") == {"GBP": 130.5}

    def test_empty_string(self):
        assert parse_fx_rates_str("") == {}

    def test_spaces_trimmed(self):
        assert parse_fx_rates_str(" USD = 155 ") == {"USD": 155.0}

    def test_case_upper(self):
        assert parse_fx_rates_str("usd=155") == {"USD": 155.0}

    def test_malformed_no_equals_skipped(self):
        result = parse_fx_rates_str("BADVAL,EUR=170")
        assert "EUR" in result
        assert "BADVAL" not in result

    def test_non_numeric_value_skipped(self):
        result = parse_fx_rates_str("USD=abc,EUR=170")
        assert result == {"EUR": 170.0}

    def test_none_returns_empty(self):
        assert parse_fx_rates_str(None) == {}


class TestBuildJpyTableFromEur:
    def test_basic_conversion(self):
        eur = {"EUR": 1.0, "USD": 1.1, "JPY": 160.0}
        result = build_jpy_table_from_eur(eur)
        assert result["JPY"] == 1.0
        assert result["USD"] == pytest.approx(160.0 / 1.1)

    def test_no_jpy_returns_empty(self):
        assert build_jpy_table_from_eur({"EUR": 1.0, "USD": 1.1}) == {}

    def test_empty_table_returns_empty(self):
        assert build_jpy_table_from_eur({}) == {}


class TestEcbXmlToEurTable:
    def _make_xml(self, rates):
        cubes = "".join(
            f'<Cube currency="{c}" rate="{r}"/>' for c, r in rates
        )
        return f'<Envelope><Cube><Cube time="2025-01-01">{cubes}</Cube></Cube></Envelope>'

    def test_parse_usd_jpy(self):
        xml = self._make_xml([("USD", "1.05"), ("JPY", "160.0")])
        result = _ecb_xml_to_eur_table(xml)
        assert result["EUR"] == 1.0
        assert result["USD"] == 1.05
        assert result["JPY"] == 160.0

    def test_empty_xml_returns_eur_default(self):
        result = _ecb_xml_to_eur_table("")
        assert result == {"EUR": 1.0}

    def test_malformed_xml(self):
        result = _ecb_xml_to_eur_table("<broken>")
        assert isinstance(result, dict)


class TestLoadSaveJson:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.json")
            data = {"key": "value", "num": 42}
            _save_json(path, data)
            loaded = _load_json(path)
            assert loaded == data

    def test_load_nonexistent_returns_none(self):
        assert _load_json("/nonexistent/path/file.json") is None

    def test_load_invalid_json_returns_none(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{bad json")
            f.flush()
            assert _load_json(f.name) is None
            os.unlink(f.name)

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sub", "dir", "test.json")
            _save_json(path, {"ok": True})
            assert os.path.exists(path)


class TestFxHelpers:
    def test_now_iso_format(self):
        result = _now_iso()
        assert "T" in result
        assert len(result) >= 19

    def test_cache_paths_returns_tuple(self):
        cache_dir, cache_file = _cache_paths()
        assert cache_dir.endswith("cache")
        assert cache_file.endswith("ecb_fx.json")

    def test_get_fx_table_manual(self):
        manual = {"USD": 150.0, "EUR": 165.0}
        table, meta = get_fx_table_jpy(manual_table=manual, auto=False)
        assert table["USD"] == 150.0
        assert meta["source"] == "manual"

    def test_get_fx_table_no_auto_no_manual(self):
        table, meta = get_fx_table_jpy(manual_table=None, auto=False)
        assert table == {}
        assert meta["source"] is None

    @patch("app.utils.fx_utils._ecb_fetch_xml")
    def test_get_fx_table_auto_live(self, mock_fetch):
        xml = '<Envelope><Cube><Cube time="2025-01-01"><Cube currency="USD" rate="1.1"/><Cube currency="JPY" rate="160"/></Cube></Cube></Envelope>'
        mock_fetch.return_value = ("2025-01-01", xml)
        with tempfile.TemporaryDirectory() as td:
            with patch("app.utils.fx_utils._cache_paths", return_value=(td, os.path.join(td, "ecb_fx.json"))):
                table, meta = get_fx_table_jpy(manual_table=None, auto=True)
                assert meta["source"] == "ecb(live)"
                assert "USD" in table

    def test_get_fx_table_auto_cache_fallback_on_failure(self):
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "ecb_fx.json")
            old_data = {"asof": "2020-01-01T00:00:00", "table": {"USD": 148.0}}
            with open(cache_file, "w") as f:
                json.dump(old_data, f)
            with patch("app.utils.fx_utils._cache_paths", return_value=(td, cache_file)):
                with patch("app.utils.fx_utils._ecb_fetch_xml", side_effect=Exception("Network error")):
                    table, meta = get_fx_table_jpy(manual_table=None, auto=True, ttl_hours=0)
                    assert meta["source"] == "ecb(cache-fallback)"
                    assert table["USD"] == 148.0


# ==========================================
# pricing_calculator
# ==========================================
class TestGetExchangeRate:
    @patch("app.utils.pricing_calculator.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"rates": {"JPY": 149.5}}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        assert get_exchange_rate("USD", "JPY") == 149.5

    @patch("app.utils.pricing_calculator.requests.get")
    def test_api_failure_returns_default(self, mock_get):
        mock_get.side_effect = Exception("Network Error")
        assert get_exchange_rate("USD", "JPY") == DEFAULT_EXCHANGE_RATE

    @patch("app.utils.pricing_calculator.requests.get")
    def test_missing_currency_returns_default(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"rates": {}}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        assert get_exchange_rate("USD", "XYZ") == DEFAULT_EXCHANGE_RATE


class TestCalculateProfit:
    @patch("app.utils.pricing_calculator.get_exchange_rate", return_value=1.0)
    def test_positive_profit(self, mock_rate):
        result = calculate_profit(
            buyma_price=20000,
            source_price=10000,
            shipping_cost=2000,
            source_currency="JPY",
        )
        assert result["profit_estimate"] > 0
        assert result["profit_rate"] > 0

    @patch("app.utils.pricing_calculator.get_exchange_rate", return_value=1.0)
    def test_loss(self, mock_rate):
        result = calculate_profit(
            buyma_price=5000,
            source_price=10000,
            shipping_cost=2000,
            source_currency="JPY",
        )
        assert result["profit_estimate"] < 0
        assert result["profit_rate"] < 0

    @patch("app.utils.pricing_calculator.get_exchange_rate", return_value=1.0)
    def test_zero_buyma_price(self, mock_rate):
        result = calculate_profit(
            buyma_price=0,
            source_price=5000,
            shipping_cost=1000,
            source_currency="JPY",
        )
        assert result["profit_rate"] == 0

    @patch("app.utils.pricing_calculator.get_exchange_rate", return_value=150.0)
    def test_usd_conversion(self, mock_rate):
        result = calculate_profit(
            buyma_price=30000,
            source_price=100.0,
            shipping_cost=10.0,
            source_currency="USD",
        )
        mock_rate.assert_called_once_with("USD", "JPY")
        # source_price_jpy = 100*150 = 15000, shipping_jpy = 10*150 = 1500
        # cost_before_customs = 16500, customs = 1650, total_cost = 18150
        # commission = 30000*0.073 = 2190, fees = 2190+200 = 2390
        # net_revenue = 30000-2390 = 27610
        # profit = 27610-18150 = 9460
        assert result["profit_estimate"] == pytest.approx(9460.0, abs=1.0)

    @patch("app.utils.pricing_calculator.get_exchange_rate", return_value=1.0)
    def test_custom_rates(self, mock_rate):
        result = calculate_profit(
            buyma_price=10000,
            source_price=5000,
            shipping_cost=0,
            customs_duty_rate=0.0,
            buyma_commission_rate=0.0,
            buyma_system_fee=0,
            source_currency="JPY",
        )
        # No fees: profit = 10000 - 5000 = 5000
        assert result["profit_estimate"] == 5000.0
        assert result["profit_rate"] == 50.0
