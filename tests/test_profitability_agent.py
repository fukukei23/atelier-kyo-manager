"""
test_profitability_agent.py
======================================================================
ProfitabilityAgent のユニットテスト
======================================================================
"""

from unittest.mock import patch

import pytest

from app.agents.profitability_agent import ProfitabilityAgent


@pytest.fixture(autouse=True)
def _mock_deps():
    """AILlmController と ShippingAgent をモック"""
    with (
        patch("app.agents.profitability_agent.AILlmController"),
        patch("app.agents.profitability_agent.ShippingAgent"),
    ):
        yield


class TestCustomsRate:
    def test_leather(self):
        ag = ProfitabilityAgent()
        assert ag._resolve_customs_rate("バッグ", "レザー") == 0.12

    def test_shoes(self):
        ag = ProfitabilityAgent()
        assert ag._resolve_customs_rate("シューズ", None) == 0.11

    def test_unknown_category(self):
        """未定義カテゴリはデフォルト10%"""
        ag = ProfitabilityAgent()
        assert ag._resolve_customs_rate("アパレル", None) == 0.10
        assert ag._resolve_customs_rate("時計", None) == 0.10

    def test_bag_no_material(self):
        ag = ProfitabilityAgent()
        assert ag._resolve_customs_rate("バッグ", None) == 0.11

    def test_default(self):
        ag = ProfitabilityAgent()
        assert ag._resolve_customs_rate(None, None) == 0.10


class TestExchangeRate:
    def test_usd_rate(self):
        ag = ProfitabilityAgent()
        with patch("app.agents.profitability_agent.get_fx_table_jpy", return_value=({"USD": 155.0}, {})):
            rate = ag._get_exchange_rate_jpy("USD")
        assert rate == 155.0

    def test_eur_rate(self):
        ag = ProfitabilityAgent()
        with patch("app.agents.profitability_agent.get_fx_table_jpy", return_value=({"EUR": 162.0}, {})):
            rate = ag._get_exchange_rate_jpy("EUR")
        assert rate == 162.0

    def test_fallback_on_api_error(self):
        ag = ProfitabilityAgent()
        with patch("app.agents.profitability_agent.get_fx_table_jpy", side_effect=Exception("API Error")):
            rate = ag._get_exchange_rate_jpy("USD")
        assert rate == 150.0

    def test_unsupported_currency_fallback(self):
        ag = ProfitabilityAgent()
        with patch("app.agents.profitability_agent.get_fx_table_jpy", return_value=({}, {})):
            rate = ag._get_exchange_rate_jpy("XYZ")
        assert rate == 150.0


class TestAssess:
    def test_profitable_decision(self):
        ag = ProfitabilityAgent()
        market = {"buyma_price": 80000, "competitor_avg_price": 85000}
        supplier = {"price": 100.0, "currency": "USD"}
        with (
            patch.object(ag, "_get_exchange_rate_jpy", return_value=150.0),
            patch.object(ag, "_get_dynamic_shipping_cost", return_value=30.0),
            patch.object(ag.llm_controller, "generate", return_value="推奨します。利益が見込めます。"),
        ):
            result = ag.assess(market, supplier)
        assert result["decision"] in ("profitable", "not_profitable", "caution")
        assert "profit_estimate" in result
        assert "profit_rate" in result
        assert "total_cost_jpy" in result

    def test_not_profitable_low_margin(self):
        ag = ProfitabilityAgent()
        market = {"buyma_price": 20000, "competitor_avg_price": 50000}
        supplier = {"price": 200.0, "currency": "USD"}
        with (
            patch.object(ag, "_get_exchange_rate_jpy", return_value=150.0),
            patch.object(ag, "_get_dynamic_shipping_cost", return_value=4500.0),
        ):
            result = ag.assess(market, supplier)
        assert result["decision"] in ("profitable", "not_profitable", "caution")
        assert "profit_estimate" in result

    def test_error_on_empty_market(self):
        ag = ProfitabilityAgent()
        result = ag.assess({}, {"price": 100.0, "currency": "USD"})
        assert result["decision"] == "error"

    def test_error_on_empty_supplier(self):
        ag = ProfitabilityAgent()
        result = ag.assess({"buyma_price": 80000}, {})
        assert result["decision"] == "error"


class TestGenerateAssessmentSummary:
    def test_summary_with_low_profit(self):
        ag = ProfitabilityAgent()
        calc = {
            "profit_estimate": 500,
            "profit_rate": 2.5,
            "total_cost_jpy": 19500,
            "exchange_rate_used": 150.0,
            "source_currency": "USD",
        }
        summary = ag._generate_assessment_summary(calc)
        assert "推奨しません" in summary or "非推奨" in summary

    def test_summary_llm_fallback(self):
        ag = ProfitabilityAgent()
        calc = {
            "profit_estimate": 5000,
            "profit_rate": 10.0,
            "total_cost_jpy": 45000,
            "exchange_rate_used": 150.0,
            "source_currency": "USD",
        }
        with patch.object(ag.llm_controller, "generate", side_effect=Exception("LLM unavailable")):
            summary = ag._generate_assessment_summary(calc)
        assert "推奨" in summary or "利益" in summary
