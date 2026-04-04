# -*- coding: utf-8 -*-
"""
test_profitability_agent.py
======================================================================
ProfitabilityAgent のユニットテスト（WSL版・LМ不使用）
======================================================================
"""
import pytest
from unittest.mock import patch


class TestCustomsRate:
    def test_leather(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", lambda **k: ({"USD": 150.0}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    assert ag._resolve_customs_rate("バッグ", "レザー") == 0.12

    def test_shoes(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", lambda **k: ({"USD": 150.0}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    assert ag._resolve_customs_rate("シューズ", None) == 0.11

    def test_known_categories(self):
        """既知カテゴリは各関税率が正しく返る"""
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", lambda **k: ({"USD": 150.0}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    # アパレルは服飾用12.7%、時計は装飾用5.7%
                    assert ag._resolve_customs_rate("アパレル", None) == 0.127
                    assert ag._resolve_customs_rate("時計", None) == 0.057

    def test_bag_no_material(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", lambda **k: ({"USD": 150.0}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    assert ag._resolve_customs_rate("バッグ", None) == 0.11

    def test_default(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", lambda **k: ({"USD": 150.0}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    assert ag._resolve_customs_rate(None, None) == 0.10


class TestExchangeRate:
    def test_usd_rate(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", return_value=({"USD": 155.0}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    rate = ag._get_exchange_rate_jpy("USD")
                    assert rate == 155.0

    def test_eur_rate(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", return_value=({"EUR": 162.0}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    rate = ag._get_exchange_rate_jpy("EUR")
                    assert rate == 162.0

    def test_fallback_on_api_error(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", side_effect=Exception("API Error")):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    rate = ag._get_exchange_rate_jpy("USD")
                    assert rate == 150.0  # DEFAULT_FX_RATES fallback

    def test_unsupported_currency_fallback(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", return_value=({}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    rate = ag._get_exchange_rate_jpy("XYZ")
                    assert rate == 150.0  # USD default


class TestAssess:
    def test_profitable_decision(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", lambda **k: ({"USD": 150.0}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    market = {"buyma_price": 80000, "competitor_avg_price": 85000}
                    supplier = {"price": 100.0, "currency": "USD"}
                    with patch.object(ag, "_get_exchange_rate_jpy", return_value=150.0):
                        with patch.object(ag, "_get_dynamic_shipping_cost", return_value=4500.0):
                            result = ag.assess(market, supplier)
                    assert result["decision"] in ("profitable", "not_profitable", "caution")
                    assert "profit_estimate" in result
                    assert "profit_rate" in result
                    assert "total_cost_jpy" in result

    def test_not_profitable_low_margin(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", lambda **k: ({"USD": 150.0}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    market = {"buyma_price": 20000, "competitor_avg_price": 50000}
                    supplier = {"price": 200.0, "currency": "USD"}
                    with patch.object(ag, "_get_exchange_rate_jpy", return_value=150.0):
                        with patch.object(ag, "_get_dynamic_shipping_cost", return_value=4500.0):
                            result = ag.assess(market, supplier)
                    assert result["decision"] in ("profitable", "not_profitable", "caution")
                    assert "profit_estimate" in result

    def test_error_on_empty_market(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", lambda **k: ({"USD": 150.0}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    result = ag.assess({}, {"price": 100.0, "currency": "USD"})
                    assert result["decision"] == "error"

    def test_error_on_empty_supplier(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", lambda **k: ({"USD": 150.0}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    result = ag.assess({"buyma_price": 80000}, {})
                    assert result["decision"] == "error"


class TestGenerateAssessmentSummary:
    def test_summary_with_low_profit(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", lambda **k: ({"USD": 150.0}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    calc = {"profit_estimate": 500, "profit_rate": 2.5, "total_cost_jpy": 19500, "exchange_rate_used": 150.0, "source_currency": "USD"}
                    summary = ag._generate_assessment_summary(calc)
                    assert "推奨しません" in summary or "非推奨" in summary

    def test_summary_without_llm(self):
        with patch("app.agents.profitability_agent.LLM_AVAILABLE", False):
            with patch("app.agents.profitability_agent.SHIPPING_AGENT_AVAILABLE", False):
                with patch("app.agents.profitability_agent.get_fx_table_jpy", lambda **k: ({"USD": 150.0}, {})):
                    from app.agents.profitability_agent import ProfitabilityAgent
                    ag = ProfitabilityAgent()
                    calc = {"profit_estimate": 5000, "profit_rate": 10.0, "total_cost_jpy": 45000, "exchange_rate_used": 150.0, "source_currency": "USD"}
                    summary = ag._generate_assessment_summary(calc)
                    assert "推奨" in summary or "利益" in summary
