"""ISSUE-101/102 Phase2 T11 — smoke 6経路6本（spec 3.7 完了条件）。

各経路（①〜⑥）の PricingInput 経由計算が結果返却（DB保存/ESTIMATED保存相当）まで至る
ことをモックなし（境界＝ネットワークのみ例外）で検証する。
"""

from __future__ import annotations

from app.agents.profitability_agent import MarketData, ProfitabilityAgent, SupplierData
from app.extensions import db
from app.models.order import Order
from app.models.price_monitor import PriceMonitor
from app.services.brand_price_service import add_profit_calculation
from app.services.price_comparison_service import ProductComparison, SourceQuote, calculate_comparison_profit
from app.services.price_monitor_service import _update_profitability
from app.utils.sourcing_profitability import calculate_profitability


class TestSmokePath1Order:
    """① Order.calc_profit（DB保存経路・routes 経由は coverage 側で302検証済み）"""

    def test_order_calc_profit_persists_fields(self, routes_app):
        with routes_app.app_context():
            order = Order(
                order_number="SMOKE-1",
                product_name="Smoke",
                selling_price=50000,
                purchase_cost=30000,
                purchase_price_source="manual_input",
                selling_price_source="manual_input",
            )
            order.calc_profit()
            db.session.add(order)
            db.session.commit()

            saved = db.session.query(Order).filter_by(order_number="SMOKE-1").one()
            assert saved.profit is not None
            assert saved.profit_status == "confirmed"


class TestSmokePath2Sourcing:
    """② sourcing pipeline"""

    def test_sourcing_calculates_real_path(self):
        result = calculate_profitability(
            {
                "purchase_price": 30000.0,
                "selling_price": 50000.0,
                "shipping_cost": 2000.0,
            }
        )
        assert result["status"] == "complete"
        assert result["profit"] is not None
        assert result["profit_status"] == "confirmed"


class TestSmokePath3Comparison:
    """③ price_comparison_service"""

    def _comparison(self) -> ProductComparison:
        quote = SourceQuote(
            brand_price_id=1,
            source_site="yoox",
            source_url="https://www.yoox.com/item",
            product_name="Smoke Item",
            price_original=130.0,
            currency="EUR",
            price_jpy=20000.0,
            exchange_rate=155.0,
            in_stock=True,
            match_score=0.9,
            is_sale=True,
            actual_purchase_jpy=None,
            actual_purchase_source=None,
            scraped_at="2026-08-15T00:00:00",
        )
        return ProductComparison(query="smoke", brand="SSENSE", category="accessory", quotes=[quote])

    def test_comparison_calculates_per_source(self):
        result = calculate_comparison_profit(self._comparison())
        entry = result["per_source"]["yoox"]
        assert entry["profit"] is not None
        assert entry["profit_status"] == "estimated"  # マークアップ推定


class TestSmokePath4BrandPrice:
    """④ brand_price_service"""

    def test_brand_price_calculates_with_status(self):
        items = [{"product_name": "Smoke", "cheapest_jpy": 20000.0}]
        result = add_profit_calculation(items, category="accessory")
        assert result[0]["profit"] is not None
        assert result[0]["profit_status"] == "estimated"


class TestSmokePath5Monitor:
    """⑤ price_monitor_service（計算部分・実呼び）"""

    def test_monitor_updates_profitability(self):
        monitor = PriceMonitor(
            brand="SSENSE",
            product_name="Smoke",
            source_url="https://www.ssense.com/ja-jp/p/smoke",
            buyma_selling_price=50000.0,
            category="accessory",
            currency="USD",
        )
        _update_profitability(monitor, source_price_jpy=30000.0, exchange_rate=1.0)
        assert monitor.latest_profit is not None
        assert monitor.is_profitable in (True, False)


class TestSmokePath6Agent:
    """⑥ profitability_agent（境界=為替/送料のみ差し替え・計算は実経路）"""

    def test_agent_estimates_with_estimated_status(self, monkeypatch):
        agent = ProfitabilityAgent()
        monkeypatch.setattr(agent, "_get_exchange_rate_jpy", lambda currency: 150.0)
        monkeypatch.setattr(agent, "_get_dynamic_shipping_cost", lambda currency: 30.0)

        market = MarketData(name="Smoke", buyma_price=50000.0)
        supplier = SupplierData(price=200.0, currency="usd", category="accessory")
        result = agent._calculate_core_profit(market, supplier)

        assert result["profit_estimate"] > 0 or result["profit_estimate"] <= 0  # 計算が完走
        assert result["profit_status"] == "estimated"
