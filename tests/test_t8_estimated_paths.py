"""ISSUE-101/102 Phase2 T8 — ③④経路の BuymaPrice.method 判定と ESTIMATED 経路。

spec 3.6: BUYMA価格の取得方法（BROWSER/MARKUP/MANUAL/API）で機械判定。
MARKUP（cheapest × markup_rate 推定）→ ESTIMATED / BROWSER → BROWSER_VERIFIED。
"""

from __future__ import annotations

from app.core.pricing.schemas import price_source_from_method
from app.services.brand_price_service import add_profit_calculation as calculate_profit
from app.services.price_comparison_service import ProductComparison, SourceQuote, calculate_comparison_profit


def _quote(cost: float = 20000.0) -> SourceQuote:
    return SourceQuote(
        brand_price_id=1,
        source_site="yoox",
        source_url="https://www.yoox.com/item",
        product_name="Test Item",
        price_original=130.0,
        currency="EUR",
        price_jpy=cost,
        exchange_rate=155.0,
        in_stock=True,
        match_score=0.9,
        is_sale=True,
        actual_purchase_jpy=None,
        actual_purchase_source=None,
        scraped_at="2026-08-14T00:00:00",
    )


def _comparison(**overrides) -> ProductComparison:
    base = {
        "query": "test item",
        "brand": "SSENSE",
        "category": "accessory",
        "quotes": [_quote()],
    }
    base.update(overrides)
    return ProductComparison(**base)


class TestPriceComparisonPath:
    """③ price_comparison_service.calculate_comparison_profit。"""

    def test_markup_estimate_is_estimated(self):
        """buyma_price 無し（マークアップ推定）→ profit_status=estimated。"""
        result = calculate_comparison_profit(_comparison())
        entry = result["per_source"]["yoox"]
        assert entry["profit"] is not None
        assert entry["profit_status"] == "estimated"

    def test_browser_verified_buyma_price_is_confirmed(self):
        """スクレイパ取得の buyma_price（method=BROWSER）→ confirmed。"""
        comp = _comparison(buyma_price=45000.0, buyma_price_method="BROWSER")
        result = calculate_comparison_profit(comp)
        entry = result["per_source"]["yoox"]
        assert entry["profit"] is not None
        assert entry["profit_status"] == "confirmed"

    def test_manual_buyma_price_is_confirmed(self):
        """手動入力の buyma_price（method=MANUAL）→ confirmed。"""
        comp = _comparison(buyma_price=45000.0, buyma_price_method="MANUAL")
        result = calculate_comparison_profit(comp)
        assert result["per_source"]["yoox"]["profit_status"] == "confirmed"


class TestBrandPricePath:
    """④ brand_price_service.calculate_profit。"""

    def test_markup_fallback_is_estimated(self):
        """buyma_price 無し（マークアップ推定）→ profit_status=estimated。"""
        items = [{"product_name": "T", "cheapest_jpy": 20000.0}]
        result = calculate_profit(items, category="accessory")
        assert result[0]["profit"] is not None
        assert result[0]["profit_status"] == "estimated"

    def test_manual_buyma_price_is_confirmed(self):
        """手動入力の buyma_price（method=manual）→ confirmed。"""
        items = [
            {
                "product_name": "T",
                "cheapest_jpy": 20000.0,
                "buyma_price": 45000.0,
                "buyma_price_method": "manual",
            }
        ]
        result = calculate_profit(items, category="accessory")
        assert result[0]["profit_status"] == "confirmed"

    def test_auto_calculated_is_estimated(self):
        """auto_calculated（マークアップ）→ estimated。"""
        items = [
            {
                "product_name": "T",
                "cheapest_jpy": 20000.0,
                "buyma_price": 30000.0,
                "buyma_price_method": "auto_calculated",
            }
        ]
        result = calculate_profit(items, category="accessory")
        assert result[0]["profit_status"] == "estimated"


class TestMethodMapping:
    """spec 3.6 の method → PriceSource 変換。"""

    def test_known_methods(self):
        assert price_source_from_method("BROWSER") is not None
        assert price_source_from_method("MARKUP").value == "estimated"
        assert price_source_from_method("MANUAL").value == "manual_input"
        assert price_source_from_method("API").value == "api_verified"

    def test_unknown_is_rejected_source(self):
        """未知の method は UNKNOWN（信頼不可・計算で拒否される安全側）。"""
        assert price_source_from_method("guess").value == "unknown"
