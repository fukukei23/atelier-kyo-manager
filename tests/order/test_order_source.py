"""ISSUE-101/102 Phase2 T6 — Order.calc_profit の source 明示と profit_status。

モック不使用（ISSUE-102 教訓: 本番経路を直接検証）。
"""

from __future__ import annotations

import pytest

from app.core.pricing import PricingInput, calculate_pricing
from app.core.pricing.schemas import PriceIntegrityError, PriceSource
from app.models.order import Order


def _order(**overrides) -> Order:
    """calc_profit 検証用の基本 Order（引数で source 系を上書き）。"""
    base = {
        "order_number": "T-T6-001",
        "product_name": "テスト商品",
        "selling_price": 50000.0,
        "purchase_cost": 30000.0,
        "customs_duty": 0.0,
        "source_type": "domestic",
    }
    base.update(overrides)
    return Order(**base)


class TestCalcProfitSources:
    """① Order.calc_profit 経路（skip撤去・source明示）。"""

    def test_manual_input_calculates_and_confirms(self):
        """実価格確認済（manual_input）→ 計算成功・profit_status=confirmed。"""
        order = _order(purchase_price_source="manual_input", selling_price_source="manual_input")
        order.calc_profit()

        assert order.profit is not None and order.profit != 0
        assert order.fees >= 0
        assert order.profit_status == "confirmed"

    def test_defaults_to_manual_input_when_unset(self):
        """source未指定（レガシー互換）→ manual_input 扱いで計算成功。"""
        order = _order()
        order.calc_profit()

        assert order.profit_status == "confirmed"

    def test_estimated_calculates_but_marks_estimated(self):
        """推定（estimated）→ 計算は実施・profit_status=estimated。"""
        order = _order(selling_price_source="estimated", purchase_price_source="manual_input")
        order.calc_profit()

        assert order.profit is not None
        assert order.profit_status == "estimated"

    def test_keyword_guess_raises(self):
        """KEYWORD_GUESS（推測）→ calc_profit が PriceIntegrityError（鉄則・許容なし）。"""
        order = _order(purchase_price_source="keyword_guess")
        with pytest.raises(PriceIntegrityError):
            order.calc_profit()

    def test_unknown_raises(self):
        """UNKNOWN（出所不明）→ PriceIntegrityError。"""
        order = _order(selling_price_source="unknown")
        with pytest.raises(PriceIntegrityError):
            order.calc_profit()


class TestEstimatedGate:
    """ESTIMATED の許容は allow_estimated 明示時のみ（spec 3.2）。"""

    def _input(self, ssrc: PriceSource) -> PricingInput:
        return PricingInput(
            purchase_price=30000.0,
            selling_price=50000.0,
            purchase_price_source=PriceSource.MANUAL_INPUT,
            selling_price_source=ssrc,
        )

    def test_estimated_rejected_by_default(self):
        """allow_estimated 未指定 → ESTIMATED も拒否（デフォルトは鉄則厳守）。"""
        with pytest.raises(PriceIntegrityError):
            calculate_pricing(self._input(PriceSource.ESTIMATED))

    def test_estimated_allowed_with_flag_and_marked(self):
        """allow_estimated=True → 計算実施・結果に estimate_mark=True。"""
        result = calculate_pricing(self._input(PriceSource.ESTIMATED), allow_estimated=True)

        assert result.estimate_mark is True
        assert result.profit is not None

    def test_reliable_sources_not_marked(self):
        """信頼済 source のみ → estimate_mark=False。"""
        result = calculate_pricing(self._input(PriceSource.MANUAL_INPUT))

        assert result.estimate_mark is False
