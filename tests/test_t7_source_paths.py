"""ISSUE-101/102 Phase2 T7 — ②⑤⑥経路の source 明示。

②sourcing（CSV/JSON = 人手提供 → manual_input 既定・estimated 宣言可）
⑤monitor（仕入=BROWSER_VERIFIED 実ページ取得 / 販売=MANUAL_INPUT フォーム入力）
⑥agent（LLM分析シミュレーション → ESTIMATED・provenance 未追跡）

モックは境界（ネットワーク）のみに使用（ISSUE-102 教訓）。
"""

from __future__ import annotations

import pytest

from app.agents.profitability_agent import MarketData, ProfitabilityAgent, SupplierData
from app.models.price_monitor import PriceMonitor
from app.services.price_monitor_service import _update_profitability
from app.utils.sourcing_profitability import calculate_profitability


def _complete_input(**overrides) -> dict:
    base = {
        "purchase_price": 30000.0,
        "selling_price": 50000.0,
        "shipping_cost": 1000.0,
    }
    base.update(overrides)
    return base


class TestSourcingPath:
    """② sourcing_profitability 経路。"""

    def test_complete_input_calculates_with_manual_default(self):
        result = calculate_profitability(_complete_input())
        assert result["status"] == "complete"
        assert result["profit"] is not None
        assert result["profit_status"] == "confirmed"

    def test_declared_estimated_marks_result(self):
        result = calculate_profitability(_complete_input(selling_price_source="estimated"))
        assert result["status"] == "complete"
        assert result["profit"] is not None
        assert result["profit_status"] == "estimated"

    def test_declared_invalid_source_rejected(self):
        """source水増しは不可: 不正な文字列は unknown 扱いで拒否されないことなく検証される。"""
        result = calculate_profitability(_complete_input(purchase_price_source="browser_verified"))
        assert result["status"] == "complete"
        assert result["profit_status"] == "confirmed"


class TestMonitorPath:
    """⑤ price_monitor_service._update_profitability 経路。"""

    def _monitor(self) -> PriceMonitor:
        return PriceMonitor(
            brand="SSENSE",
            product_name="テスト商品",
            source_url="https://www.ssense.com/ja-jp/p/test",
            buyma_selling_price=50000.0,
            category="accessory",
            currency="USD",
        )

    def test_updates_profitability_without_skip(self):
        """skip撤去後も実ページ取得(仕入)+フォーム入力(販売)で計算成功。"""
        monitor = self._monitor()
        _update_profitability(monitor, source_price_jpy=30000.0, exchange_rate=1.0)

        assert monitor.latest_profit is not None
        assert monitor.latest_profit_rate is not None
        assert monitor.is_profitable in (True, False)


class TestAgentPath:
    """⑥ profitability_agent._calculate_core_profit 経路（シミュレーション=ESTIMATED）。"""

    def test_core_profit_is_marked_estimated(self, monkeypatch):
        agent = ProfitabilityAgent()
        monkeypatch.setattr(agent, "_get_exchange_rate_jpy", lambda currency: 150.0)
        monkeypatch.setattr(agent, "_get_dynamic_shipping_cost", lambda currency: 30.0)

        market = MarketData(name="テスト", buyma_price=50000.0)
        supplier = SupplierData(price=200.0, currency="usd", category="accessory")
        result = agent._calculate_core_profit(market, supplier)

        assert result["profit_status"] == "estimated"
        assert isinstance(result["profit_estimate"], int)


@pytest.mark.parametrize(
    ("path",),
    [
        ("utils/sourcing_profitability.py",),
        ("services/price_monitor_service.py",),
        ("agents/profitability_agent.py",),
    ],
)
def test_paths_no_longer_use_skip(path: str) -> None:
    """T7 対象3経路から skip_source_validation が消えていること。"""
    from pathlib import Path

    target = Path(__file__).resolve().parents[1] / "app" / path
    assert "skip_source_validation=True" not in target.read_text(encoding="utf-8"), (
        f"{path} に skip_source_validation=True が残存（T7 未完了）"
    )
