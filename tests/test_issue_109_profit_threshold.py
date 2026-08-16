"""Issue #109: 利益閾値¥10,000の一元化

- app/config/constants.py の MIN_PROFIT_JPY / MIN_PROFIT_MARGIN_RATIO / min_acceptable_profit() が
  3箇所のハードコード（brand_price_service / price_comparison_service×2）の唯一のソースであること
- 従来挙動 max(10,000, 原価×5%) を保存すること
"""

from __future__ import annotations

import inspect

from app.config.constants import (
    MIN_PROFIT_JPY,
    MIN_PROFIT_MARGIN_RATIO,
    min_acceptable_profit,
)


class TestMinAcceptableProfit:
    def test_floor_at_10000_for_low_cost(self):
        """原価が低い場合は下限¥10,000"""
        assert min_acceptable_profit(0) == 10_000
        assert min_acceptable_profit(50_000) == 10_000  # 5% = 2,500 < 10,000

    def test_margin_dominates_for_high_cost(self):
        """原価が高い場合は原価×5%が閾値"""
        assert min_acceptable_profit(400_000) == 20_000
        assert min_acceptable_profit(1_000_000) == 50_000

    def test_boundary(self):
        """原価¥200,000で 5% = ¥10,000（ちょうど下限と一致）"""
        assert min_acceptable_profit(200_000) == 10_000

    def test_constants_values(self):
        assert MIN_PROFIT_JPY == 10_000
        assert MIN_PROFIT_MARGIN_RATIO == 0.05


class TestHardcodeEliminated:
    def test_services_use_helper_not_literal(self):
        """3箇所のハードコードがヘルパー参照に置き換わっていること"""
        import app.services.brand_price_service as bps
        import app.services.price_comparison_service as pcs

        for mod in (bps, pcs):
            src = inspect.getsource(mod)
            assert "min_acceptable_profit" in src, f"{mod.__name__} がヘルパー未使用"
            # 10_000リテラルの直接比較が残っていないこと（import行以外）
            body = "\n".join(
                ln for ln in src.splitlines() if "10_000" in ln and "import" not in ln
            )
            assert body.strip() == "", f"{mod.__name__} に10_000リテラル残存: {body}"
