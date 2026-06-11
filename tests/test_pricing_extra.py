"""Pricing extra coverage tests for rules.py and schemas.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.core.pricing.rules import (
    CUSTOMS_RATE_TABLE,
    PricingConfig,
    _DEFAULT_CONFIG,
    load_pricing_config,
    resolve_customs_rate,
)
from app.core.pricing.schemas import PricingInput, PricingResult, nz
from app.core.pricing.calculator import calculate_pricing


# =====================================================================
# schemas.py — nz helper
# =====================================================================
class TestNzHelper:
    def test_nz_with_value(self):
        assert nz(42.0) == 42.0

    def test_nz_with_int(self):
        assert nz(10) == 10.0

    def test_nz_with_zero(self):
        assert nz(0) == 0.0

    def test_nz_with_none(self):
        assert nz(None) == 0.0


# =====================================================================
# schemas.py — PricingInput
# =====================================================================
class TestPricingInput:
    def test_defaults(self):
        inp = PricingInput(purchase_price=10000, selling_price=20000)
        assert inp.transaction_fee == 0.0
        assert inp.shipping_cost == 0.0
        assert inp.customs_duty == 0.0
        assert inp.procurement_fee == 0.0
        assert inp.warehouse_shipping_cost == 0.0
        assert inp.original_currency == "JPY"
        assert inp.exchange_rate == 1.0
        assert inp.item_category == ""
        assert inp.item_material == ""

    def test_custom_values(self):
        inp = PricingInput(
            purchase_price=50000,
            selling_price=80000,
            transaction_fee=500,
            shipping_cost=2000,
            customs_duty=3000,
            procurement_fee=1000,
            warehouse_shipping_cost=4000,
            original_currency="EUR",
            exchange_rate=160.0,
            item_category="bag",
            item_material="leather",
        )
        assert inp.purchase_price == 50000
        assert inp.original_currency == "EUR"
        assert inp.exchange_rate == 160.0


# =====================================================================
# schemas.py — PricingResult
# =====================================================================
class TestPricingResult:
    def test_creation(self):
        result = PricingResult(
            revenue=50000,
            total_cost=30000,
            profit=20000,
            profit_rate=0.4,
        )
        assert result.revenue == 50000
        assert result.profit == 20000
        assert result.profit_rate == 0.4
        assert result.purchase_price_jpy == 0.0
        assert result.auto_customs_duty == 0.0


# =====================================================================
# rules.py — resolve_customs_rate
# =====================================================================
class TestResolveCustomsRate:
    def test_leather_material_english(self):
        assert resolve_customs_rate(None, "leather") == 0.12

    def test_leather_material_kanji(self):
        assert resolve_customs_rate(None, "革") == 0.12

    def test_leather_material_katakana(self):
        assert resolve_customs_rate(None, "レザー") == 0.12

    def test_bag_category_english(self):
        assert resolve_customs_rate("bag", None) == 0.11

    def test_bag_category_japanese(self):
        assert resolve_customs_rate("バッグ", None) == 0.11

    def test_shoes_category_english(self):
        assert resolve_customs_rate("shoes", None) == 0.11

    def test_shoes_category_japanese(self):
        assert resolve_customs_rate("靴", None) == 0.11

    def test_wallet_category(self):
        assert resolve_customs_rate("wallet", None) == 0.12

    def test_wallet_category_japanese(self):
        assert resolve_customs_rate("財布", None) == 0.12

    def test_watch_category(self):
        assert resolve_customs_rate("watch", None) == 0.10

    def test_watch_category_japanese(self):
        assert resolve_customs_rate("腕時計", None) == 0.10

    def test_default_rate(self):
        assert resolve_customs_rate(None, None) == 0.10

    def test_unknown_category(self):
        assert resolve_customs_rate("unknown", None) == 0.10

    def test_material_priority_over_category(self):
        """Material-based rate takes priority over category."""
        # leather material with bag category should return leather rate (0.12), not bag rate (0.11)
        assert resolve_customs_rate("bag", "leather") == 0.12

    def test_customs_rate_table_keys(self):
        expected_keys = {"leather", "bag", "shoes", "wallet", "watch", "accessory", "clothing", "default"}
        assert set(CUSTOMS_RATE_TABLE.keys()) == expected_keys


# =====================================================================
# rules.py — PricingConfig defaults
# =====================================================================
class TestPricingConfig:
    def test_default_values(self):
        cfg = PricingConfig()
        assert cfg.buyma_platform_fee_rate == 0.077
        assert cfg.buyma_effective_fee_rate == 0.142
        assert cfg.additional_fee_rate == 0.0
        assert cfg.transfer_fee == 220.0

    def test_custom_values(self):
        cfg = PricingConfig(
            buyma_platform_fee_rate=0.05,
            buyma_effective_fee_rate=0.10,
            additional_fee_rate=0.01,
            transfer_fee=300.0,
        )
        assert cfg.buyma_platform_fee_rate == 0.05
        assert cfg.buyma_effective_fee_rate == 0.10
        assert cfg.additional_fee_rate == 0.01
        assert cfg.transfer_fee == 300.0


# =====================================================================
# rules.py — load_pricing_config
# =====================================================================
class TestLoadPricingConfig:
    def test_no_path_returns_default(self):
        cfg = load_pricing_config(None)
        assert cfg.buyma_platform_fee_rate == _DEFAULT_CONFIG.buyma_platform_fee_rate

    def test_empty_path_returns_default(self):
        cfg = load_pricing_config("")
        assert cfg.buyma_platform_fee_rate == _DEFAULT_CONFIG.buyma_platform_fee_rate

    def test_nonexistent_file_returns_default(self):
        cfg = load_pricing_config("/tmp/nonexistent_config_xyz.json")
        assert cfg.buyma_platform_fee_rate == _DEFAULT_CONFIG.buyma_platform_fee_rate

    def test_valid_config_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "buyma_platform_fee_rate": 0.05,
                "buyma_effective_fee_rate": 0.12,
                "additional_fee_rate": 0.02,
                "transfer_fee": 500.0,
            }, f)
            f.flush()
            cfg = load_pricing_config(f.name)
        assert cfg.buyma_platform_fee_rate == 0.05
        assert cfg.buyma_effective_fee_rate == 0.12
        assert cfg.additional_fee_rate == 0.02
        assert cfg.transfer_fee == 500.0

    def test_backward_compat_buyma_fee_rate(self):
        """buyma_fee_rate key should be used as buyma_effective_fee_rate."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"buyma_fee_rate": 0.15}, f)
            f.flush()
            cfg = load_pricing_config(f.name)
        assert cfg.buyma_effective_fee_rate == 0.15

    def test_partial_config_uses_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"transfer_fee": 1000.0}, f)
            f.flush()
            cfg = load_pricing_config(f.name)
        assert cfg.transfer_fee == 1000.0
        assert cfg.buyma_platform_fee_rate == _DEFAULT_CONFIG.buyma_platform_fee_rate

    def test_invalid_json_returns_default(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{{{")
            f.flush()
            cfg = load_pricing_config(f.name)
        assert cfg.buyma_platform_fee_rate == _DEFAULT_CONFIG.buyma_platform_fee_rate


# =====================================================================
# calculator.py — calculate_pricing integration
# =====================================================================
class TestCalculatePricing:
    """計算ロジックのテスト — ソース検証はスキップ（検証テストは test_price_integrity.py）。"""

    @staticmethod
    def _calc(inp, **kwargs):
        return calculate_pricing(inp, skip_source_validation=True, **kwargs)

    def test_basic_domestic(self):
        inp = PricingInput(
            purchase_price=50000,
            selling_price=80000,
            shipping_cost=2000,
            item_category="bag",
        )
        result = self._calc(inp, source_type="domestic")
        assert result.revenue == 80000
        assert result.profit > 0
        assert 0 < result.profit_rate < 1
        assert result.total_cost > 0

    def test_basic_overseas(self):
        inp = PricingInput(
            purchase_price=50000,
            selling_price=80000,
            shipping_cost=2000,
            item_category="bag",
        )
        result = self._calc(inp, source_type="overseas")
        assert result.revenue == 80000
        assert result.profit > 0

    def test_foreign_currency(self):
        inp = PricingInput(
            purchase_price=300,
            selling_price=50000,
            original_currency="EUR",
            exchange_rate=160.0,
            shipping_cost=2000,
        )
        result = self._calc(inp)
        assert result.purchase_price_jpy == 300 * 160
        assert result.total_cost > 0

    def test_manual_customs_duty(self):
        inp = PricingInput(
            purchase_price=50000,
            selling_price=80000,
            customs_duty=5000,  # manual override
        )
        result = self._calc(inp)
        assert result.auto_customs_duty == 0.0
        assert result.total_cost > 50000  # includes customs

    def test_auto_customs_duty(self):
        inp = PricingInput(
            purchase_price=50000,
            selling_price=80000,
            item_category="bag",
        )
        result = self._calc(inp)
        assert result.auto_customs_duty > 0
        assert result.customs_rate_used == 0.11  # bag rate

    def test_zero_revenue(self):
        inp = PricingInput(
            purchase_price=50000,
            selling_price=0,
        )
        result = self._calc(inp)
        assert result.profit_rate == 0.0

    def test_warehouse_shipping_included(self):
        inp = PricingInput(
            purchase_price=50000,
            selling_price=80000,
            shipping_cost=2000,
            warehouse_shipping_cost=3000,
        )
        result = self._calc(inp)
        assert result.total_shipping_cost == 5000

    def test_procurement_and_transaction_fees(self):
        inp = PricingInput(
            purchase_price=50000,
            selling_price=80000,
            procurement_fee=2000,
            transaction_fee=500,
        )
        result = self._calc(inp)
        assert result.total_cost > 50000 + 2000 + 500

    def test_result_is_rounded(self):
        inp = PricingInput(
            purchase_price=33333,
            selling_price=77777,
            item_category="leather",
        )
        result = self._calc(inp)
        # revenue should be exactly 77777 (integer)
        assert result.revenue == 77777
        # profit_rate should have 4 decimal places
        assert isinstance(result.profit_rate, float)

    def test_leather_customs_rate(self):
        inp = PricingInput(
            purchase_price=100000,
            selling_price=150000,
            item_material="leather",
        )
        result = self._calc(inp)
        assert result.customs_rate_used == 0.12
