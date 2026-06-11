"""価格データ信頼性チェック — 推測データでの利益計算を防止するテスト。"""

from __future__ import annotations

import pytest

from app.core.pricing.schemas import (
    PriceData,
    PriceIntegrityError,
    PriceSource,
    PricingInput,
)
from app.core.pricing.calculator import calculate_pricing


# =====================================================================
# PriceData — 信頼性判定
# =====================================================================
class TestPriceDataReliability:
    def test_browser_verified_is_reliable(self):
        pd = PriceData(price=290.0, source=PriceSource.BROWSER_VERIFIED)
        assert pd.is_reliable() is True

    def test_manual_input_is_reliable(self):
        pd = PriceData(price=45600.0, source=PriceSource.MANUAL_INPUT)
        assert pd.is_reliable() is True

    def test_api_verified_is_reliable(self):
        pd = PriceData(price=35000.0, source=PriceSource.API_VERIFIED)
        assert pd.is_reliable() is True

    def test_keyword_guess_is_not_reliable(self):
        pd = PriceData(price=71000.0, source=PriceSource.KEYWORD_GUESS)
        assert pd.is_reliable() is False

    def test_unknown_is_not_reliable(self):
        pd = PriceData(price=50000.0, source=PriceSource.UNKNOWN)
        assert pd.is_reliable() is False


# =====================================================================
# PricingInput — validate_sources
# =====================================================================
class TestValidateSources:
    def test_both_verified_passes(self):
        inp = PricingInput(
            purchase_price=290.0,
            selling_price=50000.0,
            purchase_price_source=PriceSource.BROWSER_VERIFIED,
            selling_price_source=PriceSource.BROWSER_VERIFIED,
        )
        inp.validate_sources()  # should not raise

    def test_purchase_keyword_guess_raises(self):
        inp = PricingInput(
            purchase_price=290.0,
            selling_price=50000.0,
            purchase_price_source=PriceSource.KEYWORD_GUESS,
            selling_price_source=PriceSource.BROWSER_VERIFIED,
        )
        with pytest.raises(PriceIntegrityError, match="仕入価格"):
            inp.validate_sources()

    def test_selling_unknown_raises(self):
        inp = PricingInput(
            purchase_price=290.0,
            selling_price=50000.0,
            purchase_price_source=PriceSource.BROWSER_VERIFIED,
            selling_price_source=PriceSource.UNKNOWN,
        )
        with pytest.raises(PriceIntegrityError, match="販売価格"):
            inp.validate_sources()

    def test_both_unreliable_raises_both_errors(self):
        inp = PricingInput(
            purchase_price=290.0,
            selling_price=71000.0,
            purchase_price_source=PriceSource.KEYWORD_GUESS,
            selling_price_source=PriceSource.KEYWORD_GUESS,
        )
        with pytest.raises(PriceIntegrityError) as exc_info:
            inp.validate_sources()
        msg = str(exc_info.value)
        assert "仕入価格" in msg
        assert "販売価格" in msg

    def test_default_sources_are_unknown(self):
        inp = PricingInput(purchase_price=100.0, selling_price=200.0)
        assert inp.purchase_price_source == PriceSource.UNKNOWN
        assert inp.selling_price_source == PriceSource.UNKNOWN


# =====================================================================
# calculate_pricing — ソース検証の統合テスト
# =====================================================================
class TestCalculatorSourceValidation:
    def test_unreliable_source_blocked(self):
        """推測データはデフォルトで計算をブロックする。"""
        inp = PricingInput(
            purchase_price=290.0,
            selling_price=71000.0,
            purchase_price_source=PriceSource.KEYWORD_GUESS,
            selling_price_source=PriceSource.KEYWORD_GUESS,
        )
        with pytest.raises(PriceIntegrityError):
            calculate_pricing(inp)

    def test_verified_source_allowed(self):
        """信頼できるデータソースなら正常に計算できる。"""
        inp = PricingInput(
            purchase_price=290.0,
            selling_price=50000.0,
            purchase_price_source=PriceSource.BROWSER_VERIFIED,
            selling_price_source=PriceSource.BROWSER_VERIFIED,
        )
        result = calculate_pricing(inp)
        assert result.profit != 0 or result.revenue == result.total_cost

    def test_skip_validation_backward_compat(self):
        """skip_source_validation=True で旧来の動作を維持（レガシー用）。"""
        inp = PricingInput(
            purchase_price=290.0,
            selling_price=71000.0,
            purchase_price_source=PriceSource.KEYWORD_GUESS,
            selling_price_source=PriceSource.KEYWORD_GUESS,
        )
        result = calculate_pricing(inp, skip_source_validation=True)
        assert result.revenue == 71000.0
