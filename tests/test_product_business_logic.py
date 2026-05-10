"""
Product モデルのビジネスロジックテスト — ブランド分類・価格計算の検証
"""

from __future__ import annotations

import pytest

from app.models.product import Product


class TestClassifyBrandTier:
    """classify_brand_tier の全パターン検証"""

    # --- high tier ---
    @pytest.mark.parametrize(
        "brand_name",
        [
            "Chanel",
            "HERMES",
            "Louis Vuitton",
            "Celine",
            "Bottega Veneta",
            "Valentino",
            "Fendi",
            "Dior",
            "Givenchy",
            "Balenciaga",
            "Chanel Classic Flap Bag",  # 部分一致
        ],
    )
    def test_high_tier_brands(self, brand_name):
        assert Product.classify_brand_tier(brand_name) == "high"

    # --- medium tier ---
    @pytest.mark.parametrize(
        "brand_name",
        [
            "Gucci",
            "PRADA",
            "Saint Laurent",
            "MiuMiu",
            "Loewe",
            "Burberry",
            "Moncler",
            "Nike",
            "Jordan",
            "Adidas",
            "New Balance",
            "Cartier",
            "Tiffany",
            "Bulgari",
            "Yves Saint Laurent",  # 別名
        ],
    )
    def test_medium_tier_brands(self, brand_name):
        assert Product.classify_brand_tier(brand_name) == "medium"

    # --- low tier ---
    @pytest.mark.parametrize(
        "brand_name",
        [
            "Uniqlo",
            "ZARA",
            "H&M",
            "COS",
            "Some Unknown Brand",
            "Theory",
        ],
    )
    def test_low_tier_brands(self, brand_name):
        assert Product.classify_brand_tier(brand_name) == "low"

    def test_none_returns_low(self):
        assert Product.classify_brand_tier(None) == "low"

    def test_empty_string_returns_low(self):
        assert Product.classify_brand_tier("") == "low"

    def test_case_insensitive(self):
        assert Product.classify_brand_tier("chanel") == "high"
        assert Product.classify_brand_tier("CHANEL") == "high"
        assert Product.classify_brand_tier("ChAnEl") == "high"

    def test_accented_characters(self):
        """アクセント付きブランド名の認識"""
        assert Product.classify_brand_tier("Hermès") == "high"
        assert Product.classify_brand_tier("Céline") == "high"
        assert Product.classify_brand_tier("Miumiù") == "medium"


class TestCommissionRate:
    def test_domestic_rate(self):
        p = Product(selling_price=10000, source_type="domestic")
        assert p.commission_rate() == 0.077

    def test_overseas_rate(self):
        p = Product(selling_price=10000, source_type="overseas")
        assert p.commission_rate() == 0.055

    def test_default_is_domestic(self):
        p = Product(selling_price=10000, source_type=None)
        assert p.commission_rate() == 0.077

    def test_commission_fee_calculation(self):
        p = Product(selling_price=30000, source_type="domestic")
        expected = 30000 * 0.077
        assert p.commission_fee() == expected

    def test_transfer_fee_constant(self):
        p = Product()
        assert p.transfer_fee() == 220.0


class TestRecommendedSellingPrice:
    def test_basic_calculation(self):
        """推奨売価 = (cost * (1 + target) + 220) / (1 - rate)"""
        p = Product(
            purchase_price=10000,
            customs_duty=1000,
            shipping_cost=500,
            procurement_fee=500,
            target_profit_rate=0.10,
            source_type="domestic",
        )
        # cost = 10000 + 1000 + 500 + 500 = 12000
        # price = (12000 * 1.10 + 220) / (1 - 0.077) = 13420 / 0.923 = 14539.76
        result = p.recommended_selling_price()
        assert result is not None
        assert result > 12000  # コストより高い

    def test_zero_cost_returns_none(self):
        """コスト0 → None"""
        p = Product(
            purchase_price=0,
            customs_duty=0,
            shipping_cost=0,
            procurement_fee=0,
        )
        assert p.recommended_selling_price() is None

    def test_overseas_lower_rate_higher_price(self):
        """海外手数料(低)の方が推奨売価が低い"""
        domestic = Product(
            purchase_price=10000,
            customs_duty=1000,
            shipping_cost=500,
            procurement_fee=500,
            target_profit_rate=0.10,
            source_type="domestic",
        )
        overseas = Product(
            purchase_price=10000,
            customs_duty=1000,
            shipping_cost=500,
            procurement_fee=500,
            target_profit_rate=0.10,
            source_type="overseas",
        )
        # 海外は手数料が低いので、同じ利益を得るのに売価が低くて済む
        assert overseas.recommended_selling_price() < domestic.recommended_selling_price()

    def test_higher_target_higher_price(self):
        """目標利益率が高いほど推奨売価も高い"""
        low_target = Product(
            purchase_price=10000,
            customs_duty=1000,
            shipping_cost=500,
            procurement_fee=500,
            target_profit_rate=0.05,
            source_type="domestic",
        )
        high_target = Product(
            purchase_price=10000,
            customs_duty=1000,
            shipping_cost=500,
            procurement_fee=500,
            target_profit_rate=0.20,
            source_type="domestic",
        )
        assert high_target.recommended_selling_price() > low_target.recommended_selling_price()

    def test_result_is_rounded(self):
        """結果は整数に丸められる"""
        p = Product(
            purchase_price=10000,
            customs_duty=1000,
            target_profit_rate=0.10,
            source_type="domestic",
        )
        result = p.recommended_selling_price()
        assert result == round(result, 0)
