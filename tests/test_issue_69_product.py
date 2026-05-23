"""Issue #69: product.py カバレッジ 73→90%+

Tests for uncovered paths:
- L112-123: classify_brand_tier (high/medium/low)
- L125-126: auto_classify_tier
- L129-133: commission_rate (domestic/overseas)
- L135-137: commission_fee
- L143-151: calculate_profit / profit_rate
- L171-182: recommended_selling_price
"""

from __future__ import annotations

import pytest

from app import create_app
from app.extensions import db
from app.models.product import Product


@pytest.fixture()
def app_ctx():
    app = create_app()
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


class TestClassifyBrandTier:
    def test_none_returns_low(self):
        assert Product.classify_brand_tier(None) == "low"

    def test_empty_returns_low(self):
        assert Product.classify_brand_tier("") == "low"

    def test_high_brand_chanel(self):
        assert Product.classify_brand_tier("Chanel Classic Flap") == "high"

    def test_high_brand_hermes(self):
        assert Product.classify_brand_tier("Hermes Birkin") == "high"

    def test_high_brand_louis_vuitton(self):
        assert Product.classify_brand_tier("Louis Vuitton Neverfull") == "high"

    def test_medium_brand_gucci(self):
        assert Product.classify_brand_tier("Gucci Marmont") == "medium"

    def test_medium_brand_moncler(self):
        assert Product.classify_brand_tier("Moncler Maya") == "medium"

    def test_medium_brand_nike(self):
        assert Product.classify_brand_tier("Nike Air Jordan") == "medium"

    def test_unknown_brand_returns_low(self):
        assert Product.classify_brand_tier("Uniqlo Tee") == "low"

    def test_case_insensitive(self):
        assert Product.classify_brand_tier("CHANEL") == "high"
        assert Product.classify_brand_tier("GUCCI") == "medium"


class TestAutoClassifyTier:
    def test_sets_brand_tier(self, app_ctx):
        p = Product(name="test", purchase_price=100, selling_price=200, brand="Dior Saddle")
        p.auto_classify_tier()
        assert p.brand_tier == "high"

    def test_no_brand_sets_low(self, app_ctx):
        p = Product(name="test", purchase_price=100, selling_price=200, brand=None)
        p.auto_classify_tier()
        assert p.brand_tier == "low"


class TestCommissionRate:
    def test_domestic_rate(self, app_ctx):
        p = Product(name="test", purchase_price=100, selling_price=200, source_type="domestic")
        assert p.commission_rate() == 0.077

    def test_overseas_rate(self, app_ctx):
        p = Product(name="test", purchase_price=100, selling_price=200, source_type="overseas")
        assert p.commission_rate() == 0.055

    def test_default_is_domestic(self, app_ctx):
        p = Product(name="test", purchase_price=100, selling_price=200, source_type=None)
        assert p.commission_rate() == 0.077


class TestCommissionFee:
    def test_commission_fee_calculation(self, app_ctx):
        p = Product(name="test", purchase_price=100, selling_price=10000, source_type="domestic")
        assert p.commission_fee() == 770.0

    def test_commission_fee_zero_price(self, app_ctx):
        p = Product(name="test", purchase_price=0, selling_price=0, source_type="domestic")
        assert p.commission_fee() == 0.0


class TestCalculateProfit:
    def test_profit_calculation(self, app_ctx):
        p = Product(
            name="test",
            purchase_price=50000,
            selling_price=80000,
            transaction_fee=0,
            shipping_cost=2000,
            customs_duty=5000,
            procurement_fee=0,
            source_type="domestic",
        )
        profit = p.calculate_profit()
        assert isinstance(profit, float)

    def test_profit_rate(self, app_ctx):
        p = Product(
            name="test",
            purchase_price=50000,
            selling_price=80000,
            source_type="domestic",
        )
        rate = p.profit_rate()
        assert isinstance(rate, float)


class TestRecommendedSellingPrice:
    def test_returns_price(self, app_ctx):
        p = Product(
            name="test",
            purchase_price=50000,
            selling_price=80000,
            customs_duty=5000,
            shipping_cost=2000,
            procurement_fee=1000,
            source_type="domestic",
            target_profit_rate=0.10,
        )
        price = p.recommended_selling_price()
        assert price is not None
        assert price > 0

    def test_zero_cost_returns_none(self, app_ctx):
        p = Product(
            name="test",
            purchase_price=0,
            selling_price=0,
            customs_duty=0,
            shipping_cost=0,
            procurement_fee=0,
        )
        assert p.recommended_selling_price() is None
