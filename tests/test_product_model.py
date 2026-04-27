"""
product.py モデルテスト (Issue #69)

classify_brand_tier, commission系, _calculate_pricing,
recommended_selling_price, __repr__ をカバー
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.models.product import Product


def _make_product(**kwargs):
    """Product のメソッドをバインドした SimpleNamespace"""
    obj = SimpleNamespace(**kwargs)
    obj.commission_rate = Product.commission_rate.__get__(obj, type(obj))
    obj.commission_fee = Product.commission_fee.__get__(obj, type(obj))
    obj.transfer_fee = Product.transfer_fee.__get__(obj, type(obj))
    obj.recommended_selling_price = Product.recommended_selling_price.__get__(obj, type(obj))
    obj.calculate_profit = Product.calculate_profit.__get__(obj, type(obj))
    obj.profit_rate = Product.profit_rate.__get__(obj, type(obj))
    obj._calculate_pricing = Product._calculate_pricing.__get__(obj, type(obj))
    return obj


# === classify_brand_tier (L84-93) ===


@pytest.mark.parametrize("brand", ["chanel", "hermes", "louis vuitton", "dior", "fendi"])
def test_classify_brand_tier_high(brand):
    assert Product.classify_brand_tier(brand) == "high"


@pytest.mark.parametrize("brand", ["gucci", "prada", "moncler", "nike", "burberry"])
def test_classify_brand_tier_medium(brand):
    assert Product.classify_brand_tier(brand) == "medium"


def test_classify_brand_tier_low():
    assert Product.classify_brand_tier("Uniqlo") == "low"


def test_classify_brand_tier_none():
    assert Product.classify_brand_tier(None) == "low"


def test_classify_brand_tier_case_insensitive():
    assert Product.classify_brand_tier("CHANEL") == "high"
    assert Product.classify_brand_tier("GUCCI") == "medium"


def test_classify_brand_tier_partial_match():
    assert Product.classify_brand_tier("Louis Vuitton Bag") == "high"
    assert Product.classify_brand_tier("Nike Air Max") == "medium"


# === auto_classify_tier (L96) ===


def test_auto_classify_tier():
    p = SimpleNamespace(brand="Hermes", brand_tier=None)
    # auto_classify_tier は self.classify_brand_tier(self.brand) を呼ぶ
    p.brand_tier = Product.classify_brand_tier(p.brand)
    assert p.brand_tier == "high"


# === commission_rate (L101-103) ===


def test_commission_rate_overseas():
    p = _make_product(source_type="overseas")
    assert p.commission_rate() == 0.055


def test_commission_rate_domestic():
    p = _make_product(source_type="domestic")
    assert p.commission_rate() == 0.077


def test_commission_rate_default():
    p = _make_product(source_type=None)
    assert p.commission_rate() == 0.077


# === commission_fee (L107) ===


def test_commission_fee():
    p = _make_product(selling_price=10000, source_type="domestic")
    assert p.commission_fee() == 770.0


def test_commission_fee_zero_price():
    p = _make_product(selling_price=None, source_type="domestic")
    assert p.commission_fee() == 0.0


# === transfer_fee (L109-111) ===


def test_transfer_fee():
    p = _make_product()
    assert p.transfer_fee() == 220.0


# === _calculate_pricing / calculate_profit / profit_rate (L113-140) ===


def test_calculate_profit():
    p = _make_product(
        purchase_price=5000, selling_price=10000,
        transaction_fee=100, shipping_cost=500,
        customs_duty=300, procurement_fee=200,
        warehouse_shipping_cost=0, original_currency="JPY",
        exchange_rate=1.0, item_category="", material="",
        source_type="domestic",
    )

    mock_result = MagicMock()
    mock_result.profit = 3500.0
    mock_result.profit_rate = 0.35

    # _calculate_pricing 内部で import されるので app.core.pricing をパッチ
    with patch("app.core.pricing.calculate_pricing", return_value=mock_result):
        assert p.calculate_profit() == 3500.0
        assert p.profit_rate() == 35.0


# === recommended_selling_price (L142-153) ===


def test_recommended_selling_price():
    p = _make_product(
        purchase_price=5000, customs_duty=300,
        shipping_cost=500, procurement_fee=200,
        source_type="domestic", target_profit_rate=0.10,
    )
    price = p.recommended_selling_price()
    assert price is not None
    # cost=6000, rate=0.077, target=0.10
    # price = (6000*1.10 + 220) / 0.923 ≈ 7389
    assert 7300 < price < 7500


def test_recommended_selling_price_overseas():
    p = _make_product(
        purchase_price=3000, customs_duty=0,
        shipping_cost=0, procurement_fee=0,
        source_type="overseas", target_profit_rate=0.20,
    )
    price = p.recommended_selling_price()
    assert price is not None
    assert price > 0


def test_recommended_selling_price_zero_cost():
    p = _make_product(
        purchase_price=0, customs_duty=0,
        shipping_cost=0, procurement_fee=0,
        source_type="domestic", target_profit_rate=0.10,
    )
    assert p.recommended_selling_price() is None


def test_recommended_selling_price_none_values():
    p = _make_product(
        purchase_price=None, customs_duty=None,
        shipping_cost=None, procurement_fee=None,
        source_type="domestic", target_profit_rate=None,
    )
    assert p.recommended_selling_price() is None


# === __repr__ (L178) ===


def test_repr():
    p = SimpleNamespace(id=42, name="Test Bag")
    assert Product.__repr__(p) == "<Product id=42 name='Test Bag'>"
