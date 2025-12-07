"""
利益計算ロジックのテスト
"""
from app.core.pricing.schemas import PricingInput
from app.core.pricing.calculator import calculate_pricing
from app.core.pricing.rules import PricingConfig


def test_calculate_pricing_basic():
    """基本的な利益計算のテスト"""
    cfg = PricingConfig(
        buyma_platform_fee_rate=0.077,     # 7.7% (プラットフォーム手数料)
        buyma_effective_fee_rate=0.10,     # 10% (テスト用)
        additional_fee_rate=0.02,          # 2%
    )
    inp = PricingInput(
        purchase_price=20000,
        selling_price=30000,
        transaction_fee=0,
        shipping_cost=2000,
        customs_duty=1000,
        procurement_fee=500,
    )

    res = calculate_pricing(inp, cfg)

    # 手計算:
    # buyma_fee = 30000 * 0.10 = 3000
    # additional = 30000 * 0.02 = 600
    # total_cost = 20000 + 2000 + 1000 + 500 + 0 + 3000 + 600 = 27100
    # profit = 30000 - 27100 = 2900
    assert res.revenue == 30000
    assert res.total_cost == 27100
    assert res.profit == 2900
    assert round(res.profit_rate, 4) == round(2900/30000, 4)


def test_calculate_pricing_zero_selling_price():
    """販売価格がゼロの場合のテスト（利益率がゼロになるべき）"""
    cfg = PricingConfig(
        buyma_platform_fee_rate=0.077,
        buyma_effective_fee_rate=0.10,
        additional_fee_rate=0.02,
    )
    inp = PricingInput(
        purchase_price=20000,
        selling_price=0,
        transaction_fee=0,
        shipping_cost=2000,
        customs_duty=1000,
        procurement_fee=500,
    )

    res = calculate_pricing(inp, cfg)

    # selling_price が 0 なので revenue は 0
    # total_cost は 20000 + 2000 + 1000 + 500 = 23500
    # profit は 0 - 23500 = -23500
    # profit_rate は 0 (ゼロ除算回避)
    assert res.revenue == 0
    assert res.total_cost == 23500
    assert res.profit == -23500
    assert res.profit_rate == 0.0


def test_calculate_pricing_default_config():
    """デフォルト設定（BUYMA手数料 14.2%）でのテスト"""
    inp = PricingInput(
        purchase_price=10000,
        selling_price=20000,
        transaction_fee=100,
        shipping_cost=1500,
        customs_duty=800,
        procurement_fee=300,
    )

    res = calculate_pricing(inp)  # config は None → デフォルト使用

    # buyma_fee = 20000 * 0.142 = 2840
    # additional_fee = 20000 * 0.0 = 0
    # total_cost = 10000 + 1500 + 800 + 300 + 100 + 2840 + 0 = 15540
    # profit = 20000 - 15540 = 4460
    # profit_rate = 4460 / 20000 = 0.223
    assert res.revenue == 20000
    assert res.total_cost == 15540
    assert res.profit == 4460
    assert round(res.profit_rate, 3) == 0.223


def test_calculate_pricing_no_fees():
    """手数料がゼロの場合のテスト"""
    cfg = PricingConfig(
        buyma_platform_fee_rate=0.0,
        buyma_effective_fee_rate=0.0,
        additional_fee_rate=0.0,
    )
    inp = PricingInput(
        purchase_price=5000,
        selling_price=10000,
        transaction_fee=0,
        shipping_cost=0,
        customs_duty=0,
        procurement_fee=0,
    )

    res = calculate_pricing(inp, cfg)

    # total_cost = 5000
    # profit = 10000 - 5000 = 5000
    # profit_rate = 5000 / 10000 = 0.5
    assert res.revenue == 10000
    assert res.total_cost == 5000
    assert res.profit == 5000
    assert res.profit_rate == 0.5


def test_calculate_pricing_negative_profit():
    """赤字（負の利益）のテスト"""
    cfg = PricingConfig(
        buyma_platform_fee_rate=0.077,
        buyma_effective_fee_rate=0.15,
        additional_fee_rate=0.03,
    )
    inp = PricingInput(
        purchase_price=25000,
        selling_price=20000,
        transaction_fee=500,
        shipping_cost=3000,
        customs_duty=2000,
        procurement_fee=1000,
    )

    res = calculate_pricing(inp, cfg)

    # buyma_fee = 20000 * 0.15 = 3000
    # additional_fee = 20000 * 0.03 = 600
    # total_cost = 25000 + 3000 + 2000 + 1000 + 500 + 3000 + 600 = 35100
    # profit = 20000 - 35100 = -15100
    # profit_rate = -15100 / 20000 = -0.755
    assert res.revenue == 20000
    assert res.total_cost == 35100
    assert res.profit == -15100
    assert round(res.profit_rate, 3) == -0.755


def test_calculate_pricing_rounding():
    """小数点以下の丸め処理のテスト"""
    cfg = PricingConfig(
        buyma_platform_fee_rate=0.077,
        buyma_effective_fee_rate=0.142,
        additional_fee_rate=0.0,
    )
    inp = PricingInput(
        purchase_price=10000.555,
        selling_price=20000.777,
        transaction_fee=100.333,
        shipping_cost=1500.111,
        customs_duty=800.999,
        procurement_fee=300.222,
    )

    res = calculate_pricing(inp, cfg)

    # すべての値が小数点以下2桁に丸められていることを確認
    assert isinstance(res.revenue, float)
    assert isinstance(res.total_cost, float)
    assert isinstance(res.profit, float)
    assert isinstance(res.profit_rate, float)
    
    # revenue は selling_price を丸めたもの
    assert res.revenue == 20000.78
    
    # 計算結果も丸められている
    assert len(str(res.total_cost).split('.')[-1]) <= 2
    assert len(str(res.profit).split('.')[-1]) <= 2

