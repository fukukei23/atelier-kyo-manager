import pytest

"""
利益計算ロジックのテスト
"""

from app.core.pricing.calculator import calculate_pricing
from app.core.pricing.rules import PricingConfig, resolve_customs_rate
from app.core.pricing.schemas import PriceIntegrityError, PriceSource, PricingInput


def test_effective_fee_rate_applied():
    """ISSUE-103: 実効総手数料(buyma_effective_fee_rate)が手数料として適用されること.

    経営者判断「実効総手数料14.2%」= 販売手数料+決済手数料+その他を含む総負担率。
    修正前は source_type 別の成約手数料(domestic 7.7%/overseas 5.5%)のみで、
    海外仕入れ時に利益を約8.7pt過大評価していた。
    additional_fee_rate は実効手数料に含まれるため計算では使用しない（二重計上防止）。
    """
    cfg = PricingConfig(buyma_effective_fee_rate=0.142, transfer_fee=0.0)
    inp = PricingInput(
        purchase_price=20000,
        selling_price=30000,
        transaction_fee=0,
        shipping_cost=2000,
        customs_duty=1000,
        procurement_fee=500,
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # fee = 30000 * 0.142 = 4260（成約手数料率・additional_fee_rate は使用しない）
    # total_cost = 20000 + 2000 + 1000 + 500 + 0 + 4260 + 0 = 27760
    assert res.total_cost == 27760
    assert res.profit == 2240
    assert round(res.profit_rate, 4) == round(2240 / 30000, 4)

    # source_type に依存しない（海外仕入れでも同一の手数料率）
    res_overseas = calculate_pricing(inp, cfg, source_type="overseas")
    assert res_overseas.total_cost == 27760


def test_calculate_pricing_basic():
    """基本的な利益計算のテスト"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.10,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=20000,
        selling_price=30000,
        transaction_fee=0,
        shipping_cost=2000,
        customs_duty=1000,
        procurement_fee=500,
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # fee = 30000 * 0.10 = 3000（実効手数料は一括・additional_fee_rate は二重計上防止で不使用）
    # total_cost = 20000 + 2000 + 1000 + 500 + 0 + 3000 + 0 = 26500
    # profit = 30000 - 26500 = 3500
    assert res.revenue == 30000
    assert res.total_cost == 26500
    assert res.profit == 3500
    assert round(res.profit_rate, 4) == round(3500 / 30000, 4)


def test_calculate_pricing_zero_selling_price():
    """販売価格がゼロの場合のテスト（利益率がゼロになるべき）"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.10,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=20000,
        selling_price=0,
        transaction_fee=0,
        shipping_cost=2000,
        customs_duty=1000,
        procurement_fee=500,
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # selling_price が 0 なので revenue は 0, commission も 0
    # total_cost = 20000 + 2000 + 1000 + 500 + 0 + 0 + 0 + 0 = 23500
    # profit = 0 - 23500 = -23500
    # profit_rate は 0 (ゼロ除算回避)
    assert res.revenue == 0
    assert res.total_cost == 23500
    assert res.profit == -23500
    assert res.profit_rate == 0.0


def test_calculate_pricing_default_config():
    """デフォルト設定（実効総手数料 14.2% + 振込手数料 220円）でのテスト"""
    inp = PricingInput(
        purchase_price=10000,
        selling_price=20000,
        transaction_fee=100,
        shipping_cost=1500,
        customs_duty=800,
        procurement_fee=300,
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp)  # config は None → デフォルト使用

    # fee = 20000 * 0.142 = 2840
    # transfer_fee = 220
    # total_cost = 10000 + 1500 + 800 + 300 + 100 + 2840 + 220 = 15760
    # profit = 20000 - 15760 = 4240
    # profit_rate = 4240 / 20000 = 0.212
    assert res.revenue == 20000
    assert res.total_cost == 15760
    assert res.profit == 4240
    assert round(res.profit_rate, 3) == 0.212


def test_calculate_pricing_no_fees():
    """手数料がゼロの場合のテスト"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.0,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=5000,
        selling_price=10000,
        transaction_fee=0,
        shipping_cost=0,
        customs_duty=0,
        procurement_fee=0,
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # customs_duty=0 → 自動計算, でも category/material 空 → default 10%
    # auto_customs_duty = (5000 + 0) * 0.10 = 500
    # total_cost = 5000 + 0 + 500 + 0 + 0 + 0 + 0 + 0 = 5500
    # profit = 10000 - 5500 = 4500
    assert res.revenue == 10000
    assert res.total_cost == 5500
    assert res.profit == 4500
    assert res.profit_rate == 0.45


def test_calculate_pricing_negative_profit():
    """赤字（負の利益）のテスト"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.15,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=25000,
        selling_price=20000,
        transaction_fee=500,
        shipping_cost=3000,
        customs_duty=2000,
        procurement_fee=1000,
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # fee = 20000 * 0.15 = 3000
    # total_cost = 25000 + 3000 + 2000 + 1000 + 500(transaction) + 3000 + 0 = 34500
    # profit = 20000 - 34500 = -14500
    assert res.revenue == 20000
    assert res.total_cost == 34500
    assert res.profit == -14500
    assert round(res.profit_rate, 3) == -0.725


def test_calculate_pricing_rounding():
    """小数点以下の丸め処理のテスト"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.142,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=10000.555,
        selling_price=20000.777,
        transaction_fee=100.333,
        shipping_cost=1500.111,
        customs_duty=800.999,
        procurement_fee=300.222,
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    assert isinstance(res.revenue, float)
    assert isinstance(res.total_cost, float)
    assert isinstance(res.profit, float)
    assert isinstance(res.profit_rate, float)

    assert res.revenue == 20000.78
    assert len(str(res.total_cost).split(".")[-1]) <= 2
    assert len(str(res.profit).split(".")[-1]) <= 2


# --- FR-005 新テスト ---


def test_exchange_rate_conversion():
    """為替変換テスト（USD→JPY）"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.0,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=100,  # 100 USD
        selling_price=20000,  # 20000 JPY
        customs_duty=0,  # 自動計算
        exchange_rate=150.0,  # 1 USD = 150 JPY
        original_currency="USD",
        item_category="accessory",  # 10%
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # purchase_price_jpy = 100 * 150 = 15000
    # auto_customs_duty = (15000 + 0) * 0.10 = 1500
    # total_cost = 15000 + 0 + 1500 + 0 + 0 + 0 + 0 + 0 = 16500
    # profit = 20000 - 16500 = 3500
    assert res.purchase_price_jpy == 15000.0
    assert res.profit == 3500.0


def test_warehouse_shipping_cost():
    """転送倉庫送料合算テスト"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.0,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=10000,
        selling_price=20000,
        customs_duty=1000,
        shipping_cost=1500,  # 国内送料
        warehouse_shipping_cost=3000,  # 転送倉庫送料
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # total_shipping_cost = 1500 + 3000 = 4500
    # total_cost = 10000 + 4500 + 1000 + 0 + 0 + 0 + 0 + 0 = 15500
    # profit = 20000 - 15500 = 4500
    assert res.total_shipping_cost == 4500.0
    assert res.profit == 4500.0


def test_auto_customs_bag():
    """関税自動計算テスト（bag=11%）"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.0,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=50000,
        selling_price=80000,
        customs_duty=0,  # 0 → 自動計算
        item_category="bag",
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # customs_rate = 0.11 (bag)
    # auto_customs_duty = (50000 + 0) * 0.11 = 5500
    # total_cost = 50000 + 0 + 5500 + 0 + 0 + 0 + 0 + 0 = 55500
    # profit = 80000 - 55500 = 24500
    assert res.customs_rate_used == 0.11
    assert res.auto_customs_duty == 5500.0
    assert res.profit == 24500.0


def test_manual_customs_override():
    """関税手動入力優先テスト"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.0,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=50000,
        selling_price=80000,
        customs_duty=3000,  # 手動入力 → 自動計算を上書き
        item_category="bag",  # 自動なら11%だが手動が優先
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # customs_duty = 3000 (手動優先)
    # customs_rate_used = 0 (手動時は記録しない)
    # total_cost = 50000 + 0 + 3000 + 0 + 0 + 0 + 0 + 0 = 53000
    assert res.customs_rate_used == 0.0
    assert res.auto_customs_duty == 0.0
    assert res.total_cost == 53000.0
    assert res.profit == 27000.0


def test_overseas_source_type():
    """海外仕入れ時のテスト（手数料は実効総手数料で source_type 非依存・ISSUE-103）"""
    cfg = PricingConfig(
        transfer_fee=220.0,
    )
    inp = PricingInput(
        purchase_price=100,
        selling_price=30000,
        customs_duty=0,
        exchange_rate=150.0,
        original_currency="EUR",
        item_category="bag",
        item_material="leather",
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg, source_type="overseas")

    # purchase_price_jpy = 100 * 150 = 15000
    # auto_customs_duty = (15000 + 0) * 0.12 = 1800  (material=leather → 12%)
    # fee = 30000 * 0.142 = 4260（実効手数料は source_type 非依存）
    # total_cost = 15000 + 0 + 1800 + 0 + 0 + 4260 + 220 = 21280
    # profit = 30000 - 21280 = 8720
    assert res.purchase_price_jpy == 15000.0
    assert res.customs_rate_used == 0.12
    assert res.profit == 8720.0


def test_resolve_customs_rate():
    """関税率解決のユニットテスト"""
    assert resolve_customs_rate("bag", None) == 0.11
    assert resolve_customs_rate("shoes", None) == 0.11
    assert resolve_customs_rate("wallet", None) == 0.12
    assert resolve_customs_rate("watch", None) == 0.10
    assert resolve_customs_rate(None, "leather") == 0.12
    assert resolve_customs_rate(None, "革") == 0.12
    assert resolve_customs_rate(None, "レザー") == 0.12
    assert resolve_customs_rate(None, None) == 0.10  # default
    assert resolve_customs_rate("unknown", "cotton") == 0.10  # default


def test_backward_compatible_jpy():
    """後方互換: JPY の場合 exchange_rate を無視"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.0,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=10000,
        selling_price=20000,
        customs_duty=1000,
        original_currency="JPY",
        exchange_rate=150.0,  # JPY なので為替適用なし
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # purchase_price_jpy = 10000 (JPY の場合は為替適用なし)
    # total_cost = 10000 + 0 + 1000 + 0 + 0 + 0 + 0 + 0 = 11000
    assert res.purchase_price_jpy == 10000.0
    assert res.profit == 9000.0


# --- 境界値・エッジケーステスト ---
# （旧 test_exchange_rate_zero_skips_conversion は削除: 「レート0でUSDをそのまま円扱い」
#  する silent fallback を仕様として固定化していた ISSUE-111 のバグ擁護テスト。
#  新仕様は下方の ISSUE-111 テスト群が検証する）


def test_customs_auto_includes_warehouse_shipping():
    """関税自動計算に warehouse_shipping_cost が含まれる（ライン38の検証）"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.0,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=50000,
        selling_price=80000,
        customs_duty=0,  # 自動計算
        warehouse_shipping_cost=3000,
        item_category="bag",  # 11%
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # auto_customs_duty = (50000 + 3000) * 0.11 = 5830
    # total_shipping = 0 + 3000 = 3000
    # total_cost = 50000 + 3000 + 5830 + 0 + 0 + 0 + 0 + 0 = 58830
    assert res.auto_customs_duty == 5830.0
    assert res.total_cost == 58830.0
    assert res.profit == 21170.0


def test_domestic_vs_overseas_commission():
    """同一条件下で source_type のみ変えても手数料は同一（実効総手数料は共通・ISSUE-103）"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.142,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=30000,
        selling_price=50000,
        customs_duty=2000,
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    domestic = calculate_pricing(inp, cfg, source_type="domestic")
    overseas = calculate_pricing(inp, cfg, source_type="overseas")

    # fee = 50000 * 0.142 = 7100（source_type に依存しない）
    assert domestic.total_cost == overseas.total_cost == 30000 + 0 + 2000 + 7100
    assert domestic.profit == overseas.profit == 50000 - 39100


def test_profit_near_zero_boundary():
    """利益率がほぼ0%の境界値テスト"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.0,
        transfer_fee=0.0,
    )
    # total_cost が selling_price にほぼ一致する入力
    # total_cost = purchase_price + shipping + customs + procurement + ...
    # = 9500 + 500 + 0 + 0 + ... = 10000
    inp = PricingInput(
        purchase_price=9500,
        selling_price=10000,
        customs_duty=0,  # 自動計算: (9500+0)*0.10 = 950
        shipping_cost=500,
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # auto_customs = (9500 + 0) * 0.10 = 950
    # total_cost = 9500 + 500 + 950 + 0 + 0 + 0 + 0 + 0 = 10950
    # profit = 10000 - 10950 = -950 (赤字)
    assert res.profit < 0
    assert res.profit_rate < 0


def test_profit_exactly_zero():
    """利益がほぼゼロになる条件のテスト"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.0,
        transfer_fee=0.0,
    )
    # 全コスト = selling_price になるよう設計
    # customs自動を避け手動指定
    inp = PricingInput(
        purchase_price=8000,
        selling_price=10000,
        customs_duty=1000,  # 手動
        shipping_cost=1000,
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # total_cost = 8000 + 1000 + 1000 + 0 + 0 + 0 + 0 + 0 = 10000
    assert res.total_cost == 10000.0
    assert res.profit == 0.0
    assert res.profit_rate == 0.0


def test_material_overrides_category_for_customs():
    """素材指定がカテゴリより優先される（革素材 + bag → leather の12%が適用）"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.0,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=50000,
        selling_price=80000,
        customs_duty=0,
        item_category="bag",  # 通常は11%
        item_material="leather",  # 12%が優先
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)

    # leather優先 → 12%
    # auto_customs = (50000 + 0) * 0.12 = 6000
    assert res.customs_rate_used == 0.12
    assert res.auto_customs_duty == 6000.0


# ---- ISSUE-111: 為替レート0のサイレントフォールバック防止 ----


def test_zero_exchange_rate_foreign_currency_raises():
    """ISSUE-111: 外貨建て+exchange_rate<=0 → PriceIntegrityError（€500→¥500事故防止）"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.0,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=500,
        selling_price=100000,
        original_currency="EUR",
        exchange_rate=0.0,
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    with pytest.raises(PriceIntegrityError, match="為替"):
        calculate_pricing(inp, cfg)


def test_negative_exchange_rate_foreign_currency_raises():
    """ISSUE-111: 負の為替レートも同様に拒否"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.0,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=500,
        selling_price=100000,
        original_currency="USD",
        exchange_rate=-1.0,
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    with pytest.raises(PriceIntegrityError, match="為替"):
        calculate_pricing(inp, cfg)


def test_zero_exchange_rate_jpy_is_fine():
    """JPY建ての場合は為替レート0でも問題なし（換算しないため）"""
    cfg = PricingConfig(
        buyma_effective_fee_rate=0.0,
        transfer_fee=0.0,
    )
    inp = PricingInput(
        purchase_price=10000,
        selling_price=20000,
        customs_duty=1000,
        original_currency="JPY",
        exchange_rate=0.0,
        purchase_price_source=PriceSource.MANUAL_INPUT,
        selling_price_source=PriceSource.MANUAL_INPUT,
    )

    res = calculate_pricing(inp, cfg)
    assert res.purchase_price_jpy == 10000
