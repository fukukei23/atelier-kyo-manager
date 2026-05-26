"""FR-006 18日ルール Order model テスト"""

from datetime import datetime, timedelta

from app.models.order import (
    EXPECTED_PAYMENT_DAYS,
    PAYMENT_METHOD_EXTENSION_DAYS,
    Order,
)
from app.utils.presentation import deadline_color, deadline_message

# ---- 期限計算テスト ----


def test_calc_deadlines_basic():
    """18日ルール期限の基本計算"""
    order = Order(
        order_number="T001",
        product_name="テスト商品",
        order_date=datetime(2026, 4, 1),
        payment_method="credit_card",
        source_type="domestic",
    )
    order.calc_deadlines()

    assert order.deadline_18 == datetime(2026, 4, 19)
    assert order.extension_deadline == datetime(2026, 4, 1) + timedelta(days=45)
    assert order.expected_payment_date == datetime(2026, 4, 1) + timedelta(days=15)


def test_calc_deadlines_bank_transfer():
    """銀行振込の延長期限（90日）"""
    order = Order(
        order_number="T002",
        product_name="テスト商品",
        order_date=datetime(2026, 4, 1),
        payment_method="bank_transfer",
        source_type="overseas",
    )
    order.calc_deadlines()

    assert order.extension_deadline == datetime(2026, 4, 1) + timedelta(days=90)
    assert order.expected_payment_date == datetime(2026, 4, 1) + timedelta(days=25)


def test_calc_deadlines_d_pay():
    """d払いの延長期限（25日）"""
    order = Order(
        order_number="T003",
        product_name="テスト商品",
        order_date=datetime(2026, 4, 1),
        payment_method="d_pay",
    )
    order.calc_deadlines()

    assert order.extension_deadline == datetime(2026, 4, 1) + timedelta(days=25)


# ---- アラート閾値テスト（仕様準拠4段階）----


def test_deadline_color_green():
    """5日以上 → green"""
    order = Order(
        order_number="T010",
        product_name="テスト",
        order_date=datetime.utcnow(),
    )
    order.deadline_18 = datetime.utcnow() + timedelta(days=7)
    assert deadline_color(order) == "green"


def test_deadline_color_yellow():
    """4日以内 → yellow（注意）"""
    order = Order(
        order_number="T011",
        product_name="テスト",
        order_date=datetime.utcnow(),
    )
    order.deadline_18 = datetime.utcnow() + timedelta(days=4)
    assert deadline_color(order) == "yellow"


def test_deadline_color_orange():
    """2日以内 → orange（警告）"""
    order = Order(
        order_number="T012",
        product_name="テスト",
        order_date=datetime.utcnow(),
    )
    order.deadline_18 = datetime.utcnow() + timedelta(days=2)
    assert deadline_color(order) == "orange"


def test_deadline_color_red():
    """当日 → red（危険）"""
    order = Order(
        order_number="T013",
        product_name="テスト",
        order_date=datetime.utcnow(),
    )
    order.deadline_18 = datetime.utcnow() + timedelta(days=0)
    assert deadline_color(order) == "red"


def test_deadline_color_black():
    """期限切れ → black（キャンセル対象）"""
    order = Order(
        order_number="T014",
        product_name="テスト",
        order_date=datetime.utcnow(),
    )
    order.deadline_18 = datetime.utcnow() - timedelta(days=1)
    assert deadline_color(order) == "black"


def test_deadline_color_gray_no_deadline():
    """期限なし → gray"""
    order = Order(order_number="T015", product_name="テスト")
    assert deadline_color(order) == "gray"


# ---- メッセージテスト ----


def test_deadline_message_expired():
    """期限切れメッセージ"""
    order = Order(order_number="T020", product_name="テスト")
    order.deadline_18 = datetime.utcnow() - timedelta(days=1)
    assert "期限切れ" in deadline_message(order)


def test_deadline_message_today():
    """本日期限メッセージ"""
    order = Order(order_number="T021", product_name="テスト")
    order.deadline_18 = datetime.utcnow()
    assert "本日が期限" in deadline_message(order)


def test_deadline_message_safe():
    """安全圏メッセージ（空文字）"""
    order = Order(order_number="T022", product_name="テスト")
    order.deadline_18 = datetime.utcnow() + timedelta(days=10)
    assert deadline_message(order) == ""


# ---- 利益計算テスト（calculator.py統合）----


def test_calc_profit_basic():
    """基本的な利益計算（calculator.py委譲）"""
    order = Order(
        order_number="T030",
        product_name="テスト",
        selling_price=30000,
        purchase_cost=20000,
        customs_duty=1000,
        source_type="domestic",
    )
    order.calc_profit()

    assert order.profit is not None
    assert order.profit < 30000 - 20000 - 1000  # 手数料分減少
    assert order.profit_rate > 0


def test_calc_profit_overseas():
    """海外仕入れ時の利益計算"""
    order = Order(
        order_number="T031",
        product_name="テスト",
        selling_price=50000,
        purchase_cost=30000,
        customs_duty=3000,
        source_type="overseas",
    )
    order.calc_profit()

    assert order.profit is not None
    assert order.profit > 0


# ---- 延長申請テスト ----


def test_extension_defaults():
    """延長フィールドのデフォルト値"""
    order = Order(order_number="T040", product_name="テスト")
    # SQLiteではBooleanのdefaultが未設定時Noneになる
    assert order.extension_requested in (False, None)
    assert order.extension_reason is None


# ---- 決済方法マッピングテスト ----


def test_payment_method_extension_days():
    """決済方法別延長日数の妥当性"""
    assert PAYMENT_METHOD_EXTENSION_DAYS["credit_card"] == 45
    assert PAYMENT_METHOD_EXTENSION_DAYS["rakuten_pay"] == 45
    assert PAYMENT_METHOD_EXTENSION_DAYS["d_pay"] == 25
    assert PAYMENT_METHOD_EXTENSION_DAYS["au_pay"] == 25
    assert PAYMENT_METHOD_EXTENSION_DAYS["paidy"] == 25
    assert PAYMENT_METHOD_EXTENSION_DAYS["bank_transfer"] == 90
    assert PAYMENT_METHOD_EXTENSION_DAYS["convenience"] == 90
    assert PAYMENT_METHOD_EXTENSION_DAYS["paypay"] == 90
    assert PAYMENT_METHOD_EXTENSION_DAYS["amazon_pay"] == 90


def test_expected_payment_days():
    """仕入区分別着金日数"""
    assert EXPECTED_PAYMENT_DAYS["domestic"] == 15
    assert EXPECTED_PAYMENT_DAYS["overseas"] == 25
