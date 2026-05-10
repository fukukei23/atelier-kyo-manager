"""
Order モデルのデッドライン計算テスト — 18日ルールのビジネスロジック検証
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.models.order import Order


class TestCalcDeadlines:
    def test_18_day_rule(self):
        """注文日 + 18日 = deadline_18"""
        order = Order(order_date=datetime(2026, 5, 1))
        order.calc_deadlines()
        assert order.deadline_18 == datetime(2026, 5, 19)

    def test_extension_deadline_default(self):
        """支払い方法なし → デフォルト45日"""
        order = Order(order_date=datetime(2026, 5, 1), payment_method=None)
        order.calc_deadlines()
        assert order.extension_deadline == datetime(2026, 5, 1) + timedelta(days=45)

    def test_extension_deadline_bank_transfer(self):
        """銀行振込 → 90日"""
        order = Order(order_date=datetime(2026, 5, 1), payment_method="bank_transfer")
        order.calc_deadlines()
        assert order.extension_deadline == datetime(2026, 5, 1) + timedelta(days=90)

    def test_extension_deadline_d_pay(self):
        """d払い → 25日"""
        order = Order(order_date=datetime(2026, 5, 1), payment_method="d_pay")
        order.calc_deadlines()
        assert order.extension_deadline == datetime(2026, 5, 1) + timedelta(days=25)

    def test_expected_payment_domestic(self):
        """国内仕入れ → 入金予定15日後"""
        order = Order(order_date=datetime(2026, 5, 1), source_type="domestic")
        order.calc_deadlines()
        assert order.expected_payment_date == datetime(2026, 5, 16)

    def test_expected_payment_overseas(self):
        """海外仕入れ → 入金予定25日後"""
        order = Order(order_date=datetime(2026, 5, 1), source_type="overseas")
        order.calc_deadlines()
        assert order.expected_payment_date == datetime(2026, 5, 26)

    def test_no_order_date_does_nothing(self):
        """order_date なし → 各フィールドは未設定"""
        order = Order(order_date=None)
        order.calc_deadlines()
        assert order.deadline_18 is None
        assert order.extension_deadline is None
        assert order.expected_payment_date is None


class TestRemainingDays:
    def test_future_deadline(self):
        """未来の期限 → 正の値"""
        future = datetime.utcnow() + timedelta(days=5)
        order = Order()
        order.deadline_18 = future
        assert order.remaining_days() == 5

    def test_past_deadline(self):
        """過去の期限 → 負の値"""
        past = datetime.utcnow() - timedelta(days=3)
        order = Order()
        order.deadline_18 = past
        assert order.remaining_days() == -3

    def test_today_deadline(self):
        """今日が期限 → 0"""
        order = Order()
        order.deadline_18 = datetime.utcnow()
        assert order.remaining_days() == 0

    def test_no_deadline(self):
        """deadline_18 なし → None"""
        order = Order()
        order.deadline_18 = None
        assert order.remaining_days() is None


class TestDeadlineColor:
    def test_black_expired(self):
        """期限切れ → black"""
        order = Order()
        order.deadline_18 = datetime.utcnow() - timedelta(days=1)
        assert order.deadline_color() == "black"

    def test_red_today(self):
        """当日 → red"""
        order = Order()
        order.deadline_18 = datetime.utcnow()
        assert order.deadline_color() == "red"

    def test_orange_1_day(self):
        """残り1日 → orange"""
        order = Order()
        order.deadline_18 = datetime.utcnow() + timedelta(days=1)
        assert order.deadline_color() == "orange"

    def test_orange_2_days(self):
        """残り2日 → orange"""
        order = Order()
        order.deadline_18 = datetime.utcnow() + timedelta(days=2)
        assert order.deadline_color() == "orange"

    def test_yellow_3_days(self):
        """残り3日 → yellow"""
        order = Order()
        order.deadline_18 = datetime.utcnow() + timedelta(days=3)
        assert order.deadline_color() == "yellow"

    def test_yellow_4_days(self):
        """残り4日 → yellow"""
        order = Order()
        order.deadline_18 = datetime.utcnow() + timedelta(days=4)
        assert order.deadline_color() == "yellow"

    def test_green_5_days(self):
        """残り5日 → green"""
        order = Order()
        order.deadline_18 = datetime.utcnow() + timedelta(days=5)
        assert order.deadline_color() == "green"

    def test_green_10_days(self):
        """残り10日 → green"""
        order = Order()
        order.deadline_18 = datetime.utcnow() + timedelta(days=10)
        assert order.deadline_color() == "green"

    def test_gray_no_deadline(self):
        """期限なし → gray"""
        order = Order()
        assert order.deadline_color() == "gray"


class TestDeadlineMessage:
    def test_expired(self):
        """期限切れ"""
        order = Order()
        order.deadline_18 = datetime.utcnow() - timedelta(days=1)
        assert "期限切れ" in order.deadline_message()

    def test_today(self):
        """当日"""
        order = Order()
        order.deadline_18 = datetime.utcnow()
        msg = order.deadline_message()
        assert "本日" in msg
        assert "期限" in msg

    def test_1_day_remaining(self):
        """残り1日"""
        order = Order()
        order.deadline_18 = datetime.utcnow() + timedelta(days=1)
        msg = order.deadline_message()
        assert "わずか" in msg

    def test_3_days_remaining(self):
        """残り3日"""
        order = Order()
        order.deadline_18 = datetime.utcnow() + timedelta(days=3)
        msg = order.deadline_message()
        assert "近づ" in msg

    def test_5_days_no_message(self):
        """残り5日 → メッセージなし"""
        order = Order()
        order.deadline_18 = datetime.utcnow() + timedelta(days=5)
        assert order.deadline_message() == ""

    def test_no_deadline(self):
        """期限なし → 空文字"""
        order = Order()
        assert order.deadline_message() == ""


class TestGetExtensionDays:
    def test_known_methods(self):
        assert Order.get_extension_days("credit_card") == 45
        assert Order.get_extension_days("bank_transfer") == 90
        assert Order.get_extension_days("d_pay") == 25
        assert Order.get_extension_days("paypay") == 90

    def test_unknown_method_default(self):
        assert Order.get_extension_days("unknown_method") == 45

    def test_supported_methods_not_empty(self):
        methods = Order.supported_payment_methods()
        assert len(methods) > 0
        assert "credit_card" in methods
        assert "bank_transfer" in methods
