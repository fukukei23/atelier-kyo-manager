# ======================================================================
# File: order.py
# Purpose: Order model - F05 18日ルールダッシュボード
# ======================================================================

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import Integer, String, Float, DateTime, Text
from app.extensions import db


# 決済方法別 延長期限マッピング（日数）
PAYMENT_METHOD_EXTENSION_DAYS: dict[str, int] = {
    "credit_card": 45,
    "rakuten_pay": 45,
    "d_pay": 25,
    "au_pay": 25,
    "paidy": 25,
    "bank_transfer": 90,
    "convenience": 90,
    "paypay": 90,
    "amazon_pay": 90,
}

# 仕入区分別 平均着金日数
EXPECTED_PAYMENT_DAYS: dict[str, int] = {
    "domestic": 15,
    "overseas": 25,
}

# BUYMA手数料率
COMMISSION_RATES: dict[str, float] = {
    "domestic": 0.077,
    "overseas": 0.055,
}

TRANSFER_FEE = 220.0  # 振込手数料（楽天銀行想定）


class Order(db.Model):
    __tablename__ = "order"

    id = db.Column(Integer, primary_key=True)
    order_number = db.Column(String(64), unique=True, nullable=False, comment="BUYMA注文番号")
    product_name = db.Column(String(255), nullable=False)
    customer_name = db.Column(String(128), nullable=True)

    # 日付
    order_date = db.Column(DateTime, nullable=False, default=datetime.utcnow, comment="注文日")
    deadline_18 = db.Column(DateTime, nullable=True, comment="18日ルール期限")
    extension_deadline = db.Column(DateTime, nullable=True, comment="決済別延長期限")
    expected_payment_date = db.Column(DateTime, nullable=True, comment="入金予定日（extension_deadlineから計算）")
    shipped_date = db.Column(DateTime, nullable=True, comment="発送日")
    completed_date = db.Column(DateTime, nullable=True, comment="完了日")

    # 金額
    selling_price = db.Column(Float, nullable=False, default=0)
    purchase_cost = db.Column(Float, nullable=False, default=0, comment="仕入原価")
    customs_duty = db.Column(Float, nullable=True, default=0, comment="関税")
    fees = db.Column(Float, nullable=True, default=0, comment="手数料合計")
    profit = db.Column(Float, nullable=True, default=0, comment="利益")
    profit_rate = db.Column(Float, nullable=True, default=0, comment="利益率%")

    # 区分
    payment_method = db.Column(String(32), nullable=True, comment="決済方法")
    source_type = db.Column(String(16), nullable=True, default="domestic", comment="domestic/overseas")
    status = db.Column(String(32), default="pending", comment="pending/shipped/completed/cancelled")

    # 紐付け
    product_id = db.Column(Integer, db.ForeignKey("product.id"), nullable=True)
    partner_id = db.Column(Integer, nullable=True, comment="担当パートナーID")

    notes = db.Column(Text, nullable=True)
    created_at = db.Column(DateTime, default=datetime.utcnow)
    updated_at = db.Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # ---- 自動計算 ----
    def calc_deadlines(self) -> None:
        """注文日から各種期限を自動計算"""
        if not self.order_date:
            return

        # 18日ルール期限
        self.deadline_18 = self.order_date + timedelta(days=18)

        # 決済別延長期限
        ext_days = PAYMENT_METHOD_EXTENSION_DAYS.get(self.payment_method or "", 45)
        self.extension_deadline = self.order_date + timedelta(days=ext_days)

        # 入金予定日
        pay_days = EXPECTED_PAYMENT_DAYS.get(self.source_type or "domestic", 15)
        self.expected_payment_date = self.order_date + timedelta(days=pay_days)

    def calc_profit(self) -> None:
        """利益・手数料を自動計算"""
        rate = COMMISSION_RATES.get(self.source_type or "domestic", 0.077)
        commission = float(self.selling_price or 0) * rate
        customs = float(self.customs_duty or 0)
        self.fees = commission + TRANSFER_FEE

        selling = float(self.selling_price or 0)
        cost = float(self.purchase_cost or 0)
        self.profit = selling - cost - self.fees - customs
        self.profit_rate = (self.profit / selling * 100) if selling > 0 else 0

    def remaining_days(self) -> int | None:
        """18日ルールの残日数"""
        if not self.deadline_18:
            return None
        delta = self.deadline_18.date() - datetime.utcnow().date()
        return delta.days

    def deadline_color(self) -> str:
        """残日数に応じた色"""
        remaining = self.remaining_days()
        if remaining is None:
            return "gray"
        if remaining <= 0:
            return "red"
        if remaining <= 2:
            return "yellow"
        if remaining <= 6:
            return "orange"
        return "green"

    @staticmethod
    def get_extension_days(payment_method: str) -> int:
        """決済方法の延長日数を取得"""
        return PAYMENT_METHOD_EXTENSION_DAYS.get(payment_method, 45)

    @staticmethod
    def supported_payment_methods() -> list[str]:
        """対応決済方法一覧"""
        return list(PAYMENT_METHOD_EXTENSION_DAYS.keys())

    def __repr__(self) -> str:
        return f"<Order id={self.id} order_number={self.order_number!r}>"
