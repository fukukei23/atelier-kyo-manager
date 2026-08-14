# ======================================================================
# File: order.py
# Purpose: Order model - F05 18日ルールダッシュボード
# ======================================================================

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text

from app.config.constants import (
    EXPECTED_PAYMENT_DAYS,
    PAYMENT_METHOD_EXTENSION_DAYS,
)
from app.core.pricing import PricingInput, calculate_pricing
from app.core.pricing.schemas import PriceSource, nz
from app.core.timezone import _utcnow
from app.extensions import db
from app.models.enums import OrderStatus


class Order(db.Model):
    __tablename__ = "order"

    id = db.Column(Integer, primary_key=True)
    order_number = db.Column(String(64), unique=True, nullable=False, comment="BUYMA注文番号")
    product_name = db.Column(String(255), nullable=False)
    customer_name = db.Column(String(128), nullable=True)

    # 日付
    order_date = db.Column(DateTime, nullable=False, default=_utcnow, index=True, comment="注文日")
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
    status = db.Column(
        String(32), default=OrderStatus.PENDING, index=True, comment="pending/shipped/completed/cancelled"
    )

    # 価格出所（ISSUE-101/102 Phase2 T5: α source select + β 参照URL）
    # 値は PriceSource 値（manual_input/browser_verified/api_verified）+ 推定は "estimated"
    purchase_price_source = db.Column(
        String(32),
        nullable=False,
        default="manual_input",
        server_default="manual_input",
        comment="仕入価格の出所（manual_input/browser_verified/api_verified/estimated）",
    )
    selling_price_source = db.Column(
        String(32),
        nullable=False,
        default="manual_input",
        server_default="manual_input",
        comment="販売価格の出所（manual_input/browser_verified/api_verified/estimated）",
    )
    purchase_price_ref_url = db.Column(String(512), nullable=True, comment="仕入価格の参照URL（β・客観証拠）")
    selling_price_ref_url = db.Column(String(512), nullable=True, comment="販売価格の参照URL（β・客観証拠）")
    profit_status = db.Column(
        String(16),
        nullable=False,
        default="confirmed",
        server_default="confirmed",
        index=True,
        comment="利益計算の確度（confirmed/estimated・estimated は発注ブロック）",
    )

    # 紐付け
    product_id = db.Column(Integer, db.ForeignKey("product.id"), nullable=True)
    partner_id = db.Column(Integer, db.ForeignKey("partner.id"), nullable=True, comment="担当パートナーID")

    # 延長申請
    extension_requested = db.Column(Boolean, default=False, comment="延長申請済み")
    extension_reason = db.Column(Text, nullable=True, comment="延長理由")

    notes = db.Column(Text, nullable=True)
    created_at = db.Column(DateTime, default=_utcnow)
    updated_at = db.Column(DateTime, default=_utcnow, onupdate=_utcnow)

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
        """利益・手数料を自動計算（出所は source カラムから明示・ISSUE-101/102 Phase2 T6）"""

        inp = PricingInput(
            purchase_price=nz(self.purchase_cost),
            selling_price=nz(self.selling_price),
            customs_duty=nz(self.customs_duty),
            purchase_price_source=PriceSource(self.purchase_price_source or "manual_input"),
            selling_price_source=PriceSource(self.selling_price_source or "manual_input"),
        )
        # 推定（estimated）は計算するが profit_status=estimated で記録（T9 で発注ブロック）
        result = calculate_pricing(
            inp,
            source_type=self.source_type or "domestic",
            allow_estimated=True,
        )
        self.fees = result.total_cost - nz(self.purchase_cost) - nz(self.customs_duty)
        self.profit = result.profit
        self.profit_rate = result.profit_rate * 100
        self.profit_status = "estimated" if result.estimate_mark else "confirmed"

    # 発注確定以降の状態（推定注文はここへの遷移をブロック・T9）
    FULFILLMENT_STATUSES = ("shipped", "completed")

    def can_advance_to(self, new_status: str) -> bool:
        """ステータス遷移可否（T9: ESTIMATED 注文は発注確定以降へ遷移不可）。

        推定で計算された注文はドラフト保持。実価格 source へ更新して
        calc_profit() を再実行すれば confirmed となり遷移可能になる。
        """
        return not (self.profit_status == "estimated" and new_status in self.FULFILLMENT_STATUSES)

    def remaining_days(self) -> int | None:
        """18日ルールの残日数"""
        if not self.deadline_18:
            return None
        delta = self.deadline_18.date() - _utcnow().date()
        return delta.days

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
