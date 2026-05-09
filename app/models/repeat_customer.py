from __future__ import annotations

from datetime import datetime

from app.extensions import db


class RepeatCustomer(db.Model):
    """F13: リピーター管理"""

    __tablename__ = "repeat_customer"

    id = db.Column(db.Integer, primary_key=True)
    customer_name = db.Column(db.String(128), nullable=False, comment="顧客名")
    email = db.Column(db.String(128), comment="メール")
    phone = db.Column(db.String(32), comment="電話番号")
    total_orders = db.Column(db.Integer, default=0, comment="総注文数")
    total_spent = db.Column(db.Float, default=0, comment="総購入額")
    first_order_date = db.Column(db.DateTime, comment="初回注文日")
    last_order_date = db.Column(db.DateTime, comment="最終注文日")
    avg_order_value = db.Column(db.Float, comment="平均注文額")
    segment = db.Column(db.String(32), default="new", comment="セグメント")
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def calc_segment(self) -> str:
        n = self.total_orders or 0
        if n == 0:
            return "new"
        elif n <= 2:
            return "active"
        elif n <= 5:
            return "vip"
        elif n <= 10:
            return "at_risk"
        return "churned"

    def days_since_last(self) -> int | None:
        if self.last_order_date is None:
            return None
        return (datetime.utcnow() - self.last_order_date).days

    def segment_label(self) -> str:
        labels = {
            "new": "新規",
            "active": "アクティブ",
            "vip": "VIP",
            "at_risk": "離脱危惧",
            "churned": "離脱済",
        }
        return labels.get(self.segment or "new", "不明")

    def segment_color(self) -> str:
        colors = {
            "new": "blue",
            "active": "green",
            "vip": "purple",
            "at_risk": "yellow",
            "churned": "red",
        }
        return colors.get(self.segment or "new", "gray")

    def update_avg(self) -> None:
        if self.total_orders and self.total_orders > 0:
            self.avg_order_value = (self.total_spent or 0) / self.total_orders
