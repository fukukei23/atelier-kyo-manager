"""FR-012基盤: 発送通知ステータス管理モデル"""

from __future__ import annotations

from datetime import datetime

from app.extensions import db


class ShipmentNotification(db.Model):
    __tablename__ = "shipment_notification"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    order_id = db.Column(db.Integer, db.ForeignKey("order.id"), nullable=False, comment="注文ID")
    tracking_number = db.Column(db.String(128), nullable=True, comment="追跡番号")
    warehouse = db.Column(db.String(64), nullable=True, comment="転送倉庫: stackry/shipito/buyandship")
    carrier = db.Column(db.String(64), nullable=True, comment="配送業者: fedex/dhl/ups等")
    status = db.Column(db.String(32), default="pending", comment="pending/notified/confirmed/error")
    notification_method = db.Column(db.String(32), default="manual", comment="manual/api/rpa")
    error_message = db.Column(db.Text, nullable=True, comment="エラーメッセージ")
    notified_at = db.Column(db.DateTime, nullable=True, comment="BUYMA通知日時")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    _STATUS_LABELS = {
        "pending": "未通知",
        "notified": "通知済み",
        "confirmed": "確認済み",
        "error": "エラー",
    }

    def status_label(self) -> str:
        return self._STATUS_LABELS.get(self.status, self.status)

    def mark_notified(self) -> None:
        self.status = "notified"
        self.notified_at = datetime.utcnow()

    def mark_error(self, msg: str) -> None:
        self.status = "error"
        self.error_message = msg

    def can_notify(self) -> bool:
        return self.status in ("pending", "error")

    def __repr__(self) -> str:
        return f"<ShipmentNotification id={self.id} order_id={self.order_id} status={self.status!r}>"
