"""FR-012基盤: 発送通知ステータス管理モデル"""

from __future__ import annotations

from app.core.timezone import _utcnow
from app.extensions import db
from app.models.enums import ShipmentNotificationStatus


class ShipmentNotification(db.Model):
    __tablename__ = "shipment_notification"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    order_id = db.Column(db.Integer, db.ForeignKey("order.id"), nullable=False, index=True, comment="注文ID")
    tracking_number = db.Column(db.String(128), nullable=True, comment="追跡番号")
    warehouse = db.Column(db.String(64), nullable=True, comment="転送倉庫: stackry/shipito/buyandship")
    carrier = db.Column(db.String(64), nullable=True, comment="配送業者: fedex/dhl/ups等")
    status = db.Column(db.String(32), default=ShipmentNotificationStatus.PENDING, index=True, comment="pending/notified/confirmed/error")
    notification_method = db.Column(db.String(32), default="manual", comment="manual/api/rpa")
    error_message = db.Column(db.Text, nullable=True, comment="エラーメッセージ")
    notified_at = db.Column(db.DateTime, nullable=True, comment="BUYMA通知日時")
    created_at = db.Column(db.DateTime, default=_utcnow)
    updated_at = db.Column(db.DateTime, default=_utcnow, onupdate=_utcnow)

    def mark_notified(self) -> None:
        self.status = "notified"
        self.notified_at = _utcnow()

    def mark_error(self, msg: str) -> None:
        self.status = "error"
        self.error_message = msg

    def can_notify(self) -> bool:
        return self.status in ("pending", "error")

    def __repr__(self) -> str:
        return f"<ShipmentNotification id={self.id} order_id={self.order_id} status={self.status!r}>"
