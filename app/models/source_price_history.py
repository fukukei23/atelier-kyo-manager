from __future__ import annotations

from sqlalchemy import Boolean, DateTime, Float, Integer
from sqlalchemy.schema import Index

from app.core.timezone import _utcnow
from app.extensions import db


class SourcePriceHistory(db.Model):
    """監視対象商品の価格履歴（時系列データ）。"""

    __tablename__ = "source_price_history"

    id = db.Column(Integer, primary_key=True)
    monitor_id = db.Column(
        Integer, db.ForeignKey("price_monitor.id"), nullable=False
    )
    source_price = db.Column(Float, nullable=False)
    source_price_jpy = db.Column(Float, nullable=False)
    exchange_rate = db.Column(Float, nullable=False)
    in_stock = db.Column(Boolean, default=True)
    is_profitable = db.Column(Boolean, default=True)
    profit = db.Column(Float, nullable=True)
    profit_rate = db.Column(Float, nullable=True)
    price_changed = db.Column(Boolean, default=False)
    recorded_at = db.Column(DateTime, default=_utcnow)

    monitor = db.relationship(
        "PriceMonitor",
        backref=db.backref("price_history", lazy=True),
    )

    __table_args__ = (
        Index(
            "ix_source_price_history_monitor_time",
            "monitor_id",
            "recorded_at",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<SourcePriceHistory monitor={self.monitor_id} "
            f"price={self.source_price_jpy}>"
        )
