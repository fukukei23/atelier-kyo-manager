from __future__ import annotations

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.schema import Index

from app.core.timezone import _utcnow
from app.extensions import db


class PriceMonitor(db.Model):
    """SSENSE等のセール品を監視し、価格変動を追跡するモデル。"""

    __tablename__ = "price_monitor"

    id = db.Column(Integer, primary_key=True)
    brand_price_id = db.Column(Integer, db.ForeignKey("brand_price.id"), nullable=True)
    # 非正規化（BrandPriceは再スクレイプで上書きされるため保持）
    source_url = db.Column(String(1024), nullable=False)
    brand = db.Column(String(128), nullable=False)
    product_name = db.Column(String(255), nullable=False)
    category = db.Column(String(32), nullable=False, default="accessory")
    currency = db.Column(String(8), nullable=False, default="USD")

    # 最新の仕入れ価格
    current_source_price = db.Column(Float, nullable=True)
    current_source_price_jpy = db.Column(Float, nullable=True)

    # BUYMA販売価格（ユーザーが設定）
    buyma_selling_price = db.Column(Float, nullable=False)

    # 利益判定
    is_profitable = db.Column(Boolean, default=True)
    latest_profit = db.Column(Float, nullable=True)
    latest_profit_rate = db.Column(Float, nullable=True)

    # 監視制御
    is_active = db.Column(Boolean, default=True, nullable=False)
    check_interval_hours = db.Column(Integer, default=4, nullable=False)
    last_checked_at = db.Column(DateTime, nullable=True)
    last_price_changed_at = db.Column(DateTime, nullable=True)
    alert_sent_at = db.Column(DateTime, nullable=True)

    # タイムスタンプ
    created_at = db.Column(DateTime, default=_utcnow)
    updated_at = db.Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # リレーション
    brand_price = db.relationship("BrandPrice", backref=db.backref("price_monitor", uselist=False))

    __table_args__ = (Index("ix_price_monitor_active", "is_active", "last_checked_at"),)

    def __repr__(self) -> str:
        return f"<PriceMonitor brand={self.brand!r} profitable={self.is_profitable}>"
