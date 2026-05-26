from __future__ import annotations

from app.core.timezone import _utcnow
from app.extensions import db


class StockCheck(db.Model):
    """F10: 在庫＆価格チェック"""

    __tablename__ = "stock_check"

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"), nullable=False)
    source_url = db.Column(db.String(512), comment="仕入れ元URL")
    current_price = db.Column(db.Float, comment="現在価格")
    previous_price = db.Column(db.Float, comment="前回価格")
    in_stock = db.Column(db.Boolean, default=False, comment="在庫あり")
    checked_at = db.Column(db.DateTime, comment="チェック日時")
    price_changed = db.Column(db.Boolean, default=False, comment="価格変動あり")
    stock_changed = db.Column(db.Boolean, default=False, comment="在庫変動あり")
    notes = db.Column(db.Text, nullable=True)
    previous_in_stock = db.Column(db.Boolean, nullable=True, comment="前回在庫あり")
    created_at = db.Column(db.DateTime, default=_utcnow)
    updated_at = db.Column(db.DateTime, default=_utcnow, onupdate=_utcnow)

    product = db.relationship("Product", backref=db.backref("stock_checks", lazy=True))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "product_id": self.product_id,
            "source_url": self.source_url,
            "current_price": self.current_price,
            "previous_price": self.previous_price,
            "price_changed": self.price_changed,
            "in_stock": self.in_stock,
            "stock_changed": self.stock_changed,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
            "notes": self.notes,
        }

    def price_diff(self) -> float | None:
        if self.current_price is not None and self.previous_price is not None:
            return self.current_price - self.previous_price
        return None

    def price_diff_pct(self) -> float | None:
        if self.current_price is not None and self.previous_price is not None and self.previous_price != 0:
            return ((self.current_price - self.previous_price) / self.previous_price) * 100
        return None

