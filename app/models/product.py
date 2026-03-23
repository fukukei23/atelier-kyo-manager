# ======================================================================
# File: product.py
# Registry: app/models/product.py
# Date & Time (JST): 2026-03-21
# Version: 1.0J (moved from app/models.py)
# Purpose: Product SQLAlchemy model
# Note: Moved from app/models.py to avoid naming collision with package
# ======================================================================

from __future__ import annotations
from datetime import datetime

from sqlalchemy import Integer, String, Float, Boolean, DateTime
from app.extensions import db


class Product(db.Model):
    __tablename__ = "product"

    id = db.Column(Integer, primary_key=True)
    name = db.Column(String(255), nullable=False)

    # manage.html / 実DBに存在する列
    brand = db.Column(String(128), nullable=True)
    purchase_price = db.Column(Float, nullable=False)     # PRAGMA: NOT NULL
    selling_price = db.Column(Float, nullable=False)      # PRAGMA: NOT NULL
    supplier_url = db.Column(String(512), nullable=True)
    image_url = db.Column(String(512), nullable=True)
    stock_status = db.Column(Boolean, default=False, nullable=True)
    profit = db.Column(Float, nullable=True)              # 表示用に残存（計算メソッドも提供）
    transaction_fee = db.Column(Float, nullable=True)
    shipping_cost = db.Column(Float, nullable=True)
    customs_duty = db.Column(Float, nullable=True)
    procurement_fee = db.Column(Float, nullable=True)

    created_at = db.Column(DateTime, default=datetime.utcnow, nullable=True)
    updated_at = db.Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=True,
    )

    def calculate_profit(self) -> float:
        """
        利益の簡易計算:
          販売価格 - (仕入れ + 取引手数料 + 送料 + 関税 + 買付代行料)
        DBの profit カラムがあっても、表示はかさ算で返します。
        """
        nz = lambda x: float(x or 0.0)
        selling = nz(self.selling_price)
        costs = (
            nz(self.purchase_price)
            + nz(self.transaction_fee)
            + nz(self.shipping_cost)
            + nz(self.customs_duty)
            + nz(self.procurement_fee)
        )
        return float(selling - costs)

    def __repr__(self) -> str:
        return f"<Product id={self.id} name={self.name!r}>"
