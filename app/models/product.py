# ======================================================================
# File: product.py
# Registry: app/models/product.py
# Date & Time (JST): 2026-04-05
# Version: 2.0J (BUYMA拡張フィールド追加)
# Purpose: Product SQLAlchemy model
# ======================================================================

from __future__ import annotations
from datetime import datetime

from sqlalchemy import Integer, String, Float, Boolean, DateTime, Text
from app.extensions import db


# ブランド階層判定キーワード
_HIGH_BRAND_KEYWORDS = (
    "chanel", "hermes", "hermès", "louis vuitton", "vuitton",
    "celine", "céline", "bottega veneta", "valentino",
    "fendi", "dior", "givenchy", "balenciaga",
)
_MEDIUM_BRAND_KEYWORDS = (
    "gucci", "prada", "saint laurent", "yves saint laurent",
    "miumiu", "miumiù", "loewe", "burberry", "moncler",
    "nike", "jordan", "adidas", "new balance",
    "cartier", "tiffany", "bulgari",
)


class Product(db.Model):
    __tablename__ = "product"

    id = db.Column(Integer, primary_key=True)
    name = db.Column(String(255), nullable=False)

    # 既存フィールド
    brand = db.Column(String(128), nullable=True)
    purchase_price = db.Column(Float, nullable=False)
    selling_price = db.Column(Float, nullable=False)
    supplier_url = db.Column(String(512), nullable=True)
    image_url = db.Column(String(512), nullable=True)
    stock_status = db.Column(Boolean, default=False, nullable=True)
    profit = db.Column(Float, nullable=True)
    transaction_fee = db.Column(Float, nullable=True)
    shipping_cost = db.Column(Float, nullable=True)
    customs_duty = db.Column(Float, nullable=True)
    procurement_fee = db.Column(Float, nullable=True)

    # --- BUYMA拡張フィールド (F02) ---
    brand_tier = db.Column(String(16), nullable=True)          # high/medium/low
    source_region = db.Column(String(64), nullable=True)       # 仕入地域
    source_type = db.Column(String(16), nullable=True)         # domestic/overseas
    color = db.Column(String(64), nullable=True)
    size = db.Column(String(64), nullable=True)
    material = db.Column(String(128), nullable=True)
    description = db.Column(Text, nullable=True)               # 説明文
    retail_price = db.Column(Float, nullable=True)             # 定価
    target_profit_rate = db.Column(Float, default=0.10)        # 目標利益率
    listing_status = db.Column(String(32), default="draft")    # draft/listed/sold/archived

    created_at = db.Column(DateTime, default=datetime.utcnow, nullable=True)
    updated_at = db.Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=True,
    )

    # --- 自動ブランド階層判定 ---
    @staticmethod
    def classify_brand_tier(brand_name: str | None) -> str:
        if not brand_name:
            return "low"
        lower = brand_name.lower()
        for kw in _HIGH_BRAND_KEYWORDS:
            if kw in lower:
                return "high"
        for kw in _MEDIUM_BRAND_KEYWORDS:
            if kw in lower:
                return "medium"
        return "low"

    def auto_classify_tier(self) -> None:
        self.brand_tier = self.classify_brand_tier(self.brand)

    # --- BUYMA手数料計算 ---
    def commission_rate(self) -> float:
        """成約手数料率: 国内7.7%, 海外5.5%"""
        if self.source_type == "overseas":
            return 0.055
        return 0.077

    def commission_fee(self) -> float:
        """成約手数料額"""
        return float((self.selling_price or 0) * self.commission_rate())

    def transfer_fee(self) -> float:
        """振込手数料（楽天銀行想定: 220円）"""
        return 220.0

    def calculate_profit(self) -> float:
        """利益計算（BUYMA仕様）"""
        nz = lambda x: float(x or 0.0)
        selling = nz(self.selling_price)
        costs = (
            nz(self.purchase_price)
            + self.commission_fee()
            + self.transfer_fee()
            + nz(self.customs_duty)
            + nz(self.shipping_cost)
            + nz(self.procurement_fee)
        )
        return float(selling - costs)

    def profit_rate(self) -> float:
        """利益率%"""
        selling = float(self.selling_price or 0)
        if selling <= 0:
            return 0.0
        return self.calculate_profit() / selling * 100

    def recommended_selling_price(self) -> float | None:
        """目標利益率を満たす推奨売価"""
        nz = lambda x: float(x or 0.0)
        cost = nz(self.purchase_price) + nz(self.customs_duty) + nz(self.shipping_cost) + nz(self.procurement_fee)
        if cost <= 0:
            return None
        rate = self.commission_rate()
        target = float(self.target_profit_rate or 0.10)
        # price - price*rate - 220 = cost + (cost * target_margin)
        # price * (1 - rate) = cost * (1 + target) + 220
        price = (cost * (1 + target) + 220) / (1 - rate)
        return round(price, 0)

    def __repr__(self) -> str:
        return f"<Product id={self.id} name={self.name!r}>"
