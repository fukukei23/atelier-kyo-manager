# ======================================================================
# File: product.py
# Registry: app/models/product.py
# Date & Time (JST): 2026-04-05
# Version: 2.0J (BUYMA拡張フィールド追加)
# Purpose: Product SQLAlchemy model
# ======================================================================

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text

from app.config.constants import (
    DOMESTIC_COMMISSION_RATE,
    OVERSEAS_COMMISSION_RATE,
    TRANSFER_FEE,
)
from app.core.pricing import PricingInput, calculate_pricing
from app.core.pricing.schemas import nz
from app.extensions import db

# ブランド階層判定キーワード
_HIGH_BRAND_KEYWORDS = (
    "chanel",
    "hermes",
    "hermès",
    "louis vuitton",
    "vuitton",
    "celine",
    "céline",
    "bottega veneta",
    "valentino",
    "fendi",
    "dior",
    "givenchy",
    "balenciaga",
)
_MEDIUM_BRAND_KEYWORDS = (
    "gucci",
    "prada",
    "saint laurent",
    "yves saint laurent",
    "miumiu",
    "miumiù",
    "loewe",
    "burberry",
    "moncler",
    "nike",
    "jordan",
    "adidas",
    "new balance",
    "cartier",
    "tiffany",
    "bulgari",
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
    brand_tier = db.Column(String(16), nullable=True)  # high/medium/low
    source_region = db.Column(String(64), nullable=True)  # 仕入地域
    source_type = db.Column(String(16), nullable=True)  # domestic/overseas
    color = db.Column(String(64), nullable=True)
    size = db.Column(String(64), nullable=True)
    material = db.Column(String(128), nullable=True)
    description = db.Column(Text, nullable=True)  # 説明文
    retail_price = db.Column(Float, nullable=True)  # 定価
    target_profit_rate = db.Column(Float, default=0.10)  # 目標利益率
    listing_status = db.Column(String(32), default="draft")  # draft/listed/sold/archived

    # --- FR-005 実ベース利益計算フィールド ---
    warehouse_shipping_cost = db.Column(Float, default=0.0)  # 転送倉庫送料（海外→転送倉庫）
    original_currency = db.Column(String(8), default="JPY")  # 仕入れ通貨
    exchange_rate = db.Column(Float, default=1.0)  # 適用為替レート
    item_category = db.Column(String(64), nullable=True)  # 品目カテゴリ（関税率自動決定用）

    # --- FR-002/003 パイプライン ---
    pipeline_status = db.Column(String(16), default="pending")  # pending/running/success/partial/failed
    pipeline_error = db.Column(Text, nullable=True)
    processed_images = db.Column(Text, nullable=True)  # JSON: ["path1.png", ...]
    pipeline_run_at = db.Column(DateTime, nullable=True)

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
        """成約手数料率"""
        if self.source_type == "overseas":
            return OVERSEAS_COMMISSION_RATE
        return DOMESTIC_COMMISSION_RATE

    def commission_fee(self) -> float:
        """成約手数料額"""
        return float((self.selling_price or 0) * self.commission_rate())

    def transfer_fee(self) -> float:
        """振込手数料"""
        return TRANSFER_FEE

    def calculate_profit(self) -> float:
        """利益計算 — calculator.py に委譲（Thin Wrapper）"""
        result = self._calculate_pricing()
        return result.profit

    def profit_rate(self) -> float:
        """利益率%"""
        result = self._calculate_pricing()
        return result.profit_rate * 100

    def _calculate_pricing(self):
        """価格計算の内部メソッド"""

        inp = PricingInput(
            purchase_price=nz(self.purchase_price),
            selling_price=nz(self.selling_price),
            transaction_fee=nz(self.transaction_fee),
            shipping_cost=nz(self.shipping_cost),
            customs_duty=nz(self.customs_duty),
            procurement_fee=nz(self.procurement_fee),
            warehouse_shipping_cost=nz(self.warehouse_shipping_cost),
            original_currency=self.original_currency or "JPY",
            exchange_rate=nz(self.exchange_rate) or 1.0,
            item_category=self.item_category or "",
            item_material=self.material or "",
        )
        return calculate_pricing(inp, source_type=self.source_type or "domestic")

    def recommended_selling_price(self) -> float | None:
        """目標利益率を満たす推奨売価"""

        cost = nz(self.purchase_price) + nz(self.customs_duty) + nz(self.shipping_cost) + nz(self.procurement_fee)
        if cost <= 0:
            return None
        rate = self.commission_rate()
        target = float(self.target_profit_rate or 0.10)
        # price - price*rate - 220 = cost + (cost * target_margin)
        # price * (1 - rate) = cost * (1 + target) + 220
        price = (cost * (1 + target) + 220) / (1 - rate)
        return round(price, 0)

    # --- FR-007 出品候補リスト ---
    @classmethod
    def listing_candidates(cls, min_profit_rate: float = 10.0) -> list[Product]:
        """出品候補リストを取得

        フィルタ: 未出品 + 在庫あり + 利益率 >= min_profit_rate%
        ソート: brand_tier(high→low) → profit_rate降順
        """
        candidates = db.session.scalars(
            db.select(cls).filter(cls.listing_status != "listed").filter(cls.stock_status)
        ).all()

        tier_order = {"high": 0, "medium": 1, "low": 2}
        sorted_candidates = sorted(candidates, key=lambda p: (tier_order.get(p.brand_tier, 3), -p.profit_rate()))

        return [p for p in sorted_candidates if p.profit_rate() >= min_profit_rate]

    def __repr__(self) -> str:
        return f"<Product id={self.id} name={self.name!r}>"
