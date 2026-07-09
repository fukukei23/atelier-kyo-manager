from __future__ import annotations

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.schema import Index

from app.core.timezone import _utcnow
from app.extensions import db


class BrandPrice(db.Model):
    __tablename__ = "brand_price"

    id = db.Column(Integer, primary_key=True)
    product_id = db.Column(Integer, db.ForeignKey("product.id"), nullable=True)
    brand = db.Column(String(128), nullable=False)
    product_name = db.Column(String(255), nullable=False)
    source_site = db.Column(String(64), nullable=False)
    source_url = db.Column(String(1024), nullable=True)
    price_original = db.Column(Float, nullable=False)
    currency = db.Column(String(8), nullable=False)
    price_jpy = db.Column(Float, nullable=False)
    exchange_rate = db.Column(Float, nullable=False)
    in_stock = db.Column(Boolean, default=True)
    size_available = db.Column(String(255), nullable=True)
    buyma_price = db.Column(Float, nullable=True)
    buyma_price_source = db.Column(String(32), nullable=True, default="auto_calculated")
    buyma_matched_name = db.Column(String(255), nullable=True)
    buyma_match_score = db.Column(Float, nullable=True)
    buyma_searched_at = db.Column(DateTime, nullable=True)
    buyma_status = db.Column(String(16), nullable=True)
    # 実際の仕入れ価格（セール・アウトレット等の実際購入価格。未設定時はprice_jpyを使用）
    actual_purchase_jpy = db.Column(Float, nullable=True)
    actual_purchase_source = db.Column(String(32), nullable=True)  # "manual", "yoox", "ssense", etc.
    # 価格データ信頼性（PriceSource enum値を文字列で保存）
    color = db.Column(String(128), nullable=True)
    model_number = db.Column(String(64), nullable=True)
    purchase_price_source = db.Column(String(32), nullable=True, default="unknown")
    selling_price_source = db.Column(String(32), nullable=True, default="unknown")
    scraped_at = db.Column(DateTime, default=_utcnow)
    created_at = db.Column(DateTime, default=_utcnow)
    updated_at = db.Column(DateTime, default=_utcnow, onupdate=_utcnow)

    @classmethod
    def get_by_brand(cls, brand: str) -> list[BrandPrice]:
        return db.session.scalars(db.select(cls).filter(cls.brand == brand).order_by(cls.price_jpy)).all()

    @classmethod
    def get_latest_by_brand_site(cls, brand: str, site: str) -> BrandPrice | None:
        return db.session.scalar(
            db.select(cls).filter(cls.brand == brand, cls.source_site == site).order_by(cls.scraped_at.desc()).limit(1)
        )

    def __repr__(self) -> str:
        return f"<BrandPrice brand={self.brand!r} site={self.source_site!r} price={self.price_jpy}>"

    __table_args__ = (Index("ix_brand_price_brand_product", "brand", "product_name"),)
