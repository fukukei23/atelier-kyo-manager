from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String

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
    scraped_at = db.Column(DateTime, default=datetime.utcnow)
    created_at = db.Column(DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    @classmethod
    def get_by_brand(cls, brand: str) -> list[BrandPrice]:
        return db.session.scalars(
            db.select(cls).filter(cls.brand == brand).order_by(cls.price_jpy)
        ).all()

    @classmethod
    def get_latest_by_brand_site(cls, brand: str, site: str) -> BrandPrice | None:
        return db.session.scalar(
            db.select(cls)
            .filter(cls.brand == brand, cls.source_site == site)
            .order_by(cls.scraped_at.desc())
            .limit(1)
        )

    def __repr__(self) -> str:
        return f"<BrandPrice brand={self.brand!r} site={self.source_site!r} price={self.price_jpy}>"
