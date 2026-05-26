from __future__ import annotations

from sqlalchemy import DateTime, Float, Integer, String

from app.core.timezone import _utcnow
from app.extensions import db


class BuymaPriceHistory(db.Model):
    __tablename__ = "buyma_price_history"

    id = db.Column(Integer, primary_key=True)
    brand_price_id = db.Column(Integer, db.ForeignKey("brand_price.id"), nullable=False)
    buyma_price = db.Column(Float, nullable=False)
    buyma_price_min = db.Column(Float, nullable=True)
    buyma_price_max = db.Column(Float, nullable=True)
    buyma_price_avg = db.Column(Float, nullable=True)
    match_count = db.Column(Integer, nullable=True)
    match_score = db.Column(Float, nullable=True)
    buyma_matched_name = db.Column(String(255), nullable=True)
    source = db.Column(String(32), nullable=True, default="buyma_search")
    recorded_at = db.Column(DateTime, default=_utcnow)
