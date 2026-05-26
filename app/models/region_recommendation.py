from __future__ import annotations

from app.core.timezone import _utcnow
from app.extensions import db


class RegionRecommendation(db.Model):
    """F12: 買付先地域最適化レコメンド"""

    __tablename__ = "region_recommendation"

    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.String(64), unique=True, nullable=False, comment="地域コード")
    region_name = db.Column(db.String(128), comment="地域名")
    total_products = db.Column(db.Integer, default=0, comment="取扱商品数")
    avg_profit_rate = db.Column(db.Float, comment="平均利益率%")
    avg_shipping_days = db.Column(db.Integer, comment="平均配送日数")
    avg_customs_rate = db.Column(db.Float, comment="平均関税率%")
    risk_score = db.Column(db.Float, comment="リスクスコア0-100")
    reliability_score = db.Column(db.Float, comment="信頼性スコア0-100")
    recommendation_score = db.Column(db.Float, comment="総合推奨スコア")
    last_updated = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=_utcnow)
    updated_at = db.Column(db.DateTime, default=_utcnow, onupdate=_utcnow)

    def calc_recommendation(self) -> float | None:
        if self.reliability_score is None or self.risk_score is None or self.avg_profit_rate is None:
            return None
        return self.reliability_score * 0.4 + (100 - self.risk_score) * 0.3 + self.avg_profit_rate * 0.3

