from __future__ import annotations
from datetime import datetime, date
from app.extensions import db


class PopularityTracker(db.Model):
    """F11: 人気度トラッキング"""
    __tablename__ = "popularity_tracker"

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"), nullable=False)
    views = db.Column(db.Integer, default=0, comment="閲覧数")
    favorites = db.Column(db.Integer, default=0, comment="お気に入り数")
    inquiries = db.Column(db.Integer, default=0, comment="問い合わせ数")
    sold_count = db.Column(db.Integer, default=0, comment="販売数")
    tracking_date = db.Column(db.Date, comment="記録日")
    popularity_score = db.Column(db.Float, comment="人気スコア")
    rank = db.Column(db.Integer, comment="ランキング順位")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    product = db.relationship("Product", backref=db.backref("popularity_records", lazy=True))

    def calc_score(self) -> float:
        return (self.views or 0) * 0.1 + (self.favorites or 0) * 2 + \
               (self.inquiries or 0) * 3 + (self.sold_count or 0) * 5

    def score_label(self) -> str:
        s = self.popularity_score or 0
        if s >= 100:
            return "超人気"
        elif s >= 50:
            return "人気"
        elif s >= 20:
            return "普通"
        return "要注力"

    def score_color(self) -> str:
        s = self.popularity_score or 0
        if s >= 100:
            return "green"
        elif s >= 50:
            return "blue"
        elif s >= 20:
            return "yellow"
        return "red"

    def sold_count_or_zero(self) -> int:
        return self.sold_count or 0
