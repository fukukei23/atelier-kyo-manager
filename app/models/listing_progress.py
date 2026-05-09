# ======================================================================
# File: listing_progress.py
# Purpose: ListingProgress model - F07 品出し進捗トラッカー
# ======================================================================

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Integer, Text, func

from app.extensions import db


class ListingProgress(db.Model):
    __tablename__ = "listing_progress"

    id = db.Column(Integer, primary_key=True)

    # 日付・目標
    record_date = db.Column(Date, nullable=False, default=date.today, comment="記録日")
    listings_count = db.Column(Integer, nullable=False, default=0, comment="当日出品数")
    target_daily = db.Column(Integer, nullable=False, default=20, comment="1日目標")
    target_monthly = db.Column(Integer, nullable=False, default=600, comment="月間目標")

    # 累計（月間）
    cumulative_monthly = db.Column(Integer, nullable=False, default=0, comment="月間累計出品数")

    # メモ
    notes = db.Column(Text, nullable=True)

    created_at = db.Column(DateTime, default=datetime.utcnow)
    updated_at = db.Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # ---- ヘルパー ----
    def daily_achievement_rate(self) -> float:
        """日次達成率%"""
        if self.target_daily <= 0:
            return 0.0
        return min(self.listings_count / self.target_daily * 100, 100.0)

    def monthly_achievement_rate(self) -> float:
        """月間達成率%"""
        if self.target_monthly <= 0:
            return 0.0
        return min(self.cumulative_monthly / self.target_monthly * 100, 100.0)

    def daily_status_color(self) -> str:
        """達成率に応じた色"""
        rate = self.daily_achievement_rate()
        if rate >= 100:
            return "green"
        if rate >= 70:
            return "yellow"
        return "red"

    def monthly_status_color(self) -> str:
        """月間達成率に応じた色"""
        rate = self.monthly_achievement_rate()
        if rate >= 100:
            return "green"
        if rate >= 70:
            return "yellow"
        return "red"

    @staticmethod
    def get_monthly_summary(year: int, month: int) -> dict:
        """月間サマリーを取得"""
        start = date(year, month, 1)
        if month == 12:
            end = date(year + 1, 1, 1)
        else:
            end = date(year, month + 1, 1)

        result = (
            db.session.query(
                func.sum(ListingProgress.listings_count).label("total"),
                func.count(ListingProgress.id).label("days"),
                func.max(ListingProgress.cumulative_monthly).label("cumulative"),
            )
            .filter(
                ListingProgress.record_date >= start,
                ListingProgress.record_date < end,
            )
            .first()
        )

        if not result or result.days == 0:
            return {"total": 0, "days": 0, "cumulative": 0, "daily_avg": 0}

        return {
            "total": result.total or 0,
            "days": result.days,
            "cumulative": result.cumulative or 0,
            "daily_avg": round((result.total or 0) / result.days, 1),
        }

    def __repr__(self) -> str:
        return f"<ListingProgress id={self.id} date={self.record_date} count={self.listings_count}>"
