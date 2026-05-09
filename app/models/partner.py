# ======================================================================
# File: partner.py
# Purpose: Partner model - F06 パートナー管理
# ======================================================================

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text

from app.extensions import db


class Partner(db.Model):
    __tablename__ = "partner"

    id = db.Column(Integer, primary_key=True)
    name = db.Column(String(128), nullable=False)
    email = db.Column(String(255), nullable=True)
    phone = db.Column(String(32), nullable=True)

    # 専門領域
    active_regions = db.Column(Text, nullable=True, comment="対応地域 (CSV)")
    specialty_brands = db.Column(Text, nullable=True, comment="得意ブランド (CSV)")

    # 評価指標
    response_speed_avg = db.Column(Float, nullable=True, default=0, comment="平均応答時間(分)")
    response_speed_score = db.Column(Integer, nullable=True, default=3, comment="応答スコア 1-5")
    reliability_score = db.Column(Float, nullable=True, default=0, comment="信頼性スコア 0-100%")
    total_orders = db.Column(Integer, nullable=True, default=0)
    success_count = db.Column(Integer, nullable=True, default=0)
    cancellation_count = db.Column(Integer, nullable=True, default=0)

    # ステータス
    priority_level = db.Column(String(16), default="medium", comment="high/medium/low")
    status = db.Column(String(16), default="active", comment="active/inactive/suspended")
    notes = db.Column(Text, nullable=True)

    created_at = db.Column(DateTime, default=datetime.utcnow)
    updated_at = db.Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # ---- ヘルパー ----
    def success_rate(self) -> float:
        """成功率%"""
        if not self.total_orders or self.total_orders == 0:
            return 0.0
        return (self.success_count or 0) / self.total_orders * 100

    def cancellation_rate(self) -> float:
        """キャンセル率%"""
        if not self.total_orders or self.total_orders == 0:
            return 0.0
        return (self.cancellation_count or 0) / self.total_orders * 100

    def regions_list(self) -> list[str]:
        """対応地域をリストで取得"""
        if not self.active_regions:
            return []
        return [r.strip() for r in self.active_regions.split(",") if r.strip()]

    def brands_list(self) -> list[str]:
        """得意ブランドをリストで取得"""
        if not self.specialty_brands:
            return []
        return [b.strip() for b in self.specialty_brands.split(",") if b.strip()]

    def priority_label(self) -> str:
        labels = {"high": "高", "medium": "中", "low": "低"}
        return labels.get(self.priority_level or "medium", "中")

    def priority_color(self) -> str:
        colors = {"high": "red", "medium": "yellow", "low": "green"}
        return colors.get(self.priority_level or "medium", "gray")

    def status_label(self) -> str:
        labels = {"active": "稼働", "inactive": "停止", "suspended": "保留"}
        return labels.get(self.status or "active", "不明")

    def __repr__(self) -> str:
        return f"<Partner id={self.id} name={self.name!r}>"
