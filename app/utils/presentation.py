"""Presentation-layer helpers — labels, colors, display text.

Moved from model methods to maintain MVC separation.
Registered as Jinja2 template filters in create_app().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.listing_progress import ListingProgress
    from app.models.order import Order
    from app.models.partner import Partner
    from app.models.popularity_tracker import PopularityTracker
    from app.models.region_recommendation import RegionRecommendation
    from app.models.repeat_customer import RepeatCustomer
    from app.models.shipment_notification import ShipmentNotification
    from app.models.stock_check import StockCheck


# ---------- RepeatCustomer ----------

_SEGMENT_LABELS = {
    "new": "新規",
    "active": "アクティブ",
    "vip": "VIP",
    "at_risk": "離脱危惧",
    "churned": "離脱済",
}

_SEGMENT_COLORS = {
    "new": "blue",
    "active": "green",
    "vip": "purple",
    "at_risk": "yellow",
    "churned": "red",
}


def segment_label(obj: RepeatCustomer) -> str:
    return _SEGMENT_LABELS.get(obj.segment or "new", "不明")


def segment_color(obj: RepeatCustomer) -> str:
    return _SEGMENT_COLORS.get(obj.segment or "new", "gray")


# ---------- Partner ----------

_PRIORITY_LABELS = {"high": "高", "medium": "中", "low": "低"}
_PRIORITY_COLORS = {"high": "red", "medium": "yellow", "low": "green"}
_PARTNER_STATUS_LABELS = {"active": "稼働", "inactive": "停止", "suspended": "保留"}


def priority_label(obj: Partner) -> str:
    return _PRIORITY_LABELS.get(obj.priority_level or "medium", "中")


def priority_color(obj: Partner) -> str:
    return _PRIORITY_COLORS.get(obj.priority_level or "medium", "gray")


def partner_status_label(obj: Partner) -> str:
    return _PARTNER_STATUS_LABELS.get(obj.status or "active", "不明")


# ---------- Order ----------


def deadline_color(obj: Order) -> str:
    remaining = obj.remaining_days()
    if remaining is None:
        return "gray"
    if remaining < 0:
        return "black"
    if remaining == 0:
        return "red"
    if remaining <= 2:
        return "orange"
    if remaining <= 4:
        return "yellow"
    return "green"


def deadline_message(obj: Order) -> str:
    remaining = obj.remaining_days()
    if remaining is None:
        return ""
    if remaining < 0:
        return "期限切れです。至急対応してください。"
    if remaining == 0:
        return "本日が期限です。発送準備をお願いします。"
    if remaining <= 2:
        return "残りわずかです。発送手配を確認してください。"
    if remaining <= 4:
        return "期限が近づいています。発送準備を始めてください。"
    return ""


# ---------- PopularityTracker ----------


def score_label(obj: PopularityTracker) -> str:
    s = obj.popularity_score or 0
    if s >= 100:
        return "超人気"
    if s >= 50:
        return "人気"
    if s >= 20:
        return "普通"
    return "要注力"


def score_color(obj: PopularityTracker) -> str:
    s = obj.popularity_score or 0
    if s >= 100:
        return "green"
    if s >= 50:
        return "blue"
    if s >= 20:
        return "yellow"
    return "red"


# ---------- ShipmentNotification ----------

_SHIPMENT_STATUS_LABELS = {
    "pending": "未通知",
    "notified": "通知済み",
    "confirmed": "確認済み",
    "error": "エラー",
}


def shipment_status_label(obj: ShipmentNotification) -> str:
    return _SHIPMENT_STATUS_LABELS.get(obj.status, obj.status)


# ---------- ListingProgress ----------


def daily_status_color(obj: ListingProgress) -> str:
    rate = obj.daily_achievement_rate()
    if rate >= 100:
        return "green"
    if rate >= 70:
        return "yellow"
    return "red"


def monthly_status_color(obj: ListingProgress) -> str:
    rate = obj.monthly_achievement_rate()
    if rate >= 100:
        return "green"
    if rate >= 70:
        return "yellow"
    return "red"


# ---------- StockCheck ----------


def stock_status_label(obj: StockCheck) -> str:
    if not obj.in_stock:
        return "在庫切れ"
    if obj.price_changed:
        return "価格変動"
    return "正常"


def stock_status_color(obj: StockCheck) -> str:
    if not obj.in_stock:
        return "red"
    if obj.price_changed:
        return "yellow"
    return "green"


# ---------- RegionRecommendation ----------


def risk_label(obj: RegionRecommendation) -> str:
    r = obj.risk_score or 0
    if r <= 20:
        return "低リスク"
    if r <= 50:
        return "中リスク"
    return "高リスク"


def risk_color(obj: RegionRecommendation) -> str:
    r = obj.risk_score or 0
    if r <= 20:
        return "green"
    if r <= 50:
        return "yellow"
    return "red"


def recommendation_label(obj: RegionRecommendation) -> str:
    s = obj.recommendation_score or 0
    if s >= 70:
        return "推奨"
    if s >= 40:
        return "普通"
    return "非推奨"
