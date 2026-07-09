"""Tests for app/utils/presentation.py"""

from unittest.mock import MagicMock

from app.utils.presentation import (
    daily_status_color,
    deadline_color,
    deadline_message,
    monthly_status_color,
    partner_status_label,
    priority_color,
    priority_label,
    recommendation_label,
    risk_color,
    risk_label,
    score_color,
    score_label,
    segment_color,
    segment_label,
    shipment_status_label,
    stock_status_color,
    stock_status_label,
)


def _customer(segment):
    obj = MagicMock()
    obj.segment = segment
    return obj


def _partner(priority=None, status=None):
    obj = MagicMock()
    obj.priority_level = priority
    obj.status = status
    return obj


def _order(remaining):
    obj = MagicMock()
    obj.remaining_days.return_value = remaining
    return obj


def _popularity(score):
    obj = MagicMock()
    obj.popularity_score = score
    return obj


def _shipment(status):
    obj = MagicMock()
    obj.status = status
    return obj


def _listing_progress(daily_rate, monthly_rate):
    obj = MagicMock()
    obj.daily_achievement_rate.return_value = daily_rate
    obj.monthly_achievement_rate.return_value = monthly_rate
    return obj


def _stock(in_stock, price_changed):
    obj = MagicMock()
    obj.in_stock = in_stock
    obj.price_changed = price_changed
    return obj


def _region(risk_score=None, recommendation_score=None):
    obj = MagicMock()
    obj.risk_score = risk_score
    obj.recommendation_score = recommendation_score
    return obj


class TestSegmentHelpers:
    def test_segment_label_new(self):
        assert segment_label(_customer("new")) == "新規"

    def test_segment_label_vip(self):
        assert segment_label(_customer("vip")) == "VIP"

    def test_segment_label_churned(self):
        assert segment_label(_customer("churned")) == "離脱済"

    def test_segment_label_unknown(self):
        assert segment_label(_customer("unknown")) == "不明"

    def test_segment_label_none_defaults_new(self):
        assert segment_label(_customer(None)) == "新規"

    def test_segment_color_active(self):
        assert segment_color(_customer("active")) == "green"

    def test_segment_color_at_risk(self):
        assert segment_color(_customer("at_risk")) == "yellow"


class TestPartnerHelpers:
    def test_priority_label_high(self):
        assert priority_label(_partner(priority="high")) == "高"

    def test_priority_label_none_defaults_medium(self):
        assert priority_label(_partner(priority=None)) == "中"

    def test_priority_color_high(self):
        assert priority_color(_partner(priority="high")) == "red"

    def test_partner_status_label_active(self):
        assert partner_status_label(_partner(status="active")) == "稼働"

    def test_partner_status_label_suspended(self):
        assert partner_status_label(_partner(status="suspended")) == "保留"


class TestDeadlineHelpers:
    def test_deadline_color_none(self):
        assert deadline_color(_order(None)) == "gray"

    def test_deadline_color_expired(self):
        assert deadline_color(_order(-1)) == "black"

    def test_deadline_color_today(self):
        assert deadline_color(_order(0)) == "red"

    def test_deadline_color_urgent(self):
        assert deadline_color(_order(2)) == "orange"

    def test_deadline_color_soon(self):
        assert deadline_color(_order(4)) == "yellow"

    def test_deadline_color_ok(self):
        assert deadline_color(_order(10)) == "green"

    def test_deadline_message_expired(self):
        msg = deadline_message(_order(-1))
        assert "期限切れ" in msg

    def test_deadline_message_today(self):
        msg = deadline_message(_order(0))
        assert "本日" in msg

    def test_deadline_message_none(self):
        assert deadline_message(_order(None)) == ""

    def test_deadline_message_ok(self):
        assert deadline_message(_order(10)) == ""


class TestPopularityHelpers:
    def test_score_label_hot(self):
        assert score_label(_popularity(100)) == "超人気"

    def test_score_label_popular(self):
        assert score_label(_popularity(50)) == "人気"

    def test_score_label_normal(self):
        assert score_label(_popularity(20)) == "普通"

    def test_score_label_low(self):
        assert score_label(_popularity(5)) == "要注力"

    def test_score_color_hot(self):
        assert score_color(_popularity(100)) == "green"

    def test_score_color_low(self):
        assert score_color(_popularity(0)) == "red"


class TestShipmentHelpers:
    def test_shipment_status_label_pending(self):
        assert shipment_status_label(_shipment("pending")) == "未通知"

    def test_shipment_status_label_notified(self):
        assert shipment_status_label(_shipment("notified")) == "通知済み"

    def test_shipment_status_label_unknown(self):
        assert shipment_status_label(_shipment("something")) == "something"


class TestListingProgressHelpers:
    def test_daily_status_color_green(self):
        assert daily_status_color(_listing_progress(100, 50)) == "green"

    def test_daily_status_color_yellow(self):
        assert daily_status_color(_listing_progress(70, 50)) == "yellow"

    def test_daily_status_color_red(self):
        assert daily_status_color(_listing_progress(50, 50)) == "red"

    def test_monthly_status_color_green(self):
        assert monthly_status_color(_listing_progress(50, 100)) == "green"


class TestStockHelpers:
    def test_stock_status_label_out_of_stock(self):
        assert stock_status_label(_stock(False, False)) == "在庫切れ"

    def test_stock_status_label_price_changed(self):
        assert stock_status_label(_stock(True, True)) == "価格変動"

    def test_stock_status_label_ok(self):
        assert stock_status_label(_stock(True, False)) == "正常"

    def test_stock_status_color_out(self):
        assert stock_status_color(_stock(False, False)) == "red"

    def test_stock_status_color_ok(self):
        assert stock_status_color(_stock(True, False)) == "green"


class TestRegionHelpers:
    def test_risk_label_low(self):
        assert risk_label(_region(risk_score=10)) == "低リスク"

    def test_risk_label_medium(self):
        assert risk_label(_region(risk_score=35)) == "中リスク"

    def test_risk_label_high(self):
        assert risk_label(_region(risk_score=80)) == "高リスク"

    def test_risk_color_low(self):
        assert risk_color(_region(risk_score=10)) == "green"

    def test_recommendation_label_recommended(self):
        assert recommendation_label(_region(recommendation_score=70)) == "推奨"

    def test_recommendation_label_not_recommended(self):
        assert recommendation_label(_region(recommendation_score=20)) == "非推奨"
