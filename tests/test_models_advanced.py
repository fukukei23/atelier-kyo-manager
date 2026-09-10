"""
モデルビジネスロジック包括テスト
対象: PopularityTracker / StockCheck / RegionRecommendation / ProhibitedSource
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app import create_app
from app.extensions import db
from app.models.popularity_tracker import PopularityTracker
from app.models.prohibited_source import ProhibitedSource
from app.models.region_recommendation import RegionRecommendation
from app.models.stock_check import StockCheck
from app.utils.presentation import (
    recommendation_label,
    risk_color,
    risk_label,
    score_color,
    score_label,
    stock_status_color,
    stock_status_label,
)


@pytest.fixture(scope="function")
def app():
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test-secret",
        }
    )
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


# ─────────────────────────── PopularityTracker ───────────────────────────


class TestPopularityTrackerCalcScore:
    def test_calc_score_all_zeros(self):
        pt = PopularityTracker(views=0, favorites=0, inquiries=0, sold_count=0)
        assert pt.calc_score() == 0.0

    def test_calc_score_views_only(self):
        pt = PopularityTracker(views=10, favorites=0, inquiries=0, sold_count=0)
        assert pt.calc_score() == pytest.approx(1.0)

    def test_calc_score_favorites_only(self):
        pt = PopularityTracker(views=0, favorites=5, inquiries=0, sold_count=0)
        assert pt.calc_score() == pytest.approx(10.0)

    def test_calc_score_inquiries_only(self):
        pt = PopularityTracker(views=0, favorites=0, inquiries=4, sold_count=0)
        assert pt.calc_score() == pytest.approx(12.0)

    def test_calc_score_sold_count_only(self):
        pt = PopularityTracker(views=0, favorites=0, inquiries=0, sold_count=3)
        assert pt.calc_score() == pytest.approx(15.0)

    def test_calc_score_combined(self):
        pt = PopularityTracker(views=10, favorites=5, inquiries=2, sold_count=1)
        expected = 10 * 0.1 + 5 * 2 + 2 * 3 + 1 * 5
        assert pt.calc_score() == pytest.approx(expected)

    def test_calc_score_none_values_treated_as_zero(self):
        pt = PopularityTracker(views=None, favorites=None, inquiries=None, sold_count=None)
        assert pt.calc_score() == 0.0


class TestPopularityTrackerScoreLabel:
    def test_score_label_above_100_is_top_popular(self):
        pt = PopularityTracker(popularity_score=150.0)
        assert score_label(pt) == "超人気"

    def test_score_label_exactly_100_is_top_popular(self):
        pt = PopularityTracker(popularity_score=100.0)
        assert score_label(pt) == "超人気"

    def test_score_label_99_is_popular(self):
        pt = PopularityTracker(popularity_score=99.0)
        assert score_label(pt) == "人気"

    def test_score_label_exactly_50_is_popular(self):
        pt = PopularityTracker(popularity_score=50.0)
        assert score_label(pt) == "人気"

    def test_score_label_49_is_normal(self):
        pt = PopularityTracker(popularity_score=49.0)
        assert score_label(pt) == "普通"

    def test_score_label_exactly_20_is_normal(self):
        pt = PopularityTracker(popularity_score=20.0)
        assert score_label(pt) == "普通"

    def test_score_label_19_is_attention_needed(self):
        pt = PopularityTracker(popularity_score=19.0)
        assert score_label(pt) == "要注力"

    def test_score_label_zero_is_attention_needed(self):
        pt = PopularityTracker(popularity_score=0.0)
        assert score_label(pt) == "要注力"

    def test_score_label_none_is_attention_needed(self):
        pt = PopularityTracker(popularity_score=None)
        assert score_label(pt) == "要注力"


class TestPopularityTrackerScoreColor:
    def test_score_color_above_100_is_green(self):
        pt = PopularityTracker(popularity_score=100.0)
        assert score_color(pt) == "green"

    def test_score_color_50_to_99_is_blue(self):
        pt = PopularityTracker(popularity_score=75.0)
        assert score_color(pt) == "blue"

    def test_score_color_exactly_50_is_blue(self):
        pt = PopularityTracker(popularity_score=50.0)
        assert score_color(pt) == "blue"

    def test_score_color_20_to_49_is_yellow(self):
        pt = PopularityTracker(popularity_score=35.0)
        assert score_color(pt) == "yellow"

    def test_score_color_exactly_20_is_yellow(self):
        pt = PopularityTracker(popularity_score=20.0)
        assert score_color(pt) == "yellow"

    def test_score_color_below_20_is_red(self):
        pt = PopularityTracker(popularity_score=10.0)
        assert score_color(pt) == "red"

    def test_score_color_none_is_red(self):
        pt = PopularityTracker(popularity_score=None)
        assert score_color(pt) == "red"


class TestPopularityTrackerSoldCountOrZero:
    def test_sold_count_or_zero_with_value(self):
        pt = PopularityTracker(sold_count=5)
        assert pt.sold_count_or_zero() == 5

    def test_sold_count_or_zero_with_none(self):
        pt = PopularityTracker(sold_count=None)
        assert pt.sold_count_or_zero() == 0

    def test_sold_count_or_zero_with_zero(self):
        pt = PopularityTracker(sold_count=0)
        assert pt.sold_count_or_zero() == 0


# ─────────────────────────── StockCheck ───────────────────────────


class TestStockCheckToDict:
    def test_to_dict_basic_fields(self):
        sc = StockCheck(
            id=1,
            product_id=10,
            source_url="https://example.com/product",
            current_price=5000.0,
            previous_price=4500.0,
            price_changed=True,
            in_stock=True,
            stock_changed=False,
            notes="テスト",
        )
        result = sc.to_dict()
        assert result["id"] == 1
        assert result["product_id"] == 10
        assert result["source_url"] == "https://example.com/product"
        assert result["current_price"] == 5000.0
        assert result["previous_price"] == 4500.0
        assert result["price_changed"] is True
        assert result["in_stock"] is True
        assert result["stock_changed"] is False
        assert result["notes"] == "テスト"

    def test_to_dict_checked_at_is_isoformat(self):
        dt = datetime(2026, 4, 21, 12, 0, 0)
        sc = StockCheck(checked_at=dt)
        result = sc.to_dict()
        assert result["checked_at"] == dt.isoformat()

    def test_to_dict_checked_at_none(self):
        sc = StockCheck(checked_at=None)
        result = sc.to_dict()
        assert result["checked_at"] is None


class TestStockCheckPriceDiff:
    def test_price_diff_positive(self):
        sc = StockCheck(current_price=5000.0, previous_price=4000.0)
        assert sc.price_diff() == pytest.approx(1000.0)

    def test_price_diff_negative(self):
        sc = StockCheck(current_price=3000.0, previous_price=4000.0)
        assert sc.price_diff() == pytest.approx(-1000.0)

    def test_price_diff_zero(self):
        sc = StockCheck(current_price=4000.0, previous_price=4000.0)
        assert sc.price_diff() == pytest.approx(0.0)

    def test_price_diff_current_none_returns_none(self):
        sc = StockCheck(current_price=None, previous_price=4000.0)
        assert sc.price_diff() is None

    def test_price_diff_previous_none_returns_none(self):
        sc = StockCheck(current_price=4000.0, previous_price=None)
        assert sc.price_diff() is None

    def test_price_diff_both_none_returns_none(self):
        sc = StockCheck(current_price=None, previous_price=None)
        assert sc.price_diff() is None


class TestStockCheckPriceDiffPct:
    def test_price_diff_pct_increase(self):
        sc = StockCheck(current_price=5500.0, previous_price=5000.0)
        assert sc.price_diff_pct() == pytest.approx(10.0)

    def test_price_diff_pct_decrease(self):
        sc = StockCheck(current_price=4000.0, previous_price=5000.0)
        assert sc.price_diff_pct() == pytest.approx(-20.0)

    def test_price_diff_pct_no_change(self):
        sc = StockCheck(current_price=5000.0, previous_price=5000.0)
        assert sc.price_diff_pct() == pytest.approx(0.0)

    def test_price_diff_pct_previous_zero_returns_none(self):
        sc = StockCheck(current_price=5000.0, previous_price=0.0)
        assert sc.price_diff_pct() is None

    def test_price_diff_pct_current_none_returns_none(self):
        sc = StockCheck(current_price=None, previous_price=5000.0)
        assert sc.price_diff_pct() is None

    def test_price_diff_pct_previous_none_returns_none(self):
        sc = StockCheck(current_price=5000.0, previous_price=None)
        assert sc.price_diff_pct() is None


class TestStockCheckStatusLabel:
    def test_status_label_out_of_stock(self):
        sc = StockCheck(in_stock=False, price_changed=False)
        assert stock_status_label(sc) == "在庫切れ"

    def test_status_label_out_of_stock_overrides_price_changed(self):
        sc = StockCheck(in_stock=False, price_changed=True)
        assert stock_status_label(sc) == "在庫切れ"

    def test_status_label_price_changed(self):
        sc = StockCheck(in_stock=True, price_changed=True)
        assert stock_status_label(sc) == "価格変動"

    def test_status_label_normal(self):
        sc = StockCheck(in_stock=True, price_changed=False)
        assert stock_status_label(sc) == "正常"


class TestStockCheckStatusColor:
    def test_status_color_out_of_stock_is_red(self):
        sc = StockCheck(in_stock=False, price_changed=False)
        assert stock_status_color(sc) == "red"

    def test_status_color_price_changed_is_yellow(self):
        sc = StockCheck(in_stock=True, price_changed=True)
        assert stock_status_color(sc) == "yellow"

    def test_status_color_normal_is_green(self):
        sc = StockCheck(in_stock=True, price_changed=False)
        assert stock_status_color(sc) == "green"


# ─────────────────────────── RegionRecommendation ───────────────────────────


class TestRegionRecommendationCalcRecommendation:
    def test_calc_recommendation_valid_values(self):
        rr = RegionRecommendation(
            reliability_score=80.0,
            risk_score=20.0,
            avg_profit_rate=30.0,
        )
        expected = 80.0 * 0.4 + (100 - 20.0) * 0.3 + 30.0 * 0.3
        assert rr.calc_recommendation() == pytest.approx(expected)

    def test_calc_recommendation_reliability_none_returns_none(self):
        rr = RegionRecommendation(
            reliability_score=None,
            risk_score=20.0,
            avg_profit_rate=30.0,
        )
        assert rr.calc_recommendation() is None

    def test_calc_recommendation_risk_none_returns_none(self):
        rr = RegionRecommendation(
            reliability_score=80.0,
            risk_score=None,
            avg_profit_rate=30.0,
        )
        assert rr.calc_recommendation() is None

    def test_calc_recommendation_profit_none_returns_none(self):
        rr = RegionRecommendation(
            reliability_score=80.0,
            risk_score=20.0,
            avg_profit_rate=None,
        )
        assert rr.calc_recommendation() is None


class TestRegionRecommendationRiskLabel:
    def test_risk_label_low_risk(self):
        rr = RegionRecommendation(risk_score=10.0)
        assert risk_label(rr) == "低リスク"

    def test_risk_label_exactly_20_is_low_risk(self):
        rr = RegionRecommendation(risk_score=20.0)
        assert risk_label(rr) == "低リスク"

    def test_risk_label_21_is_medium_risk(self):
        rr = RegionRecommendation(risk_score=21.0)
        assert risk_label(rr) == "中リスク"

    def test_risk_label_exactly_50_is_medium_risk(self):
        rr = RegionRecommendation(risk_score=50.0)
        assert risk_label(rr) == "中リスク"

    def test_risk_label_51_is_high_risk(self):
        rr = RegionRecommendation(risk_score=51.0)
        assert risk_label(rr) == "高リスク"

    def test_risk_label_none_is_low_risk(self):
        rr = RegionRecommendation(risk_score=None)
        assert risk_label(rr) == "低リスク"


class TestRegionRecommendationRiskColor:
    def test_risk_color_low_is_green(self):
        rr = RegionRecommendation(risk_score=15.0)
        assert risk_color(rr) == "green"

    def test_risk_color_medium_is_yellow(self):
        rr = RegionRecommendation(risk_score=35.0)
        assert risk_color(rr) == "yellow"

    def test_risk_color_high_is_red(self):
        rr = RegionRecommendation(risk_score=75.0)
        assert risk_color(rr) == "red"


class TestRegionRecommendationLabel:
    def test_recommendation_label_recommended(self):
        rr = RegionRecommendation(recommendation_score=80.0)
        assert recommendation_label(rr) == "推奨"

    def test_recommendation_label_exactly_70_is_recommended(self):
        rr = RegionRecommendation(recommendation_score=70.0)
        assert recommendation_label(rr) == "推奨"

    def test_recommendation_label_69_is_normal(self):
        rr = RegionRecommendation(recommendation_score=69.0)
        assert recommendation_label(rr) == "普通"

    def test_recommendation_label_exactly_40_is_normal(self):
        rr = RegionRecommendation(recommendation_score=40.0)
        assert recommendation_label(rr) == "普通"

    def test_recommendation_label_39_is_not_recommended(self):
        rr = RegionRecommendation(recommendation_score=39.0)
        assert recommendation_label(rr) == "非推奨"

    def test_recommendation_label_zero_is_not_recommended(self):
        rr = RegionRecommendation(recommendation_score=0.0)
        assert recommendation_label(rr) == "非推奨"

    def test_recommendation_label_none_is_not_recommended(self):
        rr = RegionRecommendation(recommendation_score=None)
        assert recommendation_label(rr) == "非推奨"


# ─────────────────────────── ProhibitedSource ───────────────────────────


class TestProhibitedSourceIsProhibited:
    def test_is_prohibited_empty_url_returns_false(self, app):
        with app.app_context():
            result, reason = ProhibitedSource.is_prohibited("")
            assert result is False
            assert reason == ""

    def test_is_prohibited_none_url_returns_false(self, app):
        with app.app_context():
            result, reason = ProhibitedSource.is_prohibited(None)
            assert result is False
            assert reason == ""

    def test_is_prohibited_match_returns_true(self, app):
        with app.app_context():
            entry = ProhibitedSource(
                domain="prohibited.com",
                reason="テスト禁止",
                source_type="domestic",
            )
            db.session.add(entry)
            db.session.commit()

            result, reason = ProhibitedSource.is_prohibited("https://prohibited.com/item/123")
            assert result is True
            assert reason == "テスト禁止"

    def test_is_prohibited_no_match_returns_false(self, app):
        with app.app_context():
            entry = ProhibitedSource(
                domain="prohibited.com",
                reason="禁止",
                source_type="domestic",
            )
            db.session.add(entry)
            db.session.commit()

            result, reason = ProhibitedSource.is_prohibited("https://allowed.com/item/123")
            assert result is False
            assert reason == ""

    def test_is_prohibited_source_type_filter(self, app):
        with app.app_context():
            entry = ProhibitedSource(
                domain="overseas-bad.com",
                reason="海外禁止",
                source_type="overseas",
            )
            db.session.add(entry)
            db.session.commit()

            # domestic type → should NOT match overseas entry
            result, reason = ProhibitedSource.is_prohibited("https://overseas-bad.com/item", source_type="domestic")
            assert result is False

            # overseas type → should match
            result, reason = ProhibitedSource.is_prohibited("https://overseas-bad.com/item", source_type="overseas")
            assert result is True

    def test_is_prohibited_no_reason_uses_default_message(self, app):
        with app.app_context():
            entry = ProhibitedSource(
                domain="noreasonsite.com",
                reason=None,
                source_type="domestic",
            )
            db.session.add(entry)
            db.session.commit()

            result, reason = ProhibitedSource.is_prohibited("https://noreasonsite.com/product")
            assert result is True
            assert "noreasonsite.com" in reason

    def test_is_prohibited_case_insensitive(self, app):
        with app.app_context():
            entry = ProhibitedSource(
                domain="BIGSITE.COM",
                reason="大文字ドメイン禁止",
                source_type="domestic",
            )
            db.session.add(entry)
            db.session.commit()

            result, reason = ProhibitedSource.is_prohibited("https://bigsite.com/product")
            assert result is True


class TestProhibitedSourceRepr:
    def test_repr_format(self, app):
        with app.app_context():
            entry = ProhibitedSource(domain="example.com", severity="blocked")
            assert repr(entry) == "<ProhibitedSource example.com [blocked]>"
