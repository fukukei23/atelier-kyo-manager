"""
ListingProgress / RepeatCustomer モデル包括テスト
daily_achievement_rate, monthly_achievement_rate, status_color,
get_monthly_summary, calc_segment, days_since_last, segment_label,
segment_color, update_avg を完全カバー
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app import create_app
from app.extensions import db
from app.models.listing_progress import ListingProgress
from app.models.repeat_customer import RepeatCustomer


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


# ─────────────────────────── ListingProgress ───────────────────────────


class TestListingProgressDailyAchievementRate:
    def test_zero_target_returns_zero(self, app):
        lp = ListingProgress(listings_count=10, target_daily=0, target_monthly=600)
        assert lp.daily_achievement_rate() == 0.0

    def test_full_achievement(self, app):
        lp = ListingProgress(listings_count=20, target_daily=20, target_monthly=600)
        assert lp.daily_achievement_rate() == 100.0

    def test_partial_achievement(self, app):
        lp = ListingProgress(listings_count=10, target_daily=20, target_monthly=600)
        assert lp.daily_achievement_rate() == 50.0

    def test_over_target_capped_at_100(self, app):
        lp = ListingProgress(listings_count=30, target_daily=20, target_monthly=600)
        assert lp.daily_achievement_rate() == 100.0

    def test_zero_listings(self, app):
        lp = ListingProgress(listings_count=0, target_daily=20, target_monthly=600)
        assert lp.daily_achievement_rate() == 0.0


class TestListingProgressMonthlyAchievementRate:
    def test_zero_target_returns_zero(self, app):
        lp = ListingProgress(listings_count=0, target_daily=20, target_monthly=0, cumulative_monthly=100)
        assert lp.monthly_achievement_rate() == 0.0

    def test_full_achievement(self, app):
        lp = ListingProgress(listings_count=0, target_daily=20, target_monthly=600, cumulative_monthly=600)
        assert lp.monthly_achievement_rate() == 100.0

    def test_partial_achievement(self, app):
        lp = ListingProgress(listings_count=0, target_daily=20, target_monthly=600, cumulative_monthly=300)
        assert lp.monthly_achievement_rate() == 50.0

    def test_over_target_capped_at_100(self, app):
        lp = ListingProgress(listings_count=0, target_daily=20, target_monthly=600, cumulative_monthly=700)
        assert lp.monthly_achievement_rate() == 100.0

    def test_zero_cumulative(self, app):
        lp = ListingProgress(listings_count=0, target_daily=20, target_monthly=600, cumulative_monthly=0)
        assert lp.monthly_achievement_rate() == 0.0


class TestListingProgressDailyStatusColor:
    def test_green_at_100_percent(self, app):
        lp = ListingProgress(listings_count=20, target_daily=20, target_monthly=600)
        assert lp.daily_status_color() == "green"

    def test_yellow_at_70_percent(self, app):
        lp = ListingProgress(listings_count=14, target_daily=20, target_monthly=600)
        assert lp.daily_status_color() == "yellow"

    def test_red_below_70_percent(self, app):
        lp = ListingProgress(listings_count=5, target_daily=20, target_monthly=600)
        assert lp.daily_status_color() == "red"

    def test_red_at_zero(self, app):
        lp = ListingProgress(listings_count=0, target_daily=20, target_monthly=600)
        assert lp.daily_status_color() == "red"

    def test_yellow_boundary_exactly_70(self, app):
        # 14/20 = 70% → "yellow"
        lp = ListingProgress(listings_count=14, target_daily=20, target_monthly=600)
        assert lp.daily_status_color() == "yellow"


class TestListingProgressMonthlyStatusColor:
    def test_green_at_100_percent(self, app):
        lp = ListingProgress(listings_count=0, target_daily=20, target_monthly=600, cumulative_monthly=600)
        assert lp.monthly_status_color() == "green"

    def test_yellow_at_70_percent(self, app):
        lp = ListingProgress(listings_count=0, target_daily=20, target_monthly=600, cumulative_monthly=420)
        assert lp.monthly_status_color() == "yellow"

    def test_red_below_70_percent(self, app):
        lp = ListingProgress(listings_count=0, target_daily=20, target_monthly=600, cumulative_monthly=100)
        assert lp.monthly_status_color() == "red"


class TestListingProgressGetMonthlySummary:
    def test_no_records_returns_zeros(self, app):
        result = ListingProgress.get_monthly_summary(2026, 1)
        assert result == {"total": 0, "days": 0, "cumulative": 0, "daily_avg": 0}

    def test_single_record_summary(self, app):
        lp = ListingProgress(
            record_date=date(2026, 3, 1),
            listings_count=10,
            cumulative_monthly=10,
            target_daily=20,
            target_monthly=600,
        )
        db.session.add(lp)
        db.session.commit()
        result = ListingProgress.get_monthly_summary(2026, 3)
        assert result["total"] == 10
        assert result["days"] == 1
        assert result["daily_avg"] == 10.0

    def test_multiple_records_aggregated(self, app):
        for i, count in enumerate([10, 20, 30], start=1):
            lp = ListingProgress(
                record_date=date(2026, 4, i),
                listings_count=count,
                cumulative_monthly=count,
                target_daily=20,
                target_monthly=600,
            )
            db.session.add(lp)
        db.session.commit()
        result = ListingProgress.get_monthly_summary(2026, 4)
        assert result["total"] == 60
        assert result["days"] == 3
        assert result["daily_avg"] == 20.0

    def test_december_does_not_overflow(self, app):
        """12月の month+1 が 1月に正しくクロスオーバーすること"""
        lp = ListingProgress(
            record_date=date(2026, 12, 15),
            listings_count=5,
            cumulative_monthly=5,
            target_daily=20,
            target_monthly=600,
        )
        db.session.add(lp)
        db.session.commit()
        result = ListingProgress.get_monthly_summary(2026, 12)
        assert result["total"] == 5
        assert result["days"] == 1

    def test_records_outside_month_not_counted(self, app):
        """対象月外のレコードは集計されないこと"""
        lp_in = ListingProgress(
            record_date=date(2026, 5, 10),
            listings_count=15,
            cumulative_monthly=15,
            target_daily=20,
            target_monthly=600,
        )
        lp_out = ListingProgress(
            record_date=date(2026, 6, 1),
            listings_count=999,
            cumulative_monthly=999,
            target_daily=20,
            target_monthly=600,
        )
        db.session.add_all([lp_in, lp_out])
        db.session.commit()
        result = ListingProgress.get_monthly_summary(2026, 5)
        assert result["total"] == 15
        assert result["days"] == 1


class TestListingProgressRepr:
    def test_repr_contains_key_info(self, app):
        lp = ListingProgress(id=7, record_date=date(2026, 1, 1), listings_count=5, target_daily=20, target_monthly=600)
        r = repr(lp)
        assert "7" in r
        assert "5" in r


# ─────────────────────────── RepeatCustomer ───────────────────────────


class TestRepeatCustomerCalcSegment:
    def test_zero_orders_is_new(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=0)
        assert c.calc_segment() == "new"

    def test_none_orders_is_new(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=None)
        assert c.calc_segment() == "new"

    def test_one_order_is_active(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=1)
        assert c.calc_segment() == "active"

    def test_two_orders_is_active(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=2)
        assert c.calc_segment() == "active"

    def test_three_orders_is_vip(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=3)
        assert c.calc_segment() == "vip"

    def test_five_orders_is_vip(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=5)
        assert c.calc_segment() == "vip"

    def test_six_orders_is_at_risk(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=6)
        assert c.calc_segment() == "at_risk"

    def test_ten_orders_is_at_risk(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=10)
        assert c.calc_segment() == "at_risk"

    def test_eleven_orders_is_churned(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=11)
        assert c.calc_segment() == "churned"

    def test_large_orders_is_churned(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=100)
        assert c.calc_segment() == "churned"


class TestRepeatCustomerDaysSinceLast:
    def test_none_last_order_returns_none(self, app):
        c = RepeatCustomer(customer_name="A", last_order_date=None)
        assert c.days_since_last() is None

    def test_recent_last_order(self, app):
        recent = datetime.utcnow() - timedelta(days=3)
        c = RepeatCustomer(customer_name="A", last_order_date=recent)
        days = c.days_since_last()
        assert days is not None
        assert 2 <= days <= 4  # 丸め誤差を考慮

    def test_old_last_order(self, app):
        old = datetime.utcnow() - timedelta(days=100)
        c = RepeatCustomer(customer_name="A", last_order_date=old)
        days = c.days_since_last()
        assert 99 <= days <= 101


class TestRepeatCustomerSegmentLabel:
    def test_new_label(self, app):
        c = RepeatCustomer(customer_name="A", segment="new")
        assert c.segment_label() == "新規"

    def test_active_label(self, app):
        c = RepeatCustomer(customer_name="A", segment="active")
        assert c.segment_label() == "アクティブ"

    def test_vip_label(self, app):
        c = RepeatCustomer(customer_name="A", segment="vip")
        assert c.segment_label() == "VIP"

    def test_at_risk_label(self, app):
        c = RepeatCustomer(customer_name="A", segment="at_risk")
        assert c.segment_label() == "離脱危惧"

    def test_churned_label(self, app):
        c = RepeatCustomer(customer_name="A", segment="churned")
        assert c.segment_label() == "離脱済"

    def test_unknown_segment_label(self, app):
        c = RepeatCustomer(customer_name="A", segment="unknown_xyz")
        assert c.segment_label() == "不明"

    def test_none_segment_defaults_to_new_label(self, app):
        c = RepeatCustomer(customer_name="A", segment=None)
        assert c.segment_label() == "新規"


class TestRepeatCustomerSegmentColor:
    def test_new_color(self, app):
        c = RepeatCustomer(customer_name="A", segment="new")
        assert c.segment_color() == "blue"

    def test_active_color(self, app):
        c = RepeatCustomer(customer_name="A", segment="active")
        assert c.segment_color() == "green"

    def test_vip_color(self, app):
        c = RepeatCustomer(customer_name="A", segment="vip")
        assert c.segment_color() == "purple"

    def test_at_risk_color(self, app):
        c = RepeatCustomer(customer_name="A", segment="at_risk")
        assert c.segment_color() == "yellow"

    def test_churned_color(self, app):
        c = RepeatCustomer(customer_name="A", segment="churned")
        assert c.segment_color() == "red"

    def test_unknown_segment_color(self, app):
        c = RepeatCustomer(customer_name="A", segment="nope")
        assert c.segment_color() == "gray"

    def test_none_segment_defaults_to_new_color(self, app):
        c = RepeatCustomer(customer_name="A", segment=None)
        assert c.segment_color() == "blue"


class TestRepeatCustomerUpdateAvg:
    def test_update_avg_normal(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=4, total_spent=10000.0)
        c.update_avg()
        assert c.avg_order_value == 2500.0

    def test_update_avg_zero_orders_no_update(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=0, total_spent=5000.0, avg_order_value=None)
        c.update_avg()
        assert c.avg_order_value is None

    def test_update_avg_none_orders_no_update(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=None, total_spent=5000.0, avg_order_value=None)
        c.update_avg()
        assert c.avg_order_value is None

    def test_update_avg_none_spent_treated_as_zero(self, app):
        c = RepeatCustomer(customer_name="A", total_orders=2, total_spent=None)
        c.update_avg()
        assert c.avg_order_value == 0.0
