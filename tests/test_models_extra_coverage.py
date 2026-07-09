"""Models extra coverage tests for untested model files.

Uses a temp-file SQLite database per test class to avoid in-memory DB
context issues with many sequential tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app import create_app
from app.extensions import db
from app.models.brand_price import BrandPrice
from app.models.customer_inquiry import CustomerInquiry
from app.models.partner import Partner
from app.models.popularity_tracker import PopularityTracker
from app.models.product import Product
from app.models.prohibited_source import ProhibitedSource
from app.models.region_recommendation import RegionRecommendation
from app.models.repeat_customer import RepeatCustomer
from app.models.stock_check import StockCheck


@pytest.fixture(scope="function")
def app(tmp_path_factory, monkeypatch):
    """Temp-file SQLite Flask app for testing — 各テストで独立したDBを使用"""
    db_dir = tmp_path_factory.mktemp("db")
    db_path = str(db_dir / "test.db")

    # create_app前にDATABASE_URLをセット（AppConfig.get_db_url()据此）
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    application = create_app()
    application.config.update(
        {
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test-secret",
        }
    )
    with application.app_context():
        yield application
        db.session.remove()
        db.drop_all()
    if Path(db_path).exists():
        Path(db_path).unlink()


# =====================================================================
# Partner model
# =====================================================================
class TestPartnerModelExtended:
    """Partner model — edge cases not covered in test_models_extra."""

    def test_repr(self, app):
        p = Partner(name="TestPartner")
        db.session.add(p)
        db.session.flush()
        assert "TestPartner" in repr(p)

    def test_success_rate_none_counts(self, app):
        p = Partner(name="T", total_orders=10, success_count=None)
        assert p.success_rate() == 0.0

    def test_cancellation_rate_none_counts(self, app):
        p = Partner(name="T", total_orders=10, cancellation_count=None)
        assert p.cancellation_rate() == 0.0

    def test_regions_list_whitespace(self, app):
        p = Partner(name="T", active_regions="  US ,  UK , IT  ")
        assert p.regions_list() == ["US", "UK", "IT"]

    def test_brands_list_empty_string(self, app):
        p = Partner(name="T", specialty_brands="")
        assert p.brands_list() == []

    def test_regions_list_comma_only(self, app):
        p = Partner(name="T", active_regions=",,,")
        assert p.regions_list() == []

    def test_brands_list_single(self, app):
        p = Partner(name="T", specialty_brands="Gucci")
        assert p.brands_list() == ["Gucci"]

    def test_success_rate_100_percent(self, app):
        p = Partner(name="T", total_orders=50, success_count=50)
        assert p.success_rate() == 100.0

    def test_cancellation_rate_zero(self, app):
        p = Partner(name="T", total_orders=50, cancellation_count=0)
        assert p.cancellation_rate() == 0.0


# =====================================================================
# StockCheck model
# =====================================================================
class TestStockCheckModel:
    def test_price_diff_positive(self, app):
        sc = StockCheck(product_id=1, current_price=12000, previous_price=10000)
        assert sc.price_diff() == 2000.0

    def test_price_diff_negative(self, app):
        sc = StockCheck(product_id=1, current_price=8000, previous_price=10000)
        assert sc.price_diff() == -2000.0

    def test_price_diff_none_missing_current(self, app):
        sc = StockCheck(product_id=1, previous_price=10000)
        assert sc.price_diff() is None

    def test_price_diff_none_missing_previous(self, app):
        sc = StockCheck(product_id=1, current_price=10000)
        assert sc.price_diff() is None

    def test_price_diff_pct_positive(self, app):
        sc = StockCheck(product_id=1, current_price=12000, previous_price=10000)
        assert sc.price_diff_pct() == 20.0

    def test_price_diff_pct_negative(self, app):
        sc = StockCheck(product_id=1, current_price=8000, previous_price=10000)
        assert sc.price_diff_pct() == -20.0

    def test_price_diff_pct_division_by_zero(self, app):
        sc = StockCheck(product_id=1, current_price=5000, previous_price=0)
        assert sc.price_diff_pct() is None

    def test_price_diff_pct_none_when_missing(self, app):
        sc = StockCheck(product_id=1)
        assert sc.price_diff_pct() is None

    def test_to_dict_basic(self, app):
        now = datetime(2026, 5, 1, 12, 0, 0)
        sc = StockCheck(
            product_id=1,
            source_url="https://example.com",
            current_price=10000.0,
            previous_price=9000.0,
            price_changed=True,
            in_stock=True,
            stock_changed=False,
            checked_at=now,
            notes="test note",
        )
        d = sc.to_dict()
        assert d["product_id"] == 1
        assert d["current_price"] == 10000.0
        assert d["in_stock"] is True
        assert d["checked_at"] == now.isoformat()
        assert d["notes"] == "test note"

    def test_to_dict_no_checked_at(self, app):
        sc = StockCheck(product_id=1)
        d = sc.to_dict()
        assert d["checked_at"] is None

    def test_to_dict_all_keys(self, app):
        sc = StockCheck(product_id=1)
        d = sc.to_dict()
        assert set(d.keys()) == {
            "id",
            "product_id",
            "source_url",
            "current_price",
            "previous_price",
            "price_changed",
            "in_stock",
            "stock_changed",
            "checked_at",
            "notes",
        }

    def test_db_insert_and_query(self, app):
        prod = Product(
            name="SCProd",
            brand="Test",
            purchase_price=8000,
            selling_price=12000,
            retail_price=10000,
        )
        db.session.add(prod)
        db.session.flush()
        sc = StockCheck(product_id=prod.id, current_price=9500.0, previous_price=10000.0, in_stock=True)
        db.session.add(sc)
        db.session.commit()
        fetched = db.session.scalar(db.select(StockCheck).filter_by(product_id=prod.id))
        assert fetched is not None
        assert fetched.price_diff() == -500.0


# =====================================================================
# ProhibitedSource model
# =====================================================================
class TestProhibitedSourceModel:
    def test_is_prohibited_empty_url(self, app):
        result, reason = ProhibitedSource.is_prohibited("")
        assert result is False
        assert reason == ""

    def test_is_prohibited_none_url(self, app):
        result, reason = ProhibitedSource.is_prohibited(None)
        assert result is False

    def test_is_prohibited_exact_match(self, app):
        db.session.add(
            ProhibitedSource(domain="badsite.com", reason="詐欺", severity="blocked", source_type="domestic")
        )
        db.session.commit()
        result, reason = ProhibitedSource.is_prohibited("https://badsite.com/product/123")
        assert result is True
        assert "詐欺" in reason

    def test_is_prohibited_subdomain(self, app):
        db.session.add(ProhibitedSource(domain="sub-bad.com", reason="", severity="blocked", source_type="domestic"))
        db.session.commit()
        result, reason = ProhibitedSource.is_prohibited("https://shop.sub-bad.com/item")
        assert result is True
        assert "sub-bad.com" in reason

    def test_is_prohibited_no_match(self, app):
        db.session.add(ProhibitedSource(domain="nm-bad.com", reason="", severity="blocked", source_type="domestic"))
        db.session.commit()
        result, reason = ProhibitedSource.is_prohibited("https://goodsite.com/product")
        assert result is False
        assert reason == ""

    def test_is_prohibited_source_type_filter(self, app):
        db.session.add(
            ProhibitedSource(domain="tf-bad.com", reason="blocked", severity="blocked", source_type="overseas")
        )
        db.session.commit()
        result, _ = ProhibitedSource.is_prohibited("https://tf-bad.com/product", source_type="domestic")
        assert result is False
        result, _ = ProhibitedSource.is_prohibited("https://tf-bad.com/product", source_type="overseas")
        assert result is True

    def test_repr(self, app):
        ps = ProhibitedSource(domain="test.com", severity="warning")
        assert "test.com" in repr(ps)
        assert "warning" in repr(ps)

    def test_is_prohibited_case_insensitive(self, app):
        db.session.add(
            ProhibitedSource(domain="BadSite.COM", reason="test", severity="blocked", source_type="domestic")
        )
        db.session.commit()
        result, _ = ProhibitedSource.is_prohibited("https://badsite.com/page")
        assert result is True


# =====================================================================
# RegionRecommendation model
# =====================================================================
class TestRegionRecommendationModel:
    def test_calc_recommendation(self, app):
        rr = RegionRecommendation(region="US", reliability_score=80.0, risk_score=20.0, avg_profit_rate=30.0)
        expected = 80.0 * 0.4 + (100 - 20.0) * 0.3 + 30.0 * 0.3
        assert rr.calc_recommendation() == pytest.approx(expected)

    def test_calc_recommendation_missing_reliability(self, app):
        rr = RegionRecommendation(region="US", risk_score=20.0, avg_profit_rate=30.0)
        assert rr.calc_recommendation() is None

    def test_calc_recommendation_missing_risk(self, app):
        rr = RegionRecommendation(region="US", reliability_score=80.0, avg_profit_rate=30.0)
        assert rr.calc_recommendation() is None

    def test_calc_recommendation_missing_profit(self, app):
        rr = RegionRecommendation(region="US", reliability_score=80.0, risk_score=20.0)
        assert rr.calc_recommendation() is None

    def test_calc_recommendation_all_zero(self, app):
        rr = RegionRecommendation(region="US", reliability_score=0.0, risk_score=0.0, avg_profit_rate=0.0)
        # 0*0.4 + (100-0)*0.3 + 0*0.3 = 30.0
        assert rr.calc_recommendation() == pytest.approx(30.0)

    def test_db_insert(self, app):
        rr = RegionRecommendation(
            region="IT",
            region_name="Italy",
            total_products=50,
            avg_profit_rate=25.0,
            reliability_score=90.0,
            risk_score=10.0,
        )
        db.session.add(rr)
        db.session.commit()
        fetched = db.session.scalar(db.select(RegionRecommendation).filter_by(region="IT"))
        assert fetched is not None
        assert fetched.region_name == "Italy"
        assert fetched.calc_recommendation() is not None


# =====================================================================
# PopularityTracker model
# =====================================================================
class TestPopularityTrackerModel:
    def test_calc_score_basic(self, app):
        pt = PopularityTracker(product_id=1, views=100, favorites=10, inquiries=5, sold_count=2)
        expected = 100 * 0.1 + 10 * 2 + 5 * 3 + 2 * 5
        assert pt.calc_score() == pytest.approx(expected)

    def test_calc_score_all_zero(self, app):
        pt = PopularityTracker(product_id=1, views=0, favorites=0, inquiries=0, sold_count=0)
        assert pt.calc_score() == 0.0

    def test_calc_score_none_values(self, app):
        pt = PopularityTracker(product_id=1)
        assert pt.calc_score() == 0.0

    def test_sold_count_or_zero_with_value(self, app):
        pt = PopularityTracker(product_id=1, sold_count=7)
        assert pt.sold_count_or_zero() == 7

    def test_sold_count_or_zero_with_none(self, app):
        pt = PopularityTracker(product_id=1)
        assert pt.sold_count_or_zero() == 0

    def test_db_insert_with_product(self, app):
        prod = Product(name="PopProd", brand="Test", purchase_price=3000, selling_price=5000, retail_price=5000)
        db.session.add(prod)
        db.session.flush()
        pt = PopularityTracker(product_id=prod.id, views=200, favorites=15, inquiries=8, sold_count=3)
        db.session.add(pt)
        db.session.commit()
        fetched = db.session.scalar(db.select(PopularityTracker).filter_by(product_id=prod.id))
        assert fetched is not None
        assert fetched.calc_score() > 0

    def test_calc_score_views_only(self, app):
        assert PopularityTracker(product_id=1, views=500).calc_score() == pytest.approx(50.0)

    def test_calc_score_sold_only(self, app):
        assert PopularityTracker(product_id=1, sold_count=10).calc_score() == pytest.approx(50.0)


# =====================================================================
# RepeatCustomer model
# =====================================================================
class TestRepeatCustomerModel:
    def test_calc_segment_new(self, app):
        assert RepeatCustomer(customer_name="T", total_orders=0).calc_segment() == "new"

    def test_calc_segment_active(self, app):
        assert RepeatCustomer(customer_name="T", total_orders=2).calc_segment() == "active"

    def test_calc_segment_vip(self, app):
        assert RepeatCustomer(customer_name="T", total_orders=4).calc_segment() == "vip"

    def test_calc_segment_at_risk(self, app):
        assert RepeatCustomer(customer_name="T", total_orders=8).calc_segment() == "at_risk"

    def test_calc_segment_churned(self, app):
        assert RepeatCustomer(customer_name="T", total_orders=15).calc_segment() == "churned"

    def test_calc_segment_none_orders(self, app):
        assert RepeatCustomer(customer_name="T", total_orders=None).calc_segment() == "new"

    def test_days_since_last_no_date(self, app):
        assert RepeatCustomer(customer_name="T").days_since_last() is None

    def test_days_since_last_recent(self, app):
        rc = RepeatCustomer(customer_name="T", last_order_date=datetime.utcnow() - timedelta(days=3))
        assert rc.days_since_last() == 3

    def test_update_avg_basic(self, app):
        rc = RepeatCustomer(customer_name="T", total_orders=5, total_spent=50000)
        rc.update_avg()
        assert rc.avg_order_value == 10000.0

    def test_update_avg_zero_orders(self, app):
        rc = RepeatCustomer(customer_name="T", total_orders=0, total_spent=50000)
        rc.update_avg()
        assert rc.avg_order_value is None

    def test_update_avg_none_spent(self, app):
        rc = RepeatCustomer(customer_name="T", total_orders=5, total_spent=None)
        rc.update_avg()
        assert rc.avg_order_value == 0.0

    def test_db_insert(self, app):
        rc = RepeatCustomer(customer_name="John", email="j@e.com", total_orders=3, total_spent=30000, segment="vip")
        db.session.add(rc)
        db.session.commit()
        fetched = db.session.scalar(db.select(RepeatCustomer).filter_by(customer_name="John"))
        assert fetched is not None
        assert fetched.calc_segment() == "vip"
        fetched.update_avg()
        assert fetched.avg_order_value == 10000.0


# =====================================================================
# CustomerInquiry model
# =====================================================================
class TestCustomerInquiryModel:
    def test_mark_matched(self, app):
        ci = CustomerInquiry(customer_name="Taro", customer_email="t@e.com", subject="Size", message="msg")
        ci.mark_matched(faq_template_id=1, response="S, M, L, XL")
        assert ci.status == "matched"
        assert ci.faq_template_id == 1
        assert "S, M, L" in ci.response_text

    def test_mark_approved(self, app):
        ci = CustomerInquiry(subject="T", message="m", status="matched")
        ci.mark_approved()
        assert ci.status == "approved"

    def test_mark_rejected(self, app):
        ci = CustomerInquiry(subject="T", message="m")
        ci.mark_rejected(reason="Needs review")
        assert ci.status == "rejected"
        assert "Needs review" in ci.escalation_reason

    def test_mark_rejected_empty_reason(self, app):
        ci = CustomerInquiry(subject="T", message="m")
        ci.mark_rejected()
        assert ci.status == "rejected"
        assert ci.escalation_reason == ""

    def test_can_approve_matched(self, app):
        assert CustomerInquiry(subject="T", message="m", status="matched").can_approve() is True

    def test_can_approve_pending(self, app):
        assert CustomerInquiry(subject="T", message="m", status="pending").can_approve() is False

    def test_can_approve_approved(self, app):
        assert CustomerInquiry(subject="T", message="m", status="approved").can_approve() is False

    def test_repr(self, app):
        ci = CustomerInquiry(subject="T", message="m", status="pending")
        assert "pending" in repr(ci)

    def test_db_insert_and_lifecycle(self, app):
        ci = CustomerInquiry(customer_name="Hanako", customer_email="h@e.com", subject="Ship", message="When?")
        db.session.add(ci)
        db.session.commit()
        fetched = db.session.scalar(db.select(CustomerInquiry).filter_by(customer_name="Hanako"))
        assert fetched is not None
        assert fetched.status == "pending"
        fetched.mark_matched(5, "5-7 days")
        db.session.commit()
        assert fetched.can_approve() is True
        fetched.mark_approved()
        db.session.commit()
        assert fetched.status == "approved"
        assert fetched.can_approve() is False


# =====================================================================
# BrandPrice model
# =====================================================================
class TestBrandPriceModel:
    def test_repr(self, app):
        bp = BrandPrice(
            brand="Gucci",
            product_name="Belt",
            source_site="gucci.com",
            price_original=350.0,
            currency="EUR",
            price_jpy=52500.0,
            exchange_rate=150.0,
        )
        r = repr(bp)
        assert "Gucci" in r
        assert "gucci.com" in r

    def test_get_by_brand(self, app):
        bp1 = BrandPrice(
            brand="Prada",
            product_name="Bag A",
            source_site="prada.com",
            price_original=1000,
            currency="EUR",
            price_jpy=150000,
            exchange_rate=150,
        )
        bp2 = BrandPrice(
            brand="Prada",
            product_name="Bag B",
            source_site="other.com",
            price_original=800,
            currency="EUR",
            price_jpy=120000,
            exchange_rate=150,
        )
        bp3 = BrandPrice(
            brand="Gucci",
            product_name="Belt",
            source_site="gucci.com",
            price_original=400,
            currency="EUR",
            price_jpy=60000,
            exchange_rate=150,
        )
        db.session.add_all([bp1, bp2, bp3])
        db.session.commit()
        results = BrandPrice.get_by_brand("Prada")
        assert len(results) == 2
        assert results[0].price_jpy == 120000
        assert results[1].price_jpy == 150000

    def test_get_by_brand_empty(self, app):
        assert BrandPrice.get_by_brand("NonExistent") == []

    def test_get_latest_by_brand_site(self, app):
        old = BrandPrice(
            brand="LV",
            product_name="Wallet",
            source_site="lv.com",
            price_original=500,
            currency="EUR",
            price_jpy=75000,
            exchange_rate=150,
            scraped_at=datetime(2026, 1, 1),
        )
        new = BrandPrice(
            brand="LV",
            product_name="Wallet",
            source_site="lv.com",
            price_original=520,
            currency="EUR",
            price_jpy=78000,
            exchange_rate=150,
            scraped_at=datetime(2026, 3, 1),
        )
        db.session.add_all([old, new])
        db.session.commit()
        result = BrandPrice.get_latest_by_brand_site("LV", "lv.com")
        assert result is not None
        assert result.price_jpy == 78000

    def test_get_latest_no_match(self, app):
        assert BrandPrice.get_latest_by_brand_site("None", "none.com") is None

    def test_db_insert_basic(self, app):
        bp = BrandPrice(
            brand="Chanel",
            product_name="Classic",
            source_site="chanel.com",
            price_original=8000,
            currency="EUR",
            price_jpy=1200000,
            exchange_rate=150,
            in_stock=True,
            size_available="M",
        )
        db.session.add(bp)
        db.session.commit()
        fetched = db.session.scalar(db.select(BrandPrice).filter_by(brand="Chanel"))
        assert fetched is not None
        assert fetched.in_stock is True
        assert fetched.size_available == "M"
