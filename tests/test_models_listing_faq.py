"""Tests for app/models/listing_progress.py and app/models/faq_template.py"""
import pytest
from datetime import date
from app import create_app
from app.extensions import db


@pytest.fixture(scope="function")
def app():
    application = create_app()
    application.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "WTF_CSRF_ENABLED": False,
        "SECRET_KEY": "test-secret",
    })
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


class TestListingProgress:
    def test_create_default(self, app):
        with app.app_context():
            from app.models.listing_progress import ListingProgress
            lp = ListingProgress(listings_count=5)
            db.session.add(lp)
            db.session.commit()
            assert lp.id is not None
            assert lp.target_daily == 20
            assert lp.target_monthly == 600
            assert lp.listings_count == 5

    def test_create_with_all_fields(self, app):
        with app.app_context():
            from app.models.listing_progress import ListingProgress
            lp = ListingProgress(
                record_date=date(2024, 1, 15),
                listings_count=25,
                target_daily=30,
                target_monthly=900,
                cumulative_monthly=150,
                notes="Test notes",
            )
            db.session.add(lp)
            db.session.commit()
            assert lp.record_date == date(2024, 1, 15)
            assert lp.cumulative_monthly == 150
            assert lp.notes == "Test notes"

    def test_daily_achievement_rate(self, app):
        with app.app_context():
            from app.models.listing_progress import ListingProgress
            lp = ListingProgress(listings_count=10, target_daily=20)
            assert lp.daily_achievement_rate() == 50.0

    def test_daily_achievement_rate_caps_at_100(self, app):
        with app.app_context():
            from app.models.listing_progress import ListingProgress
            lp = ListingProgress(listings_count=30, target_daily=20)
            assert lp.daily_achievement_rate() == 100.0

    def test_daily_achievement_rate_zero_target(self, app):
        with app.app_context():
            from app.models.listing_progress import ListingProgress
            lp = ListingProgress(listings_count=5, target_daily=0)
            assert lp.daily_achievement_rate() == 0.0

    def test_monthly_achievement_rate(self, app):
        with app.app_context():
            from app.models.listing_progress import ListingProgress
            lp = ListingProgress(cumulative_monthly=300, target_monthly=600)
            assert lp.monthly_achievement_rate() == 50.0

    def test_monthly_achievement_rate_caps_at_100(self, app):
        with app.app_context():
            from app.models.listing_progress import ListingProgress
            lp = ListingProgress(cumulative_monthly=800, target_monthly=600)
            assert lp.monthly_achievement_rate() == 100.0

    def test_get_monthly_summary(self, app):
        with app.app_context():
            from app.models.listing_progress import ListingProgress
            lp1 = ListingProgress(
                record_date=date(2024, 1, 5),
                listings_count=20,
                cumulative_monthly=20,
            )
            lp2 = ListingProgress(
                record_date=date(2024, 1, 15),
                listings_count=25,
                cumulative_monthly=45,
            )
            db.session.add_all([lp1, lp2])
            db.session.commit()
            summary = ListingProgress.get_monthly_summary(2024, 1)
            assert summary["total"] == 45
            assert summary["days"] == 2
            assert summary["daily_avg"] == 22.5

    def test_get_monthly_summary_no_records(self, app):
        with app.app_context():
            from app.models.listing_progress import ListingProgress
            summary = ListingProgress.get_monthly_summary(2024, 6)
            assert summary["total"] == 0
            assert summary["days"] == 0

    def test_repr(self, app):
        with app.app_context():
            from app.models.listing_progress import ListingProgress
            lp = ListingProgress(listings_count=5, record_date=date(2024, 1, 1))
            assert "ListingProgress" in repr(lp)


class TestFaqTemplate:
    def test_create_default(self, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            ft = FaqTemplate(
                question_pattern="help,support",
                answer_template="Contact us at support@example.com",
            )
            db.session.add(ft)
            db.session.commit()
            assert ft.id is not None
            assert ft.category == "general"
            assert ft.is_active is True

    def test_create_with_custom_fields(self, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            ft = FaqTemplate(
                category="billing",
                question_pattern="payment,invoice,charge",
                answer_template="Billing info here",
                is_active=False,
            )
            db.session.add(ft)
            db.session.commit()
            assert ft.category == "billing"
            assert ft.is_active is False

    def test_match_positive(self, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            ft = FaqTemplate(
                question_pattern="reset,password,forgot",
                answer_template="Reset your password",
            )
            assert ft.match("I forgot my password") is True

    def test_match_negative(self, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            ft = FaqTemplate(
                question_pattern="reset,password,forgot",
                answer_template="Reset your password",
            )
            assert ft.match("How do I change my email?") is False

    def test_match_case_insensitive(self, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            ft = FaqTemplate(
                question_pattern="HELLO,WORLD",
                answer_template="Hello world",
            )
            assert ft.match("say hello to everyone") is True

    def test_match_empty_text(self, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            ft = FaqTemplate(question_pattern="test", answer_template="Answer")
            assert ft.match("") is False
            assert ft.match(None) is False

    def test_render_with_kwargs(self, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            ft = FaqTemplate(
                question_pattern="test",
                answer_template="Hello {name}, your order {order_id} is ready",
            )
            result = ft.render(name="Alice", order_id="12345")
            assert result == "Hello Alice, your order 12345 is ready"

    def test_render_missing_kwargs_fallback_to_empty(self, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            ft = FaqTemplate(
                question_pattern="test",
                answer_template="Welcome {name}, code: {code}",
            )
            # Undefined placeholders fall back to empty string
            result = ft.render(name="Bob")
            assert "Bob" in result
            assert "{code}" not in result

    def test_render_no_placeholders(self, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            ft = FaqTemplate(
                question_pattern="test",
                answer_template="No placeholders here",
            )
            result = ft.render(extra="ignored")
            assert result == "No placeholders here"
