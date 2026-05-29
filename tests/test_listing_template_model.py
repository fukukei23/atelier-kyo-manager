"""Tests for app/models/listing_template.py and app/core/timezone.py"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock
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


class TestTimezone:
    def test_utcnow_returns_naive_datetime(self):
        from app.core.timezone import _utcnow
        result = _utcnow()
        assert isinstance(result, datetime)
        assert result.tzinfo is None

    def test_utcnow_is_recent(self):
        from app.core.timezone import _utcnow
        before = datetime.utcnow()
        result = _utcnow()
        after = datetime.utcnow()
        assert before <= result <= after


class TestListingTemplate:
    def test_create_minimal(self, app):
        with app.app_context():
            from app.models.listing_template import ListingTemplate
            lt = ListingTemplate(name="Basic", template_text="Hello world")
            db.session.add(lt)
            db.session.commit()
            assert lt.id is not None
            assert lt.category == "general"
            assert lt.is_default is False

    def test_create_full(self, app):
        with app.app_context():
            from app.models.listing_template import ListingTemplate
            lt = ListingTemplate(
                name="Luxury Template",
                template_text="Brand: {{brand}}, Price: {{sellingPrice}}",
                category="luxury",
                is_default=True,
            )
            db.session.add(lt)
            db.session.commit()
            assert lt.category == "luxury"
            assert lt.is_default is True

    def test_render_substitutes_placeholders(self, app):
        with app.app_context():
            from app.models.listing_template import ListingTemplate
            lt = ListingTemplate(
                name="Test",
                template_text="{{brand}} {{productName}} {{color}}",
            )
            product = MagicMock()
            product.brand = "Gucci"
            product.name = "Bag"
            product.color = "Red"
            product.material = ""
            product.size = ""
            product.retail_price = None
            product.source_region = ""
            product.description = ""
            product.selling_price = 50000
            result = lt.render(product)
            assert "Gucci" in result
            assert "Bag" in result
            assert "Red" in result

    def test_render_selling_price_formatted(self, app):
        with app.app_context():
            from app.models.listing_template import ListingTemplate
            lt = ListingTemplate(
                name="Price Test",
                template_text="Price: {{sellingPrice}}",
            )
            product = MagicMock()
            product.brand = ""
            product.name = ""
            product.color = ""
            product.material = ""
            product.size = ""
            product.retail_price = None
            product.source_region = ""
            product.description = ""
            product.selling_price = 100000
            result = lt.render(product)
            assert "100,000" in result

    def test_render_none_fields_become_empty(self, app):
        with app.app_context():
            from app.models.listing_template import ListingTemplate
            lt = ListingTemplate(
                name="Null Test",
                template_text="Brand: {{brand}}, Material: {{material}}",
            )
            product = MagicMock()
            product.brand = None
            product.name = None
            product.color = None
            product.material = None
            product.size = None
            product.retail_price = None
            product.source_region = None
            product.description = None
            product.selling_price = None
            result = lt.render(product)
            assert "{{brand}}" not in result
            assert "{{material}}" not in result

    def test_repr(self, app):
        with app.app_context():
            from app.models.listing_template import ListingTemplate
            lt = ListingTemplate(name="MyTemplate", template_text="text")
            assert "MyTemplate" in repr(lt)

    def test_timestamps_set_on_create(self, app):
        with app.app_context():
            from app.models.listing_template import ListingTemplate
            lt = ListingTemplate(name="TS Test", template_text="text")
            db.session.add(lt)
            db.session.commit()
            assert lt.created_at is not None
            assert lt.updated_at is not None
