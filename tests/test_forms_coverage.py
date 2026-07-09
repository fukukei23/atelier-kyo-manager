"""Tests for app/forms.py"""

import pytest

from app import create_app
from app.extensions import db
from app.forms import (
    AutoResearchForm,
    CustomerForm,
    ListingTemplateForm,
    OrderForm,
    PartnerForm,
    ProductForm,
)


@pytest.fixture(scope="function")
def app():
    """In-memory SQLite Flask app for testing."""
    application = create_app()
    application.config.update(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test-secret",
        }
    )
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


class TestProductForm:
    def test_valid_minimal(self, app):
        with app.app_context():
            form = ProductForm(
                data={"name": "Test Product", "purchase_price": 1000.0, "selling_price": 1500.0},
                meta={"csrf": False},
            )
            assert form.validate(), form.errors

    def test_valid_full(self, app):
        with app.app_context():
            form = ProductForm(
                data={
                    "name": "Gucci Bag",
                    "brand": "Gucci",
                    "purchase_price": 80000.0,
                    "selling_price": 120000.0,
                    "transaction_fee": 12000.0,
                    "shipping_cost": 3000.0,
                    "customs_duty": 5000.0,
                    "original_currency": "EUR",
                    "exchange_rate": 160.0,
                    "listing_status": "draft",
                    "target_profit_rate": 15.0,
                },
                meta={"csrf": False},
            )
            assert form.validate(), form.errors

    def test_missing_name(self, app):
        with app.app_context():
            form = ProductForm(
                data={"purchase_price": 1000.0, "selling_price": 1500.0},
                meta={"csrf": False},
            )
            assert not form.validate()
            assert "name" in form.errors

    def test_missing_purchase_price(self, app):
        with app.app_context():
            form = ProductForm(
                data={"name": "Test Product", "selling_price": 1500.0},
                meta={"csrf": False},
            )
            assert not form.validate()
            assert "purchase_price" in form.errors

    def test_missing_selling_price(self, app):
        with app.app_context():
            form = ProductForm(
                data={"name": "Test Product", "purchase_price": 1000.0},
                meta={"csrf": False},
            )
            assert not form.validate()
            assert "selling_price" in form.errors

    def test_negative_price_rejected(self, app):
        with app.app_context():
            form = ProductForm(
                data={"name": "Test", "purchase_price": -100.0, "selling_price": 1500.0},
                meta={"csrf": False},
            )
            assert not form.validate()
            assert "purchase_price" in form.errors

    def test_valid_with_url(self, app):
        with app.app_context():
            form = ProductForm(
                data={
                    "name": "Test",
                    "purchase_price": 1000.0,
                    "selling_price": 1500.0,
                    "supplier_url": "https://example.com/product",
                },
                meta={"csrf": False},
            )
            assert form.validate(), form.errors

    def test_valid_profit_rate_boundary(self, app):
        with app.app_context():
            form = ProductForm(
                data={
                    "name": "Test",
                    "purchase_price": 1000.0,
                    "selling_price": 1500.0,
                    "target_profit_rate": 100.0,
                },
                meta={"csrf": False},
            )
            assert form.validate(), form.errors


class TestOrderForm:
    def test_valid(self, app):
        with app.app_context():
            form = OrderForm(
                data={
                    "order_number": "ORD-001",
                    "product_name": "Widget",
                    "selling_price": 50.0,
                    "purchase_cost": 30.0,
                },
                meta={"csrf": False},
            )
            assert form.validate(), form.errors

    def test_missing_order_number(self, app):
        with app.app_context():
            form = OrderForm(
                data={"product_name": "Widget", "selling_price": 50.0, "purchase_cost": 30.0},
                meta={"csrf": False},
            )
            assert not form.validate()
            assert "order_number" in form.errors

    def test_missing_product_name(self, app):
        with app.app_context():
            form = OrderForm(
                data={"order_number": "ORD-001", "selling_price": 50.0, "purchase_cost": 30.0},
                meta={"csrf": False},
            )
            assert not form.validate()
            assert "product_name" in form.errors

    def test_order_number_too_long(self, app):
        with app.app_context():
            form = OrderForm(
                data={
                    "order_number": "X" * 65,
                    "product_name": "Widget",
                    "selling_price": 50.0,
                    "purchase_cost": 30.0,
                },
                meta={"csrf": False},
            )
            assert not form.validate()
            assert "order_number" in form.errors


class TestPartnerForm:
    def test_valid(self, app):
        with app.app_context():
            form = PartnerForm(data={"name": "Alice"}, meta={"csrf": False})
            assert form.validate(), form.errors

    def test_valid_with_email(self, app):
        with app.app_context():
            form = PartnerForm(
                data={"name": "Alice", "email": "alice@example.com", "status": "active"},
                meta={"csrf": False},
            )
            assert form.validate(), form.errors

    def test_missing_name(self, app):
        with app.app_context():
            form = PartnerForm(data={}, meta={"csrf": False})
            assert not form.validate()
            assert "name" in form.errors

    def test_valid_with_optional_fields(self, app):
        with app.app_context():
            form = PartnerForm(
                data={"name": "Alice", "phone": "090-1234-5678", "status": "active"},
                meta={"csrf": False},
            )
            assert form.validate(), form.errors


class TestCustomerForm:
    def test_valid(self, app):
        with app.app_context():
            form = CustomerForm(data={"customer_name": "Bob"}, meta={"csrf": False})
            assert form.validate(), form.errors

    def test_missing_customer_name(self, app):
        with app.app_context():
            form = CustomerForm(data={}, meta={"csrf": False})
            assert not form.validate()
            assert "customer_name" in form.errors


class TestListingTemplateForm:
    def test_valid(self, app):
        with app.app_context():
            form = ListingTemplateForm(
                data={"name": "Default Template", "template_text": "Product: {{brand}}"},
                meta={"csrf": False},
            )
            assert form.validate(), form.errors

    def test_missing_name(self, app):
        with app.app_context():
            form = ListingTemplateForm(
                data={"template_text": "Some text"},
                meta={"csrf": False},
            )
            assert not form.validate()
            assert "name" in form.errors

    def test_missing_template_text(self, app):
        with app.app_context():
            form = ListingTemplateForm(
                data={"name": "My Template"},
                meta={"csrf": False},
            )
            assert not form.validate()
            assert "template_text" in form.errors


class TestAutoResearchForm:
    def test_valid(self, app):
        with app.app_context():
            form = AutoResearchForm(data={}, meta={"csrf": False})
            assert form.validate(), form.errors
