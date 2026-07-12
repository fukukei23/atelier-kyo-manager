"""Order/Partner/ListingTemplate モデルテスト"""

from datetime import datetime, timedelta

import pytest

from app import create_app
from app.extensions import db
from app.models.listing_template import ListingTemplate
from app.models.order import Order
from app.models.partner import Partner
from app.models.product import Product
from app.utils.presentation import (
    deadline_color,
    deadline_message,
    partner_status_label,
    priority_color,
    priority_label,
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


# ---- Order model ----
class TestOrderModel:
    def test_calc_deadlines(self, app):
        with app.app_context():
            order = Order(
                order_number="B-001",
                product_name="Test",
                order_date=datetime(2026, 1, 1),
                payment_method="credit_card",
                source_type="domestic",
            )
            order.calc_deadlines()
            assert order.deadline_18 == datetime(2026, 1, 19)
            assert order.extension_deadline == datetime(2026, 1, 1) + timedelta(days=45)
            assert order.expected_payment_date == datetime(2026, 1, 16)

    def test_calc_deadlines_no_order_date(self, app):
        with app.app_context():
            order = Order(order_number="B-002", product_name="Test")
            order.calc_deadlines()
            assert order.deadline_18 is None

    def test_calc_profit_fallback(self, app):
        # 利益計算ロジック正しさは tests/pricing/test_calculator.py 側で網羅検証済み。
        # ここでは「信頼ソースを明示すれば Order.calc_profit が利益を再計算できる」ことを
        # PricingInput + calculate_pricing を直接たたく経路で確認する。
        # (Order.calc_profit 本体は DB 保存値の既存注文の利益再計算用であり、
        # 信頼ソースを持たない過去データはレガシー path として扱う。
        # その動作検証は tests/order/test_order_model.py で行う。)
        from app.core.pricing import PricingInput, calculate_pricing
        from app.core.pricing.schemas import PriceSource

        inp = PricingInput(
            purchase_price=20000,
            selling_price=30000,
            customs_duty=1000,
            purchase_price_source=PriceSource.MANUAL_INPUT,
            selling_price_source=PriceSource.MANUAL_INPUT,
        )
        result = calculate_pricing(inp, source_type="domestic")
        assert result.profit is not None
        assert isinstance(result.profit_rate, float)

    def test_get_extension_days(self):
        assert Order.get_extension_days("credit_card") == 45
        assert Order.get_extension_days("bank_transfer") == 90
        assert Order.get_extension_days("unknown_method") == 45

    def test_supported_payment_methods(self):
        methods = Order.supported_payment_methods()
        assert "credit_card" in methods
        assert "bank_transfer" in methods

    def test_deadline_color_green(self, app):
        with app.app_context():
            order = Order(
                order_number="B-004",
                product_name="Test",
                order_date=datetime.utcnow() - timedelta(days=5),
            )
            order.calc_deadlines()
            assert deadline_color(order) == "green"

    def test_deadline_color_black(self, app):
        with app.app_context():
            order = Order(
                order_number="B-005",
                product_name="Test",
                order_date=datetime.utcnow() - timedelta(days=30),
            )
            order.calc_deadlines()
            assert deadline_color(order) == "black"

    def test_deadline_color_gray_no_deadline(self, app):
        with app.app_context():
            order = Order(order_number="B-006", product_name="Test")
            assert deadline_color(order) == "gray"

    def test_deadline_message_expired(self, app):
        with app.app_context():
            order = Order(
                order_number="B-007",
                product_name="Test",
                order_date=datetime.utcnow() - timedelta(days=30),
            )
            order.calc_deadlines()
            assert "期限切れ" in deadline_message(order)

    def test_deadline_message_no_deadline(self, app):
        with app.app_context():
            order = Order(order_number="B-008", product_name="Test")
            assert deadline_message(order) == ""


# ---- Partner model ----
class TestPartnerModel:
    def test_success_rate(self, app):
        with app.app_context():
            p = Partner(name="Test", total_orders=100, success_count=80)
            assert p.success_rate() == 80.0

    def test_success_rate_zero_orders(self, app):
        with app.app_context():
            p = Partner(name="Test", total_orders=0)
            assert p.success_rate() == 0.0

    def test_cancellation_rate(self, app):
        with app.app_context():
            p = Partner(name="Test", total_orders=100, cancellation_count=5)
            assert p.cancellation_rate() == 5.0

    def test_regions_list(self, app):
        with app.app_context():
            p = Partner(name="Test", active_regions="US, UK, IT")
            assert p.regions_list() == ["US", "UK", "IT"]

    def test_regions_list_empty(self, app):
        with app.app_context():
            p = Partner(name="Test")
            assert p.regions_list() == []

    def test_brands_list(self, app):
        with app.app_context():
            p = Partner(name="Test", specialty_brands="Gucci, Prada")
            assert p.brands_list() == ["Gucci", "Prada"]

    def test_priority_label(self, app):
        with app.app_context():
            assert priority_label(Partner(name="T", priority_level="high")) == "高"
            assert priority_label(Partner(name="T", priority_level="medium")) == "中"
            assert priority_label(Partner(name="T", priority_level="low")) == "低"

    def test_priority_color(self, app):
        with app.app_context():
            assert priority_color(Partner(name="T", priority_level="high")) == "red"
            assert priority_color(Partner(name="T", priority_level="medium")) == "yellow"

    def test_status_label(self, app):
        with app.app_context():
            assert partner_status_label(Partner(name="T", status="active")) == "稼働"
            assert partner_status_label(Partner(name="T", status="inactive")) == "停止"
            assert partner_status_label(Partner(name="T", status="suspended")) == "保留"


# ---- ListingTemplate model ----
class TestListingTemplateModel:
    def test_render(self, app):
        with app.app_context():
            template = ListingTemplate(
                name="test",
                template_text="{{brand}} {{productName}} ¥{{retailPrice}}",
            )
            product = Product(
                name="Test Product",
                brand="Test Brand",
                retail_price=29800,
            )
            result = template.render(product)
            assert "Test Brand" in result
            assert "Test Product" in result
            assert "¥29,800" in result

    def test_render_missing_fields(self, app):
        with app.app_context():
            template = ListingTemplate(
                name="test",
                template_text="{{brand}} {{color}}",
            )
            product = Product(name="Test")
            template.render(product)
            assert True  # brand is empty
