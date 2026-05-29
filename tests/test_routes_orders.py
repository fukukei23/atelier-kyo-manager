"""Tests for app/utils/order_model.py - Order calculations"""

from datetime import datetime, timezone

import pytest

from app.models.order import Order


# ---------- Order model calculation methods ----------


class TestOrderCalculations:
    def test_order_has_calc_profit(self):
        order = Order(order_number="T1", product_name="Test")
        assert hasattr(order, "calc_profit")

    def test_order_has_calc_deadlines(self):
        order = Order(order_number="T2", product_name="Test")
        assert hasattr(order, "calc_deadlines")

    def test_order_has_remaining_days(self):
        order = Order(order_number="T3", product_name="Test")
        assert hasattr(order, "remaining_days")

    def test_order_has_profit_attribute(self):
        order = Order(order_number="T4", product_name="Test", selling_price=100000)
        assert hasattr(order, "profit")

    def test_order_order_number_required(self):
        order = Order(order_number="ORD-001", product_name="Test")
        assert order.order_number == "ORD-001"


# ---------- ShippingAgent constants ----------


class TestShippingAgentConstants:
    def test_shipping_agent_instantiation(self):
        from app.utils.shipping_agent import ShippingAgent

        agent = ShippingAgent()
        assert agent is not None

    def test_brand_price_scraper_instantiation(self):
        from app.services.brand_price_scraper import BrandPriceScraper

        scraper = BrandPriceScraper()
        assert scraper is not None


# ---------- ShippingAgent method checks ----------


class TestShippingAgentMethods:
    def test_shipping_agent_get_warehouses_method(self):
        from app.utils.shipping_agent import ShippingAgent

        agent = ShippingAgent()
        assert callable(agent.get_warehouses_by_country)

    def test_shipping_agent_async_method(self):
        from app.utils.shipping_agent import ShippingAgent

        agent = ShippingAgent()
        assert callable(agent._get_warehouses_async)
