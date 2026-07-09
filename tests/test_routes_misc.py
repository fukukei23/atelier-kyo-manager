"""Tests for app/routes/misc.py"""

from unittest.mock import patch

import pytest

from app import create_app
from app.extensions import db


@pytest.fixture(scope="function")
def app():
    application = create_app()
    application.config.update(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test-secret",
            "LOGIN_DISABLED": True,
        }
    )
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


class TestLegacyRedirects:
    def test_dashboard_accessible(self, client):
        # /dashboard redirects to cashflow_dashboard which returns 200 with LOGIN_DISABLED
        resp = client.get("/dashboard", follow_redirects=True)
        assert resp.status_code in (200, 302)

    def test_listing_templates_redirects(self, client):
        resp = client.get("/listing-templates")
        assert resp.status_code == 302

    def test_region_recommendations_redirects(self, client):
        resp = client.get("/region-recommendations")
        assert resp.status_code == 302

    def test_repeat_customers_redirects(self, client):
        resp = client.get("/repeat-customers")
        assert resp.status_code == 302


class TestAutoResearchRoute:
    def test_get_auto_research_returns_200(self, client):
        resp = client.get("/auto-research")
        assert resp.status_code == 200

    def test_post_auto_research_returns_200(self, client):
        resp = client.post("/auto-research", data={})
        assert resp.status_code == 200


class TestImageCrawlerRoute:
    def test_get_image_crawler_returns_200(self, client):
        resp = client.get("/image-crawler")
        assert resp.status_code == 200


class TestWarehouseApiRoute:
    def test_missing_country_returns_400(self, client):
        resp = client.get("/api/warehouses")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data or "message" in data

    def test_shipping_unavailable_returns_503(self, client):
        import app.routes.misc as misc_module

        original = misc_module._shipping_ok
        try:
            misc_module._shipping_ok = False
            resp = client.get("/api/warehouses?country=HK")
            assert resp.status_code == 503
        finally:
            misc_module._shipping_ok = original

    @patch("app.routes.misc.ShippingAgent")
    def test_valid_country_returns_200(self, mock_agent, client):
        import app.routes.misc as misc_module

        original = misc_module._shipping_ok
        try:
            misc_module._shipping_ok = True
            mock_agent.return_value.get_warehouses_by_country.return_value = [
                {"name": "HK Warehouse", "address": "Hong Kong"}
            ]
            resp = client.get("/api/warehouses?country=HK")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["country"] == "HK"
            assert len(data["warehouses"]) == 1
        finally:
            misc_module._shipping_ok = original
