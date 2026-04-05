"""atelier-kyo-manager ルートテスト"""

import pytest
from unittest.mock import patch, MagicMock

from app import create_app
from app.extensions import db
from app.models.user import User
from app.models.product import Product


@pytest.fixture(scope="function")
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "WTF_CSRF_ENABLED": False,
        "SECRET_KEY": "test-secret",
    })
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def test_user(app):
    with app.app_context():
        u = User(username="testuser", display_name="Test", is_active=True)
        u.set_password("password123")
        db.session.add(u)
        db.session.commit()
    return {"username": "testuser", "password": "password123"}


@pytest.fixture()
def auth_client(client, app, test_user):
    client.post("/auth/login", data=test_user, follow_redirects=False)
    return client


# ---- Auth ----
class TestAuth:
    def test_login_success(self, client, test_user):
        r = client.post("/auth/login", data=test_user, follow_redirects=False)
        assert r.status_code == 302

    def test_login_wrong_password(self, client, test_user):
        r = client.post("/auth/login", data={"username": "testuser", "password": "wrong"})
        assert r.status_code == 401

    def test_login_nonexistent_user(self, client):
        r = client.post("/auth/login", data={"username": "nobody", "password": "x"})
        assert r.status_code == 401

    def test_logout(self, auth_client):
        r = auth_client.get("/auth/logout", follow_redirects=False)
        assert r.status_code == 302


# ---- Protected routes redirect ----
class TestProtectedRoutes:
    @pytest.mark.parametrize("path", [
        "/manage", "/products", "/orders", "/partners",
        "/stock-check", "/popularity", "/regions", "/customers",
        "/cashflow", "/brand-analytics", "/listing-progress",
        "/templates", "/auto-research",
    ])
    def test_unauthenticated_redirects(self, client, path):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 302
        assert "/auth/login" in r.headers.get("Location", "")


# ---- Authenticated pages 200 ----
class TestAuthenticatedPages:
    @pytest.mark.parametrize("path", [
        "/", "/manage", "/products", "/orders", "/partners",
        "/stock-check", "/popularity", "/regions", "/customers",
        "/cashflow", "/brand-analytics", "/listing-progress",
        "/templates", "/auto-research",
    ])
    def test_page_200(self, auth_client, path):
        r = auth_client.get(path)
        assert r.status_code == 200


# ---- Compat redirects ----
class TestCompatRedirects:
    def test_dashboard_redirect(self, auth_client):
        r = auth_client.get("/dashboard", follow_redirects=False)
        assert r.status_code == 302
        assert "/cashflow" in r.headers["Location"]

    def test_listing_templates_redirect(self, auth_client):
        r = auth_client.get("/listing-templates", follow_redirects=False)
        assert r.status_code == 302
        assert "/templates" in r.headers["Location"]

    def test_region_recommendations_redirect(self, auth_client):
        r = auth_client.get("/region-recommendations", follow_redirects=False)
        assert r.status_code == 302
        assert "/regions" in r.headers["Location"]

    def test_repeat_customers_redirect(self, auth_client):
        r = auth_client.get("/repeat-customers", follow_redirects=False)
        assert r.status_code == 302
        assert "/customers" in r.headers["Location"]

    def test_auto_research_underscore(self, auth_client):
        r = auth_client.get("/auto_research", follow_redirects=False)
        assert r.status_code == 302
        assert "/auto-research" in r.headers["Location"]


# ---- Stock Check API ----
class TestStockCheckAPI:
    def test_fetch_all_empty(self, auth_client):
        r = auth_client.post("/api/stock-check/fetch-all")
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True

    def test_fetch_all_requires_auth(self, client):
        r = client.post("/api/stock-check/fetch-all", follow_redirects=False)
        assert r.status_code == 302


# ---- PriceScraper mock ----
class TestPriceScraper:
    def test_fetch_success(self):
        with patch("app.services.price_scraper.requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = '<html><title>Test</title><span>¥12,345</span></html>'
            mock_resp.url = "http://example.com"
            mock_resp.headers = {}
            mock_get.return_value = mock_resp

            from app.services.price_scraper import PriceScraper
            scraper = PriceScraper()
            result = scraper.fetch("http://example.com")
            assert result["success"] is True
            assert result["price"] == 12345
            scraper.close()

    def test_fetch_connection_error(self):
        import requests as _req
        with patch("app.services.price_scraper.requests.Session.get") as mock_get:
            mock_get.side_effect = _req.ConnectionError("fail")

            from app.services.price_scraper import PriceScraper
            result = PriceScraper().fetch("http://bad.example.com")
            assert result["success"] is False

    def test_fetch_sold_out(self):
        with patch("app.services.price_scraper.requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = '<html><title>Sold</title><div>SOLD OUT</div></html>'
            mock_resp.url = "http://example.com"
            mock_resp.headers = {}
            mock_get.return_value = mock_resp

            from app.services.price_scraper import PriceScraper
            result = PriceScraper().fetch("http://example.com")
            assert result["in_stock"] is False
