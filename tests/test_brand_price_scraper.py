from __future__ import annotations

import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    from app import create_app
    app = create_app()
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["SECRET_KEY"] = "test-secret"
    with app.app_context():
        from app.extensions import db
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_client(client, app):
    from app.models.user import User
    from app.extensions import db
    with app.app_context():
        user = User(username="testuser", display_name="Test")
        user.set_password("password")
        db.session.add(user)
        db.session.commit()
    client.post("/auth/login", data={"username": "testuser", "password": "password"})
    return client


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestBrandPriceModel:
    def test_create_brand_price(self, app):
        from app.models.brand_price import BrandPrice
        from app.extensions import db
        with app.app_context():
            bp = BrandPrice(
                brand="Gucci",
                product_name="GG Marmont Bag",
                source_site="farfetch",
                price_original=1890.0,
                currency="EUR",
                price_jpy=278000.0,
                exchange_rate=147.0,
            )
            db.session.add(bp)
            db.session.commit()
            assert bp.id is not None
            assert bp.brand == "Gucci"
            assert bp.price_jpy == 278000.0

    def test_get_by_brand(self, app):
        from app.models.brand_price import BrandPrice
        from app.extensions import db
        with app.app_context():
            db.session.add(BrandPrice(
                brand="Gucci", product_name="Item A", source_site="farfetch",
                price_original=1000.0, currency="EUR", price_jpy=150000.0, exchange_rate=150.0,
            ))
            db.session.add(BrandPrice(
                brand="Prada", product_name="Item B", source_site="mytheresa",
                price_original=2000.0, currency="EUR", price_jpy=300000.0, exchange_rate=150.0,
            ))
            db.session.commit()

            gucci_items = BrandPrice.get_by_brand("Gucci")
            assert len(gucci_items) == 1
            assert gucci_items[0].brand == "Gucci"

    def test_repr(self, app):
        from app.models.brand_price import BrandPrice
        bp = BrandPrice(brand="Prada", source_site="nap", price_jpy=200000)
        assert "Prada" in repr(bp)
        assert "nap" in repr(bp)


# ---------------------------------------------------------------------------
# Service tests
# ---------------------------------------------------------------------------

class TestBrandPriceService:
    def test_save_scraped_prices(self, app):
        from app.services import brand_price_service
        from app.models.brand_price import BrandPrice
        with app.app_context():
            items = [{
                "brand": "Gucci",
                "product_name": "Test Bag",
                "source_site": "farfetch",
                "source_url": "https://example.com/item",
                "price_original": 1500.0,
                "currency": "EUR",
                "price_jpy": 225000.0,
                "exchange_rate": 150.0,
                "in_stock": True,
                "size_available": "",
                "scraped_at": datetime.utcnow().isoformat(),
            }]
            saved = brand_price_service.save_scraped_prices(items)
            assert saved == 1
            assert BrandPrice.query.count() == 1

    def test_get_price_comparison(self, app):
        from app.services import brand_price_service
        from app.extensions import db
        from app.models.brand_price import BrandPrice
        with app.app_context():
            now = datetime.utcnow()
            db.session.add(BrandPrice(
                brand="Gucci", product_name="Bag A", source_site="farfetch",
                price_original=1000.0, currency="EUR", price_jpy=150000.0,
                exchange_rate=150.0, scraped_at=now,
            ))
            db.session.add(BrandPrice(
                brand="Gucci", product_name="Bag A", source_site="mytheresa",
                price_original=950.0, currency="EUR", price_jpy=142500.0,
                exchange_rate=150.0, scraped_at=now,
            ))
            db.session.commit()

            result = brand_price_service.get_price_comparison("Gucci")
            assert len(result) == 1
            assert result[0]["product_name"] == "Bag A"
            assert "farfetch" in result[0]["sites"]
            assert "mytheresa" in result[0]["sites"]
            assert result[0]["cheapest_site"] == "mytheresa"

    def test_get_cheapest_source(self, app):
        from app.services import brand_price_service
        from app.extensions import db
        from app.models.brand_price import BrandPrice
        with app.app_context():
            now = datetime.utcnow()
            db.session.add(BrandPrice(
                brand="Prada", product_name="Wallet", source_site="nap",
                price_original=500.0, currency="JPY", price_jpy=50000.0,
                exchange_rate=1.0, scraped_at=now,
            ))
            db.session.add(BrandPrice(
                brand="Prada", product_name="Wallet", source_site="24s",
                price_original=300.0, currency="EUR", price_jpy=45000.0,
                exchange_rate=150.0, scraped_at=now,
            ))
            db.session.commit()

            cheapest = brand_price_service.get_cheapest_source("Prada")
            assert "Wallet" in cheapest
            assert cheapest["Wallet"]["site"] == "24s"

    def test_cleanup_old_records(self, app):
        from app.services import brand_price_service
        from app.extensions import db
        from app.models.brand_price import BrandPrice
        with app.app_context():
            from datetime import timedelta
            old = datetime.utcnow() - timedelta(days=100)
            db.session.add(BrandPrice(
                brand="Gucci", product_name="Old", source_site="farfetch",
                price_original=100.0, currency="EUR", price_jpy=15000.0,
                exchange_rate=150.0, scraped_at=old,
            ))
            db.session.add(BrandPrice(
                brand="Gucci", product_name="New", source_site="farfetch",
                price_original=200.0, currency="EUR", price_jpy=30000.0,
                exchange_rate=150.0, scraped_at=datetime.utcnow(),
            ))
            db.session.commit()

            removed = brand_price_service.cleanup_old_records(keep_days=90)
            assert removed == 1
            assert BrandPrice.query.count() == 1

    def test_get_available_brands(self, app):
        from app.services import brand_price_service
        from app.extensions import db
        from app.models.brand_price import BrandPrice
        with app.app_context():
            db.session.add(BrandPrice(
                brand="Ferragamo", product_name="Shoe", source_site="lvr",
                price_original=500.0, currency="EUR", price_jpy=75000.0,
                exchange_rate=150.0,
            ))
            db.session.commit()

            brands = brand_price_service.get_available_brands()
            assert "Ferragamo" in brands


# ---------------------------------------------------------------------------
# Scraper tests (mock Playwright)
# ---------------------------------------------------------------------------

class TestBrandPriceScraper:
    def test_farfetch_brand_slugs(self):
        from app.services.brand_price_scraper import _FARFETCH_BRAND_SLUGS
        assert _FARFETCH_BRAND_SLUGS["Gucci"] == "gucci"
        assert _FARFETCH_BRAND_SLUGS["Prada"] == "prada"

    def test_supported_brands(self):
        from app.services.brand_price_scraper import SUPPORTED_BRANDS
        assert "Gucci" in SUPPORTED_BRANDS
        assert "Prada" in SUPPORTED_BRANDS
        assert "Ferragamo" in SUPPORTED_BRANDS

    def test_supported_sites(self):
        from app.services.brand_price_scraper import SUPPORTED_SITES
        assert "farfetch" in SUPPORTED_SITES
        assert "gucci_official" in SUPPORTED_SITES
        assert "prada_official" in SUPPORTED_SITES
        assert "ferragamo_official" in SUPPORTED_SITES

    @patch("app.services.brand_price_scraper._load_proxies")
    def test_scrape_farfetch_returns_results(self, mock_proxies):
        from app.services.brand_price_scraper import BrandPriceScraper
        mock_proxies.return_value = []
        scraper = BrandPriceScraper()
        scraper._init_browser = MagicMock()
        scraper._close_browser = MagicMock()

        mock_page = MagicMock()
        mock_page.evaluate.return_value = [
            {"brand": "Gucci", "name": "GG Marmont Bag", "price": 150000, "url": "https://example.com/item1"},
        ]
        scraper.page = mock_page

        results = scraper._scrape_farfetch("Gucci")
        assert len(results) == 1
        assert results[0]["product_name"] == "GG Marmont Bag"
        assert results[0]["price_jpy"] == 150000.0

    @patch("app.services.brand_price_scraper._fetch_with_cffi")
    def test_scrape_official_gucci(self, mock_fetch):
        from app.services.brand_price_scraper import BrandPriceScraper
        mock_fetch.return_value = '''
        <div aria-label="Gucci NY Large Tote, ￥440,000"></div>
        <div aria-label="Gucci Small Bag, ￥302,500"></div>
        '''
        scraper = BrandPriceScraper()
        results = scraper._scrape_official_gucci("Gucci")
        assert len(results) == 2
        assert results[0]["product_name"] == "Gucci NY Large Tote"
        assert results[0]["price_jpy"] == 440000.0
        assert results[0]["source_site"] == "gucci_official"

    @patch("app.services.brand_price_scraper._fetch_with_cffi")
    def test_scrape_official_prada(self, mock_fetch):
        from app.services.brand_price_scraper import BrandPriceScraper
        mock_fetch.return_value = '''
        <a aria-label=" Prada Bonnie small handbag ¥ 407,000 colors: White"></a>
        <a aria-label=" Prada Route large tote bag ¥ 704,000 colors: Black"></a>
        '''
        scraper = BrandPriceScraper()
        results = scraper._scrape_official_prada("Prada")
        assert len(results) == 2
        assert results[0]["product_name"] == "Prada Bonnie small handbag"
        assert results[0]["price_jpy"] == 407000.0
        assert results[1]["source_site"] == "prada_official"

    @patch("app.services.brand_price_scraper._fetch_with_cffi")
    def test_scrape_official_gucci_no_match(self, mock_fetch):
        from app.services.brand_price_scraper import BrandPriceScraper
        mock_fetch.return_value = "<html><body>No products here</body></html>"
        scraper = BrandPriceScraper()
        results = scraper._scrape_official_gucci("Gucci")
        assert len(results) == 0

    @patch("app.services.brand_price_scraper._fetch_with_cffi")
    def test_scrape_official_returns_empty_for_unsupported(self, mock_fetch):
        from app.services.brand_price_scraper import BrandPriceScraper
        scraper = BrandPriceScraper()
        results = scraper._scrape_official_gucci("Ferragamo")
        assert len(results) == 0
        mock_fetch.assert_not_called()

    def test_scrape_calls_official_and_farfetch(self):
        from app.services.brand_price_scraper import BrandPriceScraper
        scraper = BrandPriceScraper()
        scraper._scrape_official_gucci = MagicMock(return_value=[{"product_name": "Gucci Bag", "source_site": "gucci_official"}])
        scraper._scrape_official_prada = MagicMock(return_value=[])
        scraper._scrape_official_ferragamo = MagicMock(return_value=[])
        scraper._scrape_farfetch = MagicMock(return_value=[{"product_name": "FF Bag", "source_site": "farfetch"}])
        scraper._init_browser = MagicMock()
        scraper._close_browser = MagicMock()

        results = scraper.scrape("Gucci", sites=["gucci_official", "farfetch"])
        assert len(results) == 2
        scraper._scrape_official_gucci.assert_called_once()
        scraper._scrape_farfetch.assert_called_once()

    def test_scrape_ferragamo_official_mock(self):
        from app.services.brand_price_scraper import BrandPriceScraper
        scraper = BrandPriceScraper()
        scraper._scrape_official_ferragamo = MagicMock(return_value=[
            {"product_name": "Hug Bucket Bag", "price_jpy": 330000.0, "source_site": "ferragamo_official"},
        ])
        results = scraper.scrape("Ferragamo", sites=["ferragamo_official"])
        assert len(results) == 1
        assert results[0]["source_site"] == "ferragamo_official"


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------

class TestBrandPriceRoutes:
    def test_dashboard_requires_login(self, client):
        resp = client.get("/brand-prices", follow_redirects=False)
        assert resp.status_code == 302

    def test_dashboard_shows_page(self, auth_client):
        resp = auth_client.get("/brand-prices")
        assert resp.status_code == 200
        assert "ブランド価格調査" in resp.data.decode()

    def test_dashboard_brand_filter(self, auth_client):
        resp = auth_client.get("/brand-prices?brand=Prada")
        assert resp.status_code == 200

    def test_api_comparison_requires_login(self, client):
        resp = client.get("/api/brand-prices/comparison")
        assert resp.status_code == 302

    def test_api_comparison_returns_json(self, auth_client):
        resp = auth_client.get("/api/brand-prices/comparison?brand=Gucci")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["brand"] == "Gucci"
