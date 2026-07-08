from __future__ import annotations

import json
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
    from app.extensions import db
    from app.models.user import User

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
        from app.extensions import db
        from app.models.brand_price import BrandPrice

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
        from app.extensions import db
        from app.models.brand_price import BrandPrice

        with app.app_context():
            db.session.add(
                BrandPrice(
                    brand="Gucci",
                    product_name="Item A",
                    source_site="farfetch",
                    price_original=1000.0,
                    currency="EUR",
                    price_jpy=150000.0,
                    exchange_rate=150.0,
                )
            )
            db.session.add(
                BrandPrice(
                    brand="Prada",
                    product_name="Item B",
                    source_site="mytheresa",
                    price_original=2000.0,
                    currency="EUR",
                    price_jpy=300000.0,
                    exchange_rate=150.0,
                )
            )
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
        from app.models.brand_price import BrandPrice
        from app.services import brand_price_service

        with app.app_context():
            items = [
                {
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
                }
            ]
            saved = brand_price_service.save_scraped_prices(items)
            assert saved == 1
            assert BrandPrice.query.count() == 1

    def test_get_price_comparison(self, app):
        from app.extensions import db
        from app.models.brand_price import BrandPrice
        from app.services import brand_price_service

        with app.app_context():
            now = datetime.utcnow()
            db.session.add(
                BrandPrice(
                    brand="Gucci",
                    product_name="Bag A",
                    source_site="farfetch",
                    price_original=1000.0,
                    currency="EUR",
                    price_jpy=150000.0,
                    exchange_rate=150.0,
                    scraped_at=now,
                )
            )
            db.session.add(
                BrandPrice(
                    brand="Gucci",
                    product_name="Bag A",
                    source_site="mytheresa",
                    price_original=950.0,
                    currency="EUR",
                    price_jpy=142500.0,
                    exchange_rate=150.0,
                    scraped_at=now,
                )
            )
            db.session.commit()

            result = brand_price_service.get_price_comparison("Gucci")
            assert len(result) == 1
            assert result[0]["product_name"] == "Bag A"
            assert "farfetch" in result[0]["sites"]
            assert "mytheresa" in result[0]["sites"]
            assert result[0]["cheapest_site"] == "mytheresa"

    def test_get_cheapest_source(self, app):
        from app.extensions import db
        from app.models.brand_price import BrandPrice
        from app.services import brand_price_service

        with app.app_context():
            now = datetime.utcnow()
            db.session.add(
                BrandPrice(
                    brand="Prada",
                    product_name="Wallet",
                    source_site="nap",
                    price_original=500.0,
                    currency="JPY",
                    price_jpy=50000.0,
                    exchange_rate=1.0,
                    scraped_at=now,
                )
            )
            db.session.add(
                BrandPrice(
                    brand="Prada",
                    product_name="Wallet",
                    source_site="24s",
                    price_original=300.0,
                    currency="EUR",
                    price_jpy=45000.0,
                    exchange_rate=150.0,
                    scraped_at=now,
                )
            )
            db.session.commit()

            cheapest = brand_price_service.get_cheapest_source("Prada")
            assert "Wallet" in cheapest
            assert cheapest["Wallet"]["site"] == "24s"

    def test_cleanup_old_records(self, app):
        from app.extensions import db
        from app.models.brand_price import BrandPrice
        from app.services import brand_price_service

        with app.app_context():
            from datetime import timedelta

            old = datetime.utcnow() - timedelta(days=100)
            db.session.add(
                BrandPrice(
                    brand="Gucci",
                    product_name="Old",
                    source_site="farfetch",
                    price_original=100.0,
                    currency="EUR",
                    price_jpy=15000.0,
                    exchange_rate=150.0,
                    scraped_at=old,
                )
            )
            db.session.add(
                BrandPrice(
                    brand="Gucci",
                    product_name="New",
                    source_site="farfetch",
                    price_original=200.0,
                    currency="EUR",
                    price_jpy=30000.0,
                    exchange_rate=150.0,
                    scraped_at=datetime.utcnow(),
                )
            )
            db.session.commit()

            removed = brand_price_service.cleanup_old_records(keep_days=90)
            assert removed == 1
            assert BrandPrice.query.count() == 1

    def test_get_available_brands(self, app):
        from app.extensions import db
        from app.models.brand_price import BrandPrice
        from app.services import brand_price_service

        with app.app_context():
            db.session.add(
                BrandPrice(
                    brand="Ferragamo",
                    product_name="Shoe",
                    source_site="lvr",
                    price_original=500.0,
                    currency="EUR",
                    price_jpy=75000.0,
                    exchange_rate=150.0,
                )
            )
            db.session.commit()

            brands = brand_price_service.get_available_brands()
            assert "Ferragamo" in brands


# ---------------------------------------------------------------------------
# Scraper tests
# ---------------------------------------------------------------------------


class TestBrandPriceScraper:
    def test_farfetch_brand_slugs(self):
        from app.services.brand_price_scraper import _FARFETCH_BRAND_SLUGS

        assert _FARFETCH_BRAND_SLUGS["Gucci"] == "gucci"
        assert _FARFETCH_BRAND_SLUGS["Prada"] == "prada"
        assert _FARFETCH_BRAND_SLUGS["Loewe"] == "loewe"
        assert _FARFETCH_BRAND_SLUGS["Balenciaga"] == "balenciaga"

    def test_supported_brands(self):
        from app.services.brand_price_scraper import SUPPORTED_BRANDS

        for b in [
            "Gucci",
            "Prada",
            "Valentino",
            "Ferragamo",
            "Loewe",
            "Balenciaga",
            "Bottega Veneta",
            "Versace",
            "Marni",
            "Chloe",
            "Celine",
        ]:
            assert b in SUPPORTED_BRANDS

    def test_supported_sites(self):
        from app.services.brand_price_scraper import SUPPORTED_SITES

        for s in [
            "farfetch",
            "gucci_official",
            "prada_official",
            "valentino_official",
            "ferragamo_official",
            "loewe_official",
            "balenciaga_official",
            "bottegaveneta_official",
            "versace_official",
            "marni_official",
            "chloe_official",
            "celine_official",
        ]:
            assert s in SUPPORTED_SITES

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
    def test_scrape_cffi_gucci(self, mock_fetch):
        from app.services.brand_price_scraper import BrandPriceScraper

        mock_fetch.return_value = '<div aria-label="Gucci NY Large Tote, € 2.450"></div>'
        scraper = BrandPriceScraper()
        results = scraper._scrape_cffi_official("Gucci")
        assert len(results) == 1
        assert results[0]["source_site"] == "gucci_official"
        assert results[0]["currency"] == "EUR"
        assert results[0]["price_original"] == 2450.0

    @patch("app.services.brand_price_scraper._fetch_with_cffi")
    def test_scrape_cffi_prada(self, mock_fetch):
        from app.services.brand_price_scraper import BrandPriceScraper

        mock_fetch.return_value = '<a aria-label=" Prada Bonnie bag € 1.950 x"></a>'
        scraper = BrandPriceScraper()
        results = scraper._scrape_cffi_official("Prada")
        assert len(results) == 1
        assert results[0]["currency"] == "EUR"
        assert results[0]["price_original"] == 1950.0

    @patch("app.services.brand_price_scraper._fetch_with_cffi")
    def test_scrape_cffi_valentino(self, mock_fetch):
        from app.services.brand_price_scraper import BrandPriceScraper

        mock_fetch.return_value = '<div data-canonical-url="Valentino Garavani Panthea Bag" data-price="2950.0"></div>'
        scraper = BrandPriceScraper()
        results = scraper._scrape_cffi_official("Valentino")
        assert len(results) == 1
        assert results[0]["source_site"] == "valentino_official"
        assert results[0]["currency"] == "EUR"
        assert results[0]["price_original"] == 2950.0

    @patch("app.services.brand_price_scraper._fetch_with_cffi")
    def test_scrape_cffi_loewe(self, mock_fetch):
        from app.services.brand_price_scraper import BrandPriceScraper

        mock_fetch.return_value = "<span>エクリプス バスケット</span><span>¥251,900</span>"
        scraper = BrandPriceScraper()
        results = scraper._scrape_cffi_official("Loewe")
        assert len(results) >= 1

    @patch("app.services.brand_price_scraper._fetch_with_cffi")
    def test_scrape_cffi_balenciaga(self, mock_fetch):
        from app.services.brand_price_scraper import BrandPriceScraper

        mock_fetch.return_value = '<h3>le city bag</h3><p itemprop="price" content="158400">¥158,400</p>'
        scraper = BrandPriceScraper()
        results = scraper._scrape_cffi_official("Balenciaga")
        assert len(results) >= 1
        assert results[0]["price_jpy"] == 158400.0

    @patch("app.services.brand_price_scraper._fetch_with_cffi")
    def test_scrape_cffi_no_match(self, mock_fetch):
        from app.services.brand_price_scraper import BrandPriceScraper

        mock_fetch.return_value = "<html><body>No products</body></html>"
        scraper = BrandPriceScraper()
        results = scraper._scrape_cffi_official("Gucci")
        assert len(results) == 0

    def test_scrape_cffi_unsupported_brand(self):
        from app.services.brand_price_scraper import BrandPriceScraper

        scraper = BrandPriceScraper()
        results = scraper._scrape_cffi_official("Ferragamo")
        assert len(results) == 0

    def test_scrape_calls_all_sources(self):
        from app.services.brand_price_scraper import BrandPriceScraper

        scraper = BrandPriceScraper()
        scraper._scrape_cffi_official = MagicMock(
            return_value=[
                {"product_name": "Gucci Bag", "source_site": "gucci_official"},
            ]
        )
        scraper._scrape_stealth_official = MagicMock(return_value=[])
        scraper._scrape_farfetch = MagicMock(
            return_value=[
                {"product_name": "FF Bag", "source_site": "farfetch"},
            ]
        )
        scraper._init_browser = MagicMock()
        scraper._close_browser = MagicMock()

        results = scraper.scrape("Gucci")
        assert len(results) == 2
        scraper._scrape_cffi_official.assert_called_once()
        scraper._scrape_farfetch.assert_called_once()

    def test_scrape_stealth_brand(self):
        from app.services.brand_price_scraper import BrandPriceScraper

        scraper = BrandPriceScraper()
        scraper._scrape_stealth_official = MagicMock(
            return_value=[
                {"product_name": "Hug Bag", "source_site": "ferragamo_official"},
            ]
        )
        results = scraper.scrape("Ferragamo", sites=["ferragamo_official"])
        assert len(results) == 1
        assert results[0]["source_site"] == "ferragamo_official"

    def test_scrape_versace_uses_stealth(self):
        from app.services.brand_price_scraper import BrandPriceScraper

        scraper = BrandPriceScraper()
        scraper._scrape_stealth_official = MagicMock(
            return_value=[
                {
                    "product_name": "Medusa Big Bag",
                    "source_site": "versace_official",
                    "currency": "EUR",
                    "price_jpy": 400000.0,
                },
            ]
        )
        results = scraper.scrape("Versace", sites=["versace_official"])
        assert len(results) == 1
        assert results[0]["source_site"] == "versace_official"
        scraper._scrape_stealth_official.assert_called_once()

    def test_scrape_marni_uses_stealth(self):
        from app.services.brand_price_scraper import BrandPriceScraper

        scraper = BrandPriceScraper()
        scraper._scrape_stealth_official = MagicMock(
            return_value=[
                {
                    "product_name": "Market Tote",
                    "source_site": "marni_official",
                    "currency": "EUR",
                    "price_jpy": 350000.0,
                },
            ]
        )
        results = scraper.scrape("Marni", sites=["marni_official"])
        assert len(results) == 1
        assert results[0]["source_site"] == "marni_official"

    def test_scrape_chloe_uses_stealth(self):
        from app.services.brand_price_scraper import BrandPriceScraper

        scraper = BrandPriceScraper()
        scraper._scrape_stealth_official = MagicMock(
            return_value=[
                {
                    "product_name": "Marcie Bag",
                    "source_site": "chloe_official",
                    "currency": "EUR",
                    "price_jpy": 380000.0,
                },
            ]
        )
        results = scraper.scrape("Chloe", sites=["chloe_official"])
        assert len(results) == 1
        assert results[0]["source_site"] == "chloe_official"

    def test_scrape_celine_uses_stealth(self):
        from app.services.brand_price_scraper import BrandPriceScraper

        scraper = BrandPriceScraper()
        scraper._scrape_stealth_official = MagicMock(
            return_value=[
                {
                    "product_name": "Triomphe Bag",
                    "source_site": "celine_official",
                    "currency": "EUR",
                    "price_jpy": 530000.0,
                },
            ]
        )
        results = scraper.scrape("Celine", sites=["celine_official"])
        assert len(results) == 1
        assert results[0]["source_site"] == "celine_official"


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


# ---------------------------------------------------------------------------
# Profit calculation tests
# ---------------------------------------------------------------------------


class TestProfitCalculation:
    def test_add_profit_calculation_basic(self):
        from app.services.brand_price_service import add_profit_calculation

        comparison = [
            {
                "product_name": "GG Marmont Bag",
                "cheapest_jpy": 100000.0,
                "cheapest_site": "gucci_official",
                "sites": {},
            }
        ]
        result = add_profit_calculation(comparison, category="bag")
        item = result[0]
        assert item["profit"] is not None
        assert item["total_cost"] > 0
        assert item["buyma_suggested_price"] == round(100000.0 * 1.3)
        assert item["shipping_cost"] == 3300.0

    def test_add_profit_calculation_custom_markup(self):
        from app.services.brand_price_service import add_profit_calculation

        comparison = [
            {
                "product_name": "Prada Bonnie",
                "cheapest_jpy": 200000.0,
                "cheapest_site": "prada_official",
                "sites": {},
            }
        ]
        result = add_profit_calculation(comparison, category="bag", markup_rate=1.5)
        item = result[0]
        assert item["buyma_suggested_price"] == round(200000.0 * 1.5)

    def test_add_profit_calculation_manual_buyma_price(self):
        from app.services.brand_price_service import add_profit_calculation

        comparison = [
            {
                "product_name": "Valentino Bag",
                "cheapest_jpy": 150000.0,
                "buyma_price": 250000.0,
                "cheapest_site": "valentino_official",
                "sites": {},
            }
        ]
        result = add_profit_calculation(comparison, category="bag")
        item = result[0]
        assert item["buyma_suggested_price"] == 250000.0

    def test_add_profit_calculation_no_price_skipped(self):
        from app.services.brand_price_service import add_profit_calculation

        comparison = [{"product_name": "Unknown", "cheapest_jpy": None, "sites": {}}]
        result = add_profit_calculation(comparison, category="bag")
        assert result[0]["profit"] is None
        assert result[0]["is_profitable"] is False

    def test_profitable_item_passes_threshold(self):
        from app.services.brand_price_service import add_profit_calculation

        # 100,000 * 1.5 = 150,000 selling price
        # cost ~= 100,000 + 3,300 + customs(11,000) + commission(~11,550) = ~126,850
        # profit ~= 23,150 > 10,000 → True
        comparison = [
            {
                "product_name": "Test Bag",
                "cheapest_jpy": 100000.0,
                "cheapest_site": "test",
                "sites": {},
            }
        ]
        result = add_profit_calculation(comparison, category="bag", markup_rate=1.5)
        assert result[0]["is_profitable"] is True

    def test_unprofitable_item_fails_threshold(self):
        from app.services.brand_price_service import add_profit_calculation

        # 500,000 * 1.01 = 505,000 selling → profit too small
        comparison = [
            {
                "product_name": "Expensive Bag",
                "cheapest_jpy": 500000.0,
                "buyma_price": 505000.0,
                "cheapest_site": "test",
                "sites": {},
            }
        ]
        result = add_profit_calculation(comparison, category="bag")
        assert result[0]["is_profitable"] is False


class TestCostTable:
    def test_get_shipping_bag(self):
        from app.config.cost_table import get_buyandship_shipping

        assert get_buyandship_shipping("bag") == 3300.0

    def test_get_shipping_wallet(self):
        from app.config.cost_table import get_buyandship_shipping

        assert get_buyandship_shipping("wallet") == 3300.0

    def test_get_shipping_default(self):
        from app.config.cost_table import get_buyandship_shipping

        assert get_buyandship_shipping(None) == 3300.0


# ---------------------------------------------------------------------------
# BUYMA price scraper tests
# ---------------------------------------------------------------------------


class TestBuymaPriceScraper:
    def test_parse_buyma_results_extracts_products(self):
        from app.services.buyma_price_scraper import _parse_buyma_results

        html = (
            '<div class="product_img" '
            'syo_id="12345" syo_name="PRADA Bonnie バッグ レザー" '
            'brand_name="PRADA" category="レディース/バッグ" price="398000">'
            "</div>"
            '<div class="product_img" '
            'syo_id="67890" syo_name="PRADA Re-Nylon トート" '
            'brand_name="PRADA" category="レディース/バッグ" price="250000">'
            "</div>"
        )
        results = _parse_buyma_results(html)
        assert len(results) == 2
        assert results[0]["name"] == "PRADA Bonnie バッグ レザー"
        assert results[0]["brand"] == "PRADA"
        assert results[0]["price_jpy"] == 398000
        assert results[0]["item_id"] == "12345"
        assert results[1]["price_jpy"] == 250000

    def test_parse_buyma_results_deduplicates(self):
        from app.services.buyma_price_scraper import _parse_buyma_results

        html = (
            '<div syo_id="111" syo_name="Item A" brand_name="PRADA" price="100">'
            "</div>"
            '<div syo_id="111" syo_name="Item A" brand_name="PRADA" price="100">'
            "</div>"
        )
        results = _parse_buyma_results(html)
        assert len(results) == 1

    def test_normalize_brand(self):
        from app.services.buyma_price_scraper import _normalize_brand

        assert _normalize_brand("PRADA") == "PRADA"
        assert _normalize_brand("prada") == "PRADA"
        assert _normalize_brand("LOEWE") == "LOEWE"

    def test_match_score_brand_mismatch_returns_zero(self):
        from app.services.buyma_price_scraper import _match_score

        score = _match_score("Bonnie Bag", "LOEWE Puzzle Bag", "Prada")
        assert score == 0.0

    def test_match_score_high_for_matching_product(self):
        from app.services.buyma_price_scraper import _match_score

        score = _match_score(
            "Bonnie",
            "PRADA【入手困難】Bonnie★レザー★ミディアム トート",
            "Prada",
        )
        assert score > 0.3

    def test_match_product_returns_cheapest_price(self):
        from app.services.buyma_price_scraper import match_product

        buyma_results = [
            {
                "name": "PRADA Bonnie レザー トート",
                "brand": "PRADA",
                "price_jpy": 400000,
                "item_id": "1",
                "item_url": "",
            },
            {
                "name": "PRADA Bonnie ミニ ショルダー",
                "brand": "PRADA",
                "price_jpy": 330000,
                "item_id": "2",
                "item_url": "",
            },
            {"name": "LOEWE Puzzle Bag", "brand": "LOEWE", "price_jpy": 200000, "item_id": "3", "item_url": ""},
        ]
        result = match_product("Bonnie", "Prada", buyma_results)
        assert result is not None
        assert result["buyma_price"] == 330000
        assert result["buyma_source"] == "buyma_search"
        assert result["match_score"] > 0.3

    def test_match_product_no_match_returns_none(self):
        from app.services.buyma_price_scraper import match_product

        buyma_results = [
            {"name": "LOEWE Puzzle Bag", "brand": "LOEWE", "price_jpy": 200000, "item_id": "1", "item_url": ""},
        ]
        result = match_product("Bonnie", "Prada", buyma_results)
        assert result is None

    def test_match_product_empty_results_returns_none(self):
        from app.services.buyma_price_scraper import match_product

        result = match_product("Bonnie", "Prada", [])
        assert result is None

    @patch("app.services.buyma_price_scraper.search_buyma")
    def test_fetch_buyma_prices_batch(self, mock_search):
        from app.services.buyma_price_scraper import fetch_buyma_prices

        mock_search.return_value = [
            {"name": "PRADA Bonnie レザー", "brand": "PRADA", "price_jpy": 398000, "item_id": "1", "item_url": ""},
        ]
        products = [
            {"product_name": "Bonnie", "brand": "Prada"},
        ]
        results = fetch_buyma_prices(products, headless=True)
        assert "Bonnie" in results
        assert results["Bonnie"]["buyma_price"] == 398000

    def test_search_buyma_route_requires_login(self, client):
        resp = client.post("/brand-prices/search-buyma", follow_redirects=False)
        assert resp.status_code == 302

    def test_search_buyma_route_rejects_unsupported_brand(self, auth_client):
        resp = auth_client.post(
            "/brand-prices/search-buyma",
            data={"brand": "UnknownBrand", "category": "bag"},
            follow_redirects=True,
        )
        assert resp.status_code == 200
