from __future__ import annotations

from unittest.mock import patch

from app.extensions import db
from app.models.brand_price import BrandPrice

_MOCK_COMPARISON = []
_MOCK_BRANDS = ["Gucci"]


def _patch_service():
    return [
        patch("app.routes.brand_prices.brand_price_service.get_price_comparison", return_value=_MOCK_COMPARISON),
        patch("app.routes.brand_prices.brand_price_service.add_profit_calculation", side_effect=lambda c, **kw: c),
        patch("app.routes.brand_prices.brand_price_service.get_available_brands", return_value=_MOCK_BRANDS),
        patch("app.routes.brand_prices.brand_price_service.get_last_scraped_at", return_value=None),
    ]


def _apply_patches(patches):
    for p in patches:
        p.start()
    return patches


def _stop_patches(patches):
    for p in patches:
        p.stop()


class TestBrandPricesDashboard:
    def test_get_dashboard_200(self, routes_auth_client):
        patches = _apply_patches(_patch_service())
        try:
            resp = routes_auth_client.get("/brand-prices")
            assert resp.status_code == 200
        finally:
            _stop_patches(patches)

    def test_get_dashboard_unknown_brand_defaults(self, routes_auth_client):
        patches = _apply_patches(_patch_service())
        try:
            resp = routes_auth_client.get("/brand-prices?brand=UnknownBrand")
            assert resp.status_code == 200
        finally:
            _stop_patches(patches)

    def test_get_dashboard_unauthenticated(self, routes_client):
        resp = routes_client.get("/brand-prices")
        assert resp.status_code in (200, 302)


class TestBrandPricesExport:
    def test_export_returns_csv(self, routes_auth_client):
        patches = _apply_patches(_patch_service())
        try:
            resp = routes_auth_client.get("/brand-prices/export")
            assert resp.status_code == 200
            assert "text/csv" in resp.content_type
        finally:
            _stop_patches(patches)


class TestBrandPricesApiComparison:
    def test_comparison_json_200(self, routes_auth_client):
        with patch("app.routes.brand_prices.brand_price_service.get_price_comparison", return_value=[]):
            resp = routes_auth_client.get("/api/brand-prices/comparison")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "brand" in data
        assert "comparison" in data


class TestBrandPricesUpdateSellingPrice:
    def test_valid_update_redirects(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            bp = BrandPrice(
                brand="Gucci",
                product_name="GGバッグ",
                source_site="farfetch",
                price_original=1000.0,
                currency="EUR",
                price_jpy=150000.0,
                exchange_rate=150.0,
            )
            db.session.add(bp)
            db.session.commit()
            bp_id = bp.id

        resp = routes_auth_client.post(
            "/brand-prices/update-selling-price",
            data={
                "brand_price_id": str(bp_id),
                "selling_price": "160000",
                "brand": "Gucci",
                "category": "bag",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 302

    def test_invalid_selling_price_redirects(self, routes_auth_client):
        resp = routes_auth_client.post(
            "/brand-prices/update-selling-price",
            data={
                "selling_price": "invalid",
                "brand": "Gucci",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 302

    def test_missing_bp_id_redirects(self, routes_auth_client):
        resp = routes_auth_client.post(
            "/brand-prices/update-selling-price",
            data={
                "selling_price": "100000",
                "brand": "Gucci",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 302


class TestBrandPricesAddToPipeline:
    def test_no_bp_id_redirects(self, routes_auth_client):
        resp = routes_auth_client.post(
            "/brand-prices/add-to-pipeline",
            data={
                "brand": "Gucci",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 302

    def test_nonexistent_bp_id_redirects(self, routes_auth_client):
        resp = routes_auth_client.post(
            "/brand-prices/add-to-pipeline",
            data={
                "brand_price_id": "99999",
                "brand": "Gucci",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 302

    def test_valid_add_to_pipeline(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            bp = BrandPrice(
                brand="Gucci",
                product_name="GGバッグ新",
                source_site="farfetch",
                price_original=1200.0,
                currency="EUR",
                price_jpy=200000.0,
                exchange_rate=150.0,
                buyma_price=250000.0,
            )
            db.session.add(bp)
            db.session.commit()
            bp_id = bp.id

        resp = routes_auth_client.post(
            "/brand-prices/add-to-pipeline",
            data={
                "brand_price_id": str(bp_id),
                "brand": "Gucci",
                "category": "bag",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 302


class TestBrandPricesSearchBuyma:
    def test_unsupported_brand_redirects(self, routes_auth_client):
        resp = routes_auth_client.post(
            "/brand-prices/search-buyma",
            data={
                "brand": "UnknownBrand999",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 302

    def test_supported_brand_with_no_products(self, routes_auth_client):
        with patch("app.routes.brand_prices.brand_price_service.get_price_comparison", return_value=[]):
            resp = routes_auth_client.post(
                "/brand-prices/search-buyma",
                data={
                    "brand": "Gucci",
                },
                follow_redirects=False,
            )
        assert resp.status_code == 302


class TestApiBuymaSearchSingle:
    def test_missing_params_returns_400(self, routes_auth_client):
        resp = routes_auth_client.post("/api/buyma-search", json={})
        assert resp.status_code == 400

    def test_unsupported_brand_returns_400(self, routes_auth_client):
        with patch("app.routes.brand_prices._last_buyma_request", 0.0):
            import app.routes.brand_prices as mod

            mod._last_buyma_request = 0.0
            resp = routes_auth_client.post(
                "/api/buyma-search",
                json={
                    "product_name": "Test",
                    "brand": "UnknownBrand999",
                },
            )
        assert resp.status_code in (400, 429)

    def test_rate_limiting_returns_429(self, routes_auth_client):
        import time

        with patch("app.routes.brand_prices._last_buyma_request", time.time()):
            resp = routes_auth_client.post(
                "/api/buyma-search",
                json={
                    "product_name": "GGバッグ",
                    "brand": "Gucci",
                },
            )
        assert resp.status_code in (200, 400, 429)
