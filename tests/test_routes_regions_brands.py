"""ルートテスト: Region Recommendations + Brand Prices"""

from __future__ import annotations

import pytest

from app.extensions import db


class TestBrandPriceRoutes:
    def test_brand_prices_dashboard_200(self, routes_auth_client):
        r = routes_auth_client.get("/brand-prices")
        assert r.status_code == 200

    def test_brand_prices_dashboard_with_brand(self, routes_auth_client):
        r = routes_auth_client.get("/brand-prices?brand=Gucci")
        assert r.status_code == 200

    def test_brand_prices_dashboard_invalid_brand(self, routes_auth_client):
        r = routes_auth_client.get("/brand-prices?brand=InvalidBrand")
        assert r.status_code == 200

    def test_brand_prices_export_csv(self, routes_auth_client):
        r = routes_auth_client.get("/brand-prices/export")
        assert r.status_code == 200
        assert "text/csv" in r.content_type

    def test_api_brand_prices_comparison(self, routes_auth_client):
        r = routes_auth_client.get("/api/brand-prices/comparison")
        assert r.status_code == 200
        data = r.get_json()
        assert "brand" in data

    def test_brand_prices_requires_auth(self, routes_client):
        r = routes_client.get("/brand-prices", follow_redirects=False)
        assert r.status_code == 302


class TestRegionRecommendationRoutes:
    def test_region_list_200(self, routes_auth_client):
        r = routes_auth_client.get("/regions")
        assert r.status_code == 200

    def test_region_new_form_200(self, routes_auth_client):
        r = routes_auth_client.get("/regions/new")
        assert r.status_code == 200

    def test_region_create_success(self, routes_auth_client, routes_app):
        r = routes_auth_client.post(
            "/regions/new",
            data={
                "region": "EU",
                "region_name": "ヨーロッパ",
                "avg_profit_rate": "25.0",
                "avg_shipping_days": "7",
                "risk_score": "30",
                "reliability_score": "80",
                "avg_customs_rate": "5.0",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        with routes_app.app_context():
            from app.models.region_recommendation import RegionRecommendation
            rr = RegionRecommendation.query.filter_by(region="EU").first()
            assert rr is not None

    def test_region_create_missing_fields(self, routes_auth_client):
        r = routes_auth_client.post("/regions/new", data={"region": "", "region_name": ""})
        assert r.status_code == 200

    def test_region_create_negative_shipping_days(self, routes_auth_client):
        r = routes_auth_client.post(
            "/regions/new",
            data={"region": "XX", "region_name": "Bad", "avg_shipping_days": "-5"},
        )
        assert r.status_code == 200

    def test_region_delete(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.region_recommendation import RegionRecommendation
            rr = RegionRecommendation(region="DEL", region_name="DeleteTarget", recommendation_score=50)
            db.session.add(rr)
            db.session.commit()
            rid = rr.id
        r = routes_auth_client.post(f"/regions/{rid}/delete", follow_redirects=True)
        assert r.status_code == 200

    def test_region_delete_404(self, routes_auth_client):
        r = routes_auth_client.post("/regions/9999/delete", follow_redirects=False)
        assert r.status_code == 302

    def test_region_requires_auth(self, routes_client):
        r = routes_client.get("/regions", follow_redirects=False)
        assert r.status_code == 302
