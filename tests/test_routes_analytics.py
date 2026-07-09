"""ルートテスト: Analytics（brand-analytics, dashboard）"""

from __future__ import annotations


class TestAnalyticsRoutes:
    def test_brand_analytics_200(self, routes_auth_client):
        r = routes_auth_client.get("/brand-analytics")
        assert r.status_code == 200

    def test_brand_analytics_requires_auth(self, routes_client):
        r = routes_client.get("/brand-analytics", follow_redirects=False)
        assert r.status_code == 302
        assert "/auth/login" in r.headers.get("Location", "")

    def test_dashboard_200(self, routes_auth_client):
        r = routes_auth_client.get("/dashboard")
        assert r.status_code == 200

    def test_dashboard_with_period_7d(self, routes_auth_client):
        r = routes_auth_client.get("/dashboard?period=7d")
        assert r.status_code == 200

    def test_dashboard_with_period_30d(self, routes_auth_client):
        r = routes_auth_client.get("/dashboard?period=30d")
        assert r.status_code == 200

    def test_dashboard_with_brand_filter(self, routes_auth_client):
        r = routes_auth_client.get("/dashboard?brand=Gucci")
        assert r.status_code == 200

    def test_dashboard_csv_export(self, routes_auth_client):
        r = routes_auth_client.get("/dashboard?export=csv")
        assert r.status_code == 200
        assert "text/csv" in r.content_type

    def test_dashboard_requires_auth(self, routes_client):
        r = routes_client.get("/dashboard", follow_redirects=False)
        assert r.status_code == 302
