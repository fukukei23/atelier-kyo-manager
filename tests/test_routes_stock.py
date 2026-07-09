"""ルートテスト: Stock Checks + Prohibited Sources"""

from __future__ import annotations

from app.extensions import db


class TestStockCheckRoutes:
    def test_stock_check_list_200(self, routes_auth_client):
        r = routes_auth_client.get("/stock-check")
        assert r.status_code == 200

    def test_stock_check_new_form_200(self, routes_auth_client):
        r = routes_auth_client.get("/stock-check/new")
        assert r.status_code == 200

    def test_stock_check_new_with_preselected(self, routes_auth_client):
        r = routes_auth_client.get("/stock-check/new?product_id=1")
        assert r.status_code == 200

    def test_stock_check_create_no_product(self, routes_auth_client):
        r = routes_auth_client.post("/stock-check/new", data={"product_id": "0", "current_price": "1000"})
        assert r.status_code == 200

    def test_stock_check_create_negative_price(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models import Product

            p = Product(name="SCProd", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = routes_auth_client.post("/stock-check/new", data={"product_id": str(pid), "current_price": "-100"})
        assert r.status_code == 200

    def test_stock_check_create_success(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models import Product

            p = Product(name="SCProd2", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = routes_auth_client.post(
            "/stock-check/new",
            data={
                "product_id": str(pid),
                "current_price": "5000",
                "source_url": "https://example.com",
                "in_stock": "y",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200

    def test_stock_check_delete(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models import Product
            from app.models.stock_check import StockCheck

            p = Product(name="SCProd3", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.flush()
            sc = StockCheck(product_id=p.id, current_price=3000)
            db.session.add(sc)
            db.session.commit()
            sid = sc.id
        r = routes_auth_client.post(f"/stock-check/{sid}/delete", follow_redirects=True)
        assert r.status_code == 200

    def test_api_quick_update_price(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models import Product
            from app.models.stock_check import StockCheck

            p = Product(name="QuickProd", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.flush()
            sc = StockCheck(product_id=p.id, current_price=3000)
            db.session.add(sc)
            db.session.commit()
            sid = sc.id
        r = routes_auth_client.post(
            "/api/stock-check/quick-update",
            json={"id": sid, "current_price": 3500},
            content_type="application/json",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert data["data"]["current_price"] == 3500

    def test_api_quick_update_missing_fields(self, routes_auth_client):
        r = routes_auth_client.post(
            "/api/stock-check/quick-update",
            json={},
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_api_quick_update_negative_price(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models import Product
            from app.models.stock_check import StockCheck

            p = Product(name="NegProd", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.flush()
            sc = StockCheck(product_id=p.id, current_price=3000)
            db.session.add(sc)
            db.session.commit()
            sid = sc.id
        r = routes_auth_client.post(
            "/api/stock-check/quick-update",
            json={"id": sid, "current_price": -100},
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_api_quick_update_not_found(self, routes_auth_client):
        r = routes_auth_client.post(
            "/api/stock-check/quick-update",
            json={"id": 9999, "current_price": 100},
            content_type="application/json",
        )
        assert r.status_code == 404

    def test_api_quick_add_check(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models import Product

            p = Product(name="QuickAddProd", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = routes_auth_client.post(
            "/api/stock-check/quick-add",
            json={"product_id": pid, "current_price": 5000, "in_stock": True},
            content_type="application/json",
        )
        assert r.status_code == 201
        data = r.get_json()
        assert data["success"] is True

    def test_api_quick_add_missing_fields(self, routes_auth_client):
        r = routes_auth_client.post(
            "/api/stock-check/quick-add",
            json={},
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_api_quick_add_invalid_product(self, routes_auth_client):
        r = routes_auth_client.post(
            "/api/stock-check/quick-add",
            json={"product_id": 9999, "current_price": 100},
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_api_products_list(self, routes_auth_client):
        r = routes_auth_client.get("/api/products-list")
        assert r.status_code == 200
        assert isinstance(r.get_json(), list)

    def test_api_fetch_stock_no_url(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models import Product
            from app.models.stock_check import StockCheck

            p = Product(name="NoUrlProd", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.flush()
            sc = StockCheck(product_id=p.id, source_url=None)
            db.session.add(sc)
            db.session.commit()
            sid = sc.id
        r = routes_auth_client.post(f"/api/stock-check/{sid}/fetch")
        assert r.status_code == 400


class TestProhibitedSourceRoutes:
    def test_check_source_empty_url(self, routes_auth_client):
        r = routes_auth_client.get("/api/check-source")
        assert r.status_code == 200
        data = r.get_json()
        assert data["prohibited"] is False

    def test_check_source_with_url(self, routes_auth_client):
        r = routes_auth_client.get("/api/check-source?url=https://legitimate-shop.com")
        assert r.status_code == 200
        data = r.get_json()
        assert data["prohibited"] is False

    def test_check_source_prohibited(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.prohibited_source import ProhibitedSource

            ps = ProhibitedSource(domain="bad-shop.com", reason="fake goods", source_type="domestic")
            db.session.add(ps)
            db.session.commit()
        r = routes_auth_client.get("/api/check-source?url=https://bad-shop.com/product/123")
        assert r.status_code == 200
        data = r.get_json()
        assert data["prohibited"] is True

    def test_list_prohibited_sources_empty(self, routes_auth_client):
        r = routes_auth_client.get("/api/prohibited-sources")
        assert r.status_code == 200
        assert isinstance(r.get_json(), list)

    def test_add_prohibited_source(self, routes_auth_client):
        r = routes_auth_client.post(
            "/api/prohibited-sources",
            json={"domain": "scam.com", "reason": "scam site", "severity": "blocked"},
            content_type="application/json",
        )
        assert r.status_code == 201
        data = r.get_json()
        assert data["domain"] == "scam.com"

    def test_add_prohibited_source_missing_domain(self, routes_auth_client):
        r = routes_auth_client.post(
            "/api/prohibited-sources",
            json={"reason": "no domain"},
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_add_prohibited_source_duplicate(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.prohibited_source import ProhibitedSource

            db.session.add(ProhibitedSource(domain="dup.com"))
            db.session.commit()
        r = routes_auth_client.post(
            "/api/prohibited-sources",
            json={"domain": "dup.com"},
            content_type="application/json",
        )
        assert r.status_code == 409

    def test_delete_prohibited_source(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.prohibited_source import ProhibitedSource

            ps = ProhibitedSource(domain="del.com")
            db.session.add(ps)
            db.session.commit()
            sid = ps.id
        r = routes_auth_client.delete(f"/api/prohibited-sources/{sid}")
        assert r.status_code == 200
        assert r.get_json()["deleted"] is True

    def test_prohibited_sources_require_auth(self, routes_client):
        r = routes_client.get("/api/prohibited-sources", follow_redirects=False)
        assert r.status_code == 302
