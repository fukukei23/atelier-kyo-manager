"""
未テストルートのカバレッジテスト
- analytics, partners, auto_orders, products
- prohibited_sources, shipment_notifications, listing_progress
- stock_checks, brand_prices, faq_templates, region_recommendations
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from app import create_app
from app.extensions import db
from app.models.user import User


# ---------------------------------------------------------------------------
# Fixtures — DATABASE_URL を事前設定して create_app() 時点で in-memory DB を使う
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def _env(monkeypatch):
    """create_app() より前に環境変数を設定し、Flask-SQLAlchemy が :memory: DB を使うようにする"""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("AK_STAGE", "test")
    monkeypatch.setenv("SECRET_KEY", "test-secret")


@pytest.fixture(scope="function")
def app(_env):
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
        }
    )
    with app.app_context():
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


# ===========================================================================
# Analytics (brand-analytics, dashboard)
# ===========================================================================
class TestAnalyticsRoutes:
    def test_brand_analytics_200(self, auth_client):
        r = auth_client.get("/brand-analytics")
        assert r.status_code == 200

    def test_brand_analytics_requires_auth(self, client):
        r = client.get("/brand-analytics", follow_redirects=False)
        assert r.status_code == 302
        assert "/auth/login" in r.headers.get("Location", "")

    def test_dashboard_200(self, auth_client):
        r = auth_client.get("/dashboard")
        assert r.status_code == 200

    def test_dashboard_with_period_7d(self, auth_client):
        r = auth_client.get("/dashboard?period=7d")
        assert r.status_code == 200

    def test_dashboard_with_period_30d(self, auth_client):
        r = auth_client.get("/dashboard?period=30d")
        assert r.status_code == 200

    def test_dashboard_with_brand_filter(self, auth_client):
        r = auth_client.get("/dashboard?brand=Gucci")
        assert r.status_code == 200

    def test_dashboard_csv_export(self, auth_client):
        r = auth_client.get("/dashboard?export=csv")
        assert r.status_code == 200
        assert "text/csv" in r.content_type

    def test_dashboard_requires_auth(self, client):
        r = client.get("/dashboard", follow_redirects=False)
        assert r.status_code == 302


# ===========================================================================
# Partners (CRUD)
# ===========================================================================
class TestPartnerRoutes:
    def test_partner_list_200(self, auth_client):
        r = auth_client.get("/partners")
        assert r.status_code == 200

    def test_partner_new_form_200(self, auth_client):
        r = auth_client.get("/partners/new")
        assert r.status_code == 200

    def test_partner_create_submit(self, auth_client, app):
        r = auth_client.post(
            "/partners/new",
            data={
                "name": "TestPartner",
                "email": "p@example.com",
                "phone": "090-1234-5678",
                "active_regions": "EU,US",
                "specialty_brands": "Gucci,Prada",
                "priority_level": "high",
                "status": "active",
                "notes": "test partner",
            },
            follow_redirects=False,
        )
        assert r.status_code == 302  # redirect to partner_list
        with app.app_context():
            from app.models.partner import Partner
            p = Partner.query.filter_by(name="TestPartner").first()
            assert p is not None
            assert p.email == "p@example.com"

    def test_partner_create_missing_name(self, auth_client):
        r = auth_client.post("/partners/new", data={"name": ""})
        assert r.status_code == 200  # re-renders form with flash

    def test_partner_edit_200(self, auth_client, app):
        with app.app_context():
            from app.models.partner import Partner
            p = Partner(name="EditTarget", email="old@example.com")
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = auth_client.get(f"/partners/{pid}/edit")
        assert r.status_code == 200

    def test_partner_edit_submit(self, auth_client, app):
        with app.app_context():
            from app.models.partner import Partner
            p = Partner(name="BeforeEdit")
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = auth_client.post(
            f"/partners/{pid}/edit",
            data={"name": "AfterEdit", "email": "new@example.com"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        with app.app_context():
            from app.models.partner import Partner
            updated = db.session.get(Partner, pid)
            assert updated.name == "AfterEdit"

    def test_partner_edit_missing_name(self, auth_client, app):
        with app.app_context():
            from app.models.partner import Partner
            p = Partner(name="NoNameEdit")
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = auth_client.post(f"/partners/{pid}/edit", data={"name": ""})
        assert r.status_code == 200

    def test_partner_delete(self, auth_client, app):
        with app.app_context():
            from app.models.partner import Partner
            p = Partner(name="DeleteTarget")
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = auth_client.post(f"/partners/{pid}/delete", follow_redirects=True)
        assert r.status_code == 200
        with app.app_context():
            from app.models.partner import Partner
            assert db.session.get(Partner, pid) is None


# ===========================================================================
# Customers (repeat customer CRUD)
# ===========================================================================
class TestCustomerRoutes:
    def test_customer_list_200(self, auth_client):
        r = auth_client.get("/customers")
        assert r.status_code == 200

    def test_customer_new_form_200(self, auth_client):
        r = auth_client.get("/customers/new")
        assert r.status_code == 200

    def test_customer_create_submit(self, auth_client, app):
        r = auth_client.post(
            "/customers/new",
            data={
                "customer_name": "TestCustomer",
                "email": "c@example.com",
                "total_orders": "3",
                "total_spent": "15000",
                "notes": "test",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200
        with app.app_context():
            from app.models.repeat_customer import RepeatCustomer
            c = RepeatCustomer.query.filter_by(customer_name="TestCustomer").first()
            assert c is not None
            assert c.total_orders == 3

    def test_customer_create_negative_orders(self, auth_client):
        r = auth_client.post(
            "/customers/new",
            data={"customer_name": "Bad", "total_orders": "-1", "total_spent": "0"},
        )
        assert r.status_code == 200  # re-renders form

    def test_customer_create_invalid_date(self, auth_client):
        r = auth_client.post(
            "/customers/new",
            data={
                "customer_name": "BadDate",
                "total_orders": "1",
                "first_order_date": "not-a-date",
            },
        )
        assert r.status_code == 200

    def test_customer_edit_200(self, auth_client, app):
        with app.app_context():
            from app.models.repeat_customer import RepeatCustomer
            c = RepeatCustomer(customer_name="EditCust", total_orders=1, total_spent=1000)
            c.segment = c.calc_segment()
            db.session.add(c)
            db.session.commit()
            cid = c.id
        r = auth_client.get(f"/customers/{cid}/edit")
        assert r.status_code == 200

    def test_customer_edit_submit(self, auth_client, app):
        with app.app_context():
            from app.models.repeat_customer import RepeatCustomer
            c = RepeatCustomer(customer_name="EditCust2", total_orders=1, total_spent=1000)
            c.segment = c.calc_segment()
            db.session.add(c)
            db.session.commit()
            cid = c.id
        r = auth_client.post(
            f"/customers/{cid}/edit",
            data={"customer_name": "UpdatedCust", "total_orders": "5", "total_spent": "5000"},
            follow_redirects=True,
        )
        assert r.status_code == 200

    def test_customer_delete(self, auth_client, app):
        with app.app_context():
            from app.models.repeat_customer import RepeatCustomer
            c = RepeatCustomer(customer_name="DeleteCust", total_orders=0, total_spent=0)
            db.session.add(c)
            db.session.commit()
            cid = c.id
        r = auth_client.post(f"/customers/{cid}/delete", follow_redirects=True)
        assert r.status_code == 200


# ===========================================================================
# Auto Orders
# ===========================================================================
class TestAutoOrderRoutes:
    def test_auto_orders_list_200(self, auth_client):
        r = auth_client.get("/auto-orders")
        assert r.status_code == 200

    def test_auto_orders_requires_auth(self, client):
        r = client.get("/auto-orders", follow_redirects=False)
        assert r.status_code == 302

    def test_start_auto_order_404(self, auth_client):
        """handle_db_error catches NotFound and redirects with 302"""
        r = auth_client.post("/auto-orders/9999/start", follow_redirects=False)
        assert r.status_code == 302  # handle_db_error catches and redirects

    def test_execute_auto_order_step_404(self, auth_client):
        r = auth_client.post("/auto-orders/9999/step", follow_redirects=False)
        assert r.status_code == 302  # handle_db_error catches and redirects

    def test_report_auto_order_error_404(self, auth_client):
        r = auth_client.post("/auto-orders/9999/error", follow_redirects=False)
        assert r.status_code == 302  # handle_db_error catches and redirects

    def test_auto_orders_start_with_order(self, auth_client, app):
        """Test start auto order with a real order."""
        with app.app_context():
            from app.models.order import Order
            o = Order(order_number="ORD-001", product_name="Test", status="pending")
            db.session.add(o)
            db.session.commit()
            oid = o.id
        r = auth_client.post(f"/auto-orders/{oid}/start", follow_redirects=True)
        assert r.status_code == 200


# ===========================================================================
# Products (manage, CRUD, CSV, pipeline)
# ===========================================================================
class TestProductRoutes:
    def test_index_200(self, auth_client):
        r = auth_client.get("/")
        assert r.status_code == 200

    def test_index_no_auth(self, client):
        r = client.get("/")
        assert r.status_code == 200  # index is public

    def test_manage_products_200(self, auth_client):
        r = auth_client.get("/manage")
        assert r.status_code == 200

    def test_product_list_200(self, auth_client):
        r = auth_client.get("/products")
        assert r.status_code == 200

    def test_product_edit_404(self, auth_client):
        r = auth_client.get("/products/9999/edit", follow_redirects=False)
        assert r.status_code == 302  # handle_db_error catches 404 and redirects

    def test_product_delete_404(self, auth_client):
        r = auth_client.post("/products/9999/delete", follow_redirects=False)
        assert r.status_code == 302  # handle_db_error catches and redirects

    def test_export_csv_200(self, auth_client):
        r = auth_client.get("/export_csv")
        assert r.status_code == 200
        assert "text/csv" in r.content_type

    def test_listing_candidates_200(self, auth_client):
        r = auth_client.get("/listing-candidates")
        assert r.status_code == 200

    def test_listing_candidates_export_200(self, auth_client):
        r = auth_client.get("/listing-candidates/export")
        assert r.status_code == 200
        assert "text/csv" in r.content_type

    def test_import_csv_no_file(self, auth_client):
        r = auth_client.post("/import_csv", follow_redirects=True)
        assert r.status_code == 200

    def test_import_csv_non_csv_file(self, auth_client):
        import io
        data = {"file": (io.BytesIO(b"not a csv"), "test.txt")}
        r = auth_client.post("/import_csv", data=data, follow_redirects=True)
        assert r.status_code == 200

    def test_form_redirect(self, client):
        r = client.get("/form", follow_redirects=False)
        assert r.status_code == 302

    def test_list_redirect(self, client):
        r = client.get("/list", follow_redirects=False)
        assert r.status_code == 302

    def test_generate_listing_404(self, auth_client):
        r = auth_client.get("/products/9999/generate-listing")
        assert r.status_code == 404

    def test_pipeline_result_404(self, auth_client):
        r = auth_client.get("/products/9999/pipeline-result")
        assert r.status_code == 404

    def test_upload_images_404(self, auth_client):
        r = auth_client.post("/products/9999/upload-images", follow_redirects=True)
        assert r.status_code == 404

    def test_run_pipeline_404(self, auth_client):
        r = auth_client.post("/products/9999/run-pipeline", follow_redirects=True)
        assert r.status_code == 404

    def test_run_pipeline_batch_no_ids(self, auth_client):
        r = auth_client.post("/run-pipeline-batch", follow_redirects=True)
        assert r.status_code == 200


# ===========================================================================
# Prohibited Sources
# ===========================================================================
class TestProhibitedSourceRoutes:
    def test_check_source_empty_url(self, auth_client):
        r = auth_client.get("/api/check-source")
        assert r.status_code == 200
        data = r.get_json()
        assert data["prohibited"] is False

    def test_check_source_with_url(self, auth_client):
        r = auth_client.get("/api/check-source?url=https://legitimate-shop.com")
        assert r.status_code == 200
        data = r.get_json()
        assert data["prohibited"] is False

    def test_check_source_prohibited(self, auth_client, app):
        with app.app_context():
            from app.models.prohibited_source import ProhibitedSource
            ps = ProhibitedSource(domain="bad-shop.com", reason="fake goods", source_type="domestic")
            db.session.add(ps)
            db.session.commit()
        r = auth_client.get("/api/check-source?url=https://bad-shop.com/product/123")
        assert r.status_code == 200
        data = r.get_json()
        assert data["prohibited"] is True

    def test_list_prohibited_sources_empty(self, auth_client):
        r = auth_client.get("/api/prohibited-sources")
        assert r.status_code == 200
        assert isinstance(r.get_json(), list)

    def test_add_prohibited_source(self, auth_client, app):
        r = auth_client.post(
            "/api/prohibited-sources",
            json={"domain": "scam.com", "reason": "scam site", "severity": "blocked"},
            content_type="application/json",
        )
        assert r.status_code == 201
        data = r.get_json()
        assert data["domain"] == "scam.com"

    def test_add_prohibited_source_missing_domain(self, auth_client):
        r = auth_client.post(
            "/api/prohibited-sources",
            json={"reason": "no domain"},
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_add_prohibited_source_duplicate(self, auth_client, app):
        with app.app_context():
            from app.models.prohibited_source import ProhibitedSource
            db.session.add(ProhibitedSource(domain="dup.com"))
            db.session.commit()
        r = auth_client.post(
            "/api/prohibited-sources",
            json={"domain": "dup.com"},
            content_type="application/json",
        )
        assert r.status_code == 409

    def test_delete_prohibited_source(self, auth_client, app):
        with app.app_context():
            from app.models.prohibited_source import ProhibitedSource
            ps = ProhibitedSource(domain="del.com")
            db.session.add(ps)
            db.session.commit()
            sid = ps.id
        r = auth_client.delete(f"/api/prohibited-sources/{sid}")
        assert r.status_code == 200
        assert r.get_json()["deleted"] is True

    def test_prohibited_sources_require_auth(self, client):
        r = client.get("/api/prohibited-sources", follow_redirects=False)
        assert r.status_code == 302


# ===========================================================================
# Shipment Notifications
# ===========================================================================
class TestShipmentNotificationRoutes:
    def test_shipment_notifications_list_200(self, auth_client):
        r = auth_client.get("/shipment-notifications")
        assert r.status_code == 200

    def test_shipment_notification_new_form_200(self, auth_client):
        r = auth_client.get("/shipment-notifications/new")
        assert r.status_code == 200

    def test_shipment_notification_create_missing_order(self, auth_client):
        r = auth_client.post(
            "/shipment-notifications/new",
            data={"order_id": "", "tracking_number": "AB123"},
        )
        assert r.status_code == 200  # re-renders with flash error

    def test_shipment_notification_create_success(self, auth_client, app):
        with app.app_context():
            from app.models.order import Order
            o = Order(order_number="SN-001", product_name="TestSN")
            db.session.add(o)
            db.session.commit()
            oid = o.id
        r = auth_client.post(
            "/shipment-notifications/new",
            data={"order_id": str(oid), "tracking_number": "TRACK123"},
            follow_redirects=True,
        )
        assert r.status_code == 200

    def test_shipment_notification_mark_notified(self, auth_client, app):
        with app.app_context():
            from app.models.order import Order
            from app.models.shipment_notification import ShipmentNotification
            o = Order(order_number="SN-002", product_name="TestSN2")
            db.session.add(o)
            db.session.flush()
            sn = ShipmentNotification(order_id=o.id, status="pending")
            db.session.add(sn)
            db.session.commit()
            sid = sn.id
        r = auth_client.post(f"/shipment-notifications/{sid}/notify", follow_redirects=True)
        assert r.status_code == 200

    def test_shipment_notification_mark_notified_already_done(self, auth_client, app):
        with app.app_context():
            from app.models.order import Order
            from app.models.shipment_notification import ShipmentNotification
            o = Order(order_number="SN-003", product_name="TestSN3")
            db.session.add(o)
            db.session.flush()
            sn = ShipmentNotification(order_id=o.id, status="notified")
            db.session.add(sn)
            db.session.commit()
            sid = sn.id
        r = auth_client.post(f"/shipment-notifications/{sid}/notify", follow_redirects=True)
        assert r.status_code == 200

    def test_shipment_notification_delete(self, auth_client, app):
        with app.app_context():
            from app.models.order import Order
            from app.models.shipment_notification import ShipmentNotification
            o = Order(order_number="SN-004", product_name="TestSN4")
            db.session.add(o)
            db.session.flush()
            sn = ShipmentNotification(order_id=o.id)
            db.session.add(sn)
            db.session.commit()
            sid = sn.id
        r = auth_client.post(f"/shipment-notifications/{sid}/delete", follow_redirects=True)
        assert r.status_code == 200

    def test_shipment_notification_404(self, auth_client):
        r = auth_client.post("/shipment-notifications/9999/notify", follow_redirects=False)
        assert r.status_code == 302  # handle_db_error catches and redirects


# ===========================================================================
# Listing Progress
# ===========================================================================
class TestListingProgressRoutes:
    def test_listing_progress_200(self, auth_client):
        r = auth_client.get("/listing-progress")
        assert r.status_code == 200

    def test_listing_progress_new_form_200(self, auth_client):
        r = auth_client.get("/listing-progress/new")
        assert r.status_code == 200

    def test_listing_progress_create(self, auth_client, app):
        from datetime import date
        today = date.today().isoformat()
        r = auth_client.post(
            "/listing-progress/new",
            data={
                "record_date": today,
                "listings_count": "5",
                "target_daily": "20",
                "target_monthly": "600",
                "cumulative_monthly": "50",
                "notes": "test record",
            },
            follow_redirects=False,
        )
        assert r.status_code == 302  # redirect to listing_progress_view

    def test_listing_progress_create_negative(self, auth_client):
        r = auth_client.post(
            "/listing-progress/new",
            data={"listings_count": "-1", "target_daily": "20"},
        )
        assert r.status_code == 200  # re-renders form

    def test_listing_progress_delete(self, auth_client, app):
        with app.app_context():
            from app.models.listing_progress import ListingProgress
            from datetime import date
            lp = ListingProgress(record_date=date.today(), listings_count=10, target_daily=20, target_monthly=600, cumulative_monthly=10)
            db.session.add(lp)
            db.session.commit()
            rid = lp.id
        r = auth_client.post(f"/listing-progress/{rid}/delete", follow_redirects=True)
        assert r.status_code == 200

    def test_listing_progress_delete_404(self, auth_client):
        r = auth_client.post("/listing-progress/9999/delete", follow_redirects=False)
        assert r.status_code == 302  # handle_db_error catches and redirects


# ===========================================================================
# Stock Checks
# ===========================================================================
class TestStockCheckRoutes:
    def test_stock_check_list_200(self, auth_client):
        r = auth_client.get("/stock-check")
        assert r.status_code == 200

    def test_stock_check_new_form_200(self, auth_client):
        r = auth_client.get("/stock-check/new")
        assert r.status_code == 200

    def test_stock_check_new_with_preselected(self, auth_client):
        r = auth_client.get("/stock-check/new?product_id=1")
        assert r.status_code == 200

    def test_stock_check_create_no_product(self, auth_client):
        r = auth_client.post("/stock-check/new", data={"product_id": "0", "current_price": "1000"})
        assert r.status_code == 200  # re-renders with error

    def test_stock_check_create_negative_price(self, auth_client, app):
        with app.app_context():
            from app.models import Product
            p = Product(name="SCProd", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = auth_client.post("/stock-check/new", data={"product_id": str(pid), "current_price": "-100"})
        assert r.status_code == 200

    def test_stock_check_create_success(self, auth_client, app):
        with app.app_context():
            from app.models import Product
            p = Product(name="SCProd2", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = auth_client.post(
            "/stock-check/new",
            data={"product_id": str(pid), "current_price": "5000", "source_url": "https://example.com", "in_stock": "y"},
            follow_redirects=True,
        )
        assert r.status_code == 200

    def test_stock_check_delete(self, auth_client, app):
        with app.app_context():
            from app.models import Product
            from app.models.stock_check import StockCheck
            p = Product(name="SCProd3", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.flush()
            sc = StockCheck(product_id=p.id, current_price=3000)
            db.session.add(sc)
            db.session.commit()
            sid = sc.id
        r = auth_client.post(f"/stock-check/{sid}/delete", follow_redirects=True)
        assert r.status_code == 200

    def test_api_quick_update_price(self, auth_client, app):
        with app.app_context():
            from app.models import Product
            from app.models.stock_check import StockCheck
            p = Product(name="QuickProd", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.flush()
            sc = StockCheck(product_id=p.id, current_price=3000)
            db.session.add(sc)
            db.session.commit()
            sid = sc.id
        r = auth_client.post(
            "/api/stock-check/quick-update",
            json={"id": sid, "current_price": 3500},
            content_type="application/json",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert data["data"]["current_price"] == 3500

    def test_api_quick_update_missing_fields(self, auth_client):
        r = auth_client.post(
            "/api/stock-check/quick-update",
            json={},
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_api_quick_update_negative_price(self, auth_client, app):
        with app.app_context():
            from app.models import Product
            from app.models.stock_check import StockCheck
            p = Product(name="NegProd", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.flush()
            sc = StockCheck(product_id=p.id, current_price=3000)
            db.session.add(sc)
            db.session.commit()
            sid = sc.id
        r = auth_client.post(
            "/api/stock-check/quick-update",
            json={"id": sid, "current_price": -100},
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_api_quick_update_not_found(self, auth_client):
        r = auth_client.post(
            "/api/stock-check/quick-update",
            json={"id": 9999, "current_price": 100},
            content_type="application/json",
        )
        assert r.status_code == 404

    def test_api_quick_add_check(self, auth_client, app):
        with app.app_context():
            from app.models import Product
            p = Product(name="QuickAddProd", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.commit()
            pid = p.id
        r = auth_client.post(
            "/api/stock-check/quick-add",
            json={"product_id": pid, "current_price": 5000, "in_stock": True},
            content_type="application/json",
        )
        assert r.status_code == 201
        data = r.get_json()
        assert data["success"] is True

    def test_api_quick_add_missing_fields(self, auth_client):
        r = auth_client.post(
            "/api/stock-check/quick-add",
            json={},
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_api_quick_add_invalid_product(self, auth_client):
        r = auth_client.post(
            "/api/stock-check/quick-add",
            json={"product_id": 9999, "current_price": 100},
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_api_products_list(self, auth_client):
        r = auth_client.get("/api/products-list")
        assert r.status_code == 200
        assert isinstance(r.get_json(), list)

    def test_api_fetch_stock_no_url(self, auth_client, app):
        with app.app_context():
            from app.models import Product
            from app.models.stock_check import StockCheck
            p = Product(name="NoUrlProd", brand="Test", purchase_price=1000, selling_price=2000)
            db.session.add(p)
            db.session.flush()
            sc = StockCheck(product_id=p.id, source_url=None)
            db.session.add(sc)
            db.session.commit()
            sid = sc.id
        r = auth_client.post(f"/api/stock-check/{sid}/fetch")
        assert r.status_code == 400


# ===========================================================================
# Brand Prices
# ===========================================================================
class TestBrandPriceRoutes:
    def test_brand_prices_dashboard_200(self, auth_client):
        r = auth_client.get("/brand-prices")
        assert r.status_code == 200

    def test_brand_prices_dashboard_with_brand(self, auth_client):
        r = auth_client.get("/brand-prices?brand=Gucci")
        assert r.status_code == 200

    def test_brand_prices_dashboard_invalid_brand(self, auth_client):
        r = auth_client.get("/brand-prices?brand=InvalidBrand")
        assert r.status_code == 200

    def test_brand_prices_export_csv(self, auth_client):
        r = auth_client.get("/brand-prices/export")
        assert r.status_code == 200
        assert "text/csv" in r.content_type

    def test_api_brand_prices_comparison(self, auth_client):
        r = auth_client.get("/api/brand-prices/comparison")
        assert r.status_code == 200
        data = r.get_json()
        assert "brand" in data

    def test_brand_prices_requires_auth(self, client):
        r = client.get("/brand-prices", follow_redirects=False)
        assert r.status_code == 302


# ===========================================================================
# FAQ Templates (additional coverage beyond existing tests)
# ===========================================================================
class TestFaqTemplateRouteExtras:
    def test_faq_templates_requires_auth(self, client):
        r = client.get("/faq-templates", follow_redirects=False)
        assert r.status_code == 302

    def test_faq_template_delete_404(self, auth_client):
        r = auth_client.post("/faq-templates/9999/delete", follow_redirects=False)
        assert r.status_code == 302  # handle_db_error catches and redirects

    def test_faq_match_api_match_multiple(self, auth_client, app):
        with app.app_context():
            from app.models.faq_template import FaqTemplate
            db.session.add(FaqTemplate(category="shipping", question_pattern="送料", answer_template="送料無料", is_active=True))
            db.session.add(FaqTemplate(category="return", question_pattern="返品", answer_template="返品OK", is_active=True))
            db.session.commit()
        r = auth_client.post("/api/faq-match", json={"text": "送料と返品について"}, content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True

    def test_faq_match_api_no_match(self, auth_client, app):
        r = auth_client.post("/api/faq-match", json={"text": "在庫ありますか"}, content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert len(data["matches"]) == 0


# ===========================================================================
# Region Recommendations
# ===========================================================================
class TestRegionRecommendationRoutes:
    def test_region_list_200(self, auth_client):
        r = auth_client.get("/regions")
        assert r.status_code == 200

    def test_region_new_form_200(self, auth_client):
        r = auth_client.get("/regions/new")
        assert r.status_code == 200

    def test_region_create_success(self, auth_client, app):
        r = auth_client.post(
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
        with app.app_context():
            from app.models.region_recommendation import RegionRecommendation
            rr = RegionRecommendation.query.filter_by(region="EU").first()
            assert rr is not None

    def test_region_create_missing_fields(self, auth_client):
        r = auth_client.post("/regions/new", data={"region": "", "region_name": ""})
        assert r.status_code == 200

    def test_region_create_negative_shipping_days(self, auth_client):
        r = auth_client.post(
            "/regions/new",
            data={"region": "XX", "region_name": "Bad", "avg_shipping_days": "-5"},
        )
        assert r.status_code == 200

    def test_region_delete(self, auth_client, app):
        with app.app_context():
            from app.models.region_recommendation import RegionRecommendation
            rr = RegionRecommendation(region="DEL", region_name="DeleteTarget", recommendation_score=50)
            db.session.add(rr)
            db.session.commit()
            rid = rr.id
        r = auth_client.post(f"/regions/{rid}/delete", follow_redirects=True)
        assert r.status_code == 200

    def test_region_delete_404(self, auth_client):
        r = auth_client.post("/regions/9999/delete", follow_redirects=False)
        assert r.status_code == 302  # handle_db_error catches and redirects

    def test_region_requires_auth(self, client):
        r = client.get("/regions", follow_redirects=False)
        assert r.status_code == 302
