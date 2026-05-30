"""ルートテスト: Shipment Notifications + Listing Progress + FAQ Templates"""

from __future__ import annotations

from datetime import date

import pytest

from app.extensions import db


class TestShipmentNotificationRoutes:
    def test_shipment_notifications_list_200(self, routes_auth_client):
        r = routes_auth_client.get("/shipment-notifications")
        assert r.status_code == 200

    def test_shipment_notification_new_form_200(self, routes_auth_client):
        r = routes_auth_client.get("/shipment-notifications/new")
        assert r.status_code == 200

    def test_shipment_notification_create_missing_order(self, routes_auth_client):
        r = routes_auth_client.post(
            "/shipment-notifications/new",
            data={"order_id": "", "tracking_number": "AB123"},
        )
        assert r.status_code == 200  # エラーflashでフォーム再描画

    def test_shipment_notification_create_success(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.order import Order
            o = Order(order_number="SN-001", product_name="TestSN")
            db.session.add(o)
            db.session.commit()
            oid = o.id
        r = routes_auth_client.post(
            "/shipment-notifications/new",
            data={"order_id": str(oid), "tracking_number": "TRACK123"},
            follow_redirects=True,
        )
        assert r.status_code == 200

    def test_shipment_notification_mark_notified(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.order import Order
            from app.models.shipment_notification import ShipmentNotification
            o = Order(order_number="SN-002", product_name="TestSN2")
            db.session.add(o)
            db.session.flush()
            sn = ShipmentNotification(order_id=o.id, status="pending")
            db.session.add(sn)
            db.session.commit()
            sid = sn.id
        r = routes_auth_client.post(f"/shipment-notifications/{sid}/notify", follow_redirects=True)
        assert r.status_code == 200

    def test_shipment_notification_mark_notified_already_done(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.order import Order
            from app.models.shipment_notification import ShipmentNotification
            o = Order(order_number="SN-003", product_name="TestSN3")
            db.session.add(o)
            db.session.flush()
            sn = ShipmentNotification(order_id=o.id, status="notified")
            db.session.add(sn)
            db.session.commit()
            sid = sn.id
        r = routes_auth_client.post(f"/shipment-notifications/{sid}/notify", follow_redirects=True)
        assert r.status_code == 200

    def test_shipment_notification_delete(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.order import Order
            from app.models.shipment_notification import ShipmentNotification
            o = Order(order_number="SN-004", product_name="TestSN4")
            db.session.add(o)
            db.session.flush()
            sn = ShipmentNotification(order_id=o.id)
            db.session.add(sn)
            db.session.commit()
            sid = sn.id
        r = routes_auth_client.post(f"/shipment-notifications/{sid}/delete", follow_redirects=True)
        assert r.status_code == 200

    def test_shipment_notification_404(self, routes_auth_client):
        r = routes_auth_client.post("/shipment-notifications/9999/notify", follow_redirects=False)
        assert r.status_code == 302  # handle_db_error が捕捉してリダイレクト


class TestListingProgressRoutes:
    def test_listing_progress_200(self, routes_auth_client):
        r = routes_auth_client.get("/listing-progress")
        assert r.status_code == 200

    def test_listing_progress_new_form_200(self, routes_auth_client):
        r = routes_auth_client.get("/listing-progress/new")
        assert r.status_code == 200

    def test_listing_progress_create(self, routes_auth_client):
        today = date.today().isoformat()
        r = routes_auth_client.post(
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
        assert r.status_code == 302

    def test_listing_progress_create_negative(self, routes_auth_client):
        r = routes_auth_client.post(
            "/listing-progress/new",
            data={"listings_count": "-1", "target_daily": "20"},
        )
        assert r.status_code == 200

    def test_listing_progress_delete(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.listing_progress import ListingProgress
            lp = ListingProgress(
                record_date=date.today(), listings_count=10,
                target_daily=20, target_monthly=600, cumulative_monthly=10,
            )
            db.session.add(lp)
            db.session.commit()
            rid = lp.id
        r = routes_auth_client.post(f"/listing-progress/{rid}/delete", follow_redirects=True)
        assert r.status_code == 200

    def test_listing_progress_delete_404(self, routes_auth_client):
        r = routes_auth_client.post("/listing-progress/9999/delete", follow_redirects=False)
        assert r.status_code == 302


class TestFaqTemplateRouteExtras:
    def test_faq_templates_requires_auth(self, routes_client):
        r = routes_client.get("/faq-templates", follow_redirects=False)
        assert r.status_code == 302

    def test_faq_template_delete_404(self, routes_auth_client):
        r = routes_auth_client.post("/faq-templates/9999/delete", follow_redirects=False)
        assert r.status_code == 302

    def test_faq_match_api_match_multiple(self, routes_auth_client, routes_app):
        with routes_app.app_context():
            from app.models.faq_template import FaqTemplate
            db.session.add(FaqTemplate(category="shipping", question_pattern="送料", answer_template="送料無料", is_active=True))
            db.session.add(FaqTemplate(category="return", question_pattern="返品", answer_template="返品OK", is_active=True))
            db.session.commit()
        r = routes_auth_client.post(
            "/api/faq-match",
            json={"text": "送料と返品について"},
            content_type="application/json",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True

    def test_faq_match_api_no_match(self, routes_auth_client):
        r = routes_auth_client.post(
            "/api/faq-match",
            json={"text": "在庫ありますか"},
            content_type="application/json",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert data["success"] is True
        assert len(data["matches"]) == 0
