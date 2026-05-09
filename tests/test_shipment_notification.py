"""Tests for ShipmentNotification model and routes (Issue #17 / FR-012)."""

import pytest

from app import create_app
from app.extensions import db as _db
from app.models.shipment_notification import ShipmentNotification

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    app = create_app()
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret"
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["LOGIN_DISABLED"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.app_context():
        _db.create_all()
        yield app
        _db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def clean_db(app):
    """Clear shipment_notification table between tests."""
    with app.app_context():
        ShipmentNotification.query.delete()
        _db.session.commit()
    yield
    with app.app_context():
        ShipmentNotification.query.delete()
        _db.session.commit()


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestShipmentNotificationModel:
    def test_status_label(self, app):
        with app.app_context():
            labels = {"pending": "未通知", "notified": "通知済み", "confirmed": "確認済み", "error": "エラー"}
            for status, expected in labels.items():
                n = ShipmentNotification(status=status)
                assert n.status_label() == expected

    def test_status_label_unknown(self, app):
        with app.app_context():
            n = ShipmentNotification(status="unknown_status")
            assert n.status_label() == "unknown_status"

    def test_mark_notified(self, app):
        with app.app_context():
            n = ShipmentNotification(status="pending")
            n.mark_notified()
            assert n.status == "notified"
            assert n.notified_at is not None

    def test_mark_error(self, app):
        with app.app_context():
            n = ShipmentNotification(status="pending")
            n.mark_error("API timeout")
            assert n.status == "error"
            assert n.error_message == "API timeout"

    def test_can_notify(self, app):
        with app.app_context():
            assert ShipmentNotification(status="pending").can_notify() is True
            assert ShipmentNotification(status="error").can_notify() is True
            assert ShipmentNotification(status="notified").can_notify() is False
            assert ShipmentNotification(status="confirmed").can_notify() is False


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------


class TestShipmentNotificationRoutes:
    def test_list_page(self, client, clean_db):
        resp = client.get("/shipment-notifications")
        assert resp.status_code == 200
        assert "発送通知" in resp.text

    def test_create_form(self, client, clean_db):
        resp = client.get("/shipment-notifications/new")
        assert resp.status_code == 200
        assert "発送通知" in resp.text

    def test_create_submit(self, app, client, clean_db):
        resp = client.post(
            "/shipment-notifications/new",
            data={
                "order_id": "1",
                "tracking_number": "TZ123456789",
                "warehouse": "stackry",
                "carrier": "fedex",
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200
        with app.app_context():
            n = ShipmentNotification.query.first()
            assert n is not None
            assert n.tracking_number == "TZ123456789"
            assert n.warehouse == "stackry"
            assert n.carrier == "fedex"

    def test_mark_notified_route(self, app, client, clean_db):
        with app.app_context():
            n = ShipmentNotification(order_id=1, status="pending")
            _db.session.add(n)
            _db.session.commit()
            sid = n.id
        resp = client.post(f"/shipment-notifications/{sid}/notify", follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            n = ShipmentNotification.query.get(sid)
            assert n.status == "notified"

    def test_delete_route(self, app, client, clean_db):
        with app.app_context():
            n = ShipmentNotification(order_id=1)
            _db.session.add(n)
            _db.session.commit()
            sid = n.id
        resp = client.post(f"/shipment-notifications/{sid}/delete", follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            assert ShipmentNotification.query.get(sid) is None
