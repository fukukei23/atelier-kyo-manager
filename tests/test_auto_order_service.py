"""Tests for AutoOrderService and routes (Issue #18 / FR-009)."""

import pytest

from app import create_app
from app.extensions import db as _db
from app.models.order import Order
from app.services.auto_order_service import (
    AutoOrderService,
    AutoOrderLog,
    OrderStatus,
    VALID_TRANSITIONS,
    STATUS_LABELS,
)


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
def sample_order(app):
    """Create a fresh order for each test."""
    with app.app_context():
        _db.session.query(Order).delete()
        _db.session.commit()
        o = Order(
            order_number="TEST-001",
            product_name="Test Product",
            status=OrderStatus.PENDING,
            selling_price=10000,
            purchase_cost=5000,
        )
        _db.session.add(o)
        _db.session.commit()
        yield o
        with app.app_context():
            _db.session.query(Order).delete()
            _db.session.commit()


# ---------------------------------------------------------------------------
# Service tests
# ---------------------------------------------------------------------------

class TestAutoOrderServiceModel:
    def test_valid_transitions_structure(self):
        """All defined statuses appear in VALID_TRANSITIONS."""
        for s in [OrderStatus.PENDING, OrderStatus.SOURCING, OrderStatus.CART_ADDED,
                  OrderStatus.CHECKOUT, OrderStatus.PAYMENT_DONE, OrderStatus.SHIPPED,
                  OrderStatus.ERROR]:
            assert s in VALID_TRANSITIONS

    def test_status_labels(self):
        assert STATUS_LABELS[OrderStatus.PENDING] == "未処理"
        assert STATUS_LABELS[OrderStatus.ERROR] == "エラー"
        assert STATUS_LABELS[OrderStatus.COMPLETED] == "完了"

    def test_advance_valid(self, app, sample_order):
        with app.app_context():
            o = Order.query.get(sample_order.id)
            svc = AutoOrderService(o)
            assert svc.advance_to(OrderStatus.SOURCING) is True
            assert o.status == OrderStatus.SOURCING

    def test_advance_invalid(self, app, sample_order):
        with app.app_context():
            o = Order.query.get(sample_order.id)
            svc = AutoOrderService(o)
            # pending → completed is not allowed
            assert svc.advance_to(OrderStatus.COMPLETED) is False
            assert o.status == OrderStatus.PENDING

    def test_advance_with_notification(self, app, sample_order):
        class MockNotifier:
            def __init__(self):
                self.messages = []
            def send(self, msg):
                self.messages.append(msg)
                return {"success": True, "error": None}

        with app.app_context():
            o = Order.query.get(sample_order.id)
            notifier = MockNotifier()
            svc = AutoOrderService(o, notification_service=notifier)
            svc.advance_to(OrderStatus.SOURCING, note="test note")
            assert len(notifier.messages) == 1
            assert "TEST-001" in notifier.messages[0]

    def test_advance_error_notification(self, app, sample_order):
        class MockNotifier:
            def __init__(self):
                self.messages = []
            def send(self, msg):
                self.messages.append(msg)
                return {"success": True, "error": None}

        with app.app_context():
            o = Order.query.get(sample_order.id)
            notifier = MockNotifier()
            svc = AutoOrderService(o, notification_service=notifier)
            svc.advance_to(OrderStatus.ERROR, note="在庫切れ")
            assert len(notifier.messages) == 1
            assert "エラー" in notifier.messages[0]

    def test_get_next_actions(self, app, sample_order):
        with app.app_context():
            o = Order.query.get(sample_order.id)
            svc = AutoOrderService(o)
            assert "start_sourcing" in svc.get_next_actions()

    def test_get_next_actions_error(self, app, sample_order):
        with app.app_context():
            o = Order.query.get(sample_order.id)
            o.status = OrderStatus.ERROR
            svc = AutoOrderService(o)
            assert "retry" in svc.get_next_actions()

    def test_get_next_actions_completed(self, app, sample_order):
        with app.app_context():
            o = Order.query.get(sample_order.id)
            o.status = OrderStatus.COMPLETED
            svc = AutoOrderService(o)
            assert svc.get_next_actions() == []

    def test_execute_step(self, app, sample_order):
        with app.app_context():
            o = Order.query.get(sample_order.id)
            svc = AutoOrderService(o)
            ok, msg = svc.execute_step()
            assert ok is True
            assert OrderStatus.SOURCING in msg

    def test_execute_step_no_handler(self, app, sample_order):
        with app.app_context():
            o = Order.query.get(sample_order.id)
            o.status = OrderStatus.COMPLETED
            svc = AutoOrderService(o)
            ok, msg = svc.execute_step()
            assert ok is False

    def test_logs_recorded(self, app, sample_order):
        AutoOrderService.logs.clear()
        with app.app_context():
            o = Order.query.get(sample_order.id)
            svc = AutoOrderService(o)
            svc.advance_to(OrderStatus.SOURCING)
            assert len(AutoOrderService.logs) == 1
            log = AutoOrderService.logs[-1]
            assert log.from_status == OrderStatus.PENDING
            assert log.to_status == OrderStatus.SOURCING

    def test_full_pipeline(self, app, sample_order):
        """Walk through the entire pipeline: pending → completed."""
        with app.app_context():
            o = Order.query.get(sample_order.id)
            svc = AutoOrderService(o)

            transitions = [
                (OrderStatus.SOURCING, OrderStatus.CART_ADDED),
                (OrderStatus.CHECKOUT, OrderStatus.PAYMENT_DONE),
                (OrderStatus.SHIPPED, OrderStatus.COMPLETED),
            ]
            # First: pending → sourcing
            assert svc.advance_to(OrderStatus.SOURCING)
            for from_s, to_s in transitions:
                o.status = from_s  # simulate being at that step
                assert svc.advance_to(to_s)

            assert o.status == OrderStatus.COMPLETED


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------

class TestAutoOrderRoutes:
    def test_list_page(self, client, sample_order, app):
        resp = client.get("/auto-orders")
        assert resp.status_code == 200
        assert "自動発注" in resp.text

    def test_start_auto_order(self, client, sample_order, app):
        resp = client.post(f"/auto-orders/{sample_order.id}/start",
                           follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            o = Order.query.get(sample_order.id)
            assert o.status == OrderStatus.SOURCING

    def test_execute_step_route(self, client, sample_order, app):
        with app.app_context():
            o = Order.query.get(sample_order.id)
            o.status = OrderStatus.PENDING
            _db.session.commit()
        # Start first (pending → sourcing)
        client.post(f"/auto-orders/{sample_order.id}/start")
        # Execute step (sourcing → cart_added)
        resp = client.post(f"/auto-orders/{sample_order.id}/step",
                           follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            o = Order.query.get(sample_order.id)
            assert o.status == OrderStatus.CART_ADDED

    def test_report_error(self, client, sample_order, app):
        with app.app_context():
            o = Order.query.get(sample_order.id)
            o.status = OrderStatus.SOURCING
            _db.session.commit()
        resp = client.post(f"/auto-orders/{sample_order.id}/error",
                           data={"error_note": "在庫切れ"},
                           follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            o = Order.query.get(sample_order.id)
            assert o.status == OrderStatus.ERROR
