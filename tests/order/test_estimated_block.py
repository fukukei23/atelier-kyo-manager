"""ISSUE-101/102 Phase2 T9 — ESTIMATED 注文の本番発注ブロック（完全ブロック・ドラフト保持）。

spec 3.2: 推定 source で計算した注文は pending から shipped/completed へ遷移不可。
実価格 source へ更新 + 再計算で confirmed に遷移して初めて発注可能。
"""

from __future__ import annotations

from app.extensions import db
from app.models.order import Order


def _order(**overrides) -> Order:
    base = {
        "order_number": "T-T9-001",
        "product_name": "テスト商品",
        "selling_price": 50000.0,
        "purchase_cost": 30000.0,
        "customs_duty": 0.0,
        "source_type": "domestic",
        "status": "pending",
        "selling_price_source": "estimated",  # 推定で作成
        "purchase_price_source": "manual_input",
    }
    base.update(overrides)
    return Order(**base)


class TestEstimatedBlockModel:
    """Order.can_advance_to による遷移ブロック。"""

    def test_estimated_blocks_shipped(self):
        order = _order()
        order.calc_profit()
        assert order.profit_status == "estimated"
        assert order.can_advance_to("shipped") is False

    def test_estimated_blocks_completed(self):
        order = _order()
        order.calc_profit()
        assert order.can_advance_to("completed") is False

    def test_estimated_allows_cancel_and_stay(self):
        """キャンセル・pending維持は推定でも可能（ドラフト保持の意味）。"""
        order = _order()
        order.calc_profit()
        assert order.can_advance_to("cancelled") is True
        assert order.can_advance_to("pending") is True

    def test_confirmed_allows_shipped(self):
        order = _order(selling_price_source="manual_input")
        order.calc_profit()
        assert order.profit_status == "confirmed"
        assert order.can_advance_to("shipped") is True

    def test_confirm_by_source_update_and_recalc(self):
        """実価格 source 更新→再計算で confirmed に遷移→発注可能（回復経路）。"""
        order = _order()
        order.calc_profit()
        assert order.can_advance_to("shipped") is False

        order.selling_price_source = "manual_input"
        order.calc_profit()
        assert order.profit_status == "confirmed"
        assert order.can_advance_to("shipped") is True


class TestEstimatedBlockRoute:
    """POST /orders/<id>/edit での遷移ブロック（フォーム経由の水増し防止）。"""

    def test_edit_status_shipped_blocked_for_estimated(self, routes_app, routes_auth_client):
        with routes_app.app_context():
            order = _order(order_number="T-T9-ROUTE")
            order.calc_deadlines()
            order.calc_profit()
            db.session.add(order)
            db.session.commit()
            oid = order.id
            assert order.profit_status == "estimated"

        r = routes_auth_client.post(
            f"/orders/{oid}/edit",
            data={
                "order_number": "T-T9-ROUTE",
                "product_name": "テスト商品",
                "order_date": "2026-08-14",
                "selling_price": "50000",
                "purchase_cost": "30000",
                "customs_duty": "0",
                "purchase_price_source": "manual_input",
                "selling_price_source": "estimated",
                "purchase_price_ref_url": "",
                "selling_price_ref_url": "",
                "payment_method": "bank_transfer",
                "source_type": "domestic",
                "status": "shipped",
                "notes": "",
            },
            follow_redirects=True,
        )
        assert r.status_code == 200

        with routes_app.app_context():
            updated = db.session.get(Order, oid)
            assert updated.status == "pending"  # ブロックされドラフト保持
