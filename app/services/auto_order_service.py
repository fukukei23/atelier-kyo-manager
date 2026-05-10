"""FR-009基盤: AI自動発注ステートマシンサービス"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------


class OrderStatus:
    PENDING = "pending"
    SOURCING = "sourcing"
    CART_ADDED = "cart_added"
    CHECKOUT = "checkout"
    PAYMENT_DONE = "payment_done"
    SHIPPED = "shipped"
    COMPLETED = "completed"
    ERROR = "error"


VALID_TRANSITIONS: dict[str, list[str]] = {
    OrderStatus.PENDING: [OrderStatus.SOURCING, OrderStatus.ERROR],
    OrderStatus.SOURCING: [OrderStatus.CART_ADDED, OrderStatus.ERROR],
    OrderStatus.CART_ADDED: [OrderStatus.CHECKOUT, OrderStatus.ERROR],
    OrderStatus.CHECKOUT: [OrderStatus.PAYMENT_DONE, OrderStatus.ERROR],
    OrderStatus.PAYMENT_DONE: [OrderStatus.SHIPPED, OrderStatus.ERROR],
    OrderStatus.SHIPPED: [OrderStatus.COMPLETED, OrderStatus.ERROR],
    OrderStatus.ERROR: [OrderStatus.PENDING],
}

STATUS_LABELS: dict[str, str] = {
    OrderStatus.PENDING: "未処理",
    OrderStatus.SOURCING: "仕入先アクセス中",
    OrderStatus.CART_ADDED: "カート投入済み",
    OrderStatus.CHECKOUT: "決済中",
    OrderStatus.PAYMENT_DONE: "決済完了",
    OrderStatus.SHIPPED: "発送済み",
    OrderStatus.COMPLETED: "完了",
    OrderStatus.ERROR: "エラー",
}


# ---------------------------------------------------------------------------
# ログ
# ---------------------------------------------------------------------------


@dataclass
class AutoOrderLog:
    order_id: int
    from_status: str
    to_status: str
    note: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# サービス
# ---------------------------------------------------------------------------


class AutoOrderService:
    """自動発注ステートマシン"""

    def __init__(self, order, notification_service=None) -> None:
        self.order = order
        self.notification_service = notification_service
        self.logs: list[AutoOrderLog] = []

    # ---- 遷移 ----

    def advance_to(self, status: str, note: Optional[str] = None) -> bool:
        """ステータス遷移。遷移不可能ならFalseを返す。"""
        from_status = getattr(self.order, "status", OrderStatus.PENDING)
        allowed = VALID_TRANSITIONS.get(from_status, [])

        if status not in allowed:
            return False

        self._log(from_status, status, note)
        self.order.status = status

        if self.notification_service:
            self._notify(from_status, status, note)

        return True

    # ---- 次アクション ----

    def get_next_actions(self) -> list[str]:
        actions_map: dict[str, list[str]] = {
            OrderStatus.PENDING: ["start_sourcing"],
            OrderStatus.SOURCING: ["add_to_cart", "report_error"],
            OrderStatus.CART_ADDED: ["checkout", "report_error"],
            OrderStatus.CHECKOUT: ["process_payment", "report_error"],
            OrderStatus.PAYMENT_DONE: ["ship_order", "report_error"],
            OrderStatus.SHIPPED: ["complete_order", "report_error"],
            OrderStatus.COMPLETED: [],
            OrderStatus.ERROR: ["retry"],
        }
        return actions_map.get(getattr(self.order, "status", ""), [])

    # ---- ステップ実行（スタブ） ----

    def execute_step(self) -> tuple[bool, str]:
        """現在ステータスに応じたアクションを実行し、次ステータスへ遷移。"""
        current = getattr(self.order, "status", OrderStatus.PENDING)
        step_map = {
            OrderStatus.PENDING: (self._do_source, OrderStatus.SOURCING),
            OrderStatus.SOURCING: (self._do_add_to_cart, OrderStatus.CART_ADDED),
            OrderStatus.CART_ADDED: (self._do_checkout, OrderStatus.CHECKOUT),
            OrderStatus.CHECKOUT: (self._do_payment, OrderStatus.PAYMENT_DONE),
            OrderStatus.PAYMENT_DONE: (self._do_ship, OrderStatus.SHIPPED),
            OrderStatus.SHIPPED: (self._do_complete, OrderStatus.COMPLETED),
        }
        entry = step_map.get(current)
        if not entry:
            return False, f"No executable step for status: {current}"

        handler, next_status = entry
        ok = handler()
        if not ok:
            return False, f"Step handler failed for status: {current}"

        success = self.advance_to(next_status, note=f"Auto step: {current} → {next_status}")
        if not success:
            return False, f"Transition {current} → {next_status} not allowed"

        return True, f"Transitioned to {next_status}"

    # ---- プライベート ----

    def _do_source(self) -> bool:
        """仕入れ先URLアクセス（スタブ）"""
        return True

    def _do_add_to_cart(self) -> bool:
        """カート投入（スタブ）"""
        return True

    def _do_checkout(self) -> bool:
        """決済開始（スタブ）"""
        return True

    def _do_payment(self) -> bool:
        """決済完了（スタブ）"""
        return True

    def _do_ship(self) -> bool:
        """発送処理（スタブ）"""
        return True

    def _do_complete(self) -> bool:
        """完了処理（スタブ）"""
        return True

    def _log(self, from_status: str, to_status: str, note: Optional[str]) -> None:
        entry = AutoOrderLog(
            order_id=getattr(self.order, "id", 0),
            from_status=from_status,
            to_status=to_status,
            note=note,
        )
        self.logs.append(entry)

    def _notify(self, from_status: str, to_status: str, note: Optional[str]) -> None:
        if to_status == OrderStatus.ERROR:
            msg = f":rotating_light: 注文 #{self.order.order_number} エラー: {note}"
        else:
            label_from = STATUS_LABELS.get(from_status, from_status)
            label_to = STATUS_LABELS.get(to_status, to_status)
            msg = f":package: 注文 #{self.order.order_number}: {label_from} → {label_to}"
        self.notification_service.send(msg)
