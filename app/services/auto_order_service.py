"""FR-009基盤: AI自動発注ステートマシンサービス"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import logging

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

from app.models.enums import (
    OrderStatus,
    STATUS_LABELS,
    VALID_TRANSITIONS,
)


# ---------------------------------------------------------------------------
# ログ
# ---------------------------------------------------------------------------


@dataclass
class AutoOrderLog:
    order_id: int
    from_status: str
    to_status: str
    note: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# サービス
# ---------------------------------------------------------------------------


class AutoOrderService:
    """自動発注ステートマシン"""

    _logger = logging.getLogger(__name__)

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
        try:
            ok = handler()
        except Exception as e:
            self._logger.error("Step handler exception for order %s: %s", getattr(self.order, "id", "?"), e, exc_info=True)
            self.advance_to(OrderStatus.ERROR, note=f"Handler exception: {type(e).__name__}")
            return False, f"Handler exception: {type(e).__name__}"

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
        self._logger.info("Order %s: %s → %s (%s)", entry.order_id, from_status, to_status, note or "")

    def _notify(self, from_status: str, to_status: str, note: Optional[str]) -> None:
        if to_status == OrderStatus.ERROR:
            msg = f":rotating_light: 注文 #{self.order.order_number} エラー: {note}"
        else:
            label_from = STATUS_LABELS.get(from_status, from_status)
            label_to = STATUS_LABELS.get(to_status, to_status)
            msg = f":package: 注文 #{self.order.order_number}: {label_from} → {label_to}"
        self.notification_service.send(msg)
