"""str Enum definitions for model status fields.

Q5 decision: Python str Enum only (DB CHECK constraints deferred).
"""
from __future__ import annotations

from enum import StrEnum


class OrderStatus(StrEnum):
    PENDING = "pending"
    SOURCING = "sourcing"
    CART_ADDED = "cart_added"
    CHECKOUT = "checkout"
    PAYMENT_DONE = "payment_done"
    SHIPPED = "shipped"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


class ListingStatus(StrEnum):
    DRAFT = "draft"
    LISTED = "listed"
    SOLD = "sold"
    ARCHIVED = "archived"


class PipelineStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class ShipmentNotificationStatus(StrEnum):
    PENDING = "pending"
    NOTIFIED = "notified"
    CONFIRMED = "confirmed"
    ERROR = "error"


# ---- Auto-order state machine ----

VALID_TRANSITIONS: dict[OrderStatus, list[OrderStatus]] = {
    OrderStatus.PENDING: [OrderStatus.SOURCING, OrderStatus.ERROR],
    OrderStatus.SOURCING: [OrderStatus.CART_ADDED, OrderStatus.ERROR],
    OrderStatus.CART_ADDED: [OrderStatus.CHECKOUT, OrderStatus.ERROR],
    OrderStatus.CHECKOUT: [OrderStatus.PAYMENT_DONE, OrderStatus.ERROR],
    OrderStatus.PAYMENT_DONE: [OrderStatus.SHIPPED, OrderStatus.ERROR],
    OrderStatus.SHIPPED: [OrderStatus.COMPLETED, OrderStatus.ERROR],
    OrderStatus.ERROR: [OrderStatus.PENDING],
}

STATUS_LABELS: dict[OrderStatus, str] = {
    OrderStatus.PENDING: "未処理",
    OrderStatus.SOURCING: "仕入先アクセス中",
    OrderStatus.CART_ADDED: "カート投入済み",
    OrderStatus.CHECKOUT: "決済中",
    OrderStatus.PAYMENT_DONE: "決済完了",
    OrderStatus.SHIPPED: "発送済み",
    OrderStatus.COMPLETED: "完了",
    OrderStatus.ERROR: "エラー",
}
