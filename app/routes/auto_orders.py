# ======================================================================
# FR-009: AI自動発注管理
# ======================================================================
from __future__ import annotations

from flask import flash, redirect, render_template, request, url_for
from flask_login import login_required

from app.extensions import db
from app.models.order import Order
from app.services.auto_order_service import AutoOrderService, OrderStatus
from app.services.notification_service import NotificationService

from . import bp


@bp.get("/auto-orders")
@login_required
def auto_orders():
    """自動発注ステータス一覧"""
    orders = Order.query.order_by(Order.order_date.desc()).all()
    return render_template("auto_orders.html", orders=orders)


@bp.post("/auto-orders/<int:oid>/start")
@login_required
def start_auto_order(oid: int):
    """自動発注開始"""
    order = Order.query.get_or_404(oid)
    order.status = OrderStatus.PENDING
    db.session.commit()

    svc = AutoOrderService(order, notification_service=NotificationService())
    ok = svc.advance_to(OrderStatus.SOURCING, note="自動発注開始")
    if ok:
        db.session.commit()
        flash(f"注文 #{order.order_number} の自動発注を開始しました。", "success")
    else:
        flash("ステータス遷移に失敗しました。", "error")
    return redirect(url_for("main.auto_orders"))


@bp.post("/auto-orders/<int:oid>/step")
@login_required
def execute_auto_order_step(oid: int):
    """自動発注ステップ実行"""
    order = Order.query.get_or_404(oid)
    svc = AutoOrderService(order, notification_service=NotificationService())
    ok, msg = svc.execute_step()
    if ok:
        db.session.commit()
        flash(f"ステップ実行成功: {msg}", "success")
    else:
        flash(f"ステップ実行失敗: {msg}", "error")
    return redirect(url_for("main.auto_orders"))


@bp.post("/auto-orders/<int:oid>/error")
@login_required
def report_auto_order_error(oid: int):
    """エラー報告"""
    order = Order.query.get_or_404(oid)
    note = request.form.get("error_note", "手動エラー報告")
    svc = AutoOrderService(order, notification_service=NotificationService())
    svc.advance_to(OrderStatus.ERROR, note=note)
    db.session.commit()
    flash(f"注文 #{order.order_number} をエラー状態にしました。", "error")
    return redirect(url_for("main.auto_orders"))
