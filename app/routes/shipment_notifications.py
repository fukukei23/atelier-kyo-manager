# ======================================================================
# FR-012基盤: 発送通知管理
# ======================================================================
from __future__ import annotations

from flask import flash, redirect, render_template, request, url_for
from flask_login import login_required

from app.extensions import db
from app.models.order import Order
from app.models.shipment_notification import ShipmentNotification
from app.utils.decorators import handle_db_error

from . import bp


@bp.get("/shipment-notifications")
@login_required
def shipment_notifications():
    """発送通知一覧"""
    notifications = ShipmentNotification.query.order_by(
        ShipmentNotification.created_at.desc()
    ).all()
    return render_template("shipment_notifications.html", notifications=notifications)


@bp.route("/shipment-notifications/new", methods=["GET", "POST"])
@login_required
@handle_db_error()
def create_shipment_notification():
    """発送通知手動登録"""
    if request.method == "POST":
        order_id = request.form.get("order_id", type=int)
        tracking_number = request.form.get("tracking_number", "").strip()
        warehouse = request.form.get("warehouse", "").strip()
        carrier = request.form.get("carrier", "").strip()
        if not order_id:
            flash("注文を選択してください。", "error")
            orders = Order.query.order_by(Order.order_date.desc()).all()
            return render_template("shipment_notification_form.html", orders=orders)
        sn = ShipmentNotification(
            order_id=order_id,
            tracking_number=tracking_number or None,
            warehouse=warehouse or None,
            carrier=carrier or None,
        )
        db.session.add(sn)
        db.session.commit()
        flash("発送通知を登録しました。", "success")
        return redirect(url_for("main.shipment_notifications"))
    orders = Order.query.order_by(Order.order_date.desc()).all()
    return render_template("shipment_notification_form.html", orders=orders)


@bp.post("/shipment-notifications/<int:sid>/notify")
@login_required
@handle_db_error("main.shipment_notifications")
def mark_shipment_notified(sid: int):
    """発送通知を通知済みに更新"""
    sn = ShipmentNotification.query.get_or_404(sid)
    if not sn.can_notify():
        flash("この通知は既に処理済みです。", "error")
        return redirect(url_for("main.shipment_notifications"))
    sn.mark_notified()
    db.session.commit()
    flash("発送通知を「通知済み」に更新しました。", "success")
    return redirect(url_for("main.shipment_notifications"))


@bp.post("/shipment-notifications/<int:sid>/delete")
@login_required
@handle_db_error("main.shipment_notifications")
def delete_shipment_notification(sid: int):
    """発送通知削除"""
    sn = ShipmentNotification.query.get_or_404(sid)
    db.session.delete(sn)
    db.session.commit()
    flash("発送通知を削除しました。", "success")
    return redirect(url_for("main.shipment_notifications"))
