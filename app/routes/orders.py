# ======================================================================
# F05: 18日ルールダッシュボード + F08: キャッシュフロー予測
# ======================================================================
from __future__ import annotations

from datetime import datetime, timedelta

from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import login_required
from sqlalchemy import desc, func

from app.extensions import db
from app.models.order import PAYMENT_METHOD_EXTENSION_DAYS, Order
from app.utils.decorators import handle_db_error

from . import bp


# ---- F05: 注文管理 ------------------------------------------------------
@bp.get("/orders")
@login_required
def order_list():
    """注文一覧（18日ルール対応）"""
    orders = Order.query.order_by(desc(Order.order_date)).all()
    return render_template("orders.html", orders=orders)


@bp.route("/orders/new", methods=["GET", "POST"])
@login_required
@handle_db_error()
def create_order():
    """注文新規登録"""
    if request.method == "POST":
        order_date_str = request.form.get("order_date", "")
        order_date = datetime.strptime(order_date_str, "%Y-%m-%d") if order_date_str else datetime.utcnow()

        selling_price = float(request.form.get("selling_price", 0) or 0)
        purchase_cost = float(request.form.get("purchase_cost", 0) or 0)
        customs_duty = float(request.form.get("customs_duty", 0) or 0)
        if selling_price < 0 or purchase_cost < 0 or customs_duty < 0:
            flash("金額に負の値は入力できません。", "error")
            return render_template(
                "order_form.html", order=None, payment_methods=list(PAYMENT_METHOD_EXTENSION_DAYS.keys())
            )

        order = Order(
            order_number=request.form.get("order_number", ""),
            product_name=request.form.get("product_name", ""),
            customer_name=request.form.get("customer_name", ""),
            order_date=order_date,
            selling_price=selling_price,
            purchase_cost=purchase_cost,
            customs_duty=customs_duty,
            payment_method=request.form.get("payment_method", ""),
            source_type=request.form.get("source_type", "domestic"),
            status=request.form.get("status", "pending"),
            notes=request.form.get("notes", ""),
        )
        order.calc_deadlines()
        order.calc_profit()
        db.session.add(order)
        db.session.commit()
        flash("注文を登録しました。", "success")
        return redirect(url_for("main.order_list"))

    return render_template("order_form.html", order=None, payment_methods=list(PAYMENT_METHOD_EXTENSION_DAYS.keys()))


@bp.route("/orders/<int:oid>/edit", methods=["GET", "POST"])
@login_required
@handle_db_error()
def edit_order(oid: int):
    """注文編集"""
    order = Order.query.get_or_404(oid)
    if request.method == "POST":
        order_date_str = request.form.get("order_date", "")
        if order_date_str:
            order.order_date = datetime.strptime(order_date_str, "%Y-%m-%d")
        order.order_number = request.form.get("order_number", order.order_number)
        order.product_name = request.form.get("product_name", order.product_name)
        order.customer_name = request.form.get("customer_name", order.customer_name)
        order.selling_price = float(request.form.get("selling_price", 0) or 0)
        order.purchase_cost = float(request.form.get("purchase_cost", 0) or 0)
        order.customs_duty = float(request.form.get("customs_duty", 0) or 0)
        if order.selling_price < 0 or order.purchase_cost < 0 or order.customs_duty < 0:
            flash("金額に負の値は入力できません。", "error")
            return render_template(
                "order_form.html", order=order, payment_methods=list(PAYMENT_METHOD_EXTENSION_DAYS.keys())
            )
        order.payment_method = request.form.get("payment_method", order.payment_method)
        order.source_type = request.form.get("source_type", "domestic")
        order.status = request.form.get("status", order.status)
        order.notes = request.form.get("notes", order.notes)
        order.calc_deadlines()
        order.calc_profit()
        db.session.commit()
        flash("注文を更新しました。", "success")
        return redirect(url_for("main.order_list"))

    return render_template("order_form.html", order=order, payment_methods=list(PAYMENT_METHOD_EXTENSION_DAYS.keys()))


@bp.post("/orders/<int:oid>/delete")
@login_required
@handle_db_error("main.order_list")
def delete_order(oid: int):
    """注文削除"""
    order = Order.query.get_or_404(oid)
    db.session.delete(order)
    db.session.commit()
    flash("注文を削除しました。", "success")
    return redirect(url_for("main.order_list"))


@bp.get("/api/orders/dashboard")
@login_required
def api_order_dashboard():
    """18日ルールダッシュボードAPI"""
    datetime.utcnow()
    orders = Order.query.filter(Order.status.in_(["pending", "shipped"])).all()
    result = []
    for o in orders:
        remaining = o.remaining_days()
        result.append(
            {
                "id": o.id,
                "order_number": o.order_number,
                "product_name": o.product_name,
                "order_date": o.order_date.isoformat() if o.order_date else None,
                "deadline_18": o.deadline_18.isoformat() if o.deadline_18 else None,
                "extension_deadline": o.extension_deadline.isoformat() if o.extension_deadline else None,
                "remaining_days": remaining,
                "color": o.deadline_color(),
                "message": o.deadline_message(),
                "extension_requested": o.extension_requested or False,
                "profit": o.profit,
                "status": o.status,
                "payment_method": o.payment_method,
            }
        )
    return jsonify(result)


@bp.post("/orders/<int:oid>/extend")
@login_required
@handle_db_error("main.order_list")
def extend_order(oid: int):
    """延長申請"""
    order = Order.query.get_or_404(oid)
    if order.extension_requested:
        flash("この注文は既に延長申請済みです。", "warning")
        return redirect(url_for("main.order_list"))
    reason = request.form.get("extension_reason", "")
    order.extension_requested = True
    order.extension_reason = reason
    db.session.commit()
    flash("延長申請を記録しました。", "success")
    return redirect(url_for("main.order_list"))


# ---- F08: キャッシュフロー予測 -------------------------------------------
@bp.get("/cashflow")
@login_required
def cashflow_dashboard():
    """キャッシュフロー予測ダッシュボード"""
    now = datetime.utcnow()
    pending_orders = Order.query.filter(Order.status.in_(["pending", "shipped"])).all()

    forecast_days = int(request.args.get("days", 30))
    daily_forecast: list[dict] = []
    running_balance = 0.0

    past_profit = (
        db.session.query(func.coalesce(func.sum(Order.profit), 0))
        .filter(
            Order.status == "completed",
            Order.completed_date >= now - timedelta(days=30),
        )
        .scalar()
        or 0
    )

    for i in range(forecast_days):
        target_date = (now + timedelta(days=i)).date()
        inflow = 0.0
        for o in pending_orders:
            if o.expected_payment_date and o.expected_payment_date.date() == target_date:
                inflow += float(o.profit or 0)
        running_balance += inflow
        daily_forecast.append(
            {
                "date": target_date.isoformat(),
                "inflow": inflow,
                "balance": running_balance,
            }
        )

    return render_template(
        "cashflow.html",
        daily_forecast=daily_forecast,
        pending_count=len(pending_orders),
        past_profit=past_profit,
        forecast_days=forecast_days,
    )
