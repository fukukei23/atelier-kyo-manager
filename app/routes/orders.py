# ======================================================================
# F05: 18日ルールダッシュボード + F08: キャッシュフロー予測
# ======================================================================
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from flask import Request, flash, jsonify, redirect, render_template, request, url_for
from flask_login import login_required
from sqlalchemy import desc, func

from app.core.timezone import _utcnow
from app.extensions import db
from app.models.order import PAYMENT_METHOD_EXTENSION_DAYS, Order
from app.utils.decorators import handle_db_error
from app.utils.presentation import deadline_color, deadline_message

from . import bp

# フォームの価格確度 select が取り得る値（ISSUE-101/102 Phase2 T6: α）
# browser_verified / api_verified はスクレイパ等システム側のみ設定可
_FORM_PRICE_SOURCES = ("manual_input", "estimated")


def _form_price_source(req: Request, field: str) -> str:
    """フォームの確度 select を検証して返す（不正値は manual_input に落とさず 400 相当へ）。

    鉄則: 出所の水増し（estimated を manual_input へ勝手変換等）はしない。
    """
    value = req.form.get(field, "manual_input")
    if value not in _FORM_PRICE_SOURCES:
        raise ValueError(f"不正な価格確度です: {value}")
    return value


def _form_ref_url(req: Request, field: str) -> str | None:
    """フォームの参照URL（β・optional）を検証して返す。"""
    url = (req.form.get(field, "") or "").strip()
    if not url:
        return None
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("参照URLは http:// または https:// で始めてください。")
    return url


# ---- F05: 注文管理 ------------------------------------------------------
@bp.get("/orders")
@login_required
def order_list():
    """注文一覧（18日ルール対応）"""
    page = request.args.get("page", 1, type=int)
    orders = Order.query.order_by(desc(Order.order_date)).paginate(page=page, per_page=50, error_out=False).items
    return render_template("orders.html", orders=orders)


@bp.route("/orders/new", methods=["GET", "POST"])
@login_required
@handle_db_error()
def create_order():
    """注文新規登録"""
    if request.method == "POST":
        order_date_str = request.form.get("order_date", "")
        order_date = (
            datetime.strptime(order_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) if order_date_str else _utcnow()
        )

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
            purchase_price_source=_form_price_source(request, "purchase_price_source"),
            selling_price_source=_form_price_source(request, "selling_price_source"),
            purchase_price_ref_url=_form_ref_url(request, "purchase_price_ref_url"),
            selling_price_ref_url=_form_ref_url(request, "selling_price_ref_url"),
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
            order.order_date = datetime.strptime(order_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
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
        order.purchase_price_source = _form_price_source(request, "purchase_price_source")
        order.selling_price_source = _form_price_source(request, "selling_price_source")
        order.purchase_price_ref_url = _form_ref_url(request, "purchase_price_ref_url")
        order.selling_price_ref_url = _form_ref_url(request, "selling_price_ref_url")
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
                "color": deadline_color(o),
                "message": deadline_message(o),
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
    now = _utcnow()
    pending_orders = Order.query.filter(Order.status.in_(["pending", "shipped"])).all()

    forecast_days = min(int(request.args.get("days", 30)), 365)
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

    inflow_by_date: dict[str, float] = {}
    for o in pending_orders:
        if o.expected_payment_date and o.profit:
            key = o.expected_payment_date.date().isoformat()
            inflow_by_date[key] = inflow_by_date.get(key, 0) + float(o.profit)

    for i in range(forecast_days):
        target_date = (now + timedelta(days=i)).date()
        inflow = inflow_by_date.get(target_date.isoformat(), 0.0)
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
