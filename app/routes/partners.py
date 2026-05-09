# ======================================================================
# F06: パートナー管理 + F13: リピーター管理
# ======================================================================
from __future__ import annotations

from flask import flash, redirect, render_template, request, url_for
from flask_login import login_required

from app.extensions import db
from app.utils.decorators import handle_db_error

from . import bp


# ---- F06: パートナー管理 ------------------------------------------------
@bp.get("/partners")
@login_required
def partner_list():
    """パートナー一覧"""
    from app.models.partner import Partner
    partners = Partner.query.order_by(Partner.priority_level.asc(), Partner.name.asc()).all()
    return render_template("partners.html", partners=partners)


@bp.route("/partners/new", methods=["GET", "POST"])
@login_required
@handle_db_error()
def create_partner():
    """パートナー新規登録"""
    from app.models.partner import Partner
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("パートナー名は必須です。", "error")
            return render_template("partner_form.html", partner=None)
        p = Partner(
            name=name,
            email=request.form.get("email", ""),
            phone=request.form.get("phone", ""),
            active_regions=request.form.get("active_regions", ""),
            specialty_brands=request.form.get("specialty_brands", ""),
            priority_level=request.form.get("priority_level", "medium"),
            status=request.form.get("status", "active"),
            notes=request.form.get("notes", ""),
        )
        db.session.add(p)
        db.session.commit()
        flash("パートナーを登録しました。", "success")
        return redirect(url_for("main.partner_list"))
    return render_template("partner_form.html", partner=None)


@bp.route("/partners/<int:pid>/edit", methods=["GET", "POST"])
@login_required
@handle_db_error()
def edit_partner(pid: int):
    """パートナー編集"""
    from app.models.partner import Partner
    p = Partner.query.get_or_404(pid)
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("パートナー名は必須です。", "error")
            return render_template("partner_form.html", partner=p)
        p.name = name
        p.email = request.form.get("email", p.email)
        p.phone = request.form.get("phone", p.phone)
        p.active_regions = request.form.get("active_regions", p.active_regions)
        p.specialty_brands = request.form.get("specialty_brands", p.specialty_brands)
        p.priority_level = request.form.get("priority_level", "medium")
        p.status = request.form.get("status", "active")
        p.notes = request.form.get("notes", "")
        db.session.commit()
        flash("パートナーを更新しました。", "success")
        return redirect(url_for("main.partner_list"))
    return render_template("partner_form.html", partner=p)


@bp.post("/partners/<int:pid>/delete")
@login_required
@handle_db_error("main.partner_list")
def delete_partner(pid: int):
    """パートナー削除"""
    from app.models.partner import Partner
    p = Partner.query.get_or_404(pid)
    db.session.delete(p)
    db.session.commit()
    flash("パートナーを削除しました。", "success")
    return redirect(url_for("main.partner_list"))


# ---- F13: リピーター管理 ------------------------------------------------
@bp.get("/customers")
@login_required
def customer_list():
    """リピーター一覧"""
    from app.models.repeat_customer import RepeatCustomer
    customers = RepeatCustomer.query.order_by(RepeatCustomer.total_orders.desc()).all()
    return render_template("repeat_customers.html", customers=customers)


@bp.route("/customers/new", methods=["GET", "POST"])
@login_required
@handle_db_error()
def create_customer():
    """顧客新規登録"""
    from app.models.repeat_customer import RepeatCustomer
    from datetime import datetime as _dt
    if request.method == "POST":
        total_orders = int(request.form.get("total_orders", 0) or 0)
        total_spent = float(request.form.get("total_spent", 0) or 0)
        if total_orders < 0 or total_spent < 0:
            flash("注文件数・合計金額に負の値は入力できません。", "error")
            return render_template("repeat_customer_form.html", customer=None)
        c = RepeatCustomer(
            customer_name=request.form.get("customer_name", ""),
            email=request.form.get("email", ""),
            phone=request.form.get("phone", ""),
            total_orders=total_orders,
            total_spent=total_spent,
        )
        fod = request.form.get("first_order_date", "")
        lod = request.form.get("last_order_date", "")
        if fod:
            c.first_order_date = _dt.strptime(fod, "%Y-%m-%d")
        if lod:
            c.last_order_date = _dt.strptime(lod, "%Y-%m-%d")
        c.notes = request.form.get("notes", "")
        c.update_avg()
        c.segment = c.calc_segment()
        db.session.add(c)
        db.session.commit()
        flash("顧客を登録しました。", "success")
        return redirect(url_for("main.customer_list"))
    return render_template("repeat_customer_form.html", customer=None)


@bp.route("/customers/<int:cid>/edit", methods=["GET", "POST"])
@login_required
@handle_db_error()
def edit_customer(cid: int):
    """顧客編集"""
    from app.models.repeat_customer import RepeatCustomer
    from datetime import datetime as _dt
    c = RepeatCustomer.query.get_or_404(cid)
    if request.method == "POST":
        c.customer_name = request.form.get("customer_name", c.customer_name)
        c.email = request.form.get("email", "")
        c.phone = request.form.get("phone", "")
        c.total_orders = int(request.form.get("total_orders", 0) or 0)
        c.total_spent = float(request.form.get("total_spent", 0) or 0)
        if c.total_orders < 0 or c.total_spent < 0:
            flash("注文件数・合計金額に負の値は入力できません。", "error")
            return render_template("repeat_customer_form.html", customer=c)
        fod = request.form.get("first_order_date", "")
        lod = request.form.get("last_order_date", "")
        if fod:
            c.first_order_date = _dt.strptime(fod, "%Y-%m-%d")
        if lod:
            c.last_order_date = _dt.strptime(lod, "%Y-%m-%d")
        c.notes = request.form.get("notes", "")
        c.update_avg()
        c.segment = c.calc_segment()
        db.session.commit()
        flash("顧客を更新しました。", "success")
        return redirect(url_for("main.customer_list"))
    return render_template("repeat_customer_form.html", customer=c)


@bp.post("/customers/<int:cid>/delete")
@login_required
@handle_db_error("main.customer_list")
def delete_customer(cid: int):
    """顧客削除"""
    from app.models.repeat_customer import RepeatCustomer
    c = RepeatCustomer.query.get_or_404(cid)
    db.session.delete(c)
    db.session.commit()
    flash("顧客を削除しました。", "success")
    return redirect(url_for("main.customer_list"))
