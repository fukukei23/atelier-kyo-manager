# ======================================================================
# F03: 出品テンプレート管理
# ======================================================================
from __future__ import annotations

from types import SimpleNamespace

from flask import flash, redirect, render_template, request, url_for
from flask_login import login_required

from app.extensions import db
from app.models.listing_template import ListingTemplate
from app.utils.decorators import handle_db_error

from . import bp


@bp.get("/templates")
@login_required
def listing_templates():
    """テンプレート一覧"""
    templates = ListingTemplate.query.order_by(ListingTemplate.is_default.desc(), ListingTemplate.id.asc()).all()
    return render_template("listing_templates.html", templates=templates)


@bp.route("/templates/new", methods=["GET", "POST"])
@bp.route("/templates/<int:tid>/edit", methods=["GET", "POST"])
@login_required
@handle_db_error()
def edit_listing_template(tid: int | None = None):
    """テンプレート新規/編集"""
    tpl = ListingTemplate.query.get(tid) if tid else None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        template_text = request.form.get("template_text", "")
        category = request.form.get("category", "general")
        is_default = "is_default" in request.form

        if not name or not template_text:
            flash("テンプレート名と本文は必須です。", "error")
            return render_template(
                "edit_listing_template.html",
                tpl=tpl
                or SimpleNamespace(name=name, template_text=template_text, category=category, is_default=is_default),
            )

        if tpl is None:
            tpl = ListingTemplate()
            db.session.add(tpl)
        tpl.name = name
        tpl.template_text = template_text
        tpl.category = category
        tpl.is_default = is_default
        db.session.commit()
        flash("テンプレートを保存しました。", "success")
        return redirect(url_for("main.listing_templates"))

    return render_template("edit_listing_template.html", tpl=tpl)


@bp.post("/templates/<int:tid>/delete")
@login_required
@handle_db_error("main.listing_templates")
def delete_listing_template(tid: int):
    """テンプレート削除"""
    tpl = ListingTemplate.query.get_or_404(tid)
    db.session.delete(tpl)
    db.session.commit()
    flash("テンプレートを削除しました。", "success")
    return redirect(url_for("main.listing_templates"))
