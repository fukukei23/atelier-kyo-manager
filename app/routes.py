# app/routes.py
from __future__ import annotations
from flask import Blueprint, render_template, request, redirect, url_for, flash
from .forms import ProductForm
from .models import Product
from .extensions import db

bp = Blueprint("main", __name__)

@bp.route("/")
def index():
    return redirect(url_for("main.product_list"))

@bp.route("/products")
def product_list():
    products = Product.query.order_by(Product.created_at.desc()).all()
    return render_template("list.html", products=products)

@bp.route("/products/new", methods=["GET", "POST"])
def product_new():
    form = ProductForm()
    if form.validate_on_submit():
        p = Product(
            name=form.name.data.strip(),
            price=(form.price.data or "").strip() or None,
            source_url=(form.source_url.data or "").strip() or None,
            image_urls=form.parse_image_urls(),
            note=(form.note.data or "").strip() or None,
        )
        db.session.add(p)
        db.session.commit()
        flash("商品を登録しました。", "success")
        return redirect(url_for("main.product_list"))
    return render_template("form.html", form=form)

@bp.route("/products/<int:product_id>/edit", methods=["GET", "POST"])
def product_edit(product_id: int):
    p = Product.query.get_or_404(product_id)
    form = ProductForm(
        name=p.name,
        price=p.price,
        source_url=p.source_url,
        image_urls="\n".join(p.image_urls or []),
        note=p.note,
    )
    if form.validate_on_submit():
        p.name = form.name.data.strip()
        p.price = (form.price.data or "").strip() or None
        p.source_url = (form.source_url.data or "").strip() or None
        p.image_urls = form.parse_image_urls()
        p.note = (form.note.data or "").strip() or None
        db.session.commit()
        flash("商品を更新しました。", "success")
        return redirect(url_for("main.product_list"))
    return render_template("form.html", form=form, editing=True)

@bp.route("/products/<int:product_id>/delete", methods=["POST"])
def product_delete(product_id: int):
    p = Product.query.get_or_404(product_id)
    db.session.delete(p)
    db.session.commit()
    flash("商品を削除しました。", "info")
    return redirect(url_for("main.product_list"))
