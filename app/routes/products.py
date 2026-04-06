# ======================================================================
# F01: 商品管理 + F02: BUYMA拡張 + CSV入出力
# ======================================================================
from __future__ import annotations

import csv
import io
from typing import List

from flask import flash, make_response, redirect, render_template, request, url_for
from flask_login import login_required

from app.extensions import db
from app.forms import ProductForm
from app.models import Product

from . import bp


# ---- 互換ルート（旧テンプレ保護） ---------------------------------------
@bp.get("/form")
def form_view():
    """旧: url_for('main.form_view') → /manage に転送"""
    return redirect(url_for("main.manage_products"))


@bp.get("/list")
def list_view():
    """旧: url_for('main.list_view') → /products に転送"""
    return redirect(url_for("main.product_list"))


# ---- 画面ルーティング ---------------------------------------------------
@bp.get("/")
def index():
    """トップページ"""
    return render_template("index.html")


@bp.route("/manage", methods=["GET", "POST"])
@login_required
def manage_products():
    """
    商品登録/更新 + 一覧
    - POST: 新規登録（最小実装。編集は /products/<id>/edit）
    - GET : フォーム + 一覧
    """
    form = ProductForm()

    if form.validate_on_submit():
        try:
            product = Product(
                name=form.name.data,
                brand=form.brand.data,
                purchase_price=form.purchase_price.data,
                selling_price=form.selling_price.data,
                transaction_fee=form.transaction_fee.data,
                shipping_cost=form.shipping_cost.data,
                customs_duty=form.customs_duty.data,
                procurement_fee=form.procurement_fee.data,
                supplier_url=form.supplier_url.data,
                image_url=form.image_url.data,
                stock_status=bool(form.stock_status.data),
                # --- BUYMA拡張 (F02) ---
                source_type=form.source_type.data or None,
                source_region=form.source_region.data or None,
                color=form.color.data or None,
                size=form.size.data or None,
                material=form.material.data or None,
                description=form.description.data or None,
                retail_price=form.retail_price.data or None,
                target_profit_rate=(form.target_profit_rate.data or 10.0) / 100.0,
                listing_status=form.listing_status.data or "draft",
            )
            product.auto_classify_tier()
            db.session.add(product)
            db.session.commit()
            flash("商品を登録しました。", "success")
            return redirect(url_for("main.manage_products"))
        except Exception as e:
            db.session.rollback()
            flash(f"登録に失敗しました: {e}", "error")

    products = Product.query.order_by(Product.id.desc()).all()
    return render_template("products/manage.html", form=form, products=products)


@bp.get("/products")
@login_required
def product_list():
    """登録データ一覧（シンプル）"""
    products = Product.query.order_by(Product.id.desc()).all()
    return render_template("list.html", products=products)


@bp.route("/products/<int:product_id>/edit", methods=["GET", "POST"])
@login_required
def edit_product(product_id: int):
    """商品編集"""
    product = Product.query.get_or_404(product_id)
    form = ProductForm(obj=product)

    if form.validate_on_submit():
        try:
            form.populate_obj(product)
            product.stock_status = bool(form.stock_status.data)
            product.target_profit_rate = (form.target_profit_rate.data or 10.0) / 100.0
            product.auto_classify_tier()
            db.session.commit()
            flash("商品を更新しました。", "success")
            return redirect(url_for("main.manage_products"))
        except Exception as e:
            db.session.rollback()
            flash(f"更新に失敗しました: {e}", "error")

    products = Product.query.order_by(Product.id.desc()).all()
    return render_template("products/manage.html", form=form, products=products)


@bp.post("/products/<int:id>/delete")
@login_required
def delete_product(id: int):
    """商品削除"""
    product = Product.query.get_or_404(id)
    try:
        db.session.delete(product)
        db.session.commit()
        flash("商品を削除しました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"削除に失敗しました: {e}", "error")
    return redirect(url_for("main.manage_products"))


# ---- CSV 入出力 --------------------------------------------------------
@bp.post("/import_csv")
@login_required
def import_csv():
    """CSV インポート"""
    file = request.files.get("file")
    if not file:
        flash("CSV ファイルが選択されていません。", "error")
        return redirect(url_for("main.manage_products"))

    try:
        data = file.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(data))

        count = 0
        for row in reader:
            if not row.get("name"):
                continue
            if row.get("purchase_price") in (None, "") or row.get("selling_price") in (None, ""):
                continue

            p = Product(
                name=row.get("name"),
                brand=row.get("brand"),
                purchase_price=float(row.get("purchase_price", 0) or 0),
                selling_price=float(row.get("selling_price", 0) or 0),
                transaction_fee=float(row.get("transaction_fee", 0) or 0),
                shipping_cost=float(row.get("shipping_cost", 0) or 0),
                customs_duty=float(row.get("customs_duty", 0) or 0),
                procurement_fee=float(row.get("procurement_fee", 0) or 0),
                supplier_url=row.get("supplier_url"),
                image_url=row.get("image_url"),
                stock_status=str(row.get("stock_status", "")).strip() in ("1", "true", "True", "yes", "on"),
                source_type=row.get("source_type") or None,
                source_region=row.get("source_region") or None,
                color=row.get("color") or None,
                size=row.get("size") or None,
                material=row.get("material") or None,
                description=row.get("description") or None,
                retail_price=float(row.get("retail_price", 0) or 0) or None,
                target_profit_rate=float(row.get("target_profit_rate", 10) or 10) / 100.0,
                listing_status=row.get("listing_status") or "draft",
            )
            p.auto_classify_tier()
            db.session.add(p)
            count += 1

        db.session.commit()
        flash(f"CSV を {count} 件取り込みました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"CSV 取り込みに失敗しました: {e}", "error")

    return redirect(url_for("main.manage_products"))


@bp.get("/export_csv")
@login_required
def export_csv():
    """CSV エクスポート"""
    products: List[Product] = Product.query.order_by(Product.id.asc()).all()
    headers = [
        "id", "name", "brand", "purchase_price", "selling_price",
        "transaction_fee", "shipping_cost", "customs_duty", "procurement_fee",
        "supplier_url", "image_url", "stock_status", "brand_tier",
        "source_type", "source_region", "color", "size", "material",
        "description", "retail_price", "target_profit_rate",
        "listing_status", "created_at", "updated_at",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()

    for p in products:
        writer.writerow({
            "id": p.id,
            "name": p.name,
            "brand": p.brand,
            "purchase_price": p.purchase_price,
            "selling_price": p.selling_price,
            "transaction_fee": p.transaction_fee,
            "shipping_cost": p.shipping_cost,
            "customs_duty": p.customs_duty,
            "procurement_fee": p.procurement_fee,
            "supplier_url": p.supplier_url,
            "image_url": p.image_url,
            "stock_status": int(bool(p.stock_status)),
            "brand_tier": p.brand_tier or "",
            "source_type": p.source_type or "",
            "source_region": p.source_region or "",
            "color": p.color or "",
            "size": p.size or "",
            "material": p.material or "",
            "description": p.description or "",
            "retail_price": p.retail_price or "",
            "target_profit_rate": round((p.target_profit_rate or 0) * 100, 1),
            "listing_status": p.listing_status or "draft",
            "created_at": (p.created_at.isoformat(sep=" ", timespec="seconds") if p.created_at else ""),
            "updated_at": (p.updated_at.isoformat(sep=" ", timespec="seconds") if p.updated_at else ""),
        })

    resp = make_response(buf.getvalue())
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=products_export.csv"
    return resp


# ---- F02: BUYMA拡張（出品テンプレート連携）-------------------------------
@bp.get("/products/<int:product_id>/generate-listing")
@login_required
def generate_listing(product_id: int):
    """商品の出品文をプレビュー生成"""
    product = Product.query.get_or_404(product_id)
    from app.services.template_service import generate_listing_text
    try:
        text = generate_listing_text(product)
    except Exception:
        text = "テンプレートが登録されていません。先にテンプレートを作成してください。"
    return render_template("listing_preview.html", product=product, listing_text=text)


@bp.get("/generate-buyma-csv")
@login_required
def generate_buyma_csv():
    """BUYMA用CSV一括生成"""
    from app.services.template_service import generate_buyma_csv as _gen_csv
    products = Product.query.filter(Product.listing_status.in_(["draft", "listed"])).all()
    csv_text = _gen_csv(products)
    resp = make_response(csv_text)
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=buyma_listing.csv"
    return resp
