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
                # --- FR-005 実ベース利益計算 ---
                warehouse_shipping_cost=form.warehouse_shipping_cost.data or 0.0,
                original_currency=form.original_currency.data or "JPY",
                exchange_rate=form.exchange_rate.data or 1.0,
                item_category=form.item_category.data or None,
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

    # ファイル名・拡張子チェック
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        flash("CSV ファイルのみアップロード可能です。", "error")
        return redirect(url_for("main.manage_products"))

    # ファイルサイズチェック（10MB上限）
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > 10 * 1024 * 1024:
        flash("ファイルサイズは10MB以下にしてください。", "error")
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
                # --- FR-005 実ベース利益計算 ---
                warehouse_shipping_cost=float(row.get("warehouse_shipping_cost", 0) or 0),
                original_currency=row.get("original_currency", "JPY") or "JPY",
                exchange_rate=float(row.get("exchange_rate", 1) or 1),
                item_category=row.get("item_category") or None,
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
        # --- FR-005 実ベース利益計算 ---
        "warehouse_shipping_cost", "original_currency", "exchange_rate", "item_category",
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
            # --- FR-005 実ベース利益計算 ---
            "warehouse_shipping_cost": p.warehouse_shipping_cost or 0,
            "original_currency": p.original_currency or "JPY",
            "exchange_rate": p.exchange_rate or 1.0,
            "item_category": p.item_category or "",
        })

    resp = make_response(buf.getvalue())
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=products_export.csv"
    return resp


# ---- FR-007: 出品候補リスト ------------------------------------------------
@bp.get("/listing-candidates")
@login_required
def listing_candidates():
    """出品候補リスト（画面表示）"""
    min_rate = request.args.get("min_profit_rate", 10.0, type=float)
    candidates = Product.listing_candidates(min_profit_rate=min_rate)
    avg_rate = sum(p.profit_rate() for p in candidates) / len(candidates) if candidates else 0
    total_profit = sum(p.calculate_profit() for p in candidates)
    return render_template(
        "listing_candidates.html",
        candidates=candidates,
        min_profit_rate=min_rate,
        avg_profit_rate=avg_rate,
        total_profit=total_profit,
    )


@bp.get("/listing-candidates/export")
@login_required
def export_listing_candidates():
    """出品候補リスト CSV エクスポート"""
    min_rate = request.args.get("min_profit_rate", 10.0, type=float)
    candidates = Product.listing_candidates(min_profit_rate=min_rate)

    headers = [
        "id", "name", "brand", "brand_tier", "purchase_price", "selling_price",
        "profit", "profit_rate", "stock_status", "listing_status",
        "target_profit_rate", "source_type", "item_category",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()

    for p in candidates:
        writer.writerow({
            "id": p.id,
            "name": p.name,
            "brand": p.brand or "",
            "brand_tier": p.brand_tier or "",
            "purchase_price": p.purchase_price,
            "selling_price": p.selling_price,
            "profit": round(p.calculate_profit(), 0),
            "profit_rate": round(p.profit_rate(), 1),
            "stock_status": int(bool(p.stock_status)),
            "listing_status": p.listing_status or "draft",
            "target_profit_rate": round((p.target_profit_rate or 0) * 100, 1),
            "source_type": p.source_type or "",
            "item_category": p.item_category or "",
        })

    resp = make_response(buf.getvalue())
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=listing_candidates.csv"
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
    csv_text, skipped = _gen_csv(products)
    resp = make_response(csv_text)
    resp.headers["Content-Type"] = "text/csv; charset=utf-8-sig"
    resp.headers["Content-Disposition"] = "attachment; filename=buyma_listing.csv"
    flash(f"CSV生成完了: {len(products) - skipped}件出力, {skipped}件スキップ（説明文なし）", "info")
    return resp


# ---- FR-002/003: パイプライン統合 ----------------------------------------
@bp.post("/products/<int:product_id>/run-pipeline")
@login_required
def run_pipeline(product_id: int):
    """画像収集→背景除去→説明文生成→出品文生成の統合パイプライン"""
    product = Product.query.get_or_404(product_id)
    from app.services.pipeline_service import PipelineService
    svc = PipelineService()
    site_key = request.form.get("site_key", "")
    result = svc.run(product_id=product_id, site_key=site_key)

    # Slack通知
    from flask import current_app
    from app.services.notification_service import NotificationService
    ns = NotificationService(current_app._get_current_object())
    ns.send_pipeline_result(
        product_name=product.name,
        status=result.status,
        elapsed_sec=result.elapsed_sec,
        errors="; ".join(result.errors) if result.errors else None,
    )

    if result.status == "failed":
        flash(f"パイプライン実行に失敗しました: {'; '.join(result.errors)}", "error")
    elif result.status == "partial":
        flash(f"パイプライン部分成功（{result.elapsed_sec}秒）。一部エラー: {'; '.join(result.errors)}", "warning")
    else:
        flash(f"パイプライン完了（{result.elapsed_sec}秒）。画像{len(result.processed_paths)}枚処理。", "success")

    return redirect(url_for("main.pipeline_result", product_id=product_id))


@bp.post("/run-pipeline-batch")
@login_required
def run_pipeline_batch():
    """FR-008: 複数商品のパイプライン一括実行"""
    product_ids = request.form.getlist("product_ids", type=int)
    site_key = request.form.get("site_key", "")
    if not product_ids:
        flash("商品が選択されていません。", "error")
        return redirect(url_for("main.listing_candidates"))

    from app.services.pipeline_service import PipelineService
    svc = PipelineService()
    batch = svc.run_batch(product_ids=product_ids, site_key=site_key)

    flash(
        f"一括実行完了（{batch.elapsed_sec}秒）: "
        f"成功{batch.success}件, 部分成功{batch.partial}件, "
        f"失敗{batch.failed}件, スキップ{batch.skipped}件",
        "success" if batch.failed == 0 else "warning",
    )
    return redirect(url_for("main.listing_candidates"))


@bp.get("/products/<int:product_id>/pipeline-result")
@login_required
def pipeline_result(product_id: int):
    """パイプライン実行結果の表示"""
    product = Product.query.get_or_404(product_id)
    import json as _json
    processed = []
    if product.processed_images:
        try:
            processed = _json.loads(product.processed_images)
        except (ValueError, TypeError):
            processed = []

    from app.services.template_service import generate_listing_text
    try:
        listing_text = generate_listing_text(product)
    except Exception:
        listing_text = ""

    return render_template(
        "pipeline_result.html",
        product=product,
        processed_images=processed,
        listing_text=listing_text,
    )


@bp.post("/products/<int:product_id>/upload-images")
@login_required
def upload_images(product_id: int):
    """手動画像アップロード（スクレイピングブロック時フォールバック）"""
    product = Product.query.get_or_404(product_id)
    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        flash("画像ファイルが選択されていません。", "error")
        return redirect(url_for("main.pipeline_result", product_id=product_id))

    from app.services.pipeline_service import PipelineService
    svc = PipelineService()
    saved = svc.save_uploaded_images(product_id, files)

    if saved:
        flash(f"{len(saved)}枚の画像をアップロードしました。", "success")
    else:
        flash("有効な画像ファイルがありませんでした。", "error")

    return redirect(url_for("main.pipeline_result", product_id=product_id))
