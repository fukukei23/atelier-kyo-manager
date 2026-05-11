# ======================================================================
# F01: 商品管理 + F02: BUYMA拡張 + CSV入出力
# ======================================================================
from __future__ import annotations

import json as _json

from flask import current_app, flash, make_response, redirect, render_template, request, url_for
from flask_login import login_required

from app.extensions import db
from app.forms import ProductForm
from app.models import Product
from app.services.notification_service import NotificationService
from app.services.pipeline_service import PipelineService
from app.services.product_csv_service import (
    export_candidates_csv,
    export_products_csv,
    import_products_from_csv,
)
from app.services.template_service import generate_buyma_csv as _gen_csv
from app.services.template_service import generate_listing_text
from app.utils.decorators import handle_db_error

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
@handle_db_error()
def manage_products():
    """
    商品登録/更新 + 一覧
    - POST: 新規登録（最小実装。編集は /products/<id>/edit）
    - GET : フォーム + 一覧
    """
    form = ProductForm()

    if form.validate_on_submit():
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
@handle_db_error()
def edit_product(product_id: int):
    """商品編集"""
    product = Product.query.get_or_404(product_id)
    form = ProductForm(obj=product)

    if form.validate_on_submit():
        form.populate_obj(product)
        product.stock_status = bool(form.stock_status.data)
        product.target_profit_rate = (form.target_profit_rate.data or 10.0) / 100.0
        product.auto_classify_tier()
        db.session.commit()
        flash("商品を更新しました。", "success")
        return redirect(url_for("main.manage_products"))

    products = Product.query.order_by(Product.id.desc()).all()
    return render_template("products/manage.html", form=form, products=products)


@bp.post("/products/<int:id>/delete")
@login_required
@handle_db_error("main.manage_products")
def delete_product(id: int):
    """商品削除"""
    product = Product.query.get_or_404(id)
    db.session.delete(product)
    db.session.commit()
    flash("商品を削除しました。", "success")
    return redirect(url_for("main.manage_products"))


# ---- CSV 入出力 --------------------------------------------------------
@bp.post("/import_csv")
@login_required
@handle_db_error()
def import_csv():
    """CSV インポート"""
    file = request.files.get("file")
    if not file:
        flash("CSV ファイルが選択されていません。", "error")
        return redirect(url_for("main.manage_products"))

    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        flash("CSV ファイルのみアップロード可能です。", "error")
        return redirect(url_for("main.manage_products"))

    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > 10 * 1024 * 1024:
        flash("ファイルサイズは10MB以下にしてください。", "error")
        return redirect(url_for("main.manage_products"))

    data = file.read().decode("utf-8-sig")
    count = import_products_from_csv(data, db.session)
    flash(f"CSV を {count} 件取り込みました。", "success")
    return redirect(url_for("main.manage_products"))


@bp.get("/export_csv")
@login_required
def export_csv():
    """CSV エクスポート"""
    products: list[Product] = Product.query.order_by(Product.id.asc()).all()
    csv_text = export_products_csv(products)
    resp = make_response(csv_text)
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
    csv_text = export_candidates_csv(candidates)
    resp = make_response(csv_text)
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=listing_candidates.csv"
    return resp


# ---- F02: BUYMA拡張（出品テンプレート連携）-------------------------------
@bp.get("/products/<int:product_id>/generate-listing")
@login_required
def generate_listing(product_id: int):
    """商品の出品文をプレビュー生成"""
    product = Product.query.get_or_404(product_id)
    try:
        text = generate_listing_text(product)
    except Exception:
        text = "テンプレートが登録されていません。先にテンプレートを作成してください。"
    return render_template("listing_preview.html", product=product, listing_text=text)


@bp.get("/generate-buyma-csv")
@login_required
def generate_buyma_csv():
    """BUYMA用CSV一括生成"""
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
    svc = PipelineService()
    site_key = request.form.get("site_key", "")
    result = svc.run(product_id=product_id, site_key=site_key)

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
    processed = []
    if product.processed_images:
        try:
            processed = _json.loads(product.processed_images)
        except (ValueError, TypeError):
            processed = []

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
    Product.query.get_or_404(product_id)
    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        flash("画像ファイルが選択されていません。", "error")
        return redirect(url_for("main.pipeline_result", product_id=product_id))

    svc = PipelineService()
    saved = svc.save_uploaded_images(product_id, files)

    if saved:
        flash(f"{len(saved)}枚の画像をアップロードしました。", "success")
    else:
        flash("有効な画像ファイルがありませんでした。", "error")

    return redirect(url_for("main.pipeline_result", product_id=product_id))
