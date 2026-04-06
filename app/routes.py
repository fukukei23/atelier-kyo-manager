# ======================================================================
# プロジェクト : atelier-kyo-manager
# ファイル名   : app/routes.py   ← 全置換
# 目的         : 画面ルーティング / CSV入出力 / 互換リダイレクト / API(倉庫一覧)
# 対応テンプレ :
#   - templates/index.html
#   - templates/products/manage.html
#   - templates/list.html
#   - templates/auto_research.html
#   - templates/image_crawler.html
# 依存         : forms.ProductForm, forms.AutoResearchForm, models.Product, extensions.db
# メモ(運用)   :
#   - 旧リンク main.form_view / main.list_view は互換リダイレクトを用意
#   - Playwright 未導入でも /api/warehouses は 503 を返すだけでアプリは起動継続
#   - CSV 取り込みはヘッダ名がモデルのカラム名に一致している行のみ反映
# 日付         : 2025-08-23 (JST)
# ======================================================================

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List

from flask import (
    Blueprint,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
    flash,
    make_response,
)

from flask_login import login_required

from .extensions import db, csrf
from app.models import Product  # Product is exported from app.models package
from app.forms import ProductForm, AutoResearchForm

# ---- Blueprint --------------------------------------------------------
bp = Blueprint("main", __name__)

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
    # シンプル版テンプレは templates/list.html を想定
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
            # Boolean は populate_obj で正しく入るが念のため明示
            product.stock_status = bool(form.stock_status.data)
            # BUYMA拡張: 目標利益率を%→小数に変換 + 自動階層判定
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
    """
    CSV インポート
    - 受領: form-data の file=.csv
    - 受理ヘッダ例:
      name,brand,purchase_price,selling_price,transaction_fee,shipping_cost,customs_duty,procurement_fee,supplier_url,image_url,stock_status
    """
    file = request.files.get("file")
    if not file:
        flash("CSV ファイルが選択されていません。", "error")
        return redirect(url_for("main.manage_products"))

    try:
        data = file.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(data))

        count = 0
        for row in reader:
            # 必須チェック
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
                # --- BUYMA拡張 (F02) ---
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
    """
    CSV エクスポート
    - 出力ヘッダはインポートと互換
    """
    products: List[Product] = Product.query.order_by(Product.id.asc()).all()
    headers = [
        "id",
        "name",
        "brand",
        "purchase_price",
        "selling_price",
        "transaction_fee",
        "shipping_cost",
        "customs_duty",
        "procurement_fee",
        "supplier_url",
        "image_url",
        "stock_status",
        "brand_tier",
        "source_type",
        "source_region",
        "color",
        "size",
        "material",
        "description",
        "retail_price",
        "target_profit_rate",
        "listing_status",
        "created_at",
        "updated_at",
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

# ---- 自動リサーチ / 個別リサーチ ----------------------------------------
@bp.route("/auto-research", methods=["GET", "POST"])
@login_required
def auto_research():
    """
    自動リサーチ（フォーム未使用でも CSRF のため form を渡す）
    """
    form = AutoResearchForm()
    return render_template("auto_research.html", form=form)

@bp.get("/image-crawler")
@login_required
def image_crawler():
    """個別リサーチ画面"""
    return render_template("image_crawler.html")

# ---- API: 倉庫一覧（Buyandship） ---------------------------------------
try:
    from .utils.shipping_agent import ShippingAgent  # 依存が無い環境でもファイルは存在
    _shipping_agent_import_ok = True
except Exception:
    ShippingAgent = None  # type: ignore
    _shipping_agent_import_ok = False

@bp.get("/api/warehouses")
@login_required
@csrf.exempt
def api_warehouses():
    """
    GET /api/warehouses?country=HK
    - 成功: JSON (list of warehouses)
    - エラー:
        400: country 未指定
        503: ShippingAgent 未利用（Playwright 未導入など）
        500: その他例外
    """
    country = (request.args.get("country") or "").strip().upper()
    if not country:
        return jsonify({"error": "country is required (e.g. HK, TW)"}), 400

    if not _shipping_agent_import_ok or ShippingAgent is None:
        return jsonify({"error": "ShippingAgent is unavailable on this environment."}), 503

    try:
        agent = ShippingAgent()  # 認証は .env の BUYANDSHIP_EMAIL/PASSWORD を参照
        warehouses = agent.get_warehouses_by_country(country)
        return jsonify({"country": country, "warehouses": warehouses})
    except Exception as e:
        # 依存（playwright など）未導入や認証失敗時にここへ
        return jsonify({"error": f"failed to fetch warehouses: {e}"}), 500

# ---- F03: 出品テンプレート管理 -------------------------------------------
@bp.get("/templates")
@login_required
def listing_templates():
    """テンプレート一覧"""
    from app.models.listing_template import ListingTemplate
    templates = ListingTemplate.query.order_by(ListingTemplate.is_default.desc(), ListingTemplate.id.asc()).all()
    return render_template("listing_templates.html", templates=templates)

@bp.route("/templates/new", methods=["GET", "POST"])
@bp.route("/templates/<int:tid>/edit", methods=["GET", "POST"])
@login_required
def edit_listing_template(tid: int | None = None):
    """テンプレート新規/編集"""
    from app.models.listing_template import ListingTemplate
    tpl = ListingTemplate.query.get(tid) if tid else None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        template_text = request.form.get("template_text", "")
        category = request.form.get("category", "general")
        is_default = "is_default" in request.form

        if not name or not template_text:
            flash("テンプレート名と本文は必須です。", "error")
            from types import SimpleNamespace
            return render_template("edit_listing_template.html",
                                   tpl=tpl or SimpleNamespace(name=name,
                                                              template_text=template_text,
                                                              category=category,
                                                              is_default=is_default))

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
def delete_listing_template(tid: int):
    from app.models.listing_template import ListingTemplate
    tpl = ListingTemplate.query.get_or_404(tid)
    try:
        db.session.delete(tpl)
        db.session.commit()
        flash("テンプレートを削除しました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"削除に失敗しました: {e}", "error")
    return redirect(url_for("main.listing_templates"))

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

# ---- F04: 禁制品買付先チェックAPI -----------------------------------------
@bp.get("/api/check-source")
@login_required
def api_check_source():
    """買付先URLが禁止対象かチェック"""
    from app.models.prohibited_source import ProhibitedSource
    url = (request.args.get("url") or "").strip()
    source_type = (request.args.get("source_type") or "domestic").strip()
    prohibited, reason = ProhibitedSource.is_prohibited(url, source_type)
    return jsonify({"prohibited": prohibited, "reason": reason, "url": url})

@bp.get("/api/prohibited-sources")
@login_required
def api_list_prohibited_sources():
    from app.models.prohibited_source import ProhibitedSource
    items = ProhibitedSource.query.order_by(ProhibitedSource.id.asc()).all()
    return jsonify([{"id": s.id, "domain": s.domain, "reason": s.reason,
                     "severity": s.severity, "source_type": s.source_type} for s in items])

@bp.post("/api/prohibited-sources")
@login_required
def api_add_prohibited_source():
    from app.models.prohibited_source import ProhibitedSource
    data = request.get_json(force=True)
    domain = (data.get("domain") or "").strip()
    if not domain:
        return jsonify({"error": "domain is required"}), 400
    existing = ProhibitedSource.query.filter_by(domain=domain).first()
    if existing:
        return jsonify({"error": "already exists"}), 409
    src = ProhibitedSource(
        domain=domain,
        reason=data.get("reason", ""),
        severity=data.get("severity", "blocked"),
        source_type=data.get("source_type", "domestic"),
    )
    db.session.add(src)
    db.session.commit()
    return jsonify({"id": src.id, "domain": src.domain}), 201

@bp.delete("/api/prohibited-sources/<int:sid>")
@login_required
def api_delete_prohibited_source(sid: int):
    from app.models.prohibited_source import ProhibitedSource
    src = ProhibitedSource.query.get_or_404(sid)
    db.session.delete(src)
    db.session.commit()
    return jsonify({"deleted": True})

# =====================================================================
# F05: 18日ルールダッシュボード
# =====================================================================
@bp.get("/orders")
@login_required
def order_list():
    """注文一覧（18日ルール対応）"""
    from app.models.order import Order
    from sqlalchemy import desc
    orders = Order.query.order_by(desc(Order.order_date)).all()
    return render_template("orders.html", orders=orders)

@bp.route("/orders/new", methods=["GET", "POST"])
@login_required
def create_order():
    """注文新規登録"""
    from app.models.order import Order, PAYMENT_METHOD_EXTENSION_DAYS
    if request.method == "POST":
        try:
            from datetime import datetime as _dt
            order_date_str = request.form.get("order_date", "")
            order_date = _dt.strptime(order_date_str, "%Y-%m-%d") if order_date_str else _dt.utcnow()

            order = Order(
                order_number=request.form.get("order_number", ""),
                product_name=request.form.get("product_name", ""),
                customer_name=request.form.get("customer_name", ""),
                order_date=order_date,
                selling_price=float(request.form.get("selling_price", 0) or 0),
                purchase_cost=float(request.form.get("purchase_cost", 0) or 0),
                customs_duty=float(request.form.get("customs_duty", 0) or 0),
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
        except Exception as e:
            db.session.rollback()
            flash(f"登録に失敗しました: {e}", "error")

    return render_template("order_form.html", order=None,
                           payment_methods=list(PAYMENT_METHOD_EXTENSION_DAYS.keys()))

@bp.route("/orders/<int:oid>/edit", methods=["GET", "POST"])
@login_required
def edit_order(oid: int):
    """注文編集"""
    from app.models.order import Order, PAYMENT_METHOD_EXTENSION_DAYS
    order = Order.query.get_or_404(oid)
    if request.method == "POST":
        try:
            from datetime import datetime as _dt
            order_date_str = request.form.get("order_date", "")
            if order_date_str:
                order.order_date = _dt.strptime(order_date_str, "%Y-%m-%d")
            order.order_number = request.form.get("order_number", order.order_number)
            order.product_name = request.form.get("product_name", order.product_name)
            order.customer_name = request.form.get("customer_name", order.customer_name)
            order.selling_price = float(request.form.get("selling_price", 0) or 0)
            order.purchase_cost = float(request.form.get("purchase_cost", 0) or 0)
            order.customs_duty = float(request.form.get("customs_duty", 0) or 0)
            order.payment_method = request.form.get("payment_method", order.payment_method)
            order.source_type = request.form.get("source_type", "domestic")
            order.status = request.form.get("status", order.status)
            order.notes = request.form.get("notes", order.notes)
            order.calc_deadlines()
            order.calc_profit()
            db.session.commit()
            flash("注文を更新しました。", "success")
            return redirect(url_for("main.order_list"))
        except Exception as e:
            db.session.rollback()
            flash(f"更新に失敗しました: {e}", "error")

    return render_template("order_form.html", order=order,
                           payment_methods=list(PAYMENT_METHOD_EXTENSION_DAYS.keys()))

@bp.post("/orders/<int:oid>/delete")
@login_required
def delete_order(oid: int):
    """注文削除"""
    from app.models.order import Order
    order = Order.query.get_or_404(oid)
    try:
        db.session.delete(order)
        db.session.commit()
        flash("注文を削除しました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"削除に失敗しました: {e}", "error")
    return redirect(url_for("main.order_list"))

@bp.get("/api/orders/dashboard")
@login_required
def api_order_dashboard():
    """18日ルールダッシュボードAPI"""
    from app.models.order import Order
    from datetime import datetime as _dt, timedelta
    now = _dt.utcnow()
    orders = Order.query.filter(Order.status.in_(["pending", "shipped"])).all()
    result = []
    for o in orders:
        remaining = o.remaining_days()
        result.append({
            "id": o.id,
            "order_number": o.order_number,
            "product_name": o.product_name,
            "order_date": o.order_date.isoformat() if o.order_date else None,
            "deadline_18": o.deadline_18.isoformat() if o.deadline_18 else None,
            "extension_deadline": o.extension_deadline.isoformat() if o.extension_deadline else None,
            "remaining_days": remaining,
            "color": o.deadline_color(),
            "profit": o.profit,
            "status": o.status,
            "payment_method": o.payment_method,
        })
    return jsonify(result)

# =====================================================================
# F06: パートナー管理
# =====================================================================
@bp.get("/partners")
@login_required
def partner_list():
    """パートナー一覧"""
    from app.models.partner import Partner
    partners = Partner.query.order_by(Partner.priority_level.asc(), Partner.name.asc()).all()
    return render_template("partners.html", partners=partners)

@bp.route("/partners/new", methods=["GET", "POST"])
@login_required
def create_partner():
    """パートナー新規登録"""
    from app.models.partner import Partner
    if request.method == "POST":
        try:
            p = Partner(
                name=request.form.get("name", ""),
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
        except Exception as e:
            db.session.rollback()
            flash(f"登録に失敗しました: {e}", "error")
    return render_template("partner_form.html", partner=None)

@bp.route("/partners/<int:pid>/edit", methods=["GET", "POST"])
@login_required
def edit_partner(pid: int):
    """パートナー編集"""
    from app.models.partner import Partner
    p = Partner.query.get_or_404(pid)
    if request.method == "POST":
        try:
            p.name = request.form.get("name", p.name)
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
        except Exception as e:
            db.session.rollback()
            flash(f"更新に失敗しました: {e}", "error")
    return render_template("partner_form.html", partner=p)

@bp.post("/partners/<int:pid>/delete")
@login_required
def delete_partner(pid: int):
    """パートナー削除"""
    from app.models.partner import Partner
    p = Partner.query.get_or_404(pid)
    try:
        db.session.delete(p)
        db.session.commit()
        flash("パートナーを削除しました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"削除に失敗しました: {e}", "error")
    return redirect(url_for("main.partner_list"))

# =====================================================================
# F07: 品出し進捗トラッカー
# =====================================================================
@bp.get("/listing-progress")
@login_required
def listing_progress_view():
    """品出し進捗一覧"""
    from app.models.listing_progress import ListingProgress
    from datetime import date as _date
    today = _date.today()
    records = ListingProgress.query.filter(
        ListingProgress.record_date >= today.replace(day=1)
    ).order_by(ListingProgress.record_date.desc()).all()

    # 月間サマリー
    summary = ListingProgress.get_monthly_summary(today.year, today.month)
    return render_template("listing_progress.html", records=records, summary=summary)

@bp.route("/listing-progress/new", methods=["GET", "POST"])
@login_required
def create_listing_progress():
    """品出し進捗登録"""
    from app.models.listing_progress import ListingProgress
    if request.method == "POST":
        try:
            from datetime import date as _date
            record_date_str = request.form.get("record_date", "")
            record_date = _date.fromisoformat(record_date_str) if record_date_str else _date.today()

            lp = ListingProgress(
                record_date=record_date,
                listings_count=int(request.form.get("listings_count", 0) or 0),
                target_daily=int(request.form.get("target_daily", 20) or 20),
                target_monthly=int(request.form.get("target_monthly", 600) or 600),
                cumulative_monthly=int(request.form.get("cumulative_monthly", 0) or 0),
                notes=request.form.get("notes", ""),
            )
            db.session.add(lp)
            db.session.commit()
            flash("進捗を登録しました。", "success")
            return redirect(url_for("main.listing_progress_view"))
        except Exception as e:
            db.session.rollback()
            flash(f"登録に失敗しました: {e}", "error")
    return render_template("listing_progress_form.html", record=None)

@bp.post("/listing-progress/<int:rid>/delete")
@login_required
def delete_listing_progress(rid: int):
    """進捗記録削除"""
    from app.models.listing_progress import ListingProgress
    r = ListingProgress.query.get_or_404(rid)
    try:
        db.session.delete(r)
        db.session.commit()
        flash("進捗記録を削除しました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"削除に失敗しました: {e}", "error")
    return redirect(url_for("main.listing_progress_view"))

# =====================================================================
# F08: キャッシュフロー予測
# =====================================================================
@bp.get("/cashflow")
@login_required
def cashflow_dashboard():
    """キャッシュフロー予測ダッシュボード"""
    from app.models.order import Order, EXPECTED_PAYMENT_DAYS
    from datetime import datetime as _dt, timedelta
    from sqlalchemy import func

    now = _dt.utcnow()
    # 未完了注文
    pending_orders = Order.query.filter(Order.status.in_(["pending", "shipped"])).all()

    # 予測データ生成（30日先）
    forecast_days = int(request.args.get("days", 30))
    daily_forecast: list[dict] = []
    running_balance = 0.0

    # 過去30日の完了済み利益合計
    past_profit = db.session.query(func.coalesce(func.sum(Order.profit), 0)).filter(
        Order.status == "completed",
        Order.completed_date >= now - timedelta(days=30),
    ).scalar() or 0

    # 未完了注文の入金予定を日付別に集計
    for i in range(forecast_days):
        target_date = (now + timedelta(days=i)).date()
        inflow = 0.0
        for o in pending_orders:
            if o.expected_payment_date and o.expected_payment_date.date() == target_date:
                inflow += float(o.profit or 0)
        running_balance += inflow
        daily_forecast.append({
            "date": target_date.isoformat(),
            "inflow": inflow,
            "balance": running_balance,
        })

    return render_template("cashflow.html",
                           daily_forecast=daily_forecast,
                           pending_count=len(pending_orders),
                           past_profit=past_profit,
                           forecast_days=forecast_days)

# =====================================================================
# F09: ブランド階層別利益率ダッシュボード
# =====================================================================
@bp.get("/brand-analytics")
@login_required
def brand_analytics():
    """ブランド階層別利益率ダッシュボード"""
    from sqlalchemy import func
    # 階層別集計
    tier_stats = db.session.query(
        Product.brand_tier,
        func.count(Product.id).label("count"),
        func.avg(Product.target_profit_rate).label("avg_target_rate"),
        func.avg(Product.selling_price).label("avg_selling"),
        func.avg(Product.purchase_price).label("avg_cost"),
    ).group_by(Product.brand_tier).all()

    tier_data = {}
    for ts in tier_stats:
        tier = ts.brand_tier or "low"
        selling = float(ts.avg_selling or 0)
        cost = float(ts.avg_cost or 0)
        profit = selling - cost
        rate = (profit / selling * 100) if selling > 0 else 0
        tier_data[tier] = {
            "count": ts.count,
            "avg_selling": round(selling, 0),
            "avg_cost": round(cost, 0),
            "avg_profit": round(profit, 0),
            "avg_profit_rate": round(rate, 1),
            "avg_target_rate": round(float(ts.avg_target_rate or 0) * 100, 1),
        }

    # 全商品
    products = Product.query.order_by(Product.brand_tier, Product.id.desc()).all()
    return render_template("brand_analytics.html", tier_data=tier_data, products=products)

# =====================================================================
# F10: 在庫＆価格チェック
# =====================================================================
@bp.get("/stock-check")
@login_required
def stock_check_list():
    """在庫＆価格チェック一覧"""
    from app.models.stock_check import StockCheck
    checks = StockCheck.query.order_by(StockCheck.checked_at.desc()).all()
    return render_template("stock_check.html", checks=checks)

@bp.route("/stock-check/new", methods=["GET", "POST"])
@login_required
def create_stock_check():
    """在庫チェック登録"""
    from app.models.stock_check import StockCheck
    from datetime import datetime as _dt
    if request.method == "POST":
        try:
            sc = StockCheck(
                product_id=int(request.form.get("product_id", 0)),
                source_url=request.form.get("source_url", ""),
                current_price=float(request.form.get("current_price", 0) or 0),
                in_stock="in_stock" in request.form,
                checked_at=_dt.utcnow(),
                notes=request.form.get("notes", ""),
            )
            db.session.add(sc)
            db.session.commit()
            flash("在庫チェックを登録しました。", "success")
            return redirect(url_for("main.stock_check_list"))
        except Exception as e:
            db.session.rollback()
            flash(f"登録に失敗しました: {e}", "error")
    return render_template("stock_check_form.html")

@bp.post("/stock-check/<int:sid>/delete")
@login_required
def delete_stock_check(sid: int):
    """在庫チェック削除"""
    from app.models.stock_check import StockCheck
    sc = StockCheck.query.get_or_404(sid)
    try:
        db.session.delete(sc)
        db.session.commit()
        flash("チェック記録を削除しました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"削除に失敗しました: {e}", "error")
    return redirect(url_for("main.stock_check_list"))


# ---- F10 API: スクレイピング自動取得 ------------------------------------
@bp.post("/api/stock-check/<int:sid>/fetch")
@login_required
def api_fetch_stock(sid: int):
    """単一レコードの価格・在庫を自動取得"""
    from datetime import datetime as _dt
    from app.models.stock_check import StockCheck
    from app.services.price_scraper import PriceScraper

    sc = StockCheck.query.get_or_404(sid)
    if not sc.source_url:
        return jsonify({"success": False, "message": "source_url未設定"}), 400

    scraper = PriceScraper()
    try:
        result = scraper.fetch(sc.source_url)
        if result["success"]:
            sc.previous_price = sc.current_price
            sc.previous_in_stock = sc.in_stock
            if result["price"] is not None:
                sc.current_price = result["price"]
                sc.price_changed = sc.previous_price is not None and sc.previous_price != result["price"]
            sc.in_stock = result["in_stock"]
            sc.stock_changed = sc.previous_in_stock is not None and sc.previous_in_stock != result["in_stock"]
            sc.checked_at = _dt.utcnow()
            if result["title"]:
                sc.notes = f"タイトル: {result['title']}"
            db.session.commit()
            return jsonify({"success": True, "data": sc.to_dict(), "scraping": result})
        return jsonify({"success": False, "message": result["error"]}), 502
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        scraper.close()


@bp.post("/api/stock-check/fetch-all")
@login_required
def api_fetch_all_stocks():
    """全レコードの価格・在庫を一括取得"""
    from datetime import datetime as _dt
    from app.models.stock_check import StockCheck
    from app.services.price_scraper import PriceScraper

    stocks = StockCheck.query.filter(StockCheck.source_url.isnot(None)).all()
    if not stocks:
        return jsonify({"success": True, "message": "対象なし", "total": 0})

    scraper = PriceScraper()
    ok_count = err_count = 0
    results = []
    try:
        for sc in stocks:
            r = scraper.fetch(sc.source_url)
            if r["success"]:
                sc.previous_price = sc.current_price
                sc.previous_in_stock = sc.in_stock
                if r["price"] is not None:
                    sc.current_price = r["price"]
                    sc.price_changed = sc.previous_price is not None and sc.previous_price != r["price"]
                sc.in_stock = r["in_stock"]
                sc.stock_changed = sc.previous_in_stock is not None and sc.previous_in_stock != r["in_stock"]
                sc.checked_at = _dt.utcnow()
                ok_count += 1
            else:
                err_count += 1
            results.append({"id": sc.id, "success": r["success"], "error": r.get("error")})
        db.session.commit()
        return jsonify({"success": True, "total": len(stocks), "ok": ok_count, "err": err_count, "results": results})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        scraper.close()

# =====================================================================
# F11: 人気度トラッキング
# =====================================================================
@bp.get("/popularity")
@login_required
def popularity_list():
    """人気度トラッキング一覧"""
    from app.models.popularity_tracker import PopularityTracker
    from sqlalchemy import func
    trackers = PopularityTracker.query.order_by(PopularityTracker.popularity_score.desc()).all()
    avg_score = db.session.query(func.avg(PopularityTracker.popularity_score)).scalar() or 0
    top_count = sum(1 for t in trackers if (t.popularity_score or 0) >= 100)
    low_count = sum(1 for t in trackers if (t.popularity_score or 0) < 20)
    summary = {
        "total_products": len(trackers),
        "avg_score": round(float(avg_score), 1),
        "top_count": top_count,
        "low_count": low_count,
    }
    return render_template("popularity.html", trackers=trackers, summary=summary)

@bp.route("/popularity/new", methods=["GET", "POST"])
@login_required
def create_popularity():
    """人気度記録登録"""
    from app.models.popularity_tracker import PopularityTracker
    from datetime import date as _date
    if request.method == "POST":
        try:
            pt = PopularityTracker(
                product_id=int(request.form.get("product_id", 0)),
                views=int(request.form.get("views", 0) or 0),
                favorites=int(request.form.get("favorites", 0) or 0),
                inquiries=int(request.form.get("inquiries", 0) or 0),
                sold_count=int(request.form.get("sold_count", 0) or 0),
                tracking_date=_date.today(),
            )
            pt.popularity_score = pt.calc_score()
            db.session.add(pt)
            db.session.commit()
            flash("人気度を記録しました。", "success")
            return redirect(url_for("main.popularity_list"))
        except Exception as e:
            db.session.rollback()
            flash(f"登録に失敗しました: {e}", "error")
    return render_template("popularity_form.html")

@bp.post("/popularity/<int:tid>/delete")
@login_required
def delete_popularity(tid: int):
    """人気度記録削除"""
    from app.models.popularity_tracker import PopularityTracker
    pt = PopularityTracker.query.get_or_404(tid)
    try:
        db.session.delete(pt)
        db.session.commit()
        flash("記録を削除しました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"削除に失敗しました: {e}", "error")
    return redirect(url_for("main.popularity_list"))

# =====================================================================
# F12: 買付先地域最適化レコメンド
# =====================================================================
@bp.get("/regions")
@login_required
def region_list():
    """買付先地域レコメンド一覧"""
    from app.models.region_recommendation import RegionRecommendation
    regions = RegionRecommendation.query.order_by(RegionRecommendation.recommendation_score.desc()).all()
    return render_template("region_recommendations.html", regions=regions)

@bp.route("/regions/new", methods=["GET", "POST"])
@login_required
def create_region():
    """地域登録"""
    from app.models.region_recommendation import RegionRecommendation
    from datetime import datetime as _dt
    if request.method == "POST":
        try:
            rr = RegionRecommendation(
                region=request.form.get("region", ""),
                region_name=request.form.get("region_name", ""),
                avg_profit_rate=float(request.form.get("avg_profit_rate", 0) or 0),
                avg_shipping_days=int(request.form.get("avg_shipping_days", 0) or 0),
                avg_customs_rate=float(request.form.get("avg_customs_rate", 0) or 0),
                risk_score=float(request.form.get("risk_score", 50) or 50),
                reliability_score=float(request.form.get("reliability_score", 50) or 50),
                last_updated=_dt.utcnow(),
            )
            rr.recommendation_score = rr.calc_recommendation() or 0
            db.session.add(rr)
            db.session.commit()
            flash("地域を登録しました。", "success")
            return redirect(url_for("main.region_list"))
        except Exception as e:
            db.session.rollback()
            flash(f"登録に失敗しました: {e}", "error")
    return render_template("region_form.html")

@bp.post("/regions/<int:rid>/delete")
@login_required
def delete_region(rid: int):
    """地域削除"""
    from app.models.region_recommendation import RegionRecommendation
    rr = RegionRecommendation.query.get_or_404(rid)
    try:
        db.session.delete(rr)
        db.session.commit()
        flash("地域を削除しました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"削除に失敗しました: {e}", "error")
    return redirect(url_for("main.region_list"))

# =====================================================================
# F13: リピーター管理
# =====================================================================
@bp.get("/customers")
@login_required
def customer_list():
    """リピーター一覧"""
    from app.models.repeat_customer import RepeatCustomer
    customers = RepeatCustomer.query.order_by(RepeatCustomer.total_orders.desc()).all()
    return render_template("repeat_customers.html", customers=customers)

@bp.route("/customers/new", methods=["GET", "POST"])
@login_required
def create_customer():
    """顧客新規登録"""
    from app.models.repeat_customer import RepeatCustomer
    from datetime import datetime as _dt
    if request.method == "POST":
        try:
            c = RepeatCustomer(
                customer_name=request.form.get("customer_name", ""),
                email=request.form.get("email", ""),
                phone=request.form.get("phone", ""),
                total_orders=int(request.form.get("total_orders", 0) or 0),
                total_spent=float(request.form.get("total_spent", 0) or 0),
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
        except Exception as e:
            db.session.rollback()
            flash(f"登録に失敗しました: {e}", "error")
    return render_template("repeat_customer_form.html", customer=None)

@bp.route("/customers/<int:cid>/edit", methods=["GET", "POST"])
@login_required
def edit_customer(cid: int):
    """顧客編集"""
    from app.models.repeat_customer import RepeatCustomer
    from datetime import datetime as _dt
    c = RepeatCustomer.query.get_or_404(cid)
    if request.method == "POST":
        try:
            c.customer_name = request.form.get("customer_name", c.customer_name)
            c.email = request.form.get("email", "")
            c.phone = request.form.get("phone", "")
            c.total_orders = int(request.form.get("total_orders", 0) or 0)
            c.total_spent = float(request.form.get("total_spent", 0) or 0)
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
        except Exception as e:
            db.session.rollback()
            flash(f"更新に失敗しました: {e}", "error")
    return render_template("repeat_customer_form.html", customer=c)

@bp.post("/customers/<int:cid>/delete")
@login_required
def delete_customer(cid: int):
    """顧客削除"""
    from app.models.repeat_customer import RepeatCustomer
    c = RepeatCustomer.query.get_or_404(cid)
    try:
        db.session.delete(c)
        db.session.commit()
        flash("顧客を削除しました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"削除に失敗しました: {e}", "error")
    return redirect(url_for("main.customer_list"))


# =====================================================================
# 互換リダイレクト（ナビURL → 実際のルート）
# =====================================================================
@bp.get("/dashboard")
def dashboard_redirect():
    """旧: /dashboard → /cashflow"""
    return redirect(url_for("main.cashflow_dashboard"))

@bp.get("/listing-templates")
def listing_templates_redirect():
    """旧: /listing-templates → /templates"""
    return redirect(url_for("main.listing_templates"))

@bp.get("/region-recommendations")
def region_recommendations_redirect():
    """旧: /region-recommendations → /regions"""
    return redirect(url_for("main.region_list"))

@bp.get("/repeat-customers")
def repeat_customers_redirect():
    """旧: /repeat-customers → /customers"""
    return redirect(url_for("main.customer_list"))

@bp.get("/auto_research")
def auto_research_redirect():
    """旧: /auto_research → /auto-research"""
    return redirect(url_for("main.auto_research"))
