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

from .extensions import db
from .models import Product
from .forms import ProductForm, AutoResearchForm

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
            )
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
def product_list():
    """登録データ一覧（シンプル）"""
    products = Product.query.order_by(Product.id.desc()).all()
    # シンプル版テンプレは templates/list.html を想定
    return render_template("list.html", products=products)

@bp.route("/products/<int:product_id>/edit", methods=["GET", "POST"])
def edit_product(product_id: int):
    """商品編集"""
    product = Product.query.get_or_404(product_id)
    form = ProductForm(obj=product)

    if form.validate_on_submit():
        try:
            form.populate_obj(product)
            # Boolean は populate_obj で正しく入るが念のため明示
            product.stock_status = bool(form.stock_status.data)
            db.session.commit()
            flash("商品を更新しました。", "success")
            return redirect(url_for("main.manage_products"))
        except Exception as e:
            db.session.rollback()
            flash(f"更新に失敗しました: {e}", "error")

    return render_template("products/manage.html", form=form, products=[])

@bp.post("/products/<int:id>/delete")
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
            )
            db.session.add(p)
            count += 1

        db.session.commit()
        flash(f"CSV を {count} 件取り込みました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"CSV 取り込みに失敗しました: {e}", "error")

    return redirect(url_for("main.manage_products"))

@bp.get("/export_csv")
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
            "created_at": (p.created_at.isoformat(sep=" ", timespec="seconds") if p.created_at else ""),
            "updated_at": (p.updated_at.isoformat(sep=" ", timespec="seconds") if p.updated_at else ""),
        })

    resp = make_response(buf.getvalue())
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=products_export.csv"
    return resp

# ---- 自動リサーチ / 個別リサーチ ----------------------------------------
@bp.route("/auto-research", methods=["GET", "POST"])
def auto_research():
    """
    自動リサーチ（フォーム未使用でも CSRF のため form を渡す）
    """
    form = AutoResearchForm()
    return render_template("auto_research.html", form=form)

@bp.get("/image-crawler")
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
