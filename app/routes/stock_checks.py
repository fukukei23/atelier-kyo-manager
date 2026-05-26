# ======================================================================
# F10: 在庫＆価格チェック
# ======================================================================
from __future__ import annotations

from datetime import datetime

import markupsafe
from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import login_required
from sqlalchemy.orm import joinedload

from app.core.timezone import _utcnow
from app.extensions import csrf, db
from app.models import Product
from app.models.stock_check import StockCheck
from app.services.price_scraper import PriceScraper
from app.utils.decorators import handle_api_error, handle_db_error
from app.utils.errors import safe_error_msg

from . import bp


@bp.get("/stock-check")
@login_required
def stock_check_list():
    """在庫＆価格チェック一覧"""
    checks = StockCheck.query.options(joinedload(StockCheck.product)).order_by(StockCheck.checked_at.desc()).paginate(page=request.args.get("page", 1, type=int), per_page=50, error_out=False).items
    return render_template("stock_check.html", checks=checks)


@bp.route("/stock-check/new", methods=["GET", "POST"])
@login_required
@handle_db_error()
def create_stock_check():
    """在庫チェック登録"""
    if request.method == "POST":
        product_id = int(request.form.get("product_id", 0))
        current_price = float(request.form.get("current_price", 0) or 0)  # noqa: E741
        if product_id <= 0:
            flash("商品を選択してください。", "error")
            return render_template("stock_check_form.html", preselected_id=None)
        if current_price < 0:
            flash("価格に負の値は入力できません。", "error")
            return render_template("stock_check_form.html", preselected_id=product_id)
        product = db.session.get(Product, product_id)
        if not product:
            flash("指定された商品が存在しません。", "error")
            return render_template("stock_check_form.html", preselected_id=None)
        sc = StockCheck(
            product_id=product_id,
            source_url=request.form.get("source_url", ""),
            current_price=current_price,
            in_stock="in_stock" in request.form,
            checked_at=_utcnow(),
            notes=request.form.get("notes", ""),
        )
        db.session.add(sc)
        db.session.commit()
        flash("在庫チェックを登録しました。", "success")
        return redirect(url_for("main.stock_check_list"))
    preselected_id = request.args.get("product_id", type=int)
    return render_template("stock_check_form.html", preselected_id=preselected_id)


@bp.post("/stock-check/<int:sid>/delete")
@login_required
@handle_db_error("main.stock_check_list")
def delete_stock_check(sid: int):
    """在庫チェック削除"""
    sc = StockCheck.query.get_or_404(sid)
    db.session.delete(sc)
    db.session.commit()
    flash("チェック記録を削除しました。", "success")
    return redirect(url_for("main.stock_check_list"))


@bp.post("/api/stock-check/<int:sid>/fetch")
@login_required
def api_fetch_stock(sid: int):
    """単一レコードの価格・在庫を自動取得"""
    sc = StockCheck.query.get_or_404(sid)
    if not sc.source_url:
        return jsonify({"success": False, "message": "source_url未設定"}), 400

    scraper = PriceScraper()
    try:
        result = scraper.fetch_with_retry(sc.source_url)
        if result["success"]:
            sc.previous_price = sc.current_price
            sc.previous_in_stock = sc.in_stock
            if result["price"] is not None:
                sc.current_price = result["price"]
                sc.price_changed = sc.previous_price is not None and sc.previous_price != result["price"]
            sc.in_stock = result["in_stock"]
            sc.stock_changed = sc.previous_in_stock is not None and sc.previous_in_stock != result["in_stock"]
            sc.checked_at = _utcnow()
            if result["title"]:
                sc.notes = f"タイトル: {result['title']}"
            db.session.commit()
            return jsonify({"success": True, "data": sc.to_dict(), "scraping": result})
        return jsonify({"success": False, "message": result["error"]}), 502
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": safe_error_msg(e, context="stock_fetch")}), 500
    finally:
        scraper.close()


# ---- F10-a: クイック価格入力API -----------------------------------------
@bp.post("/api/stock-check/quick-update")
@login_required
@handle_api_error()
def api_quick_update_price():
    """インライン価格更新"""
    data = request.get_json(silent=True) or {}
    sid = data.get("id")
    new_price = data.get("current_price")
    if sid is None or new_price is None:
        return jsonify({"success": False, "error": "id, current_price required"}), 400
    try:
        new_price = float(new_price)
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "current_price must be a number"}), 400
    if new_price < 0:
        return jsonify({"success": False, "error": "current_price must be >= 0"}), 400
    sc = db.session.get(StockCheck, sid)
    if not sc:
        return jsonify({"success": False, "error": "not found"}), 404
    if sc.current_price != new_price:
        sc.previous_price = sc.current_price
        sc.price_changed = sc.previous_price is not None and sc.previous_price != new_price
    sc.current_price = new_price
    sc.checked_at = _utcnow()
    db.session.commit()
    return jsonify({"success": True, "data": sc.to_dict()})


@bp.post("/api/stock-check/quick-add")
@login_required
@handle_api_error(status_code=500)
def api_quick_add_check():
    """クイック追加（商品ID・価格・在庫のみ）"""
    data = request.get_json(silent=True) or {}
    pid = data.get("product_id")
    price = data.get("current_price")
    if pid is None or price is None:
        return jsonify({"success": False, "error": "product_id, current_price required"}), 400
    try:
        price = float(price)
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "current_price must be a number"}), 400
    if price < 0:
        return jsonify({"success": False, "error": "current_price must be >= 0"}), 400
    product = db.session.get(Product, pid)
    if not product:
        return jsonify({"success": False, "error": f"product_id {pid} not found"}), 400
    source_url = str(data.get("source_url", ""))[:2000]
    sc = StockCheck(
        product_id=pid,
        source_url=markupsafe.escape(source_url),
        current_price=price,
        in_stock=bool(data.get("in_stock", True)),
        checked_at=_utcnow(),
        notes=str(markupsafe.escape(data.get("notes", ""))),
    )
    db.session.add(sc)
    db.session.commit()
    return jsonify({"success": True, "id": sc.id}), 201


@bp.get("/api/products-list")
@login_required
def api_products_list():
    """商品一覧（クイック入力用ドロップダウン）"""
    prods = Product.query.order_by(Product.brand, Product.name).all()
    return jsonify(
        [{"id": p.id, "name": f"{p.brand or ''} {p.name}", "supplier_url": p.supplier_url or ""} for p in prods]
    )


@bp.post("/api/stock-check/fetch-all")
@login_required
def api_fetch_all_stocks():
    """全レコードの価格・在庫を一括取得"""
    stocks = StockCheck.query.filter(StockCheck.source_url.isnot(None)).all()
    if not stocks:
        return jsonify({"success": True, "message": "対象なし", "total": 0})

    scraper = PriceScraper()
    ok_count = err_count = 0
    results = []
    try:
        for sc in stocks:
            r = scraper.fetch_cached(sc.source_url)
            if r["success"]:
                sc.previous_price = sc.current_price
                sc.previous_in_stock = sc.in_stock
                if r["price"] is not None:
                    sc.current_price = r["price"]
                    sc.price_changed = sc.previous_price is not None and sc.previous_price != r["price"]
                sc.in_stock = r["in_stock"]
                sc.stock_changed = sc.previous_in_stock is not None and sc.previous_in_stock != r["in_stock"]
                sc.checked_at = _utcnow()
                ok_count += 1
            else:
                err_count += 1
            results.append({"id": sc.id, "success": r["success"], "error": r.get("error")})
        db.session.commit()
        return jsonify({"success": True, "total": len(stocks), "ok": ok_count, "err": err_count, "results": results})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": safe_error_msg(e, context="fetch_all_stocks")}), 500
    finally:
        scraper.close()
