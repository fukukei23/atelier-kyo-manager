# ======================================================================
# F03: 出品テンプレート + F04: 禁制品API + F07: 品出し進捗
# F09: ブランド分析 + F10: 在庫チェック + F11: 人気度 + F12: 地域レコメンド
# ======================================================================
from __future__ import annotations

from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import login_required
from sqlalchemy import func
from sqlalchemy.orm import joinedload
from types import SimpleNamespace
from datetime import datetime

from app.extensions import db, csrf
from app.models import Product

from . import bp


# ---- F03: 出品テンプレート管理 -------------------------------------------
@bp.get("/templates")
@login_required
def listing_templates():
    """テンプレート一覧"""
    from app.models.listing_template import ListingTemplate
    templates = ListingTemplate.query.order_by(
        ListingTemplate.is_default.desc(), ListingTemplate.id.asc()
    ).all()
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
    """テンプレート削除"""
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


# ---- F07: 品出し進捗トラッカー -------------------------------------------
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
            listings_count = int(request.form.get("listings_count", 0) or 0)
            target_daily = int(request.form.get("target_daily", 20) or 20)
            target_monthly = int(request.form.get("target_monthly", 600) or 600)
            cumulative_monthly = int(request.form.get("cumulative_monthly", 0) or 0)
            if any(v < 0 for v in [listings_count, target_daily, target_monthly, cumulative_monthly]):
                flash("出品数・目標値に負の値は入力できません。", "error")
                return render_template("listing_progress_form.html", record=None)
            lp = ListingProgress(
                record_date=record_date,
                listings_count=listings_count,
                target_daily=target_daily,
                target_monthly=target_monthly,
                cumulative_monthly=cumulative_monthly,
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


# ---- F09: ブランド階層別利益率ダッシュボード -------------------------------
@bp.get("/brand-analytics")
@login_required
def brand_analytics():
    """ブランド階層別利益率ダッシュボード"""
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

    products = Product.query.order_by(Product.brand_tier, Product.id.desc()).all()
    return render_template("brand_analytics.html", tier_data=tier_data, products=products)


# ---- F10: 在庫＆価格チェック ---------------------------------------------
@bp.get("/stock-check")
@login_required
def stock_check_list():
    """在庫＆価格チェック一覧"""
    from app.models.stock_check import StockCheck
    checks = StockCheck.query.options(
        joinedload(StockCheck.product)
    ).order_by(StockCheck.checked_at.desc()).all()
    return render_template("stock_check.html", checks=checks)


@bp.route("/stock-check/new", methods=["GET", "POST"])
@login_required
def create_stock_check():
    """在庫チェック登録"""
    from app.models.stock_check import StockCheck
    from datetime import datetime as _dt
    if request.method == "POST":
        try:
            product_id = int(request.form.get("product_id", 0))
            current_price = float(request.form.get("current_price", 0) or 0)
            if product_id <= 0:
                flash("商品を選択してください。", "error")
                return render_template("stock_check_form.html", preselected_id=None)
            if current_price < 0:
                flash("価格に負の値は入力できません。", "error")
                return render_template("stock_check_form.html", preselected_id=product_id)
            product = Product.query.get(product_id)
            if not product:
                flash("指定された商品が存在しません。", "error")
                return render_template("stock_check_form.html", preselected_id=None)
            sc = StockCheck(
                product_id=product_id,
                source_url=request.form.get("source_url", ""),
                current_price=current_price,
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
    preselected_id = request.args.get("product_id", type=int)
    return render_template("stock_check_form.html", preselected_id=preselected_id)


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


# ---- F10-a: クイック価格入力API -----------------------------------------
@bp.post("/api/stock-check/quick-update")
@login_required
@csrf.exempt
def api_quick_update_price():
    """インライン価格更新"""
    from app.models.stock_check import StockCheck
    data = request.get_json(silent=True) or {}
    sid = data.get("id")
    new_price = data.get("current_price")
    if sid is None or new_price is None:
        return jsonify({"success": False, "error": "id, current_price required"}), 400
    # 価格バリデーション: 数値型で0以上であること
    try:
        new_price = float(new_price)
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "current_price must be a number"}), 400
    if new_price < 0:
        return jsonify({"success": False, "error": "current_price must be >= 0"}), 400
    sc = StockCheck.query.get(sid)
    if not sc:
        return jsonify({"success": False, "error": "not found"}), 404
    try:
        if sc.current_price != new_price:
            sc.previous_price = sc.current_price
            sc.price_changed = sc.previous_price is not None and sc.previous_price != new_price
        sc.current_price = new_price
        sc.checked_at = datetime.utcnow()
        db.session.commit()
        return jsonify({"success": True, "data": sc.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@bp.post("/api/stock-check/quick-add")
@login_required
@csrf.exempt
def api_quick_add_check():
    """クイック追加（商品ID・価格・在庫のみ）"""
    from app.models.stock_check import StockCheck
    data = request.get_json(silent=True) or {}
    pid = data.get("product_id")
    price = data.get("current_price")
    if pid is None or price is None:
        return jsonify({"success": False, "error": "product_id, current_price required"}), 400
    # バリデーション: 価格は0以上の数値
    try:
        price = float(price)
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "current_price must be a number"}), 400
    if price < 0:
        return jsonify({"success": False, "error": "current_price must be >= 0"}), 400
    # バリデーション: product_id存在チェック
    product = Product.query.get(pid)
    if not product:
        return jsonify({"success": False, "error": f"product_id {pid} not found"}), 400
    # source_urlサニタイズ（HTMLタグ除去）
    source_url = str(data.get("source_url", ""))[:2000]
    source_url = source_url.replace("<", "&lt;").replace(">", "&gt;")
    try:
        sc = StockCheck(
            product_id=pid,
            source_url=source_url,
            current_price=price,
            in_stock=bool(data.get("in_stock", True)),
            checked_at=datetime.utcnow(),
            notes=str(data.get("notes", "")).replace("<", "&lt;").replace(">", "&gt;"),
        )
        db.session.add(sc)
        db.session.commit()
        return jsonify({"success": True, "id": sc.id}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@bp.get("/api/products-list")
@login_required
def api_products_list():
    """商品一覧（クイック入力用ドロップダウン）"""
    prods = Product.query.order_by(Product.brand, Product.name).all()
    return jsonify([{"id": p.id, "name": f"{p.brand or ''} {p.name}",
                     "supplier_url": p.supplier_url or ""} for p in prods])


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


# ---- F11: 人気度トラッキング ---------------------------------------------
@bp.get("/popularity")
@login_required
def popularity_list():
    """人気度トラッキング一覧"""
    from app.models.popularity_tracker import PopularityTracker
    trackers = PopularityTracker.query.options(
        joinedload(PopularityTracker.product)
    ).order_by(PopularityTracker.popularity_score.desc()).all()
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
            views = int(request.form.get("views", 0) or 0)
            favorites = int(request.form.get("favorites", 0) or 0)
            inquiries = int(request.form.get("inquiries", 0) or 0)
            sold_count = int(request.form.get("sold_count", 0) or 0)
            if any(v < 0 for v in [views, favorites, inquiries, sold_count]):
                flash("閲覧数・お気に入り・問い合わせ・販売数に負の値は入力できません。", "error")
                return render_template("popularity_form.html")
            pt = PopularityTracker(
                product_id=int(request.form.get("product_id", 0)),
                views=views,
                favorites=favorites,
                inquiries=inquiries,
                sold_count=sold_count,
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


# ---- F12: 買付先地域最適化レコメンド --------------------------------------
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
            avg_profit_rate = float(request.form.get("avg_profit_rate", 0) or 0)
            avg_shipping_days = int(request.form.get("avg_shipping_days", 0) or 0)
            risk_score = float(request.form.get("risk_score", 50) or 50)
            reliability_score = float(request.form.get("reliability_score", 50) or 50)
            if avg_shipping_days < 0:
                flash("配送日数に負の値は入力できません。", "error")
                return render_template("region_form.html")
            rr = RegionRecommendation(
                region=request.form.get("region", ""),
                region_name=request.form.get("region_name", ""),
                avg_profit_rate=avg_profit_rate,
                avg_shipping_days=avg_shipping_days,
                avg_customs_rate=float(request.form.get("avg_customs_rate", 0) or 0),
                risk_score=risk_score,
                reliability_score=reliability_score,
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
