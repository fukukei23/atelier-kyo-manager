# ======================================================================
# F03: 出品テンプレート + F04: 禁制品API + F07: 品出し進捗
# F09: ブランド分析 + F10: 在庫チェック + F11: 人気度 + F12: 地域レコメンド
# ======================================================================
from __future__ import annotations

import csv
from io import StringIO

from flask import flash, jsonify, redirect, render_template, request, url_for, Response
from flask_login import login_required
from sqlalchemy import func
from sqlalchemy.orm import joinedload
from types import SimpleNamespace
from datetime import datetime, timedelta

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


# ---- FR-010基盤: FAQテンプレート管理 ------------------------------------
@bp.get("/faq-templates")
@login_required
def faq_templates():
    """FAQテンプレート一覧"""
    from app.models.faq_template import FaqTemplate
    faqs = FaqTemplate.query.order_by(FaqTemplate.category, FaqTemplate.id.asc()).all()
    return render_template("faq_templates.html", faqs=faqs)


@bp.route("/faq-templates/new", methods=["GET", "POST"])
@login_required
def create_faq_template():
    """FAQテンプレート新規作成"""
    from app.models.faq_template import FaqTemplate
    if request.method == "POST":
        category = request.form.get("category", "general")
        question_pattern = request.form.get("question_pattern", "").strip()
        answer_template = request.form.get("answer_template", "").strip()
        if not question_pattern or not answer_template:
            flash("キーワードパターンと返答テンプレートは必須です。", "error")
            return render_template("faq_template_form.html", faq=None)
        try:
            faq = FaqTemplate(
                category=category,
                question_pattern=question_pattern,
                answer_template=answer_template,
            )
            db.session.add(faq)
            db.session.commit()
            flash("FAQテンプレートを登録しました。", "success")
            return redirect(url_for("main.faq_templates"))
        except Exception as e:
            db.session.rollback()
            flash(f"登録に失敗しました: {e}", "error")
    return render_template("faq_template_form.html", faq=None)


@bp.post("/faq-templates/<int:fid>/delete")
@login_required
def delete_faq_template(fid: int):
    """FAQテンプレート削除"""
    from app.models.faq_template import FaqTemplate
    faq = FaqTemplate.query.get_or_404(fid)
    try:
        db.session.delete(faq)
        db.session.commit()
        flash("FAQテンプレートを削除しました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"削除に失敗しました: {e}", "error")
    return redirect(url_for("main.faq_templates"))


@bp.post("/api/faq-match")
@login_required
def api_faq_match():
    """問い合わせ内容からFAQテンプレートを検索"""
    from app.models.faq_template import FaqTemplate
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"success": False, "error": "text required"}), 400
    faqs = FaqTemplate.query.filter_by(is_active=True).all()
    matches = []
    for faq in faqs:
        if faq.match(text):
            matches.append({
                "id": faq.id,
                "category": faq.category,
                "answer_template": faq.answer_template,
            })
    return jsonify({"success": True, "matches": matches})


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


# ---- FR-018: Analyticsダッシュボード -------------------------------------
@bp.get("/dashboard")
@login_required
def dashboard():
    """Analyticsダッシュボード: KPI + パイプライン + 在庫 + ブランド階層"""
    # フィルター
    query = Product.query
    period = request.args.get("period", "")
    if period == "7d":
        query = query.filter(Product.created_at >= datetime.utcnow() - timedelta(days=7))
    elif period == "30d":
        query = query.filter(Product.created_at >= datetime.utcnow() - timedelta(days=30))
    brand = request.args.get("brand", "")
    if brand:
        query = query.filter(Product.brand == brand)

    # CSVエクスポート
    if request.args.get("export") == "csv":
        products = query.order_by(Product.created_at.desc()).all()
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["商品名", "ブランド", "仕入価格", "販売価格", "利益率", "パイプライン状態", "在庫", "出品状態"])
        for p in products:
            margin = ""
            if p.purchase_price and p.selling_price and p.purchase_price > 0:
                margin = f"{((p.selling_price - p.purchase_price) / p.purchase_price * 100):.1f}%"
            writer.writerow([
                p.name, p.brand, p.purchase_price, p.selling_price,
                margin, p.pipeline_status or "",
                "あり" if p.stock_status else "なし", p.listing_status or "",
            ])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Response(
            output.getvalue(),
            mimetype="text/csv; charset=utf-8-sig",
            headers={"Content-Disposition": f"attachment; filename=dashboard_{ts}.csv"},
        )

    # KPI
    total_products = query.count()
    products = query.all()
    rates = [p.profit_rate() for p in products]
    profits = [p.calculate_profit() for p in products]
    kpi = {
        "total_products": total_products,
        "avg_profit_rate": sum(rates) / len(rates) if rates else 0,
        "total_profit": sum(profits),
    }

    # パイプライン進捗
    pipeline_rows = db.session.query(
        Product.pipeline_status, func.count(Product.id)
    ).group_by(Product.pipeline_status).all()
    pipeline_summary = {"pending": 0, "running": 0, "success": 0, "partial": 0, "failed": 0}
    for status, cnt in pipeline_rows:
        key = status or "pending"
        if key in pipeline_summary:
            pipeline_summary[key] = cnt

    # 在庫ステータス
    in_stock = Product.query.filter(Product.stock_status == True).count()
    stock_summary = {"in_stock": in_stock, "out_of_stock": total_products - in_stock}

    # 出品ステータス
    listing_rows = db.session.query(
        Product.listing_status, func.count(Product.id)
    ).group_by(Product.listing_status).all()
    listing_summary = {"draft": 0, "listed": 0, "sold": 0, "archived": 0}
    for status, cnt in listing_rows:
        key = status or "draft"
        if key in listing_summary:
            listing_summary[key] = cnt

    # ブランド階層別利益率
    tier_labels = []
    tier_rates = []
    for tier in ("high", "medium", "low"):
        tier_prods = [p for p in products if (p.brand_tier or "low") == tier]
        if tier_prods:
            avg = sum(p.profit_rate() for p in tier_prods) / len(tier_prods)
        else:
            avg = 0
        label_map = {"high": "ハイブランド", "medium": "ミドル", "low": "ロー"}
        tier_labels.append(label_map[tier])
        tier_rates.append(round(avg, 1))

    # ブランド一覧（フィルター用）
    brands = [r[0] for r in db.session.query(Product.brand).distinct().all() if r[0]]

    return render_template(
        "dashboard.html",
        kpi=kpi,
        pipeline_summary=pipeline_summary,
        stock_summary=stock_summary,
        listing_summary=listing_summary,
        tier_labels=tier_labels,
        tier_rates=tier_rates,
        period=period,
        brand=brand,
        brands=brands,
    )


# ---- FR-012基盤: 発送通知管理 -------------------------------------------
@bp.get("/shipment-notifications")
@login_required
def shipment_notifications():
    """発送通知一覧"""
    from app.models.shipment_notification import ShipmentNotification
    notifications = ShipmentNotification.query.order_by(
        ShipmentNotification.created_at.desc()
    ).all()
    return render_template("shipment_notifications.html", notifications=notifications)


@bp.route("/shipment-notifications/new", methods=["GET", "POST"])
@login_required
def create_shipment_notification():
    """発送通知手動登録"""
    from app.models.order import Order
    from app.models.shipment_notification import ShipmentNotification
    if request.method == "POST":
        order_id = request.form.get("order_id", type=int)
        tracking_number = request.form.get("tracking_number", "").strip()
        warehouse = request.form.get("warehouse", "").strip()
        carrier = request.form.get("carrier", "").strip()
        if not order_id:
            flash("注文を選択してください。", "error")
            orders = Order.query.order_by(Order.order_date.desc()).all()
            return render_template("shipment_notification_form.html", orders=orders)
        try:
            sn = ShipmentNotification(
                order_id=order_id,
                tracking_number=tracking_number or None,
                warehouse=warehouse or None,
                carrier=carrier or None,
            )
            db.session.add(sn)
            db.session.commit()
            flash("発送通知を登録しました。", "success")
            return redirect(url_for("main.shipment_notifications"))
        except Exception as e:
            db.session.rollback()
            flash(f"登録に失敗しました: {e}", "error")
    orders = Order.query.order_by(Order.order_date.desc()).all()
    return render_template("shipment_notification_form.html", orders=orders)


@bp.post("/shipment-notifications/<int:sid>/notify")
@login_required
def mark_shipment_notified(sid: int):
    """発送通知を通知済みに更新"""
    from app.models.shipment_notification import ShipmentNotification
    sn = ShipmentNotification.query.get_or_404(sid)
    if not sn.can_notify():
        flash("この通知は既に処理済みです。", "error")
        return redirect(url_for("main.shipment_notifications"))
    try:
        sn.mark_notified()
        db.session.commit()
        flash("発送通知を「通知済み」に更新しました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"更新に失敗しました: {e}", "error")
    return redirect(url_for("main.shipment_notifications"))


@bp.post("/shipment-notifications/<int:sid>/delete")
@login_required
def delete_shipment_notification(sid: int):
    """発送通知削除"""
    from app.models.shipment_notification import ShipmentNotification
    sn = ShipmentNotification.query.get_or_404(sid)
    try:
        db.session.delete(sn)
        db.session.commit()
        flash("発送通知を削除しました。", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"削除に失敗しました: {e}", "error")
    return redirect(url_for("main.shipment_notifications"))


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
        result = scraper.fetch_with_retry(sc.source_url)
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
            r = scraper.fetch_cached(sc.source_url)
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


# ---- FR-009: AI自動発注管理 -----------------------------------------------
@bp.get("/auto-orders")
@login_required
def auto_orders():
    """自動発注ステータス一覧"""
    from app.models.order import Order
    orders = Order.query.order_by(Order.order_date.desc()).all()
    return render_template("auto_orders.html", orders=orders)


@bp.post("/auto-orders/<int:oid>/start")
@login_required
def start_auto_order(oid: int):
    """自動発注開始"""
    from app.models.order import Order
    from app.services.auto_order_service import AutoOrderService, OrderStatus
    from app.services.notification_service import NotificationService

    order = Order.query.get_or_404(oid)
    order.status = OrderStatus.PENDING
    db.session.commit()

    svc = AutoOrderService(order, notification_service=NotificationService())
    ok = svc.advance_to(OrderStatus.SOURCING, note="自動発注開始")
    if ok:
        db.session.commit()
        flash(f"注文 #{order.order_number} の自動発注を開始しました。", "success")
    else:
        flash("ステータス遷移に失敗しました。", "error")
    return redirect(url_for("main.auto_orders"))


@bp.post("/auto-orders/<int:oid>/step")
@login_required
def execute_auto_order_step(oid: int):
    """自動発注ステップ実行"""
    from app.models.order import Order
    from app.services.auto_order_service import AutoOrderService
    from app.services.notification_service import NotificationService

    order = Order.query.get_or_404(oid)
    svc = AutoOrderService(order, notification_service=NotificationService())
    ok, msg = svc.execute_step()
    if ok:
        db.session.commit()
        flash(f"ステップ実行成功: {msg}", "success")
    else:
        flash(f"ステップ実行失敗: {msg}", "error")
    return redirect(url_for("main.auto_orders"))


@bp.post("/auto-orders/<int:oid>/error")
@login_required
def report_auto_order_error(oid: int):
    """エラー報告"""
    from app.models.order import Order
    from app.services.auto_order_service import AutoOrderService, OrderStatus
    from app.services.notification_service import NotificationService

    order = Order.query.get_or_404(oid)
    note = request.form.get("error_note", "手動エラー報告")
    svc = AutoOrderService(order, notification_service=NotificationService())
    svc.advance_to(OrderStatus.ERROR, note=note)
    db.session.commit()
    flash(f"注文 #{order.order_number} をエラー状態にしました。", "error")
    return redirect(url_for("main.auto_orders"))
