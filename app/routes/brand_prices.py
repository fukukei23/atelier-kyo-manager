from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta

from flask import flash, jsonify, make_response, redirect, render_template, request, session, url_for
from flask_login import login_required

from app.core.timezone import _utcnow
from app.services import brand_price_service
from app.services.brand_price_scraper import SUPPORTED_BRANDS, SUPPORTED_SITES, BrandPriceScraper
from app.services import price_comparison_service
from app.services.buyma_price_scraper import (
    BRAND_THRESHOLDS,
    BUYMA_UNAVAILABLE_BRANDS,
    CACHE_DAYS,
    DEFAULT_THRESHOLD,
    get_shared_searcher,
)
from app.utils.errors import safe_error_msg

from . import bp

logger = logging.getLogger(__name__)

_last_buyma_request: float = 0.0
_buyma_rate_lock = threading.Lock()
_BUIMA_MIN_INTERVAL = 2.0


@bp.get("/brand-prices")
@login_required
def brand_prices_dashboard():
    brand = request.args.get("brand", "Gucci")
    if brand not in SUPPORTED_BRANDS:
        brand = SUPPORTED_BRANDS[0]

    category = request.args.get("category", "bag")
    comparison = brand_price_service.get_price_comparison(brand)
    comparison = brand_price_service.add_profit_calculation(comparison, category=category)
    available_brands = brand_price_service.get_available_brands()
    last_scraped = brand_price_service.get_last_scraped_at(brand)

    return render_template(
        "brand_prices.html",
        brand=brand,
        category=category,
        comparison=comparison,
        supported_brands=SUPPORTED_BRANDS,
        available_brands=available_brands,
        last_scraped=last_scraped,
        supported_sites=SUPPORTED_SITES,
    )


@bp.post("/brand-prices/scrape")
@login_required
def brand_prices_scrape():
    brand = request.form.get("brand", "Gucci")
    if brand not in SUPPORTED_BRANDS:
        flash("対応していないブランドです", "error")
        return redirect(url_for("main.brand_prices_dashboard", brand=brand))

    # Celery タスクを投入（非同期）
    try:
        from app.tasks.scrape_tasks import scrape_brand_prices

        # 同時実行制限: セッション内の未完了タスクが5件以上なら拒否
        pending = session.get("pending_tasks", [])
        if len(pending) >= 5:
            flash("同時実行数の上限に達しています。少し待ってから再試行してください。", "warning")
            return redirect(url_for("main.brand_prices_dashboard", brand=brand))

        task = scrape_brand_prices.delay(brand)
        pending.append(task.id)
        session["pending_tasks"] = pending
        return jsonify({"task_id": task.id, "brand": brand, "status": "started"})
    except Exception as e:
        # Redis未起動等のフォールバック: 同期実行
        logger.warning("Celery unavailable, falling back to sync: %s", e)
        try:
            scraper = BrandPriceScraper(headless=True)
            results = scraper.scrape(brand=brand)
            if not results:
                flash(f"{brand} の価格データを取得できませんでした（ブロックされた可能性）", "warning")
            else:
                saved = brand_price_service.save_scraped_prices(results)
                flash(f"{brand} の価格データを {saved} 件取得しました", "success")
        except Exception as ex:
            logger.error("Scraping error: %s", ex, exc_info=True)
            flash(f"スクレイピングエラー: {safe_error_msg(ex, context='brand_scrape')}", "error")
        return redirect(url_for("main.brand_prices_dashboard", brand=brand))


@bp.post("/brand-prices/scrape-sale")
@login_required
def brand_prices_scrape_sale():
    """YOOX/SSENSE セール価格を取得。"""
    brand = request.form.get("brand", "Gucci")
    category = request.form.get("category", "bag")
    if brand not in SUPPORTED_BRANDS:
        flash("対応していないブランドです", "error")
        return redirect(url_for("main.brand_prices_dashboard", brand=brand, category=category))

    # Celery タスクを投入（非同期）
    try:
        from app.tasks.scrape_tasks import scrape_sale_prices

        # 同時実行制限
        pending = session.get("pending_tasks", [])
        if len(pending) >= 5:
            flash("同時実行数の上限に達しています。少し待ってから再試行してください。", "warning")
            return redirect(url_for("main.brand_prices_dashboard", brand=brand, category=category))

        task = scrape_sale_prices.delay(brand, category)
        pending.append(task.id)
        session["pending_tasks"] = pending
        return jsonify({"task_id": task.id, "brand": brand, "status": "started"})
    except Exception as e:
        # Redis未起動等のフォールバック: 同期実行
        logger.warning("Celery unavailable, falling back to sync: %s", e)
        try:
            from app.services.sale_scraper import SaleScraper
            scraper = SaleScraper()
            results = scraper.scrape(brand)
            if not results:
                flash(f"{brand} のセール価格を取得できませんでした", "warning")
            else:
                saved = brand_price_service.save_scraped_prices(results)
                flash(f"{brand} のセール価格を {saved} 件取得しました (YOOX/SSENSE)", "success")
        except Exception as ex:
            logger.error("Sale scrape error: %s", ex, exc_info=True)
            flash(f"セール価格取得エラー: {safe_error_msg(ex, context='sale_scrape')}", "error")
        return redirect(url_for("main.brand_prices_dashboard", brand=brand, category=category))


@bp.post("/brand-prices/update-selling-price")
@login_required
def brand_prices_update_selling_price():
    """BUYMA販売価格を手動更新。"""
    from app.extensions import db
    from app.models.brand_price import BrandPrice

    bp_id = request.form.get("brand_price_id", type=int)
    selling_price = request.form.get("selling_price", type=float)
    brand = request.form.get("brand", "Gucci")
    category = request.form.get("category", "bag")

    if bp_id and selling_price and selling_price > 0:
        record = db.session.get(BrandPrice, bp_id)
        if record:
            record.buyma_price = selling_price
            record.buyma_price_source = "manual"
            db.session.commit()
            flash(f"販売価格を ¥{selling_price:,.0f} に更新しました", "success")
    else:
        flash("販売価格の更新に失敗しました", "error")

    return redirect(url_for("main.brand_prices_dashboard", brand=brand, category=category))


@bp.post("/brand-prices/update-purchase-price")
@login_required
def brand_prices_update_purchase_price():
    """実際の仕入れ価格（セール価格等）を手動設定。"""
    from app.extensions import db
    from app.models.brand_price import BrandPrice

    bp_id = request.form.get("brand_price_id", type=int)
    actual_price = request.form.get("actual_purchase_price", type=float)
    brand = request.form.get("brand", "Gucci")
    category = request.form.get("category", "bag")

    if bp_id and actual_price and actual_price > 0:
        record = db.session.get(BrandPrice, bp_id)
        if record and record.brand == brand:
            record.actual_purchase_jpy = actual_price
            record.actual_purchase_source = "manual"
            db.session.commit()
            flash(f"仕入れ価格を ¥{actual_price:,.0f} に更新しました", "success")
    else:
        flash("仕入れ価格の更新に失敗しました", "error")

    return redirect(url_for("main.brand_prices_dashboard", brand=brand, category=category))


@bp.post("/brand-prices/add-to-pipeline")
@login_required
def brand_prices_add_to_pipeline():
    """利益が出る商品を出品候補（Product）に追加。"""
    from app.extensions import db
    from app.models.brand_price import BrandPrice
    from app.models.product import Product

    bp_id = request.form.get("brand_price_id", type=int)
    brand = request.form.get("brand", "Gucci")
    category = request.form.get("category", "bag")

    if not bp_id:
        flash("商品の指定がありません", "error")
        return redirect(url_for("main.brand_prices_dashboard", brand=brand, category=category))

    record = db.session.get(BrandPrice, bp_id)
    if not record:
        flash("対象レコードが見つかりません", "error")
        return redirect(url_for("main.brand_prices_dashboard", brand=brand, category=category))

    existing = db.session.scalar(
        db.select(Product).filter(
            Product.brand == record.brand,
            Product.name == record.product_name,
        ).limit(1)
    )
    if existing:
        flash(f"「{record.product_name}」は既に出品候補に存在します", "warning")
        return redirect(url_for("main.brand_prices_dashboard", brand=brand, category=category))

    product = Product(
        name=record.product_name,
        brand=record.brand,
        purchase_price=record.price_jpy,
        selling_price=record.buyma_price or round(record.price_jpy * 1.3),
        source_type="overseas",
        source_region="EU",
        item_category=category,
        pipeline_status="pending",
    )
    db.session.add(product)
    db.session.flush()
    record.product_id = product.id
    db.session.commit()

    flash(f"「{record.product_name}」を出品候補に追加しました", "success")
    return redirect(url_for("main.brand_prices_dashboard", brand=brand, category=category))


@bp.get("/api/brand-prices/comparison")
@login_required
def api_brand_prices_comparison():
    brand = request.args.get("brand", "Gucci")
    comparison = brand_price_service.get_price_comparison(brand)
    return jsonify({"brand": brand, "comparison": comparison})


@bp.get("/brand-prices/export")
@login_required
def brand_prices_export():
    import csv
    import io

    brand = request.args.get("brand", "Gucci")
    category = request.args.get("category", "bag")
    comparison = brand_price_service.get_price_comparison(brand)
    comparison = brand_price_service.add_profit_calculation(comparison, category=category)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "商品名", "仕入れ原価(JPY)", "送料", "関税", "総コスト",
        "販売価格", "利益額", "利益率", "判定",
        "BUYMA商品名", "マッチスコア", "状態",
    ])
    for item in comparison:
        if item.get("profit") is None:
            continue
        writer.writerow([
            item["product_name"],
            int(item.get("cheapest_jpy", 0)),
            int(item.get("shipping_cost", 0)),
            int(item.get("customs_duty", 0)),
            int(item.get("total_cost", 0)),
            int(item.get("buyma_suggested_price", 0)),
            int(item.get("profit", 0)),
            f"{item.get('profit_rate', 0) * 100:.1f}%",
            "OK" if item.get("is_profitable") else "NG",
            item.get("buyma_matched_name", ""),
            f"{item.get('buyma_match_score', 0) * 100:.0f}%" if item.get("buyma_match_score") else "",
            item.get("buyma_status", ""),
        ])

    output = make_response(buf.getvalue())
    output.headers["Content-Type"] = "text/csv; charset=utf-8-sig"
    output.headers["Content-Disposition"] = f"attachment; filename=brand_prices_{brand}_{category}.csv"
    return output


@bp.post("/brand-prices/search-buyma")
@login_required
def brand_prices_search_buyma():
    """BUYMA販売価格を自動検索してDBに反映。"""
    from app.extensions import db
    from app.models.brand_price import BrandPrice

    brand = request.form.get("brand", "Gucci")
    category = request.args.get("category", request.form.get("category", "bag"))
    if brand not in SUPPORTED_BRANDS:
        flash("対応していないブランドです", "error")
        return redirect(url_for("main.brand_prices_dashboard", brand=brand, category=category))

    if brand in BUYMA_UNAVAILABLE_BRANDS:
        flash(f"{brand} はBUYMAに出品がないため、販売価格の自動取得はできません", "warning")
        return redirect(url_for("main.brand_prices_dashboard", brand=brand, category=category))

    comparison = brand_price_service.get_price_comparison(brand)
    products = [
        {"product_name": item["product_name"], "brand": brand}
        for item in comparison
        if item.get("cheapest_jpy")
    ]

    if not products:
        flash(f"{brand} の商品データがありません。先に価格を取得してください。", "warning")
        return redirect(url_for("main.brand_prices_dashboard", brand=brand, category=category))

    return render_template(
        "buyma_search_progress.html",
        brand=brand,
        category=category,
        products=products[:20],
    )


@bp.post("/api/buyma-search")
@login_required
def api_buyma_search_single():
    """1商品ずつBUYMA検索。JSから呼ばれるAPI。キャッシュ・部分保存対応。"""
    from app.extensions import db
    from app.models.brand_price import BrandPrice

    data = request.get_json(force=True)
    product_name = data.get("product_name", "")
    brand = data.get("brand", "")

    # AA: サーバー側レート制限（2秒間隔）
    global _last_buyma_request
    with _buyma_rate_lock:
        now = time.time()
        if now - _last_buyma_request < _BUIMA_MIN_INTERVAL:
            return jsonify({"status": "rate_limited", "message": "too fast"}), 429
        _last_buyma_request = now

    if not product_name or not brand:
        return jsonify({"status": "error", "message": "missing params"}), 400

    if brand not in SUPPORTED_BRANDS:
        return jsonify({"status": "error", "message": "unsupported brand"}), 400

    if brand in BUYMA_UNAVAILABLE_BRANDS:
        return jsonify({"status": "skipped", "reason": "unavailable_brand"})

    rows = db.session.execute(
        db.select(BrandPrice).filter(
            BrandPrice.brand == brand,
            BrandPrice.product_name == product_name,
        )
    ).scalars().all()

    # #4 キャッシュ判定
    if rows:
        newest = max(rows, key=lambda r: r.buyma_searched_at or datetime.min)
        if newest.buyma_searched_at and newest.buyma_status == "matched":
            age = _utcnow() - newest.buyma_searched_at
            if age.days < CACHE_DAYS:
                return jsonify({
                    "status": "cached",
                    "buyma_price": newest.buyma_price,
                    "buyma_name": newest.buyma_matched_name,
                    "match_score": newest.buyma_match_score,
                })

    # #9 ブランド別閾値
    threshold = BRAND_THRESHOLDS.get(brand, DEFAULT_THRESHOLD)

    searcher = get_shared_searcher()
    try:
        match = searcher.search_single(product_name, brand, threshold=threshold)
    except Exception as e:
        logger.error(f"BUYMA search error for {product_name}: {e}")
        err_type = "unknown"
        err_msg = str(e).lower()
        if "timeout" in err_msg or "timed out" in err_msg:
            err_type = "timeout"
        elif "403" in err_msg or "blocked" in err_msg:
            err_type = "blocked"
        elif "429" in err_msg or "rate" in err_msg:
            err_type = "rate_limited"
        elif "navigation" in err_msg or "goto" in err_msg:
            err_type = "navigation"
        for row in rows:
            row.buyma_status = "error"
            row.buyma_searched_at = _utcnow()
        db.session.commit()
        return jsonify({"status": "error", "message": str(e), "error_type": err_type})

    now = _utcnow()

    if not match:
        for row in rows:
            row.buyma_status = "no_match"
            row.buyma_searched_at = now
        db.session.commit()
        return jsonify({"status": "no_match"})

    # マッチ結果を全カラムに保存（#5 + #8）+ 履歴保存（#A）
    from app.models.buyma_price_history import BuymaPriceHistory

    for row in rows:
        if row.buyma_price and row.buyma_price != match["buyma_price"]:
            db.session.add(BuymaPriceHistory(
                brand_price_id=row.id,
                buyma_price=row.buyma_price,
                buyma_matched_name=row.buyma_matched_name,
                match_score=row.buyma_match_score,
                source=row.buyma_price_source or "previous",
            ))
        row.buyma_price = match["buyma_price"]
        row.buyma_price_source = f"buyma_search({match['match_score']:.2f})"
        row.buyma_matched_name = match["buyma_name"]
        row.buyma_match_score = match["match_score"]
        row.buyma_searched_at = now
        row.buyma_status = "matched"
    db.session.commit()

    return jsonify({
        "status": "matched",
        "buyma_price": match["buyma_price"],
        "buyma_name": match["buyma_name"],
        "match_score": match["match_score"],
        "match_count": match["match_count"],
    })


@bp.get("/api/tasks/<task_id>/status")
@login_required
def task_status(task_id: str):
    """Celery タスクの実行状態を返す."""
    from flask import current_app, session

    # 所有者検証: セッションに記録されたtask_idのみアクセス可能
    user_tasks = session.get("pending_tasks", [])
    if task_id not in user_tasks:
        return jsonify({"state": "FORBIDDEN", "result": None}), 403

    celery_app = getattr(current_app, "celery", None)
    if celery_app is None:
        return jsonify({"state": "UNAVAILABLE", "result": None})

    result = celery_app.AsyncResult(task_id)
    response = {"state": result.state}

    if result.state == "SUCCESS":
        response["result"] = result.result
        # 完了したタスクはセッションから除去
        session["pending_tasks"] = [t for t in user_tasks if t != task_id]
    elif result.state == "FAILURE":
        response["result"] = "エラーが発生しました"
        session["pending_tasks"] = [t for t in user_tasks if t != task_id]
    else:
        response["result"] = None

    return jsonify(response)


# ---------------------------------------------------------------------------
# Multi-source price comparison (product-centric)
# ---------------------------------------------------------------------------


@bp.get("/price-comparison")
@login_required
def price_comparison_search():
    """商品中心のマルチサイト価格比較ページ。"""
    brand = request.args.get("brand", "Gucci")
    query = request.args.get("q", "").strip()
    category = request.args.get("category", "wallet")

    comparison = None
    profit_data = None

    if query:
        comparison = price_comparison_service.compare_product(
            brand=brand,
            product_query=query,
            category=category,
        )
        profit_data = price_comparison_service.calculate_comparison_profit(
            comparison, category=category,
        )

    return render_template(
        "price_comparison.html",
        brand=brand,
        query=query,
        category=category,
        supported_brands=SUPPORTED_BRANDS,
        comparison=comparison,
        profit_data=profit_data,
    )


@bp.get("/api/price-comparison/search")
@login_required
def api_price_comparison_search():
    """AJAX商品検索API（autocomplete用）。"""
    brand = request.args.get("brand", "")
    query = request.args.get("q", "").strip()
    if not brand or not query or len(query) < 2:
        return jsonify({"results": []})
    results = price_comparison_service.search_products(brand, query)
    return jsonify({"results": results})


@bp.post("/price-comparison/scrape-and-compare")
@login_required
def price_comparison_scrape_and_compare():
    """全ソースをスクレイプ → 比較ページにリダイレクト。"""
    brand = request.form.get("brand", "Gucci")
    category = request.form.get("category", "wallet")
    query = request.form.get("q", "")

    # Celery非同期スクレイプがあれば利用、なければ同期
    try:
        from flask import current_app
        celery_app = getattr(current_app, "celery", None)
        if celery_app:
            task = celery_app.send_task(
                "app.tasks.scrape_brand_prices",
                args=[brand],
            )
            flash(f"スクレイプ開始: {brand}（タスクID: {task.id}）", "info")
            time.sleep(3)
        else:
            scraper = BrandPriceScraper()
            items = scraper.scrape(brand)
            brand_price_service.save_scraped_prices(items)
            flash(f"スクレイプ完了: {brand}（{len(items)}件）", "success")
    except Exception as e:
        flash(f"スクレイプエラー: {safe_error_msg(e)}", "error")

    return redirect(url_for(
        "main.price_comparison_search",
        brand=brand, q=query, category=category,
    ))
