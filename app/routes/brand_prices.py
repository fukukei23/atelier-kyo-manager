from __future__ import annotations

import logging

from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import login_required

from app.services import brand_price_service
from app.services.brand_price_scraper import SUPPORTED_BRANDS, SUPPORTED_SITES, BrandPriceScraper

from . import bp

logger = logging.getLogger(__name__)


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

    try:
        scraper = BrandPriceScraper(headless=True)
        results = scraper.scrape(brand=brand)

        if not results:
            flash(f"{brand} の価格データを取得できませんでした（ブロックされた可能性）", "warning")
            return redirect(url_for("main.brand_prices_dashboard", brand=brand))

        saved = brand_price_service.save_scraped_prices(results)
        flash(f"{brand} の価格データを {saved} 件取得しました", "success")
    except Exception as e:
        logger.error(f"Scraping error: {e}")
        flash(f"スクレイピングエラー: {e}", "error")

    return redirect(url_for("main.brand_prices_dashboard", brand=brand))


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
