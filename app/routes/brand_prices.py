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

    comparison = brand_price_service.get_price_comparison(brand)
    available_brands = brand_price_service.get_available_brands()
    last_scraped = brand_price_service.get_last_scraped_at(brand)

    return render_template(
        "brand_prices.html",
        brand=brand,
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


@bp.get("/api/brand-prices/comparison")
@login_required
def api_brand_prices_comparison():
    brand = request.args.get("brand", "Gucci")
    comparison = brand_price_service.get_price_comparison(brand)
    return jsonify({"brand": brand, "comparison": comparison})
