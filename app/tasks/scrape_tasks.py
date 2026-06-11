"""スクレイピング Celery タスク.

重いスクレイピング処理をバックグラウンドで実行する。
Flask app context 内で動作するよう celery_app の ContextTask に依存。
"""

from __future__ import annotations

import logging

from app.core.celery_app import celery

logger = logging.getLogger(__name__)


@celery.task(bind=True, name="scrape_brand_prices")
def scrape_brand_prices(self, brand: str, sites: list[str] | None = None) -> dict:
    """公式ブランドサイトの価格をスクレイピングして保存する."""
    try:
        from app.services.brand_price_scraper import BrandPriceScraper
        from app.services import brand_price_service

        scraper = BrandPriceScraper(headless=True)
        results = scraper.scrape(brand=brand, sites=sites)

        if not results:
            return {"brand": brand, "saved": 0, "error": "no_results"}

        saved = brand_price_service.save_scraped_prices(results)
        logger.info("scrape_brand_prices: %s → %d件保存", brand, saved)
        return {"brand": brand, "saved": saved}

    except Exception as exc:
        logger.error("scrape_brand_prices error: %s", exc, exc_info=True)
        return {"brand": brand, "saved": 0, "error": "scrape_error"}


@celery.task(bind=True, name="scrape_sale_prices")
def scrape_sale_prices(self, brand: str, category: str = "bag") -> dict:
    """YOOX/SSENSE セール価格をスクレイピングして保存する."""
    try:
        from app.services.sale_scraper import SaleScraper
        from app.services import brand_price_service

        scraper = SaleScraper()
        results = scraper.scrape(brand, category=category)

        if not results:
            return {"brand": brand, "saved": 0, "error": "no_results"}

        saved = brand_price_service.save_scraped_prices(results)
        logger.info("scrape_sale_prices: %s → %d件保存", brand, saved)
        return {"brand": brand, "saved": saved}

    except Exception as exc:
        logger.error("scrape_sale_prices error: %s", exc, exc_info=True)
        return {"brand": brand, "saved": 0, "error": "scrape_error"}


@celery.task(bind=True, name="run_ssense_buyma_pipeline")
def run_ssense_buyma_pipeline(self, brand: str, category: str = "sunglasses") -> dict:
    """SSENSE→BUYMA自動価格検証パイプライン。"""
    try:
        from app.services.ssense_buyma_pipeline import SsenseBuymaPipeline

        pipeline = SsenseBuymaPipeline()
        summary = pipeline.run(brand, category)
        logger.info(
            "pipeline: %s/%s → scraped=%d, buyma=%d, profitable=%d",
            brand,
            category,
            summary.total_scraped,
            summary.buyma_matched,
            summary.verified_profitable,
        )
        return {
            "brand": brand,
            "total_scraped": summary.total_scraped,
            "buyma_matched": summary.buyma_matched,
            "verified_profitable": summary.verified_profitable,
            "errors": summary.errors,
        }

    except Exception as exc:
        logger.error("pipeline error: %s", exc, exc_info=True)
        return {"brand": brand, "error": str(exc)}
