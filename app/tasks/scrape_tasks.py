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
        results = scraper.scrape(brand)

        if not results:
            return {"brand": brand, "saved": 0, "error": "no_results"}

        saved = brand_price_service.save_scraped_prices(results)
        logger.info("scrape_sale_prices: %s → %d件保存", brand, saved)
        return {"brand": brand, "saved": saved}

    except Exception as exc:
        logger.error("scrape_sale_prices error: %s", exc, exc_info=True)
        return {"brand": brand, "saved": 0, "error": "scrape_error"}
