from __future__ import annotations

import click


def register_commands(app):
    @app.cli.command("scrape-brand-prices")
    @click.argument("brand")
    def scrape_brand_prices(brand):
        from app.services import brand_price_service
        from app.services.brand_price_scraper import SUPPORTED_BRANDS, BrandPriceScraper

        if brand not in SUPPORTED_BRANDS:
            click.echo(f"Unsupported brand: {brand}")
            return

        click.echo(f"Scraping {brand}...")
        scraper = BrandPriceScraper(headless=True)
        results = scraper.scrape(brand=brand)

        if not results:
            click.echo(f"No results for {brand}")
            return

        saved = brand_price_service.save_scraped_prices(results)
        click.echo(f"Saved {saved} records for {brand}")
