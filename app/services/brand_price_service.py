from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import func

from app.extensions import db
from app.models.brand_price import BrandPrice

logger = logging.getLogger(__name__)


def save_scraped_prices(items: list[dict]) -> int:
    saved = 0
    for item in items:
        bp = BrandPrice(
            brand=item["brand"],
            product_name=item.get("product_name", ""),
            source_site=item["source_site"],
            source_url=item.get("source_url", ""),
            price_original=item["price_original"],
            currency=item["currency"],
            price_jpy=item["price_jpy"],
            exchange_rate=item["exchange_rate"],
            in_stock=item.get("in_stock", True),
            size_available=item.get("size_available", ""),
            scraped_at=datetime.fromisoformat(item["scraped_at"]) if isinstance(item.get("scraped_at"), str) else item.get("scraped_at", datetime.utcnow()),
        )
        db.session.add(bp)
        saved += 1
    db.session.commit()
    logger.info(f"Saved {saved} brand price records")
    return saved


def get_price_comparison(brand: str) -> list[dict]:
    subq = (
        db.session.query(
            BrandPrice.product_name,
            BrandPrice.source_site,
            func.max(BrandPrice.scraped_at).label("latest"),
        )
        .filter(BrandPrice.brand == brand)
        .group_by(BrandPrice.product_name, BrandPrice.source_site)
        .subquery()
    )

    rows = db.session.execute(
        db.select(BrandPrice)
        .join(
            subq,
            (BrandPrice.product_name == subq.c.product_name)
            & (BrandPrice.source_site == subq.c.source_site)
            & (BrandPrice.scraped_at == subq.c.latest),
        )
        .filter(BrandPrice.brand == brand)
        .order_by(BrandPrice.product_name, BrandPrice.price_jpy)
    ).scalars().all()

    grouped: dict[str, list[BrandPrice]] = {}
    for r in rows:
        grouped.setdefault(r.product_name, []).append(r)

    result = []
    for name, prices in grouped.items():
        entry = {"product_name": name, "sites": {}, "cheapest_site": None, "cheapest_jpy": None}
        for p in prices:
            entry["sites"][p.source_site] = {
                "price_jpy": p.price_jpy,
                "price_original": p.price_original,
                "currency": p.currency,
                "in_stock": p.in_stock,
                "scraped_at": p.scraped_at.isoformat() if p.scraped_at else None,
            }
            if entry["cheapest_jpy"] is None or p.price_jpy < entry["cheapest_jpy"]:
                entry["cheapest_jpy"] = p.price_jpy
                entry["cheapest_site"] = p.source_site
        result.append(entry)

    return result


def get_cheapest_source(brand: str) -> dict[str, dict]:
    comparison = get_price_comparison(brand)
    cheapest: dict[str, dict] = {}
    for item in comparison:
        name = item["product_name"]
        site = item["cheapest_site"]
        if site and site in item["sites"]:
            cheapest[name] = {
                "site": site,
                "price_jpy": item["cheapest_jpy"],
                **item["sites"][site],
            }
    return cheapest


def cleanup_old_records(keep_days: int = 90) -> int:
    cutoff = datetime.utcnow() - timedelta(days=keep_days)
    count = db.session.execute(
        db.delete(BrandPrice).where(BrandPrice.scraped_at < cutoff)
    ).rowcount
    db.session.commit()
    logger.info(f"Cleaned up {count} old brand price records (older than {keep_days} days)")
    return count


def get_available_brands() -> list[str]:
    rows = db.session.execute(
        db.select(BrandPrice.brand).distinct().order_by(BrandPrice.brand)
    ).scalars().all()
    return list(rows)


def get_last_scraped_at(brand: str) -> datetime | None:
    row = db.session.scalar(
        db.select(func.max(BrandPrice.scraped_at)).filter(BrandPrice.brand == brand)
    )
    return row
