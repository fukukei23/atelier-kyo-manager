from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import func

from app.core.timezone import _utcnow
from app.config.cost_table import DEFAULT_MARKUP_RATE, get_buyandship_shipping
from app.core.pricing.calculator import calculate_pricing
from app.core.pricing.schemas import PricingInput
from app.extensions import db
from app.models.brand_price import BrandPrice

logger = logging.getLogger(__name__)


def save_scraped_prices(items: list[dict]) -> int:
    saved = 0
    for item in items:
        existing = db.session.scalar(
            db.select(BrandPrice).filter(
                BrandPrice.brand == item["brand"],
                BrandPrice.product_name == item.get("product_name", ""),
                BrandPrice.source_site == item["source_site"],
            ).limit(1)
        )
        if existing:
            existing.price_original = item["price_original"]
            existing.currency = item["currency"]
            existing.price_jpy = item["price_jpy"]
            existing.exchange_rate = item["exchange_rate"]
            existing.in_stock = item.get("in_stock", True)
            existing.size_available = item.get("size_available", "")
            existing.scraped_at = datetime.fromisoformat(item["scraped_at"]) if isinstance(item.get("scraped_at"), str) else item.get("scraped_at", _utcnow())
            saved += 1
            continue
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
            scraped_at=datetime.fromisoformat(item["scraped_at"]) if isinstance(item.get("scraped_at"), str) else item.get("scraped_at", _utcnow()),
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
        entry = {"product_name": name, "sites": {}, "cheapest_site": None, "cheapest_jpy": None, "cheapest_price_id": None}
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
                entry["cheapest_price_id"] = p.id
            if p.buyma_price and not entry.get("buyma_price"):
                entry["buyma_price"] = p.buyma_price
            if p.buyma_matched_name and not entry.get("buyma_matched_name"):
                entry["buyma_matched_name"] = p.buyma_matched_name
            if p.buyma_match_score is not None and "buyma_match_score" not in entry:
                entry["buyma_match_score"] = p.buyma_match_score
            if p.buyma_status and not entry.get("buyma_status"):
                entry["buyma_status"] = p.buyma_status
            if p.buyma_searched_at and not entry.get("buyma_searched_at"):
                entry["buyma_searched_at"] = p.buyma_searched_at.isoformat()
            # 実際仕入れ価格（セール・アウトレット等）
            if p.actual_purchase_jpy and "actual_purchase_jpy" not in entry:
                entry["actual_purchase_jpy"] = p.actual_purchase_jpy
                entry["actual_purchase_source"] = p.actual_purchase_source
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
    cutoff = _utcnow() - timedelta(days=keep_days)
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


def cleanup_buyma_cache(max_age_days: int = 30) -> int:
    """BUYMA検索キャッシュが古くなったレコードのbuyma_status/buyma_searched_atをリセット。"""
    cutoff = _utcnow() - timedelta(days=max_age_days)
    count = db.session.execute(
        db.update(BrandPrice)
        .where(
            BrandPrice.buyma_searched_at < cutoff,
            BrandPrice.buyma_status == "matched",
        )
        .values(buyma_status=None, buyma_searched_at=None)
    ).rowcount
    db.session.commit()
    logger.info(f"Reset {count} stale BUYMA cache entries (older than {max_age_days} days)")
    return count


def get_last_scraped_at(brand: str) -> datetime | None:
    row = db.session.scalar(
        db.select(func.max(BrandPrice.scraped_at)).filter(BrandPrice.brand == brand)
    )
    return row


def add_profit_calculation(
    comparison: list[dict],
    category: str | None = None,
    markup_rate: float = DEFAULT_MARKUP_RATE,
) -> list[dict]:
    """価格比較データに利益計算結果を付与する。

    各商品の cheapest_jpy を仕入れ原価とし、BUYMA販売価格（buyma_price または
    マークアップ価格）から利益・利益率・安全判定を計算する。

    actual_purchase_jpy が設定されている場合は、EU定価ではなく実際の仕入れ価格
    （セール・アウトレット・代理購入等）を原価として使用する。

    経営者判断に基づく計算式:
      仕入れ原価 = 実際仕入れ価格(JPY) + Buyandship送料 + 関税
      利益 = BUYMA販売価格 - 総コスト（既存 calculator.py 使用）
      安全判定 = 利益 > max(10,000, 原価 × 5%)
    """
    for item in comparison:
        # 実際仕入れ価格があれば優先、なければEU定価
        cheapest = item.get("actual_purchase_jpy") or item.get("cheapest_jpy")
        if cheapest is None or cheapest <= 0:
            item["profit"] = None
            item["profit_rate"] = None
            item["is_profitable"] = False
            item["buyma_suggested_price"] = None
            continue

        shipping = get_buyandship_shipping(category)
        buyma_price = item.get("buyma_price") or round(cheapest * markup_rate)

        inp = PricingInput(
            purchase_price=cheapest,
            selling_price=buyma_price,
            shipping_cost=shipping,
            original_currency="JPY",
            item_category=category or "",
        )
        result = calculate_pricing(inp)

        item["buyma_suggested_price"] = buyma_price
        item["shipping_cost"] = shipping
        item["total_cost"] = round(result.total_cost, 2)
        item["customs_duty"] = round(result.auto_customs_duty, 2)
        item["customs_rate"] = round(result.customs_rate_used, 4)
        item["profit"] = round(result.profit, 2)
        item["profit_rate"] = round(result.profit_rate, 4)
        item["is_profitable"] = result.profit > max(10_000, result.total_cost * 0.05)

    return comparison
