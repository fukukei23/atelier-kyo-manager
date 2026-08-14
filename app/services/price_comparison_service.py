"""
Product-centric multi-source price comparison service.

Queries BrandPrice records across all sites (official, Farfetch, YOOX, SSENSE)
for a given brand, fuzzy-matches by product name, and returns a unified
comparison with profit calculations.

Usage:
    from app.services.price_comparison_service import compare_product, search_products
    result = compare_product(brand="Gucci", product_query="Continental Wallet GG Supreme")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app import db
from app.core.pricing.calculator import calculate_pricing
from app.core.pricing.schemas import PricingInput
from app.models.brand_price import BrandPrice
from app.utils.product_matching import cross_site_match_score

logger = logging.getLogger(__name__)

# Default minimum match score (0.0-1.0)
DEFAULT_MIN_SCORE = 0.30

# Sale site identifiers
SALE_SITES = {"yoox", "ssense"}

# BUYMA markup default (1.5x = ~11-14% profit rate)
DEFAULT_MARKUP_RATE = 1.5

# Default shipping cost (Buyandship base)
DEFAULT_SHIPPING_JPY = 2500


@dataclass
class SourceQuote:
    """Single price quote from one source site."""

    brand_price_id: int
    source_site: str
    source_url: str
    product_name: str  # name as it appears on that site
    price_original: float  # price in original currency
    currency: str
    price_jpy: float
    exchange_rate: float
    in_stock: bool
    match_score: float  # fuzzy match confidence 0.0-1.0
    is_sale: bool  # True if from YOOX/SSENSE
    actual_purchase_jpy: float | None  # sale/override price if set
    actual_purchase_source: str | None
    scraped_at: str | None

    @property
    def effective_cost_jpy(self) -> float:
        """Effective procurement cost (actual_purchase_jpy if set, else price_jpy)."""
        return self.actual_purchase_jpy if self.actual_purchase_jpy else self.price_jpy


@dataclass
class ProductComparison:
    """Unified comparison result for one product across all sources."""

    query: str  # original user query
    brand: str
    category: str | None = None
    quotes: list[SourceQuote] = field(default_factory=list)
    buyma_price: float | None = None
    buyma_matched_name: str | None = None
    buyma_match_score: float | None = None
    buyma_item_url: str | None = None

    @property
    def cheapest_quote(self) -> SourceQuote | None:
        """Return the cheapest in-stock procurement source."""
        in_stock = [q for q in self.quotes if q.in_stock]
        if not in_stock:
            return None
        return min(in_stock, key=lambda q: q.effective_cost_jpy)

    @property
    def cheapest_cost_jpy(self) -> float | None:
        q = self.cheapest_quote
        return q.effective_cost_jpy if q else None

    @property
    def profit_jpy(self) -> float | None:
        if self.cheapest_cost_jpy is None or self.buyma_price is None:
            return None
        return self.buyma_price - self.cheapest_cost_jpy

    @property
    def profit_rate(self) -> float | None:
        if self.cheapest_cost_jpy is None or self.buyma_price is None:
            return None
        if self.cheapest_cost_jpy == 0:
            return None
        return (self.buyma_price - self.cheapest_cost_jpy) / self.cheapest_cost_jpy

    @property
    def is_profitable(self) -> bool:
        """Check if profit >= ¥10,000 and >= 5% of cost."""
        p = self.profit_jpy
        c = self.cheapest_cost_jpy
        if p is None or c is None:
            return False
        return p >= max(10_000, c * 0.05)


# ---------------------------------------------------------------------------
# Core comparison functions
# ---------------------------------------------------------------------------


def compare_product(
    brand: str,
    product_query: str,
    category: str | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    include_buyma: bool = True,
) -> ProductComparison:
    """
    Main entry point. Given a brand + product name query, return a
    ProductComparison with all matching quotes from all sources.

    Args:
        brand: Brand name (e.g. "Gucci")
        product_query: Product name to search for (e.g. "Continental Wallet GG Supreme")
        category: Item category for shipping calculation (e.g. "wallet")
        min_score: Minimum fuzzy match score (0.0-1.0) to include a quote
        include_buyma: If True, also fetch BUYMA selling price for profit calc

    Returns:
        ProductComparison with quotes sorted by cost (cheapest first)
    """
    comparison = ProductComparison(query=product_query, brand=brand, category=category)

    # 1. Query all BrandPrice records for this brand
    all_prices = (
        db.session.execute(
            db.select(BrandPrice).filter(BrandPrice.brand == brand).order_by(BrandPrice.scraped_at.desc())
        )
        .scalars()
        .all()
    )

    # Deduplicate: keep latest per (product_name, source_site) combo
    seen: dict[tuple[str, str], BrandPrice] = {}
    for bp in all_prices:
        key = (bp.product_name, bp.source_site)
        if key not in seen:
            seen[key] = bp

    # 2. Score each against the query and keep >= min_score
    scored: list[tuple[float, SourceQuote]] = []
    for bp in seen.values():
        score = cross_site_match_score(
            query_name=product_query,
            candidate_name=bp.product_name,
            brand=brand,
        )
        if score < min_score:
            continue

        is_sale = bp.source_site.lower() in SALE_SITES
        quote = SourceQuote(
            brand_price_id=bp.id,
            source_site=bp.source_site,
            source_url=bp.source_url or "",
            product_name=bp.product_name,
            price_original=bp.price_original,
            currency=bp.currency,
            price_jpy=bp.price_jpy,
            exchange_rate=bp.exchange_rate,
            in_stock=bool(bp.in_stock),
            match_score=round(score, 3),
            is_sale=is_sale,
            actual_purchase_jpy=bp.actual_purchase_jpy,
            actual_purchase_source=bp.actual_purchase_source,
            scraped_at=bp.scraped_at.isoformat() if bp.scraped_at else None,
        )

        # Prefer BUYMA selling price if available
        if bp.buyma_price and comparison.buyma_price is None:
            comparison.buyma_price = bp.buyma_price
            comparison.buyma_matched_name = bp.buyma_matched_name
            comparison.buyma_match_score = bp.buyma_match_score

        scored.append((score, quote))

    # Sort by score descending, then by effective cost ascending
    scored.sort(key=lambda x: (-x[0], x[1].effective_cost_jpy))
    comparison.quotes = [q for _, q in scored]

    return comparison


def find_matching_quotes(
    brand: str,
    product_query: str,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[SourceQuote]:
    """
    Lightweight version of compare_product - returns only the matching quotes
    without BUYMA enrichment. Useful for autocomplete / product picker.
    """
    result = compare_product(brand, product_query, min_score=min_score, include_buyma=False)
    return result.quotes


def search_products(
    brand: str,
    query: str,
    limit: int = 20,
) -> list[dict]:
    """
    Lightweight product search for autocomplete.
    Returns distinct product names from BrandPrice matching the query substring.
    """
    rows = db.session.execute(
        db.select(BrandPrice.product_name, db.func.count(BrandPrice.id).label("source_count"))
        .filter(
            BrandPrice.brand == brand,
            BrandPrice.product_name.ilike(f"%{query}%"),
        )
        .group_by(BrandPrice.product_name)
        .order_by(db.func.count(BrandPrice.id).desc())
        .limit(limit)
    ).all()

    return [
        {
            "product_name": r.product_name,
            "source_count": r.source_count,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# BUYMA enrichment
# ---------------------------------------------------------------------------


def enrich_with_buyma(comparison: ProductComparison) -> ProductComparison:
    """
    Search BUYMA for the comparison's query and attach selling price + metadata.
    Uses the existing BuymaPriceSearcher.
    """
    from app.services.buyma_price_scraper import BuymaPriceSearcher

    try:
        searcher = BuymaPriceSearcher()
        results = searcher.search(
            query=comparison.query,
            brand=comparison.brand,
            limit=3,
        )
        if results:
            best = max(results, key=lambda r: r.get("score", 0))
            comparison.buyma_price = best.get("price")
            comparison.buyma_matched_name = best.get("name")
            comparison.buyma_match_score = best.get("score")
            comparison.buyma_item_url = best.get("url")
    except Exception as e:
        logger.warning(f"BUYMA enrichment failed: {e}")

    return comparison


# ---------------------------------------------------------------------------
# Profit calculation
# ---------------------------------------------------------------------------


def calculate_comparison_profit(
    comparison: ProductComparison,
    category: str | None = None,
    markup_rate: float = DEFAULT_MARKUP_RATE,
) -> dict:
    """
    Calculate profit for each quote in the comparison using the pricing calculator.

    Returns:
        dict with per_source data, best_source, and summary
    """
    if comparison.buyma_price is None:
        # Estimate BUYMA price from cheapest source
        cheapest = comparison.cheapest_cost_jpy
        if cheapest is None:
            return {"error": "No price data available"}
        estimated_buyma = round(cheapest * markup_rate)
    else:
        estimated_buyma = comparison.buyma_price

    per_source = {}
    for quote in comparison.quotes:
        cost = quote.effective_cost_jpy
        inp = PricingInput(
            purchase_price=cost,
            selling_price=estimated_buyma,
            shipping_cost=DEFAULT_SHIPPING_JPY,
            original_currency="JPY",
            item_category=category or "",
        )
        # Phase1(ISSUE-101/102): 一時skip・Phase2 でsource明示後に撤去
        result = calculate_pricing(inp, skip_source_validation=True)
        per_source[quote.source_site] = {
            "cost_jpy": cost,
            "sell_jpy": estimated_buyma,
            "profit": round(result.profit, 2),
            "profit_rate": round(result.profit_rate, 4),
            "total_cost": round(result.total_cost, 2),
            "customs_duty": round(result.auto_customs_duty, 2),
            "is_profitable": result.profit > max(10_000, result.total_cost * 0.05),
            "in_stock": quote.in_stock,
        }

    best_quote = comparison.cheapest_quote
    best_source_name = best_quote.source_site if best_quote else None
    best_profit_data = per_source.get(best_source_name, {}) if best_source_name else {}

    return {
        "per_source": per_source,
        "best_source": best_source_name,
        "best_profit": best_profit_data.get("profit"),
        "best_profit_rate": best_profit_data.get("profit_rate"),
        "buyma_price": comparison.buyma_price,
        "estimated_buyma": estimated_buyma,
        "is_profitable": comparison.is_profitable,
    }
