"""
SSENSE + BUYMA 自動価格検証パイプライン

SSENSEのサングラス（任意カテゴリ）商品をスクレイプし、
BUYMA検索で実際の出品価格を確認。両方BROWSER_VERIFIEDの
データのみで利益計算する。

フロー:
  Phase 1: SSENSE PLP巡回（商品一覧取得）
  Phase 2: SSENSE PDP巡回（色・型番取得）
  Phase 3: BUYMA検索（出品価格取得）
  Phase 4: 利益計算（PriceSource.BROWSER_VERIFIED）
  Phase 5: 結果保存（BrandPrice レコード）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.core.pricing.calculator import calculate_pricing
from app.core.pricing.schemas import PriceSource, PricingInput, PricingResult
from app.services.sale_scraper import _scrape_ssense, scrape_ssense_pdp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SsenseProduct:
    """SSENSEから取得した1商品。"""

    url: str
    name: str
    price_usd: float
    price_jpy: float
    exchange_rate: float
    sku: str
    brand: str
    color: str | None = None
    model_number: str | None = None


@dataclass
class PipelineResult:
    """パイプライン1商品の結果。"""

    ssense: SsenseProduct
    buyma_match: dict[str, Any] | None = None
    pricing: PricingResult | None = None
    is_verified: bool = False
    error: str | None = None


@dataclass
class PipelineSummary:
    """パイプライン全体のサマリー。"""

    brand: str
    category: str
    total_scraped: int = 0
    buyma_matched: int = 0
    verified_profitable: int = 0
    errors: int = 0
    results: list[PipelineResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class SsenseBuymaPipeline:
    """SSENSE→BUYMA自動価格検証パイプライン。

    Usage:
        pipeline = SsenseBuymaPipeline()
        summary = pipeline.run("Gucci", category="sunglasses")
    """

    def __init__(self) -> None:
        self._buyma_searcher: Any = None

    def run(self, brand: str, category: str = "sunglasses") -> PipelineSummary:
        """パイプライン全体を実行。"""
        summary = PipelineSummary(brand=brand, category=category)

        # Phase 1: SSENSE PLP
        products = self._phase1_ssense(brand, category)
        summary.total_scraped = len(products)
        if not products:
            logger.warning("[pipeline] Phase 1: no products for %s/%s", brand, category)
            return summary

        # Phase 2: SSENSE PDP (色・型番)
        products = self._phase2_pdp(products)

        # Phase 3: BUYMA検索
        buyma_results = self._phase3_buyma(products, brand)

        # Phase 4: 利益計算
        results = self._phase4_calculate(products, buyma_results)

        # Phase 5: 結果保存
        saved = self._phase5_save(results, brand, category)

        summary.buyma_matched = sum(1 for r in results if r.buyma_match)
        summary.verified_profitable = sum(
            1 for r in results if r.is_verified and r.pricing and r.pricing.profit > 0
        )
        summary.errors = sum(1 for r in results if r.error)
        summary.results = results

        logger.info(
            "[pipeline] %s/%s: scraped=%d, buyma=%d, profitable=%d, errors=%d, saved=%d",
            brand,
            category,
            summary.total_scraped,
            summary.buyma_matched,
            summary.verified_profitable,
            summary.errors,
            saved,
        )
        return summary

    # --- Phase 1: SSENSE PLP ---

    def _phase1_ssense(self, brand: str, category: str) -> list[SsenseProduct]:
        """SSENSE PLPから商品一覧を取得。"""
        raw_items = _scrape_ssense(brand, item_limit=50, category=category)

        products: list[SsenseProduct] = []
        for item in raw_items:
            products.append(
                SsenseProduct(
                    url=item.get("source_url", ""),
                    name=item.get("product_name", ""),
                    price_usd=item.get("price_original", 0.0),
                    price_jpy=item.get("price_jpy", 0.0),
                    exchange_rate=item.get("exchange_rate", 1.0),
                    sku=str(item.get("sku", "")),
                    brand=brand,
                )
            )
        logger.info("[pipeline] Phase 1: %d products from SSENSE/%s", len(products), category)
        return products

    # --- Phase 2: SSENSE PDP ---

    def _phase2_pdp(self, products: list[SsenseProduct]) -> list[SsenseProduct]:
        """各商品のPDPから色・型番を取得。"""
        for prod in products:
            if not prod.url:
                continue
            try:
                pdp = scrape_ssense_pdp(prod.url)
                if pdp:
                    prod.color = pdp.get("color")
                    prod.model_number = pdp.get("model_number")
                    if pdp.get("full_name"):
                        prod.name = pdp["full_name"]
            except Exception as exc:
                logger.debug("[pipeline] PDP error for %s: %s", prod.url, exc)
        logger.info("[pipeline] Phase 2: PDP enriched %d products", len(products))
        return products

    # --- Phase 3: BUYMA検索 ---

    def _phase3_buyma(
        self, products: list[SsenseProduct], brand: str
    ) -> dict[str, dict[str, Any]]:
        """BUYMA検索で各商品の出品価格を取得。"""
        from app.services.buyma_price_scraper import BuymaPriceSearcher

        if self._buyma_searcher is None:
            self._buyma_searcher = BuymaPriceSearcher()

        results: dict[str, dict[str, Any]] = {}
        threshold = 0.25  # 緩めの閾値（パイプラインは自動検証が目的）

        for prod in products:
            try:
                match = self._buyma_searcher.search_single(prod.name, brand, threshold)
                if match:
                    results[prod.name] = match
            except Exception as exc:
                logger.debug("[pipeline] BUYMA search error for %s: %s", prod.name, exc)

        logger.info(
            "[pipeline] Phase 3: %d/%d BUYMA matches", len(results), len(products)
        )
        return results

    # --- Phase 4: 利益計算 ---

    def _phase4_calculate(
        self,
        products: list[SsenseProduct],
        buyma_results: dict[str, dict[str, Any]],
    ) -> list[PipelineResult]:
        """BROWSER_VERIFIEDデータで利益計算。"""
        from app.config.cost_table import get_buyandship_shipping

        results: list[PipelineResult] = []

        for prod in products:
            result = PipelineResult(ssense=prod)

            # BUYMAマッチ確認
            buyma = buyma_results.get(prod.name)
            if not buyma:
                result.error = "BUYMA price not found"
                results.append(result)
                continue

            result.buyma_match = buyma

            # 価格取得
            purchase_price = prod.price_jpy
            buyma_price = buyma.get("price", 0.0)
            if isinstance(buyma_price, str):
                try:
                    buyma_price = float(buyma_price.replace(",", ""))
                except ValueError:
                    result.error = f"Invalid BUYMA price: {buyma_price}"
                    results.append(result)
                    continue

            if purchase_price <= 0 or buyma_price <= 0:
                result.error = f"Invalid prices: purchase={purchase_price}, buyma={buyma_price}"
                results.append(result)
                continue

            # 為替レートを使ってJPY換算（USD商品の場合）
            if prod.exchange_rate and prod.exchange_rate > 0:
                purchase_price_jpy = prod.price_usd * prod.exchange_rate
            else:
                purchase_price_jpy = purchase_price

            try:
                shipping = get_buyandship_shipping("accessory")
                inp = PricingInput(
                    purchase_price=purchase_price_jpy,
                    selling_price=buyma_price,
                    shipping_cost=shipping,
                    warehouse_shipping_cost=25.0 * (prod.exchange_rate or 150.0),  # SSENSE送料$25
                    original_currency="JPY",
                    exchange_rate=1.0,  # 既にJPY換算済み
                    item_category="accessory",
                    purchase_price_source=PriceSource.BROWSER_VERIFIED,
                    selling_price_source=PriceSource.BROWSER_VERIFIED,
                )
                pricing = calculate_pricing(inp, source_type="overseas")
                result.pricing = pricing
                result.is_verified = True
            except Exception as exc:
                result.error = f"Pricing error: {exc}"

            results.append(result)

        return results

    # --- Phase 5: 結果保存 ---

    def _phase5_save(
        self, results: list[PipelineResult], brand: str, category: str
    ) -> int:
        """BrandPriceレコードに保存。"""
        from app.extensions import db
        from app.models.brand_price import BrandPrice

        saved = 0
        for r in results:
            if not r.is_verified:
                continue
            try:
                bp = BrandPrice(
                    brand=brand,
                    product_name=r.ssense.name,
                    source_site="ssense",
                    source_url=r.ssense.url,
                    price_original=r.ssense.price_usd,
                    currency="USD",
                    price_jpy=r.ssense.price_jpy,
                    exchange_rate=r.ssense.exchange_rate,
                    actual_purchase_jpy=r.ssense.price_jpy,
                    actual_purchase_source="ssense_pipeline",
                    color=r.ssense.color,
                    model_number=r.ssense.model_number,
                    purchase_price_source="browser_verified",
                    buyma_price=r.buyma_match.get("price") if r.buyma_match else None,
                    buyma_matched_name=r.buyma_match.get("name") if r.buyma_match else None,
                    buyma_match_score=r.buyma_match.get("score") if r.buyma_match else None,
                    buyma_status="pipeline_verified",
                    selling_price_source="browser_verified",
                )
                db.session.add(bp)
                saved += 1
            except Exception as exc:
                logger.warning("[pipeline] Save error for %s: %s", r.ssense.name, exc)

        if saved > 0:
            try:
                db.session.commit()
            except Exception as exc:
                db.session.rollback()
                logger.error("[pipeline] DB commit error: %s", exc)
                return 0

        logger.info("[pipeline] Phase 5: saved %d records", saved)
        return saved
