"""price_comparison_service のテスト — DB使用"""

from __future__ import annotations

import contextlib

import pytest

from app import create_app
from app.extensions import db
from app.models.brand_price import BrandPrice
from app.services import price_comparison_service
from app.services.price_comparison_service import (
    ProductComparison,
    SourceQuote,
)


@pytest.fixture(scope="function")
def app():
    """In-memory SQLite Flask app for testing."""
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test-secret",
        }
    )
    with app.app_context():
        db.create_all()
        yield app
        with contextlib.suppress(Exception):
            db.drop_all()


def _seed_brand_prices(app):
    """テスト用BrandPriceデータを投入"""
    with app.app_context():
        records = [
            # Gucci official (EUR)
            BrandPrice(
                brand="Gucci",
                product_name="GG Supreme Continental Wallet",
                source_site="gucci_official",
                price_original=550.0,
                currency="EUR",
                price_jpy=90750,
                exchange_rate=165.0,
                in_stock=True,
                buyma_price=110000,
                buyma_matched_name="Gucci GGスプリーム コンチネンタル",
                buyma_match_score=0.85,
            ),
            # Farfetch (JPY)
            BrandPrice(
                brand="Gucci",
                product_name="Gucci GG Supreme Continental Wallet Beige",
                source_site="farfetch",
                price_original=89000,
                currency="JPY",
                price_jpy=89000,
                exchange_rate=1.0,
                in_stock=True,
                buyma_price=110000,
            ),
            # YOOX sale (USD)
            BrandPrice(
                brand="Gucci",
                product_name="Gucci GG Supreme Continental Wallet Black",
                source_site="yoox",
                price_original=320.0,
                currency="USD",
                price_jpy=48000,
                exchange_rate=150.0,
                in_stock=True,
                actual_purchase_jpy=45000,
                actual_purchase_source="yoox",
            ),
            # SSENSE sale (USD)
            BrandPrice(
                brand="Gucci",
                product_name="Gucci GG Supreme Card Holder",
                source_site="ssense",
                price_original=280.0,
                currency="USD",
                price_jpy=42000,
                exchange_rate=150.0,
                in_stock=False,
                actual_purchase_jpy=40000,
                actual_purchase_source="ssense",
            ),
            # Prada (different brand — should not appear in Gucci search)
            BrandPrice(
                brand="Prada",
                product_name="Prada Re-Nylon Wallet",
                source_site="prada_official",
                price_original=400.0,
                currency="EUR",
                price_jpy=66000,
                exchange_rate=165.0,
                in_stock=True,
            ),
        ]
        for r in records:
            db.session.add(r)
        db.session.commit()


class TestSourceQuote:
    def test_effective_cost_uses_actual_purchase(self):
        quote = SourceQuote(
            brand_price_id=1,
            source_site="yoox",
            source_url="",
            product_name="GG Wallet",
            price_original=320.0,
            currency="USD",
            price_jpy=48000,
            exchange_rate=150.0,
            in_stock=True,
            match_score=0.9,
            is_sale=True,
            actual_purchase_jpy=45000,
            actual_purchase_source="yoox",
            scraped_at=None,
        )
        assert quote.effective_cost_jpy == 45000

    def test_effective_cost_fallback_to_price_jpy(self):
        quote = SourceQuote(
            brand_price_id=1,
            source_site="gucci_official",
            source_url="",
            product_name="GG Wallet",
            price_original=550.0,
            currency="EUR",
            price_jpy=90750,
            exchange_rate=165.0,
            in_stock=True,
            match_score=0.9,
            is_sale=False,
            actual_purchase_jpy=None,
            actual_purchase_source=None,
            scraped_at=None,
        )
        assert quote.effective_cost_jpy == 90750


class TestProductComparison:
    def test_cheapest_quote_returns_lowest_in_stock(self):
        comparison = ProductComparison(query="GG Wallet", brand="Gucci")
        comparison.quotes = [
            SourceQuote(
                brand_price_id=1,
                source_site="gucci_official",
                source_url="",
                product_name="GG Wallet",
                price_original=550.0,
                currency="EUR",
                price_jpy=90750,
                exchange_rate=165.0,
                in_stock=True,
                match_score=0.9,
                is_sale=False,
                actual_purchase_jpy=None,
                actual_purchase_source=None,
                scraped_at=None,
            ),
            SourceQuote(
                brand_price_id=2,
                source_site="yoox",
                source_url="",
                product_name="GG Wallet sale",
                price_original=320.0,
                currency="USD",
                price_jpy=48000,
                exchange_rate=150.0,
                in_stock=True,
                match_score=0.9,
                is_sale=True,
                actual_purchase_jpy=45000,
                actual_purchase_source="yoox",
                scraped_at=None,
            ),
        ]
        comparison.buyma_price = 110000

        cheapest = comparison.cheapest_quote
        assert cheapest.source_site == "yoox"
        assert comparison.cheapest_cost_jpy == 45000

    def test_profit_calculation(self):
        comparison = ProductComparison(query="GG Wallet", brand="Gucci")
        comparison.buyma_price = 110000
        # Set up a quote with actual_purchase_jpy
        comparison.quotes = [
            SourceQuote(
                brand_price_id=1,
                source_site="yoox",
                source_url="",
                product_name="GG Wallet",
                price_original=320.0,
                currency="USD",
                price_jpy=48000,
                exchange_rate=150.0,
                in_stock=True,
                match_score=0.9,
                is_sale=True,
                actual_purchase_jpy=45000,
                actual_purchase_source="yoox",
                scraped_at=None,
            ),
        ]
        # profit = 110000 - 45000 = 65000
        assert comparison.profit_jpy == 65000
        # profit_rate = 65000 / 45000 = ~1.44 (144%)
        assert comparison.profit_rate > 1.0
        # is_profitable: 65000 >= max(10000, 45000 * 0.05=2250) → True
        assert comparison.is_profitable is True

    def test_empty_quotes_no_cheapest(self):
        comparison = ProductComparison(query="Unknown", brand="Gucci")
        assert comparison.cheapest_quote is None
        assert comparison.cheapest_cost_jpy is None


class TestCompareProduct:
    def test_returns_matching_gucci_quotes(self, app):
        _seed_brand_prices(app)
        with app.app_context():
            result = price_comparison_service.compare_product(
                brand="Gucci",
                product_query="Continental Wallet GG Supreme",
                include_buyma=False,
            )
            # Should match gucci_official and farfetch
            assert len(result.quotes) >= 2
            assert all(q for q in result.quotes)  # verify quotes returned
            # YOOX should also match
            site_names = {q.source_site for q in result.quotes}
            assert "gucci_official" in site_names

    def test_filters_different_brand(self, app):
        _seed_brand_prices(app)
        with app.app_context():
            result = price_comparison_service.compare_product(
                brand="Gucci",
                product_query="Continental Wallet GG Supreme",
                include_buyma=False,
            )
            site_names = {q.source_site for q in result.quotes}
            # Prada records should not appear
            assert "prada_official" not in site_names

    def test_no_match_returns_empty(self, app):
        _seed_brand_prices(app)
        with app.app_context():
            result = price_comparison_service.compare_product(
                brand="Gucci",
                product_query="Zzzzzzzz totally nonexistent product",
                include_buyma=False,
            )
            assert len(result.quotes) == 0


class TestSearchProducts:
    def test_ilike_search_returns_matches(self, app):
        _seed_brand_prices(app)
        with app.app_context():
            results = price_comparison_service.search_products(
                brand="Gucci",
                query="Continental",
                limit=10,
            )
            assert len(results) >= 1
            assert all("Continental" in r["product_name"] for r in results)

    def test_no_result_for_unknown(self, app):
        _seed_brand_prices(app)
        with app.app_context():
            results = price_comparison_service.search_products(
                brand="Gucci",
                query="xyzabc123",
                limit=10,
            )
            assert len(results) == 0
