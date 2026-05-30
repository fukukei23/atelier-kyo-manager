"""BrandPriceService のテスト — DB使用"""

from __future__ import annotations

from datetime import datetime

import pytest

from app import create_app
from app.extensions import db
from app.models.brand_price import BrandPrice
from app.services import brand_price_service


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
        try:
            db.drop_all()
        except Exception:
            pass  # テーブルがない場合は無視


class TestBrandPriceService:
    def test_save_scraped_prices_new(self, app):
        items = [
            {
                "brand": "Prada",
                "product_name": "Bag X",
                "source_site": "prada.com",
                "price_original": 1000,
                "currency": "EUR",
                "price_jpy": 150000,
                "exchange_rate": 150,
                "in_stock": True,
                "scraped_at": "2026-05-01T12:00:00",
            }
        ]
        saved = brand_price_service.save_scraped_prices(items)
        assert saved == 1
        fetched = db.session.scalar(db.select(BrandPrice).filter_by(brand="Prada"))
        assert fetched is not None
        assert fetched.price_jpy == 150000

    def test_save_scraped_prices_update_existing(self, app):
        bp = BrandPrice(
            brand="Prada", product_name="Bag X", source_site="prada.com",
            price_original=1000, currency="EUR", price_jpy=150000, exchange_rate=150,
        )
        db.session.add(bp)
        db.session.commit()

        items = [
            {
                "brand": "Prada",
                "product_name": "Bag X",
                "source_site": "prada.com",
                "price_original": 1100,
                "currency": "EUR",
                "price_jpy": 165000,
                "exchange_rate": 150,
                "scraped_at": "2026-05-15T12:00:00",
            }
        ]
        saved = brand_price_service.save_scraped_prices(items)
        assert saved == 1
        fetched = db.session.scalar(db.select(BrandPrice).filter_by(brand="Prada"))
        assert fetched.price_jpy == 165000

    def test_save_scraped_prices_multiple(self, app):
        items = [
            {
                "brand": "Gucci",
                "product_name": f"Item {i}",
                "source_site": "gucci.com",
                "price_original": 500 + i * 100,
                "currency": "EUR",
                "price_jpy": 75000 + i * 15000,
                "exchange_rate": 150,
                "scraped_at": "2026-05-01T12:00:00",
            }
            for i in range(3)
        ]
        saved = brand_price_service.save_scraped_prices(items)
        assert saved == 3

    def test_get_available_brands(self, app):
        db.session.add_all([
            BrandPrice(brand="Chanel", product_name="A", source_site="s1",
                       price_original=1, currency="EUR", price_jpy=1, exchange_rate=1),
            BrandPrice(brand="Gucci", product_name="B", source_site="s2",
                       price_original=1, currency="EUR", price_jpy=1, exchange_rate=1),
            BrandPrice(brand="Gucci", product_name="C", source_site="s3",
                       price_original=1, currency="EUR", price_jpy=1, exchange_rate=1),
        ])
        db.session.commit()
        brands = brand_price_service.get_available_brands()
        assert brands == ["Chanel", "Gucci"]

    def test_get_available_brands_empty(self, app):
        assert brand_price_service.get_available_brands() == []

    def test_cleanup_old_records(self, app):
        old = BrandPrice(
            brand="Old", product_name="Old", source_site="s",
            price_original=1, currency="EUR", price_jpy=1, exchange_rate=1,
            scraped_at=datetime(2020, 1, 1),
        )
        recent = BrandPrice(
            brand="Recent", product_name="Recent", source_site="s",
            price_original=1, currency="EUR", price_jpy=1, exchange_rate=1,
            scraped_at=datetime.utcnow(),
        )
        db.session.add_all([old, recent])
        db.session.commit()

        count = brand_price_service.cleanup_old_records(keep_days=90)
        assert count == 1
        remaining = db.session.scalars(db.select(BrandPrice)).all()
        assert len(remaining) == 1
        assert remaining[0].brand == "Recent"

    def test_cleanup_old_records_none_to_clean(self, app):
        recent = BrandPrice(
            brand="R", product_name="R", source_site="s",
            price_original=1, currency="EUR", price_jpy=1, exchange_rate=1,
            scraped_at=datetime.utcnow(),
        )
        db.session.add(recent)
        db.session.commit()
        count = brand_price_service.cleanup_old_records(keep_days=90)
        assert count == 0

    def test_get_last_scraped_at(self, app):
        dt1 = datetime(2026, 1, 1)
        dt2 = datetime(2026, 5, 1)
        db.session.add_all([
            BrandPrice(brand="LV", product_name="A", source_site="s1",
                       price_original=1, currency="EUR", price_jpy=1, exchange_rate=1, scraped_at=dt1),
            BrandPrice(brand="LV", product_name="B", source_site="s2",
                       price_original=1, currency="EUR", price_jpy=1, exchange_rate=1, scraped_at=dt2),
        ])
        db.session.commit()
        result = brand_price_service.get_last_scraped_at("LV")
        assert result == dt2

    def test_get_last_scraped_at_no_brand(self, app):
        result = brand_price_service.get_last_scraped_at("None")
        assert result is None

    def test_cleanup_buyma_cache(self, app):
        old_cached = BrandPrice(
            brand="B", product_name="P", source_site="s",
            price_original=1, currency="EUR", price_jpy=1, exchange_rate=1,
            buyma_status="matched",
            buyma_searched_at=datetime(2020, 1, 1),
        )
        fresh_cached = BrandPrice(
            brand="B", product_name="P2", source_site="s",
            price_original=1, currency="EUR", price_jpy=1, exchange_rate=1,
            buyma_status="matched",
            buyma_searched_at=datetime.utcnow(),
        )
        db.session.add_all([old_cached, fresh_cached])
        db.session.commit()

        count = brand_price_service.cleanup_buyma_cache(max_age_days=30)
        assert count == 1
        db.session.refresh(old_cached)
        assert old_cached.buyma_status is None
        assert old_cached.buyma_searched_at is None
        db.session.refresh(fresh_cached)
        assert fresh_cached.buyma_status == "matched"

    def test_get_cheapest_source(self, app):
        db.session.add_all([
            BrandPrice(brand="LV", product_name="Wallet", source_site="site_a",
                       price_original=500, currency="EUR", price_jpy=75000, exchange_rate=150,
                       scraped_at=datetime(2026, 5, 1)),
            BrandPrice(brand="LV", product_name="Wallet", source_site="site_b",
                       price_original=450, currency="EUR", price_jpy=67500, exchange_rate=150,
                       scraped_at=datetime(2026, 5, 1)),
        ])
        db.session.commit()
        result = brand_price_service.get_cheapest_source("LV")
        assert "Wallet" in result
        assert result["Wallet"]["site"] == "site_b"
        assert result["Wallet"]["price_jpy"] == 67500

    def test_get_price_comparison(self, app):
        db.session.add_all([
            BrandPrice(brand="G", product_name="Bag", source_site="s1",
                       price_original=1, currency="EUR", price_jpy=10000, exchange_rate=1,
                       scraped_at=datetime(2026, 5, 1)),
            BrandPrice(brand="G", product_name="Bag", source_site="s2",
                       price_original=1, currency="EUR", price_jpy=8000, exchange_rate=1,
                       scraped_at=datetime(2026, 5, 1)),
        ])
        db.session.commit()
        result = brand_price_service.get_price_comparison("G")
        assert len(result) == 1
        assert result[0]["cheapest_jpy"] == 8000
        assert result[0]["cheapest_site"] == "s2"

    def test_get_price_comparison_empty(self, app):
        result = brand_price_service.get_price_comparison("None")
        assert result == []