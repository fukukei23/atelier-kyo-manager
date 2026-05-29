"""Services extra coverage tests for untested service files."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from app import create_app
from app.extensions import db
from app.models.brand_price import BrandPrice
from app.models.listing_template import ListingTemplate
from app.models.product import Product
from app.services import brand_price_service
from app.services.notification_service import NotificationService
from app.services.template_service import _fallback_template, generate_buyma_csv


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
        db.session.remove()
        db.drop_all()


# =====================================================================
# NotificationService
# =====================================================================
class TestNotificationService:
    def test_init_no_app(self):
        svc = NotificationService()
        assert svc.webhook_url is None

    def test_init_with_app(self):
        flask_app = create_app()
        flask_app.config["SLACK_WEBHOOK_URL"] = "https://hooks.slack.com/test"
        svc = NotificationService(flask_app)
        assert svc.webhook_url == "https://hooks.slack.com/test"

    def test_init_app_method(self):
        flask_app = create_app()
        flask_app.config["SLACK_WEBHOOK_URL"] = "https://hooks.slack.com/test2"
        svc = NotificationService()
        svc.init_app(flask_app)
        assert svc.webhook_url == "https://hooks.slack.com/test2"

    def test_send_no_webhook_url(self):
        svc = NotificationService()
        result = svc.send("test message")
        assert result["success"] is False
        assert "未設定" in result["error"]

    def test_send_with_channel(self):
        svc = NotificationService()
        with patch("app.services.notification_service.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()
            svc.webhook_url = "https://hooks.slack.com/test"
            result = svc.send("hello", channel="#general")
            assert result["success"] is True
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["channel"] == "#general"

    def test_send_network_error(self):
        svc = NotificationService()
        import requests as req
        with patch("app.services.notification_service.requests.post") as mock_post:
            mock_post.side_effect = req.exceptions.ConnectionError("network fail")
            svc.webhook_url = "https://hooks.slack.com/test"
            result = svc.send("hello")
            assert result["success"] is False
            assert "network fail" in result["error"]

    def test_send_order_status_error(self):
        svc = NotificationService()
        with patch("app.services.notification_service.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()
            svc.webhook_url = "https://hooks.slack.com/test"
            result = svc.send_order_status("B-001", "pending", "error", product_name="Shoes", error="Timeout")
            assert result["success"] is True
            payload = mock_post.call_args[1]["json"]
            assert "エラー" in payload["text"]

    def test_send_order_status_same_status(self):
        svc = NotificationService()
        with patch("app.services.notification_service.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()
            svc.webhook_url = "https://hooks.slack.com/test"
            result = svc.send_order_status("B-001", "shipped", "shipped", product_name="Bag")
            assert result["success"] is True
            payload = mock_post.call_args[1]["json"]
            assert "ステータス確認" in payload["text"]

    def test_send_order_status_transition(self):
        svc = NotificationService()
        with patch("app.services.notification_service.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()
            svc.webhook_url = "https://hooks.slack.com/test"
            result = svc.send_order_status("B-001", "pending", "shipped", product_name="Wallet")
            assert result["success"] is True
            payload = mock_post.call_args[1]["json"]
            assert "pending" in payload["text"]
            assert "shipped" in payload["text"]

    def test_send_pipeline_result_success(self):
        svc = NotificationService()
        with patch("app.services.notification_service.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()
            svc.webhook_url = "https://hooks.slack.com/test"
            result = svc.send_pipeline_result("Test Product", "success", 12.5)
            assert result["success"] is True
            payload = mock_post.call_args[1]["json"]
            assert "完了" in payload["text"]

    def test_send_pipeline_result_partial(self):
        svc = NotificationService()
        with patch("app.services.notification_service.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()
            svc.webhook_url = "https://hooks.slack.com/test"
            result = svc.send_pipeline_result("Test Product", "partial", 8.0, errors="Image failed")
            assert result["success"] is True
            payload = mock_post.call_args[1]["json"]
            assert "部分成功" in payload["text"]

    def test_send_pipeline_result_failed(self):
        svc = NotificationService()
        with patch("app.services.notification_service.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()
            svc.webhook_url = "https://hooks.slack.com/test"
            result = svc.send_pipeline_result("Test Product", "failed", 5.0, errors="Crash")
            assert result["success"] is True
            payload = mock_post.call_args[1]["json"]
            assert "失敗" in payload["text"]

    def test_send_pipeline_result_unknown_status(self):
        svc = NotificationService()
        with patch("app.services.notification_service.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()
            svc.webhook_url = "https://hooks.slack.com/test"
            result = svc.send_pipeline_result("Test Product", "weird", 1.0)
            assert result["success"] is True
            payload = mock_post.call_args[1]["json"]
            assert "不明" in payload["text"]


# =====================================================================
# TemplateService
# =====================================================================
class TestTemplateService:
    def test_fallback_template(self, app):
        with app.app_context():
            t = _fallback_template()
            assert "{{brand}}" in t.template_text
            assert "{{productName}}" in t.template_text
            assert "{{color}}" in t.template_text
            assert "{{size}}" in t.template_text

    def test_generate_buyma_csv_basic(self, app):
        with app.app_context():
            prod = Product(
                name="Test Bag",
                brand="Gucci",
                color="Black",
                size="M",
                material="Leather",
                retail_price=100000,
                selling_price=150000,
                source_region="IT",
                description="Beautiful bag",
                item_category="bag",
                purchase_price=80000,
            )
            csv_text, skipped = generate_buyma_csv([prod])
            assert skipped == 0
            assert "﻿" in csv_text  # BOM
            assert "Gucci" in csv_text
            assert "Test Bag" in csv_text

    def test_generate_buyma_csv_skip_no_description(self, app):
        with app.app_context():
            prod = Product(
                name="NoDesc",
                brand="Brand",
                purchase_price=1000,
                selling_price=2000,
            )
            csv_text, skipped = generate_buyma_csv([prod])
            assert skipped == 1
            assert "NoDesc" not in csv_text

    def test_generate_buyma_csv_with_processed_images(self, app):
        with app.app_context():
            prod = Product(
                name="ImgProd",
                brand="Brand",
                description="With images",
                purchase_price=1000,
                selling_price=2000,
                processed_images=json.dumps(["img1.jpg", "img2.jpg"]),
            )
            csv_text, skipped = generate_buyma_csv([prod])
            assert skipped == 0
            assert "img1.jpg" in csv_text
            assert "img2.jpg" in csv_text

    def test_generate_buyma_csv_processed_images_fallback(self, app):
        with app.app_context():
            prod = Product(
                name="FallbackImg",
                brand="Brand",
                description="test",
                purchase_price=1000,
                selling_price=2000,
                image_url="original.jpg",
                processed_images="invalid json",
            )
            csv_text, _ = generate_buyma_csv([prod])
            assert "original.jpg" in csv_text

    def test_generate_buyma_csv_empty_list(self, app):
        with app.app_context():
            csv_text, skipped = generate_buyma_csv([])
            assert skipped == 0
            assert "﻿" in csv_text  # BOM + headers only

    def test_generate_buyma_csv_multiple_products(self, app):
        with app.app_context():
            p1 = Product(name="P1", brand="B1", description="d1", purchase_price=100, selling_price=200)
            p2 = Product(name="P2", brand="B2", description="d2", purchase_price=200, selling_price=400)
            p3 = Product(name="P3", brand="B3", purchase_price=300, selling_price=600)  # no desc
            csv_text, skipped = generate_buyma_csv([p1, p2, p3])
            assert skipped == 1
            assert "P1" in csv_text
            assert "P2" in csv_text
            assert "P3" not in csv_text

    def test_generate_buyma_csv_headers(self, app):
        with app.app_context():
            csv_text, _ = generate_buyma_csv([])
            reader = csv.reader(io.StringIO(csv_text.lstrip("﻿")))
            headers = next(reader)
            assert "商品名" in headers
            assert "ブランド" in headers
            assert "説明文" in headers


# =====================================================================
# BrandPriceService
# =====================================================================
class TestBrandPriceService:
    def test_save_scraped_prices_new(self, app):
        with app.app_context():
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
        with app.app_context():
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
        with app.app_context():
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
        with app.app_context():
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
        with app.app_context():
            assert brand_price_service.get_available_brands() == []

    def test_cleanup_old_records(self, app):
        with app.app_context():
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
        with app.app_context():
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
        with app.app_context():
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
        with app.app_context():
            result = brand_price_service.get_last_scraped_at("None")
            assert result is None

    def test_cleanup_buyma_cache(self, app):
        with app.app_context():
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
        with app.app_context():
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
        with app.app_context():
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
        with app.app_context():
            result = brand_price_service.get_price_comparison("None")
            assert result == []
