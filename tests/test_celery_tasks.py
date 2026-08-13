"""Tests for Celery task infrastructure and scrape tasks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestCeleryApp:
    """Celery アプリ定義のテスト."""

    def test_celery_app_imports(self):
        from app.core.celery_app import celery

        assert celery.main == "atelier_kyo"

    def test_celery_config_has_broker(self):
        from app.core.celery_app import celery

        assert "redis" in celery.conf.broker_url

    def test_celery_config_has_backend(self):
        from app.core.celery_app import celery

        assert "redis" in celery.conf.result_backend

    def test_make_celery_creates_instance(self):
        from app.core.celery_app import make_celery

        c = make_celery()
        assert c is not None
        assert c.main == "atelier_kyo"

    def test_all_app_tasks_registered(self):
        """worker 起動相当: include 指定モジュールが import され 5 タスク全て登録される.

        新規タスクモジュール追加時は celery_app.py の include にも追記すること
        （autodiscover は本プロジェクト構造 app/tasks/<name>_tasks.py では動かない）。
        """
        from app.core.celery_app import celery

        celery.loader.import_default_modules()
        expected = {
            "scrape_brand_prices",
            "scrape_sale_prices",
            "run_ssense_buyma_pipeline",
            "monitor_prices_periodic",
            "monitor_prices_single",
        }
        registered = {t for t in celery.tasks if not t.startswith("celery.")}
        assert expected <= registered, f"未登録タスク: {expected - registered}"

    def test_beat_schedule_tasks_registered(self):
        """beat_schedule の全 task 名が celery.tasks に実在する（孤儿タスク名の再発防止）."""
        from app.core.celery_app import celery

        celery.loader.import_default_modules()
        beat_names = {v["task"] for v in celery.conf.beat_schedule.values()}
        registered = set(celery.tasks)
        missing = beat_names - registered
        assert not missing, f"beat が送出するタスクが未登録: {missing}"


class TestScrapeTasks:
    """スクレイピングタスクのテスト."""

    @patch("app.services.brand_price_scraper.BrandPriceScraper")
    @patch("app.services.brand_price_service.save_scraped_prices")
    def test_scrape_brand_prices_success(self, mock_save, mock_scraper_cls):
        from app.tasks.scrape_tasks import scrape_brand_prices

        mock_scraper = MagicMock()
        mock_scraper.scrape.return_value = [{"product_name": "Test Bag"}]
        mock_scraper_cls.return_value = mock_scraper
        mock_save.return_value = 1

        result = scrape_brand_prices("Gucci")

        assert result["brand"] == "Gucci"
        assert result["saved"] == 1
        mock_scraper.scrape.assert_called_once_with(brand="Gucci", sites=None)

    @patch("app.services.brand_price_scraper.BrandPriceScraper")
    @patch("app.services.brand_price_service.save_scraped_prices")
    def test_scrape_brand_prices_no_results(self, mock_save, mock_scraper_cls):
        from app.tasks.scrape_tasks import scrape_brand_prices

        mock_scraper = MagicMock()
        mock_scraper.scrape.return_value = []
        mock_scraper_cls.return_value = mock_scraper

        result = scrape_brand_prices("Prada")

        assert result["brand"] == "Prada"
        assert result["saved"] == 0
        assert result["error"] == "no_results"

    @patch("app.services.brand_price_scraper.BrandPriceScraper")
    @patch("app.services.brand_price_service.save_scraped_prices")
    def test_scrape_brand_prices_error_sanitized(self, mock_save, mock_scraper_cls):
        from app.tasks.scrape_tasks import scrape_brand_prices

        mock_scraper = MagicMock()
        mock_scraper.scrape.side_effect = RuntimeError("connection failed /var/secret")
        mock_scraper_cls.return_value = mock_scraper

        result = scrape_brand_prices("Gucci")

        assert result["saved"] == 0
        assert result["error"] == "scrape_error"
        # 機密情報が漏洩しないことを確認
        assert "/var/" not in result["error"]
        assert "connection" not in result["error"]

    @patch("app.services.sale_scraper.SaleScraper")
    @patch("app.services.brand_price_service.save_scraped_prices")
    def test_scrape_sale_prices_success(self, mock_save, mock_scraper_cls):
        from app.tasks.scrape_tasks import scrape_sale_prices

        mock_scraper = MagicMock()
        mock_scraper.scrape.return_value = [{"product_name": "Sale Bag"}]
        mock_scraper_cls.return_value = mock_scraper
        mock_save.return_value = 3

        result = scrape_sale_prices("Valentino", "shoes")

        assert result["brand"] == "Valentino"
        assert result["saved"] == 3

    @patch("app.services.sale_scraper.SaleScraper")
    @patch("app.services.brand_price_service.save_scraped_prices")
    def test_scrape_sale_prices_no_results(self, mock_save, mock_scraper_cls):
        from app.tasks.scrape_tasks import scrape_sale_prices

        mock_scraper = MagicMock()
        mock_scraper.scrape.return_value = []
        mock_scraper_cls.return_value = mock_scraper

        result = scrape_sale_prices("Chloe")

        assert result["saved"] == 0
        assert result["error"] == "no_results"


class TestTaskStatusRoute:
    """GET /api/tasks/<task_id>/status のテスト."""

    def test_returns_forbidden_for_unknown_task(self, routes_auth_client):
        """セッションにないtask_idは403."""
        resp = routes_auth_client.get("/api/tasks/unknown-task-id/status")
        assert resp.status_code == 403
        data = resp.get_json()
        assert data["state"] == "FORBIDDEN"

    def test_returns_unavailable_when_no_celery(self, routes_app, routes_auth_client):
        """Redis未接続でもUNAVAILABLEでエラーにならない."""
        with routes_app.test_request_context():
            with routes_auth_client.session_transaction() as sess:
                sess["pending_tasks"] = ["test-task-id"]
            routes_app.celery = None
            resp = routes_auth_client.get("/api/tasks/test-task-id/status")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["state"] == "UNAVAILABLE"
