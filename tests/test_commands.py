"""Tests for app/commands.py - Issue #87 カバレッジ向上"""
from unittest.mock import MagicMock, patch

import pytest

from app import create_app
from app.commands import register_commands


@pytest.fixture(scope="function")
def app():
    application = create_app()
    application.config["TESTING"] = True
    register_commands(application)
    return application


@pytest.fixture(scope="function")
def runner(app):
    return app.test_cli_runner()


# ── register_commands ─────────────────────────────────────────────────────────

class TestRegisterCommands:
    def test_register_commands_adds_cli_command(self, app):
        """scrape-brand-prices コマンドが登録されている"""
        commands = [cmd.name for cmd in app.cli.commands.values()]
        assert "scrape-brand-prices" in commands

    def test_register_commands_does_not_raise(self):
        """register_commands は例外なく完了する"""
        application = create_app()
        application.config["TESTING"] = True
        register_commands(application)  # should not raise


# ── scrape-brand-prices コマンド ──────────────────────────────────────────────

class TestScrapeBrandPricesCommand:
    def test_unsupported_brand_prints_error(self, runner):
        """サポートされていないブランドを指定すると警告メッセージが出る"""
        result = runner.invoke(args=["scrape-brand-prices", "UNKNOWN_BRAND_XYZ"])
        assert result.exit_code == 0
        assert "Unsupported brand" in result.output
        assert "UNKNOWN_BRAND_XYZ" in result.output

    def test_supported_brand_no_results(self, runner):
        """サポートブランドで結果なしの場合に適切なメッセージが出る"""
        with patch("app.services.brand_price_scraper.BrandPriceScraper") as MockScraper:
            mock_instance = MagicMock()
            mock_instance.scrape.return_value = []
            MockScraper.return_value = mock_instance
            result = runner.invoke(args=["scrape-brand-prices", "Prada"])
        assert result.exit_code == 0
        assert "No results for Prada" in result.output

    def test_supported_brand_with_results(self, runner):
        """サポートブランドで結果ありの場合に Saved N records が出る"""
        fake_results = [{"brand": "Prada", "price": 100000}]
        with patch("app.services.brand_price_scraper.BrandPriceScraper") as MockScraper, \
             patch("app.services.brand_price_service.save_scraped_prices", return_value=1) as mock_save:
            mock_instance = MagicMock()
            mock_instance.scrape.return_value = fake_results
            MockScraper.return_value = mock_instance
            result = runner.invoke(args=["scrape-brand-prices", "Prada"])
        assert result.exit_code == 0
        assert "Scraping Prada" in result.output
        assert "Saved 1 records" in result.output
        mock_save.assert_called_once_with(fake_results)

    def test_scraping_message_shown(self, runner):
        """実行時に 'Scraping BRAND...' メッセージが表示される"""
        with patch("app.services.brand_price_scraper.BrandPriceScraper") as MockScraper:
            mock_instance = MagicMock()
            mock_instance.scrape.return_value = []
            MockScraper.return_value = mock_instance
            result = runner.invoke(args=["scrape-brand-prices", "Gucci"])
        assert "Scraping Gucci" in result.output

    def test_headless_true_passed_to_scraper(self, runner):
        """BrandPriceScraper は headless=True で初期化される"""
        with patch("app.services.brand_price_scraper.BrandPriceScraper") as MockScraper:
            mock_instance = MagicMock()
            mock_instance.scrape.return_value = []
            MockScraper.return_value = mock_instance
            runner.invoke(args=["scrape-brand-prices", "Prada"])
        MockScraper.assert_called_once_with(headless=True)
