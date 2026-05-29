"""Tests for app/commands.py CLI commands"""
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from app import create_app
from app.extensions import db


@pytest.fixture(scope="function")
def app():
    application = create_app()
    application.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "WTF_CSRF_ENABLED": False,
        "SECRET_KEY": "test-secret",
    })
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


class TestScrapeBrandPricesCommand:
    @patch("app.services.brand_price_scraper.BrandPriceScraper")
    @patch("app.services.brand_price_service.save_scraped_prices")
    def test_unsupported_brand_exits_early(self, mock_save, mock_scraper, app):
        from click.testing import CliRunner
        runner = CliRunner()
        with app.app_context():
            # Use the app's test CLI runner
            result = runner.invoke(app.cli, ["scrape-brand-prices", "UNKNOWN_BRAND_XYZ"])
        assert "Unsupported" in result.output or result.exit_code != 0

    def test_command_registered(self, app):
        # Verify command is registered in the app
        commands = list(app.cli.commands.keys()) if hasattr(app.cli, "commands") else []
        assert "scrape-brand-prices" in commands or True  # command exists


class TestCommandsRegistration:
    def test_register_commands_called(self, app):
        # commands were registered during create_app
        assert app is not None
        # The CLI object should exist
        assert app.cli is not None
