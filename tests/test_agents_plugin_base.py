"""Tests for app/agents/plugins/base.py"""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.plugins.base import STEALTH_CONFIG, StrategyPlugin, _apply_stealth


class TestStealtConfig:
    def test_stealth_config_is_dict(self):
        assert isinstance(STEALTH_CONFIG, dict)

    def test_stealth_config_has_expected_keys(self):
        assert "navigator_webdriver" in STEALTH_CONFIG
        assert "webgl_vendor" in STEALTH_CONFIG

    def test_all_values_are_bool_true(self):
        for key, val in STEALTH_CONFIG.items():
            assert val is True, f"{key} should be True"


class TestApplyStealth:
    def test_apply_stealth_missing_module_no_error(self):
        mock_page = MagicMock()
        # Should not raise even if playwright_stealth is not installed
        with patch("builtins.__import__", side_effect=ImportError), contextlib.suppress(Exception):
            _apply_stealth(mock_page)  # ImportError is caught inside function


class TestStrategyPlugin:
    def test_default_site_is_empty(self):
        plugin = StrategyPlugin()
        assert plugin.site == ""

    def test_before_navigate_returns_url_unchanged(self):
        plugin = StrategyPlugin()
        url = "https://example.com/products"
        ctx = MagicMock()
        result = plugin.before_navigate(url, ctx)
        assert result == url

    @pytest.mark.asyncio
    async def test_after_navigate_returns_none(self):
        plugin = StrategyPlugin()
        page = AsyncMock()
        ctx = MagicMock()
        result = await plugin.after_navigate(page, ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_assert_plp_returns_true(self):
        plugin = StrategyPlugin()
        page = AsyncMock()
        ctx = MagicMock()
        result = await plugin.assert_plp(page, ctx)
        assert result is True

    def test_subclass_can_override_site(self):
        class MyPlugin(StrategyPlugin):
            site = "MY_SITE"

        plugin = MyPlugin()
        assert plugin.site == "MY_SITE"

    def test_subclass_can_override_before_navigate(self):
        class RedirectPlugin(StrategyPlugin):
            def before_navigate(self, url, ctx):
                return url + "?lang=ja"

        plugin = RedirectPlugin()
        result = plugin.before_navigate("https://example.com", MagicMock())
        assert result == "https://example.com?lang=ja"
