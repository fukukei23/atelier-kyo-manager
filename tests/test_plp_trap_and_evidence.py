"""Tests for plp_trap and plp_evidence modules."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.browser.plp_evidence import save_materialize_failure_evidence


class TestPlpTrapMixin:
    def _make_mixin(self, nav_driver=None):
        from app.agents.browser.plp_trap import PlpTrapMixin

        class ConcreteTrap(PlpTrapMixin):
            pass

        obj = ConcreteTrap()
        obj.page = MagicMock()
        obj.site_config = {"trap_detection": {"trap_url_patterns": ["/legal"]}}
        obj.logger = MagicMock()
        obj._navigation_driver = nav_driver
        return obj

    def test_is_trap_page_with_nav_driver(self):
        nav = MagicMock()
        nav._looks_like_trap_or_legal.return_value = True
        m = self._make_mixin(nav_driver=nav)
        assert m._is_trap_page("https://example.com/legal", {}) is True
        nav._looks_like_trap_or_legal.assert_called_once_with("https://example.com/legal", m.site_config)

    def test_is_trap_page_without_nav_driver(self):
        m = self._make_mixin(nav_driver=None)
        with patch("app.agents.browser.navigation_driver.NavigationDriver") as MockNav:
            mock_instance = MagicMock()
            mock_instance._looks_like_trap_or_legal.return_value = False
            MockNav.return_value = mock_instance
            result = m._is_trap_page("https://example.com/product", {})
            assert result is False

    def test_looks_like_trap_or_legal_legacy_wrapper(self):
        nav = MagicMock()
        nav._looks_like_trap_or_legal.return_value = True
        m = self._make_mixin(nav_driver=nav)
        assert m._looks_like_trap_or_legal("https://example.com/legal") is True

    @pytest.mark.asyncio
    async def test_recover_from_trap_go_back_success(self):
        m = self._make_mixin()
        m.page.url = "https://example.com/product"
        m.page.go_back = AsyncMock()
        m.page.wait_for_timeout = AsyncMock()
        trap_config = {"recovery_actions": [{"action": "go_back", "max_attempts": 1}]}
        # After go_back, page is no longer a trap
        m._is_trap_page = MagicMock(return_value=False)
        result = await m._recover_from_trap(target_url="https://example.com/product", trap_config=trap_config)
        assert result is True

    @pytest.mark.asyncio
    async def test_recover_from_trap_goto_target_success(self):
        m = self._make_mixin()
        m.page.url = "https://example.com/product"
        m.page.goto = AsyncMock()
        m.page.wait_for_timeout = AsyncMock()
        trap_config = {"recovery_actions": [{"action": "goto_target", "max_attempts": 1}]}
        # After goto_target, page is no longer a trap
        m._is_trap_page = MagicMock(return_value=False)
        result = await m._recover_from_trap(target_url="https://example.com/product", trap_config=trap_config)
        assert result is True
        m.page.goto.assert_called_once()

    @pytest.mark.asyncio
    async def test_recover_from_trap_all_actions_fail(self):
        m = self._make_mixin()
        m.page.url = "https://example.com/legal"
        m.page.go_back = AsyncMock()
        m.page.goto = AsyncMock()
        m.page.reload = AsyncMock()
        m.page.wait_for_timeout = AsyncMock()
        trap_config = {
            "recovery_actions": [
                {"action": "go_back", "max_attempts": 1},
                {"action": "goto_target", "max_attempts": 1},
                {"action": "reload", "max_attempts": 1},
            ]
        }
        m._is_trap_page = MagicMock(return_value=True)
        with patch("app.agents.browser.navigation_driver.NavigationDriver") as MockNav:
            mock_nav = MagicMock()
            mock_nav._force_plp_recover = AsyncMock()
            MockNav.return_value = mock_nav
            result = await m._recover_from_trap(target_url="https://example.com/product", trap_config=trap_config)
            assert result is False

    @pytest.mark.asyncio
    async def test_recover_from_trap_empty_actions(self):
        m = self._make_mixin()
        with patch("app.agents.browser.navigation_driver.NavigationDriver") as MockNav:
            mock_nav = MagicMock()
            mock_nav._force_plp_recover = AsyncMock()
            MockNav.return_value = mock_nav
            result = await m._recover_from_trap(target_url="", trap_config={"recovery_actions": []})
            assert result is False

    @pytest.mark.asyncio
    async def test_recover_from_trap_exception_in_action(self):
        m = self._make_mixin()
        m.page.go_back = AsyncMock(side_effect=Exception("Navigation failed"))
        m.page.wait_for_timeout = AsyncMock()
        trap_config = {"recovery_actions": [{"action": "go_back", "max_attempts": 1}]}
        m._is_trap_page = MagicMock(return_value=True)
        with patch("app.agents.browser.navigation_driver.NavigationDriver") as MockNav:
            mock_nav = MagicMock()
            mock_nav._force_plp_recover = AsyncMock()
            MockNav.return_value = mock_nav
            result = await m._recover_from_trap(target_url="", trap_config=trap_config)
            assert result is False


class TestSaveMaterializeFailureEvidence:
    @pytest.mark.asyncio
    async def test_saves_dom_snapshot(self, tmp_path):
        page = MagicMock()
        page.url = "https://example.com/plp"
        page.content = AsyncMock(return_value="<html><body>Test</body></html>")
        page.locator = MagicMock()
        page.locator.return_value.count = AsyncMock(return_value=0)
        page.screenshot = AsyncMock()
        page.title = AsyncMock(return_value="Test Page")
        page.context = MagicMock()
        page.context.cookies = AsyncMock(return_value=[])
        page.evaluate = AsyncMock(return_value=0)

        run_context = MagicMock()
        run_context.get_path = lambda name: str(tmp_path / name)

        logger = MagicMock()
        plp_config = {"product_tiles": [".tile"], "product_link": []}

        await save_materialize_failure_evidence(
            page=page,
            run_context=run_context,
            tile_selector_str=".tile",
            plp_config=plp_config,
            logger=logger,
        )

        dom_file = tmp_path / "plp_failure_dom.html"
        assert dom_file.exists()
        assert "Test" in dom_file.read_text()

    @pytest.mark.asyncio
    async def test_saves_selector_counts(self, tmp_path):
        page = MagicMock()
        page.url = "https://example.com/plp"
        page.content = AsyncMock(return_value="<html></html>")

        mock_locator = MagicMock()
        mock_locator.count = AsyncMock(return_value=5)
        page.locator = MagicMock(return_value=mock_locator)

        page.screenshot = AsyncMock()
        page.title = AsyncMock(return_value="Test")
        page.context = MagicMock()
        page.context.cookies = AsyncMock(return_value=[])
        page.evaluate = AsyncMock(return_value=0)

        run_context = MagicMock()
        run_context.get_path = lambda name: str(tmp_path / name)

        logger = MagicMock()
        plp_config = {"product_tiles": [".tile"], "product_link": []}

        await save_materialize_failure_evidence(
            page=page,
            run_context=run_context,
            tile_selector_str=".tile",
            plp_config=plp_config,
            logger=logger,
        )

        counts_file = tmp_path / "plp_failure_selector_counts.json"
        assert counts_file.exists()
        data = json.loads(counts_file.read_text())
        assert data["url"] == "https://example.com/plp"
        assert data["selector_counts"][".tile"] == 5

    @pytest.mark.asyncio
    async def test_no_error_when_run_context_is_none(self):
        page = MagicMock()
        page.url = "https://example.com/plp"
        page.content = AsyncMock(return_value="<html></html>")
        page.locator = MagicMock()
        page.locator.return_value.count = AsyncMock(return_value=0)
        page.screenshot = AsyncMock()
        page.title = AsyncMock(return_value="Test")
        page.context = MagicMock()
        page.context.cookies = AsyncMock(return_value=[])
        page.evaluate = AsyncMock(return_value=0)

        logger = MagicMock()
        plp_config = {"product_tiles": [], "product_link": []}

        await save_materialize_failure_evidence(
            page=page,
            run_context=None,
            tile_selector_str="",
            plp_config=plp_config,
            logger=logger,
        )
        # Should not raise

    @pytest.mark.asyncio
    async def test_handles_page_content_exception(self, tmp_path):
        page = MagicMock()
        page.url = "https://example.com/plp"
        page.content = AsyncMock(side_effect=Exception("Page crashed"))
        page.locator = MagicMock()
        page.locator.return_value.count = AsyncMock(side_effect=Exception("Crashed"))
        page.screenshot = AsyncMock(side_effect=Exception("Crashed"))

        run_context = MagicMock()
        run_context.get_path = lambda name: str(tmp_path / name)

        logger = MagicMock()
        plp_config = {"product_tiles": [], "product_link": []}

        await save_materialize_failure_evidence(
            page=page,
            run_context=run_context,
            tile_selector_str="",
            plp_config=plp_config,
            logger=logger,
        )
        # Should not raise - best effort
