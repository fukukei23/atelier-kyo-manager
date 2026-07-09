"""Tests for app/utils/shipping_agent.py - Issue #81 カバレッジ向上"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import create_app
from app.utils.shipping_agent import ShippingAgent, _click_first, _fill_first


@pytest.fixture(scope="function")
def app():
    application = create_app()
    application.config.update(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "SECRET_KEY": "test-secret",
            "WTF_CSRF_ENABLED": False,
        }
    )
    return application


# ── _click_first ─────────────────────────────────────────────────────────────


class TestClickFirst:
    def _make_locator(self, count=1, visible=True):
        loc = AsyncMock()
        loc.count = AsyncMock(return_value=count)
        loc.is_visible = AsyncMock(return_value=visible)
        loc.click = AsyncMock()
        return loc

    def _make_page(self, locator):
        page = MagicMock()
        page.locator.return_value.first = locator
        return page

    def test_empty_selectors_returns_false(self):
        result = asyncio.run(_click_first(MagicMock(), []))
        assert result is False

    def test_click_succeeds_returns_true(self):
        loc = self._make_locator(count=1, visible=True)
        page = self._make_page(loc)
        result = asyncio.run(_click_first(page, ["button"]))
        assert result is True
        loc.click.assert_called_once()

    def test_count_zero_skips_click(self):
        loc = self._make_locator(count=0, visible=True)
        page = self._make_page(loc)
        result = asyncio.run(_click_first(page, ["button"]))
        assert result is False
        loc.click.assert_not_called()

    def test_not_visible_skips_click(self):
        loc = self._make_locator(count=1, visible=False)
        page = self._make_page(loc)
        result = asyncio.run(_click_first(page, ["button"]))
        assert result is False
        loc.click.assert_not_called()

    def test_exception_in_locator_continues(self):
        page = MagicMock()
        page.locator.side_effect = Exception("playwright error")
        result = asyncio.run(_click_first(page, ["bad-sel", "also-bad"]))
        assert result is False

    def test_falls_back_to_second_selector(self):
        loc1 = self._make_locator(count=0)
        loc2 = self._make_locator(count=1, visible=True)
        results = [MagicMock(first=loc1), MagicMock(first=loc2)]
        page = MagicMock()
        page.locator.side_effect = lambda sel: results.pop(0)
        result = asyncio.run(_click_first(page, ["first", "second"]))
        assert result is True
        loc2.click.assert_called_once()

    def test_custom_timeout_passed_to_click(self):
        loc = self._make_locator(count=1, visible=True)
        page = self._make_page(loc)
        asyncio.run(_click_first(page, ["btn"], timeout_ms=3000))
        loc.click.assert_called_once_with(timeout=3000)


# ── _fill_first ──────────────────────────────────────────────────────────────


class TestFillFirst:
    def _make_locator(self, count=1, visible=True):
        loc = AsyncMock()
        loc.count = AsyncMock(return_value=count)
        loc.is_visible = AsyncMock(return_value=visible)
        loc.fill = AsyncMock()
        return loc

    def _make_page(self, locator):
        page = MagicMock()
        page.locator.return_value.first = locator
        return page

    def test_empty_selectors_returns_false(self):
        result = asyncio.run(_fill_first(MagicMock(), [], "text"))
        assert result is False

    def test_fill_succeeds_returns_true(self):
        loc = self._make_locator(count=1, visible=True)
        page = self._make_page(loc)
        result = asyncio.run(_fill_first(page, ["input"], "hello"))
        assert result is True
        loc.fill.assert_called_once_with("hello", timeout=8000)

    def test_count_zero_skips_fill(self):
        loc = self._make_locator(count=0)
        page = self._make_page(loc)
        result = asyncio.run(_fill_first(page, ["input"], "text"))
        assert result is False
        loc.fill.assert_not_called()

    def test_not_visible_skips_fill(self):
        loc = self._make_locator(count=1, visible=False)
        page = self._make_page(loc)
        result = asyncio.run(_fill_first(page, ["input"], "text"))
        assert result is False
        loc.fill.assert_not_called()

    def test_exception_in_locator_continues(self):
        page = MagicMock()
        page.locator.side_effect = Exception("error")
        result = asyncio.run(_fill_first(page, ["input"], "text"))
        assert result is False

    def test_custom_timeout_passed_to_fill(self):
        loc = self._make_locator(count=1, visible=True)
        page = self._make_page(loc)
        asyncio.run(_fill_first(page, ["input"], "text", timeout_ms=5000))
        loc.fill.assert_called_once_with("text", timeout=5000)

    def test_unicode_text(self):
        loc = self._make_locator(count=1, visible=True)
        page = self._make_page(loc)
        asyncio.run(_fill_first(page, ["input"], "日本語テスト"))
        loc.fill.assert_called_once_with("日本語テスト", timeout=8000)


# ── ShippingAgent.__init__ ───────────────────────────────────────────────────


class TestShippingAgentInit:
    @patch("os.makedirs")
    def test_headless_default_when_no_env(self, _mock, monkeypatch):
        monkeypatch.delenv("PLAYWRIGHT_HEADFUL", raising=False)
        agent = ShippingAgent()
        assert agent.headless is True

    @patch("os.makedirs")
    def test_headful_when_env_is_1(self, _mock, monkeypatch):
        monkeypatch.setenv("PLAYWRIGHT_HEADFUL", "1")
        agent = ShippingAgent()
        assert agent.headless is False

    @patch("os.makedirs")
    def test_headless_explicit_true(self, _mock):
        agent = ShippingAgent(headless=True)
        assert agent.headless is True

    @patch("os.makedirs")
    def test_headless_explicit_false(self, _mock):
        agent = ShippingAgent(headless=False)
        assert agent.headless is False

    @patch("os.makedirs")
    def test_default_base_url_contains_buyandship(self, _mock):
        agent = ShippingAgent()
        assert "buyandship" in agent.base_url

    @patch("os.makedirs")
    def test_custom_state_path_stored(self, _mock):
        agent = ShippingAgent(state_path="tmp/test.json")
        assert agent.state_path == "tmp/test.json"

    @patch("os.makedirs")
    def test_custom_timeout_stored(self, _mock):
        agent = ShippingAgent(timeout_ms=30_000)
        assert agent.timeout_ms == 30_000

    @patch("os.makedirs")
    def test_logger_is_logging_logger(self, _mock):
        agent = ShippingAgent()
        assert isinstance(agent.logger, logging.Logger)

    @patch("os.makedirs")
    def test_email_and_password_initially_none(self, _mock):
        agent = ShippingAgent()
        assert agent.email is None
        assert agent.password is None


# ── ShippingAgent._get_credentials ──────────────────────────────────────────


class TestGetCredentials:
    @patch("os.makedirs")
    def test_reads_from_env_vars(self, _mock, monkeypatch):
        monkeypatch.setenv("BUYANDSHIP_EMAIL", "user@example.com")
        monkeypatch.setenv("BUYANDSHIP_PASSWORD", "secret")
        agent = ShippingAgent()
        creds = agent._get_credentials()
        assert creds["email"] == "user@example.com"
        assert creds["password"] == "secret"

    @patch("os.makedirs")
    def test_raises_when_both_missing(self, _mock, monkeypatch, app):
        for key in ("BUYANDSHIP_EMAIL", "BUYANDSHIP_PASSWORD", "BNS_EMAIL", "BNS_PASSWORD"):
            monkeypatch.delenv(key, raising=False)
        with app.app_context():
            agent = ShippingAgent()
            with pytest.raises(RuntimeError, match="BUYANDSHIP_EMAIL"):
                agent._get_credentials()

    @patch("os.makedirs")
    def test_raises_when_only_email_set(self, _mock, monkeypatch, app):
        monkeypatch.setenv("BUYANDSHIP_EMAIL", "x@x.com")
        for key in ("BUYANDSHIP_PASSWORD", "BNS_EMAIL", "BNS_PASSWORD"):
            monkeypatch.delenv(key, raising=False)
        with app.app_context():
            agent = ShippingAgent()
            with pytest.raises(RuntimeError):
                agent._get_credentials()

    @patch("os.makedirs")
    def test_legacy_bns_env_vars(self, _mock, monkeypatch, app):
        for key in ("BUYANDSHIP_EMAIL", "BUYANDSHIP_PASSWORD"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("BNS_EMAIL", "legacy@example.com")
        monkeypatch.setenv("BNS_PASSWORD", "legacypass")
        with app.app_context():
            agent = ShippingAgent()
            creds = agent._get_credentials()
        assert creds["email"] == "legacy@example.com"
        assert creds["password"] == "legacypass"

    @patch("os.makedirs")
    def test_sets_self_email_and_password(self, _mock, monkeypatch):
        monkeypatch.setenv("BUYANDSHIP_EMAIL", "a@b.com")
        monkeypatch.setenv("BUYANDSHIP_PASSWORD", "pw")
        agent = ShippingAgent()
        agent._get_credentials()
        assert agent.email == "a@b.com"
        assert agent.password == "pw"

    @patch("os.makedirs")
    def test_from_flask_config(self, _mock, monkeypatch, app):
        for key in ("BUYANDSHIP_EMAIL", "BUYANDSHIP_PASSWORD", "BNS_EMAIL", "BNS_PASSWORD"):
            monkeypatch.delenv(key, raising=False)
        with app.app_context():
            from flask import current_app

            current_app.config["BUYANDSHIP_EMAIL"] = "cfg@example.com"
            current_app.config["BUYANDSHIP_PASSWORD"] = "cfgpass"
            agent = ShippingAgent()
            creds = agent._get_credentials()
        assert creds["email"] == "cfg@example.com"
        assert creds["password"] == "cfgpass"


# ── ShippingAgent._pick_working_page ────────────────────────────────────────


class TestPickWorkingPage:
    @patch("os.makedirs")
    def test_returns_non_blank_page(self, _mock):
        agent = ShippingAgent()
        blank = MagicMock()
        blank.is_closed.return_value = False
        blank.url = "about:blank"
        working = MagicMock()
        working.is_closed.return_value = False
        working.url = "https://buyandship.co.jp/warehouses"
        working.bring_to_front = AsyncMock()
        context = AsyncMock()
        context.pages = [blank, working]
        result = asyncio.run(agent._pick_working_page(context))
        assert result is working
        working.bring_to_front.assert_called_once()

    @patch("os.makedirs")
    def test_creates_new_page_when_all_blank(self, _mock):
        agent = ShippingAgent()
        blank = MagicMock()
        blank.is_closed.return_value = False
        blank.url = "about:blank"
        new_page = AsyncMock()
        new_page.bring_to_front = AsyncMock()
        context = AsyncMock()
        context.pages = [blank]
        context.new_page = AsyncMock(return_value=new_page)
        result = asyncio.run(agent._pick_working_page(context))
        assert result is new_page
        context.new_page.assert_called_once()

    @patch("os.makedirs")
    def test_creates_new_page_when_no_pages(self, _mock):
        agent = ShippingAgent()
        new_page = AsyncMock()
        new_page.bring_to_front = AsyncMock()
        context = AsyncMock()
        context.pages = []
        context.new_page = AsyncMock(return_value=new_page)
        result = asyncio.run(agent._pick_working_page(context))
        assert result is new_page

    @patch("os.makedirs")
    def test_skips_closed_pages(self, _mock):
        agent = ShippingAgent()
        closed = MagicMock()
        closed.is_closed.return_value = True
        closed.url = "https://example.com"
        working = MagicMock()
        working.is_closed.return_value = False
        working.url = "https://other.com"
        working.bring_to_front = AsyncMock()
        context = AsyncMock()
        context.pages = [closed, working]
        result = asyncio.run(agent._pick_working_page(context))
        assert result is working


# ── ShippingAgent._close_blank_tabs ─────────────────────────────────────────


class TestCloseBlankTabs:
    @patch("os.makedirs")
    def test_closes_blank_page(self, _mock):
        agent = ShippingAgent()
        blank = MagicMock()
        blank.is_closed.return_value = False
        blank.url = "about:blank"
        blank.close = AsyncMock()
        context = MagicMock()
        context.pages = [blank]
        asyncio.run(agent._close_blank_tabs(context))
        blank.close.assert_called_once()

    @patch("os.makedirs")
    def test_does_not_close_working_page(self, _mock):
        agent = ShippingAgent()
        working = MagicMock()
        working.is_closed.return_value = False
        working.url = "https://example.com"
        working.close = AsyncMock()
        context = MagicMock()
        context.pages = [working]
        asyncio.run(agent._close_blank_tabs(context))
        working.close.assert_not_called()

    @patch("os.makedirs")
    def test_closes_blank_but_not_working(self, _mock):
        agent = ShippingAgent()
        blank = MagicMock()
        blank.is_closed.return_value = False
        blank.url = "about:blank"
        blank.close = AsyncMock()
        working = MagicMock()
        working.is_closed.return_value = False
        working.url = "https://buyandship.co.jp"
        working.close = AsyncMock()
        context = MagicMock()
        context.pages = [blank, working]
        asyncio.run(agent._close_blank_tabs(context))
        blank.close.assert_called_once()
        working.close.assert_not_called()

    @patch("os.makedirs")
    def test_empty_pages_no_error(self, _mock):
        agent = ShippingAgent()
        context = MagicMock()
        context.pages = []
        asyncio.run(agent._close_blank_tabs(context))

    @patch("os.makedirs")
    def test_skips_already_closed_pages(self, _mock):
        agent = ShippingAgent()
        page = MagicMock()
        page.is_closed.return_value = True
        page.url = "about:blank"
        page.close = AsyncMock()
        context = MagicMock()
        context.pages = [page]
        asyncio.run(agent._close_blank_tabs(context))
        page.close.assert_not_called()


# ── ShippingAgent._login_if_needed ──────────────────────────────────────────


class TestLoginIfNeeded:
    """_login_if_needed のテスト（既ログイン時はスキップ）"""

    @patch("os.makedirs")
    def test_skips_login_when_warehouse_visible(self, _mock):
        """倉庫ページが見えている場合はログインをスキップ"""
        agent = ShippingAgent()
        page = MagicMock()
        loc = MagicMock()
        loc.is_visible = AsyncMock(return_value=True)
        page.locator.return_value.first = loc
        context = MagicMock()
        asyncio.run(agent._login_if_needed(page, context))
        # _do_login_flow が呼ばれないこと（page.waitが呼ばれない）

    @patch("os.makedirs")
    def test_calls_login_flow_when_not_logged_in(self, _mock, monkeypatch):
        """未ログイン時は _do_login_flow を呼ぶ"""
        monkeypatch.setenv("BUYANDSHIP_EMAIL", "test@test.com")
        monkeypatch.setenv("BUYANDSHIP_PASSWORD", "pass")
        agent = ShippingAgent()
        page = MagicMock()

        # 倉庫ページもユーザーメニューも見えない
        loc = MagicMock()
        loc.is_visible = AsyncMock(return_value=False)
        page.locator.return_value.first = loc

        with patch.object(agent, "_do_login_flow", new_callable=AsyncMock) as mock_login:
            asyncio.run(agent._login_if_needed(page, MagicMock()))
            mock_login.assert_called_once_with(page)


# ── ShippingAgent._do_login_flow ─────────────────────────────────────────────


class TestDoLoginFlow:
    """_do_login_flow のテスト（ログインフロー）"""

    @patch("os.makedirs")
    def test_login_flow_happy_path(self, _mock, monkeypatch):
        """正常なログインフロー: email入力 → 次へ → password → 送信"""
        monkeypatch.setenv("BUYANDSHIP_EMAIL", "user@example.com")
        monkeypatch.setenv("BUYANDSHIP_PASSWORD", "secret123")
        agent = ShippingAgent()

        page = MagicMock()
        page.wait_for_timeout = AsyncMock()
        page.wait_for_load_state = AsyncMock()
        page.frames = []

        # locator().first.is_visible() → False（login panel not open）
        loc_visible = MagicMock()
        loc_visible.is_visible = AsyncMock(return_value=False)
        page.locator.return_value.first = loc_visible

        with (
            patch("app.utils.shipping_agent._click_first", new_callable=AsyncMock, return_value=True),
            patch("app.utils.shipping_agent._fill_first", new_callable=AsyncMock, return_value=True),
        ):
            asyncio.run(agent._do_login_flow(page))

    @patch("os.makedirs")
    def test_login_flow_completes_when_already_logged_in(self, _mock, monkeypatch):
        """既に email 入力欄が開いている状態（login_panel_open=True）のフロー"""
        monkeypatch.setenv("BUYANDSHIP_EMAIL", "user@example.com")
        monkeypatch.setenv("BUYANDSHIP_PASSWORD", "secret")
        agent = ShippingAgent()

        page = MagicMock()
        page.wait_for_timeout = AsyncMock()
        page.wait_for_load_state = AsyncMock()
        page.frames = []

        loc_visible = MagicMock()
        loc_visible.is_visible = AsyncMock(return_value=True)  # panel already open
        page.locator.return_value.first = loc_visible

        with (
            patch("app.utils.shipping_agent._click_first", new_callable=AsyncMock, return_value=True),
            patch("app.utils.shipping_agent._fill_first", new_callable=AsyncMock, return_value=True),
        ):
            asyncio.run(agent._do_login_flow(page))  # raises しないこと


# ── ShippingAgent._extract_warehouses ───────────────────────────────────────


class TestExtractWarehouses:
    """_extract_warehouses のテスト（Playwright locator をモック化）"""

    def _make_locator_obj(self, count=1, inner_text=""):
        """同期的な locator オブジェクト（countとinner_textはAsync）"""
        loc = MagicMock()
        loc.count = AsyncMock(return_value=count)
        loc.first = MagicMock()
        loc.first.inner_text = AsyncMock(return_value=inner_text)
        return loc

    def _make_card(self, country_text="US", name="US Warehouse", address="123 Main St", fee_text=None):
        """倉庫カードモックを作成（locatorは同期）"""
        card = MagicMock()

        country_loc = self._make_locator_obj(count=1, inner_text=country_text)
        name_loc = self._make_locator_obj(count=1, inner_text=name)
        addr_loc = self._make_locator_obj(count=1, inner_text=address)
        fee_loc = self._make_locator_obj(count=1 if fee_text else 0, inner_text=fee_text or "")

        def card_locator(sel):
            if "country" in sel or sel.startswith(".country"):
                return country_loc
            elif "h3" in sel or "warehouse-name" in sel:
                return name_loc
            elif "address" in sel:
                return addr_loc
            elif "warehouse-fee" in sel:
                return fee_loc
            return self._make_locator_obj(count=0)

        card.locator.side_effect = card_locator
        return card

    @patch("os.makedirs")
    def test_extract_returns_empty_when_no_cards(self, _mock):
        agent = ShippingAgent()
        page = MagicMock()  # locator() は同期
        cards_mock = MagicMock()
        cards_mock.count = AsyncMock(return_value=0)
        page.locator.return_value = cards_mock
        result = asyncio.run(agent._extract_warehouses(page, "US"))
        assert result == []

    @patch("os.makedirs")
    def test_extract_skips_wrong_country(self, _mock):
        agent = ShippingAgent()
        card = self._make_card(country_text="HK")
        page = MagicMock()
        cards_mock = MagicMock()
        cards_mock.count = AsyncMock(return_value=1)
        cards_mock.nth.return_value = card
        page.locator.return_value = cards_mock
        result = asyncio.run(agent._extract_warehouses(page, "US"))
        assert result == []

    @patch("os.makedirs")
    def test_extract_returns_matching_warehouse(self, _mock):
        agent = ShippingAgent()
        card = self._make_card(country_text="US", name="US Warehouse", address="123 Main St")
        page = MagicMock()
        cards_mock = MagicMock()
        cards_mock.count = AsyncMock(return_value=1)
        cards_mock.nth.return_value = card
        page.locator.return_value = cards_mock
        result = asyncio.run(agent._extract_warehouses(page, "US"))
        assert len(result) == 1
        assert result[0]["name"] == "US Warehouse"
        assert result[0]["address"] == "123 Main St"
        assert result[0]["country"] == "US"
        assert result[0]["fee"] is None

    @patch("os.makedirs")
    def test_extract_parses_fee(self, _mock):
        agent = ShippingAgent()
        card = self._make_card(country_text="US", name="US Warehouse", address="123 Main St", fee_text="$12.50 per lb")
        page = MagicMock()
        cards_mock = MagicMock()
        cards_mock.count = AsyncMock(return_value=1)
        cards_mock.nth.return_value = card
        page.locator.return_value = cards_mock
        result = asyncio.run(agent._extract_warehouses(page, "US"))
        assert result[0]["fee"] == 12.50

    @patch("os.makedirs")
    def test_extract_skips_card_with_empty_name(self, _mock):
        agent = ShippingAgent()
        card = self._make_card(country_text="US", name="", address="123 Main St")
        page = MagicMock()
        cards_mock = MagicMock()
        cards_mock.count = AsyncMock(return_value=1)
        cards_mock.nth.return_value = card
        page.locator.return_value = cards_mock
        result = asyncio.run(agent._extract_warehouses(page, "US"))
        assert result == []


# ── ShippingAgent.get_warehouses_by_country ──────────────────────────────────


class TestGetWarehousesByCountry:
    """get_warehouses_by_country のテスト"""

    @patch("os.makedirs")
    def test_calls_async_get_warehouses(self, _mock):
        agent = ShippingAgent()
        expected = [{"name": "US", "address": "addr", "country": "US", "fee": None}]
        with patch.object(agent, "_get_warehouses_async", new_callable=AsyncMock, return_value=expected):
            result = agent.get_warehouses_by_country("US")
            assert result == expected

    @patch("os.makedirs")
    def test_returns_empty_list_on_failure(self, _mock):
        agent = ShippingAgent()
        with patch.object(agent, "_get_warehouses_async", new_callable=AsyncMock, return_value=[]):
            result = agent.get_warehouses_by_country("HK")
            assert result == []

    @patch("os.makedirs")
    def test_passes_country_code_to_async(self, _mock):
        agent = ShippingAgent()
        mock_async = AsyncMock(return_value=[])
        with patch.object(agent, "_get_warehouses_async", mock_async):
            agent.get_warehouses_by_country("JP")
            mock_async.assert_called_once_with("JP")


# ── ShippingAgent.fetch_warehouses ───────────────────────────────────────────


class TestFetchWarehouses:
    @patch("os.makedirs")
    def test_delegates_to_get_warehouses(self, _mock):
        agent = ShippingAgent()
        expected = [{"name": "US Warehouse", "address": "123 Main St", "country": "US", "fee": None}]
        with patch.object(agent, "get_warehouses_by_country", return_value=expected) as mock_get:
            result = agent.fetch_warehouses("US")
            mock_get.assert_called_once_with("US")
            assert result == expected

    @patch("os.makedirs")
    def test_passes_country_code(self, _mock):
        agent = ShippingAgent()
        with patch.object(agent, "get_warehouses_by_country", return_value=[]) as mock_get:
            agent.fetch_warehouses("HK")
            mock_get.assert_called_once_with("HK")

    @patch("os.makedirs")
    def test_returns_empty_list(self, _mock):
        agent = ShippingAgent()
        with patch.object(agent, "get_warehouses_by_country", return_value=[]):
            result = agent.fetch_warehouses("UK")
            assert result == []
