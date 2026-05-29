"""Tests for app/utils/shipping_agent.py"""

import os
from unittest.mock import MagicMock, patch

import pytest


# ---------- ShippingAgent class tests ----------


class TestShippingAgentClass:
    def test_class_exists(self):
        from app.utils.shipping_agent import ShippingAgent

        assert ShippingAgent is not None

    def test_instantiation(self):
        from app.utils.shipping_agent import ShippingAgent

        agent = ShippingAgent()
        assert agent is not None

    def test_default_headless(self):
        # 環境変数をクリアしてテスト
        with patch.dict(os.environ, {"PLAYWRIGHT_HEADFUL": ""}, clear=False):
            # unset PLAYWRIGHT_HEADFUL as well
            env_without = {k: v for k, v in os.environ.items() if k != "PLAYWRIGHT_HEADFUL"}
            with patch.dict(os.environ, env_without, clear=True):
                from app.utils.shipping_agent import ShippingAgent

                agent = ShippingAgent()
                assert agent.headless is True

    def test_explicit_headless_false(self):
        from app.utils.shipping_agent import ShippingAgent

        agent = ShippingAgent(headless=False)
        assert agent.headless is False

    def test_get_warehouses_method_exists(self):
        from app.utils.shipping_agent import ShippingAgent

        agent = ShippingAgent()
        assert callable(agent.get_warehouses_by_country)

    def test_fetch_warehouses_alias(self):
        from app.utils.shipping_agent import ShippingAgent

        agent = ShippingAgent()
        assert callable(agent.fetch_warehouses)


# ---------- Authentication source code check ----------


class TestShippingAgentAuth:
    def test_credentials_from_env(self):
        import inspect
        from app.utils.shipping_agent import ShippingAgent

        source = inspect.getsource(ShippingAgent._get_credentials)
        assert "BUYANDSHIP_EMAIL" in source
        assert "BUYANDSHIP_PASSWORD" in source

    def test_fallback_env_bns(self):
        import inspect
        from app.utils.shipping_agent import ShippingAgent

        source = inspect.getsource(ShippingAgent._get_credentials)
        # Should have fallback from BNS_EMAIL
        assert "BNS_EMAIL" in source or "BUYANDSHIP_EMAIL" in source


# ---------- Method existence checks ----------


class TestShippingAgentMethods:
    def test_private_methods_exist(self):
        from app.utils.shipping_agent import ShippingAgent

        agent = ShippingAgent()
        assert hasattr(agent, "_get_credentials")
        assert hasattr(agent, "_pick_working_page")
        assert hasattr(agent, "_close_blank_tabs")
        assert hasattr(agent, "_open_target_url")
        assert hasattr(agent, "_login_if_needed")
        assert hasattr(agent, "_do_login_flow")
        assert hasattr(agent, "_extract_warehouses")
        assert hasattr(agent, "_get_warehouses_async")


# ---------- Helper functions ----------


class TestShippingAgentHelpers:
    def test_click_first_function_exists(self):
        from app.utils.shipping_agent import _click_first

        assert callable(_click_first)

    def test_fill_first_function_exists(self):
        from app.utils.shipping_agent import _fill_first

        assert callable(_fill_first)
