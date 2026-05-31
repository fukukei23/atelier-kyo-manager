from __future__ import annotations

import os
import pytest
from unittest.mock import patch, AsyncMock

from app.utils.shipping_agent import ShippingAgent, WAREHOUSE_LIST_URL


@pytest.fixture(autouse=True)
def patch_makedirs():
    with patch("os.makedirs"):
        yield


def test_init_headless_true():
    agent = ShippingAgent(headless=True)
    assert agent.headless is True


def test_init_headless_false():
    agent = ShippingAgent(headless=False)
    assert agent.headless is False


def test_init_default_headless_no_env():
    with patch.dict(os.environ, {}, clear=True):
        agent = ShippingAgent()
        assert agent.headless is True


def test_init_headful_env():
    with patch.dict(os.environ, {"PLAYWRIGHT_HEADFUL": "1"}):
        agent = ShippingAgent()
        assert agent.headless is False


def test_init_default_base_url():
    agent = ShippingAgent()
    assert agent.base_url == WAREHOUSE_LIST_URL


def test_init_default_timeout():
    agent = ShippingAgent()
    assert agent.timeout_ms == 60_000


def test_init_custom_timeout():
    agent = ShippingAgent(timeout_ms=30_000)
    assert agent.timeout_ms == 30_000


def test_get_credentials_from_env(routes_app):
    with routes_app.app_context():
        with patch.dict(os.environ, {"BUYANDSHIP_EMAIL": "test@example.com", "BUYANDSHIP_PASSWORD": "secret123"}):
            agent = ShippingAgent()
            result = agent._get_credentials()
    assert result == {"email": "test@example.com", "password": "secret123"}


def test_get_credentials_missing_raises(routes_app):
    clean = {k: "" for k in ["BUYANDSHIP_EMAIL", "BUYANDSHIP_PASSWORD", "BNS_EMAIL", "BNS_PASSWORD"]}
    with routes_app.app_context():
        with patch.dict(os.environ, clean, clear=True):
            agent = ShippingAgent()
            with pytest.raises(RuntimeError, match="BUYANDSHIP_EMAIL"):
                agent._get_credentials()


def test_get_credentials_legacy_env(routes_app):
    clean = {"BUYANDSHIP_EMAIL": "", "BUYANDSHIP_PASSWORD": ""}
    with routes_app.app_context():
        with patch.dict(os.environ, {**clean, "BNS_EMAIL": "legacy@example.com", "BNS_PASSWORD": "legacypass"}, clear=True):
            agent = ShippingAgent()
            result = agent._get_credentials()
    assert result["email"] == "legacy@example.com"
    assert result["password"] == "legacypass"


def test_get_warehouses_by_country():
    expected = [{"id": 1, "country": "US", "name": "Warehouse A"}]
    with patch.object(ShippingAgent, "_get_warehouses_async", new_callable=AsyncMock) as mock_async:
        mock_async.return_value = expected
        agent = ShippingAgent()
        result = agent.get_warehouses_by_country("US")
        mock_async.assert_awaited_once_with("US")
        assert result == expected


def test_fetch_warehouses_delegates():
    dummy = [{"id": 2, "country": "JP", "name": "Warehouse B"}]
    with patch.object(ShippingAgent, "get_warehouses_by_country", return_value=dummy) as mock_get:
        agent = ShippingAgent()
        result = agent.fetch_warehouses("JP")
        mock_get.assert_called_once_with("JP")
        assert result == dummy


def test_get_warehouses_default_country():
    with patch.object(ShippingAgent, "_get_warehouses_async", new_callable=AsyncMock) as mock_async:
        mock_async.return_value = []
        agent = ShippingAgent()
        agent.get_warehouses_by_country()
        mock_async.assert_awaited_once_with("US")
