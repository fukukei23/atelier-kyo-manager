"""Tests for app/integrations/forward2me_client.py"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from app.integrations.forward2me_client import Forward2meClient


@pytest.fixture
def client():
    return Forward2meClient(api_base="https://api.forward2me.com", api_key="test-key-123")


class TestForward2meClientInit:
    def test_trailing_slash_stripped(self):
        c = Forward2meClient(api_base="https://api.example.com/", api_key="key")
        assert c.api_base == "https://api.example.com"

    def test_default_timeout(self):
        c = Forward2meClient(api_base="https://api.example.com", api_key="key")
        assert c.timeout == 10.0

    def test_custom_timeout(self):
        c = Forward2meClient(api_base="https://api.example.com", api_key="key", timeout=30.0)
        assert c.timeout == 30.0

    def test_api_key_stored(self, client):
        assert client.api_key == "test-key-123"


class TestForward2meClientHeaders:
    def test_headers_include_bearer(self, client):
        headers = client._headers()
        assert headers["Authorization"] == "Bearer test-key-123"

    def test_headers_content_type(self, client):
        headers = client._headers()
        assert headers["Content-Type"] == "application/json"

    def test_headers_accept_json(self, client):
        headers = client._headers()
        assert headers["Accept"] == "application/json"


class TestListParcels:
    @pytest.mark.asyncio
    async def test_list_parcels_returns_list(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [{"id": "P001", "status": "in_warehouse"}]

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_http
            mock_client_cls.return_value.__aexit__.return_value = None

            result = await client.list_parcels()
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["id"] == "P001"

    @pytest.mark.asyncio
    async def test_list_parcels_with_updated_since(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = []

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_http
            mock_client_cls.return_value.__aexit__.return_value = None

            since = datetime(2024, 1, 1)
            result = await client.list_parcels(updated_since=since)
            call_kwargs = mock_http.get.call_args[1]
            assert "updated_since" in call_kwargs["params"]

    @pytest.mark.asyncio
    async def test_list_parcels_non_list_response_returns_empty(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"error": "bad format"}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_http
            mock_client_cls.return_value.__aexit__.return_value = None

            result = await client.list_parcels()
            assert result == []


class TestGetParcel:
    @pytest.mark.asyncio
    async def test_get_parcel_returns_dict(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"id": "P001", "tracking": "ABC123"}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_http
            mock_client_cls.return_value.__aexit__.return_value = None

            result = await client.get_parcel("P001")
            assert result["id"] == "P001"

    @pytest.mark.asyncio
    async def test_get_parcel_non_dict_returns_empty(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [1, 2, 3]

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.get.return_value = mock_resp
            mock_client_cls.return_value.__aenter__.return_value = mock_http
            mock_client_cls.return_value.__aexit__.return_value = None

            result = await client.get_parcel("P999")
            assert result == {}
