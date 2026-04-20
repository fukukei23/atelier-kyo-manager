"""pricing_calculator テスト"""

import pytest
from unittest.mock import patch

from app.utils.pricing_calculator import (
    calculate_profit,
    get_exchange_rate,
    DEFAULT_EXCHANGE_RATE,
    BUYMA_COMMISSION_RATE,
    BUYMA_SYSTEM_FEE,
)


class TestGetExchangeRate:
    @patch("app.utils.pricing_calculator.requests.get")
    def test_success(self, mock_get):
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"rates": {"JPY": 155.0}}
        mock_get.return_value = mock_resp

        rate = get_exchange_rate("USD", "JPY")
        assert rate == 155.0

    @patch("app.utils.pricing_calculator.requests.get")
    def test_currency_not_found_returns_default(self, mock_get):
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"rates": {}}
        mock_get.return_value = mock_resp

        rate = get_exchange_rate("USD", "XYZ")
        assert rate == DEFAULT_EXCHANGE_RATE

    @patch("app.utils.pricing_calculator.requests.get")
    def test_request_failure_returns_default(self, mock_get):
        import requests as _req
        mock_get.side_effect = _req.ConnectionError("fail")

        rate = get_exchange_rate("USD", "JPY")
        assert rate == DEFAULT_EXCHANGE_RATE


class TestCalculateProfit:
    @patch("app.utils.pricing_calculator.get_exchange_rate")
    def test_normal_profit(self, mock_rate):
        mock_rate.return_value = 150.0
        result = calculate_profit(
            buyma_price=30000, source_price=100, shipping_cost=50,
            customs_duty_rate=0.1, source_currency="USD"
        )
        assert result["profit_estimate"] > 0
        assert result["profit_rate"] > 0

    @patch("app.utils.pricing_calculator.get_exchange_rate")
    def test_zero_buyma_price(self, mock_rate):
        mock_rate.return_value = 150.0
        result = calculate_profit(
            buyma_price=0, source_price=100, shipping_cost=10,
            source_currency="USD"
        )
        assert result["profit_estimate"] < 0
        assert result["profit_rate"] == 0.0

    @patch("app.utils.pricing_calculator.get_exchange_rate")
    def test_negative_profit(self, mock_rate):
        mock_rate.return_value = 150.0
        result = calculate_profit(
            buyma_price=15000, source_price=120, shipping_cost=20,
            source_currency="USD"
        )
        assert result["profit_estimate"] < 0
        assert result["profit_rate"] < 0

    @patch("app.utils.pricing_calculator.get_exchange_rate")
    def test_zero_customs_duty(self, mock_rate):
        mock_rate.return_value = 100.0
        result = calculate_profit(
            buyma_price=10000, source_price=30, shipping_cost=10,
            customs_duty_rate=0.0, source_currency="USD"
        )
        assert result["profit_estimate"] == 5070.0
        assert result["profit_rate"] == 50.7

    @patch("app.utils.pricing_calculator.get_exchange_rate")
    def test_high_customs_duty(self, mock_rate):
        mock_rate.return_value = 100.0
        result = calculate_profit(
            buyma_price=10000, source_price=30, shipping_cost=10,
            customs_duty_rate=0.3, source_currency="USD"
        )
        assert result["profit_estimate"] == 3870.0
        assert result["profit_rate"] == 38.7

    @patch("app.utils.pricing_calculator.get_exchange_rate")
    def test_default_rate_fallback(self, mock_rate):
        mock_rate.return_value = DEFAULT_EXCHANGE_RATE
        result = calculate_profit(
            buyma_price=26800, source_price=120, shipping_cost=20,
            source_currency="USD"
        )
        mock_rate.assert_called_once_with("USD", "JPY")
        assert isinstance(result["profit_estimate"], float)
        assert isinstance(result["profit_rate"], float)
