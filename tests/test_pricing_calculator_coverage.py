"""pricing_calculator のテスト — 外部APIはmock"""

from unittest.mock import patch

from app.utils.pricing_calculator import calculate_profit, get_exchange_rate


class TestGetExchangeRate:
    @patch("app.utils.pricing_calculator.requests.get")
    def test_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"rates": {"JPY": 149.5}}
        mock_get.return_value.raise_for_status = lambda: None
        rate = get_exchange_rate("USD", "JPY")
        assert rate == 149.5

    @patch("app.utils.pricing_calculator.requests.get")
    def test_api_failure_returns_default(self, mock_get):
        import requests as req

        mock_get.side_effect = req.exceptions.RequestException("timeout")
        rate = get_exchange_rate("USD", "JPY")
        assert rate == 150.0

    @patch("app.utils.pricing_calculator.requests.get")
    def test_rate_not_found_returns_default(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"rates": {}}
        mock_get.return_value.raise_for_status = lambda: None
        rate = get_exchange_rate("USD", "XYZ")
        assert rate == 150.0

    @patch("app.utils.pricing_calculator.requests.get")
    def test_unexpected_error_returns_default(self, mock_get):
        mock_get.side_effect = RuntimeError("boom")
        rate = get_exchange_rate("USD", "JPY")
        assert rate == 150.0


class TestCalculateProfit:
    @patch("app.utils.pricing_calculator.get_exchange_rate", return_value=150.0)
    def test_basic_calculation(self, mock_rate):
        result = calculate_profit(
            buyma_price=30000,
            source_price=100,
            shipping_cost=20,
        )
        assert "profit_estimate" in result
        assert "profit_rate" in result
        # source=100*150=15000, ship=20*150=3000, duty=(15000+3000)*0.1=1800
        # total_cost=19800, commission=30000*0.073=2190, fees=2190+200=2390
        # net=30000-2390=27610, profit=27610-19800=7810
        assert result["profit_estimate"] == 7810.0
        assert result["profit_rate"] > 0

    @patch("app.utils.pricing_calculator.get_exchange_rate", return_value=150.0)
    def test_zero_buyma_price(self, mock_rate):
        result = calculate_profit(buyma_price=0, source_price=100, shipping_cost=20)
        assert result["profit_rate"] == 0

    @patch("app.utils.pricing_calculator.get_exchange_rate", return_value=1.0)
    def test_loss_scenario(self, mock_rate):
        result = calculate_profit(buyma_price=100, source_price=80, shipping_cost=30)
        assert result["profit_estimate"] < 0

    @patch("app.utils.pricing_calculator.get_exchange_rate", return_value=1.0)
    def test_custom_rates(self, mock_rate):
        result = calculate_profit(
            buyma_price=10000,
            source_price=50,
            shipping_cost=10,
            customs_duty_rate=0.0,
            buyma_commission_rate=0.05,
            buyma_system_fee=0,
        )
        assert result["profit_estimate"] > 0
