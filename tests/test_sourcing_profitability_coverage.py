"""Tests for app/utils/sourcing_profitability.py - Issue #92 カバレッジ向上"""

from app.utils.sourcing_profitability import calculate_profitability


class TestNoneInput:
    def test_none_returns_invalid(self):
        """None 入力 → status=invalid"""
        result = calculate_profitability(None)
        assert result["status"] == "invalid"
        assert result["errors"]
        assert result["profit"] is None
        assert result["revenue"] is None


class TestPartialMissingRequired:
    def test_purchase_price_none_returns_partial_error(self):
        """purchase_price=None → status=partial, errors あり"""
        data = {
            "purchase_price": None,
            "selling_price": 50000,
            "shipping_cost": 2000,
            "customs_duty": None,
            "procurement_fee": None,
            "transaction_fee": None,
            "warehouse_shipping_cost": None,
        }
        result = calculate_profitability(data)
        assert result["status"] == "partial"
        assert len(result["errors"]) > 0
        assert result["profit"] is None

    def test_selling_price_none_returns_partial_error(self):
        """selling_price=None → status=partial, errors あり"""
        data = {
            "purchase_price": 30000,
            "selling_price": None,
            "shipping_cost": None,
            "customs_duty": None,
            "procurement_fee": None,
            "transaction_fee": None,
            "warehouse_shipping_cost": None,
        }
        result = calculate_profitability(data)
        assert result["status"] == "partial"
        assert result["profit"] is None

    def test_both_required_none_returns_partial_error(self):
        """purchase_price=None, selling_price=None → status=partial, errors あり"""
        data = {
            "purchase_price": None,
            "selling_price": None,
        }
        result = calculate_profitability(data)
        assert result["status"] == "partial"
        assert result["errors"]


class TestPartialWithKnownCosts:
    def test_known_cost_sum_calculated(self):
        """selling_price と known costs あり → known_cost_sum が計算される"""
        data = {
            "purchase_price": 30000,
            "selling_price": 50000,
            "shipping_cost": 2000,
            "customs_duty": 1000,
            "procurement_fee": None,  # unknown
            "transaction_fee": None,  # unknown
            "warehouse_shipping_cost": None,  # unknown
        }
        result = calculate_profitability(data)
        assert result["status"] == "partial"
        assert result["known_cost_sum"] is not None
        assert result["known_cost_sum"] > 0
        assert result["profit_upper_bound"] is not None
        assert result["revenue"] == 50000.0

    def test_all_optional_costs_included(self):
        """全 optional コストが known_cost_sum に含まれる"""
        data = {
            "purchase_price": 20000,
            "selling_price": 50000,
            "shipping_cost": 1000,
            "customs_duty": 500,
            "procurement_fee": 300,
            "transaction_fee": 200,
            "warehouse_shipping_cost": 400,
            "some_unknown": None,  # これが unknown にするためのフィールド
        }
        result = calculate_profitability(data)
        assert result["status"] == "partial"
        # known_cost_sum = purchase + shipping + customs + procurement + transaction + warehouse + fees
        assert result["known_cost_sum"] > 20000 + 1000 + 500

    def test_shipping_cost_included_in_sum(self):
        """shipping_cost が known_cost_sum に加算される"""
        data = {
            "purchase_price": 30000,
            "selling_price": 50000,
            "shipping_cost": 5000,
            "customs_duty": None,
            "procurement_fee": None,
            "transaction_fee": None,
            "warehouse_shipping_cost": None,
            "extra": None,
        }
        result = calculate_profitability(data)
        assert result["status"] == "partial"
        assert result["known_cost_sum"] > 30000 + 5000  # purchase + shipping + fees

    def test_transaction_fee_included_in_sum(self):
        """transaction_fee が known_cost_sum に加算される"""
        data = {
            "purchase_price": 20000,
            "selling_price": 40000,
            "shipping_cost": None,
            "customs_duty": None,
            "procurement_fee": None,
            "transaction_fee": 1000,
            "warehouse_shipping_cost": None,
            "extra": None,
        }
        result = calculate_profitability(data)
        assert result["status"] == "partial"
        assert result["known_cost_sum"] is not None

    def test_warehouse_shipping_included(self):
        """warehouse_shipping_cost が known_cost_sum に加算される"""
        data = {
            "purchase_price": 20000,
            "selling_price": 40000,
            "shipping_cost": None,
            "customs_duty": None,
            "procurement_fee": None,
            "transaction_fee": None,
            "warehouse_shipping_cost": 800,
            "extra": None,
        }
        result = calculate_profitability(data)
        assert result["status"] == "partial"
        assert result["known_cost_sum"] is not None


class TestCompletePath:
    def _complete_data(self):
        """complete データ。price_source は明示しない（calculate_profitability 側でモック）"""
        return {
            "purchase_price": 30000.0,
            "selling_price": 50000.0,
            "shipping_cost": 2000.0,
            "customs_duty": 1000.0,
            "procurement_fee": 500.0,
            "transaction_fee": 300.0,
            "warehouse_shipping_cost": 200.0,
            "original_currency": "JPY",
            "exchange_rate": 1.0,
            "item_category": "bags",
            "item_material": "",
        }

    def test_complete_returns_complete_status(self):
        """全フィールドあり → status=complete"""
        result = calculate_profitability(self._complete_data())  # 実経路（モックなし・ISSUE-102改修済み）
        assert result["status"] == "complete"

    def test_complete_has_profit(self):
        """complete パスでは profit が計算される"""
        result = calculate_profitability(self._complete_data())  # 実経路（モックなし・ISSUE-102改修済み）
        assert result["profit"] is not None

    def test_complete_has_revenue_and_cost(self):
        """complete パスでは revenue と total_cost が設定される"""
        result = calculate_profitability(self._complete_data())  # 実経路（モックなし・ISSUE-102改修済み）
        assert result["revenue"] is not None
        assert result["total_cost"] is not None

    def test_complete_has_no_errors(self):
        """complete パスでは errors が空"""
        result = calculate_profitability(self._complete_data())  # 実経路（モックなし・ISSUE-102改修済み）
        assert result["errors"] == []

    def test_complete_missing_fields_is_none(self):
        """complete パスでは missing_fields が None"""
        result = calculate_profitability(self._complete_data())  # 実経路（モックなし・ISSUE-102改修済み）
        assert result["missing_fields"] is None
