"""sourcing_input_schema テスト"""

import pytest

from app.utils.sourcing_input_schema import validate_sourcing_input


class TestValidateSourcingInput:
    def test_valid_complete(self):
        result = validate_sourcing_input({
            "purchase_price": 10000,
            "selling_price": 20000,
        })
        assert result["valid"] is True
        assert result["status"] == "complete"
        assert result["normalized"]["purchase_price"] == 10000.0

    def test_valid_with_optional_fields(self):
        result = validate_sourcing_input({
            "purchase_price": 10000,
            "selling_price": 20000,
            "shipping_cost": 1500,
            "customs_duty": 500,
        })
        assert result["valid"] is True
        assert result["normalized"]["shipping_cost"] == 1500.0

    def test_missing_required_field(self):
        result = validate_sourcing_input({"purchase_price": 10000})
        assert result["valid"] is False
        assert result["status"] == "invalid"
        assert len(result["errors"]) > 0

    def test_unknown_optional_field(self):
        result = validate_sourcing_input({
            "purchase_price": 10000,
            "selling_price": 20000,
            "shipping_cost": "unknown",
        })
        assert result["valid"] is True
        assert result["status"] == "partial"
        assert result["normalized"]["shipping_cost"] is None

    def test_unknown_required_field(self):
        result = validate_sourcing_input({
            "purchase_price": "unknown",
            "selling_price": 20000,
        })
        assert result["valid"] is False
        assert result["status"] == "invalid"

    def test_invalid_type(self):
        result = validate_sourcing_input({
            "purchase_price": "not_a_number",
            "selling_price": 20000,
        })
        assert result["valid"] is False
        assert result["status"] == "invalid"

    def test_negative_value(self):
        result = validate_sourcing_input({
            "purchase_price": -100,
            "selling_price": 20000,
        })
        assert result["valid"] is False
        assert result["status"] == "invalid"

    def test_empty_dict(self):
        result = validate_sourcing_input({})
        assert result["valid"] is False
        assert result["status"] == "invalid"
