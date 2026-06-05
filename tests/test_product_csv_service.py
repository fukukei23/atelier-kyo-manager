"""Tests for product_csv_service module."""

from __future__ import annotations

import csv
import io
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.services.product_csv_service import (
    CANDIDATE_HEADERS,
    EXPORT_HEADERS,
    _candidate_to_row,
    _product_to_row,
    _row_to_product,
    export_candidates_csv,
    export_products_csv,
    import_products_from_csv,
)


def _make_product(**overrides):
    """Create a mock Product with sensible defaults."""
    p = MagicMock()
    p.id = overrides.get("id", 1)
    p.name = overrides.get("name", "Test Product")
    p.brand = overrides.get("brand", "TestBrand")
    p.purchase_price = overrides.get("purchase_price", 10000.0)
    p.selling_price = overrides.get("selling_price", 15000.0)
    p.transaction_fee = overrides.get("transaction_fee", 500.0)
    p.shipping_cost = overrides.get("shipping_cost", 2000.0)
    p.customs_duty = overrides.get("customs_duty", 0.0)
    p.procurement_fee = overrides.get("procurement_fee", 0.0)
    p.supplier_url = overrides.get("supplier_url", "https://example.com/item")
    p.image_url = overrides.get("image_url", "https://example.com/img.jpg")
    p.stock_status = overrides.get("stock_status", True)
    p.brand_tier = overrides.get("brand_tier", "mid")
    p.source_type = overrides.get("source_type", "online")
    p.source_region = overrides.get("source_region", "EU")
    p.color = overrides.get("color", "Black")
    p.size = overrides.get("size", "M")
    p.material = overrides.get("material", "Cotton")
    p.description = overrides.get("description", "A test product")
    p.retail_price = overrides.get("retail_price", 20000.0)
    p.target_profit_rate = overrides.get("target_profit_rate", 0.10)
    p.listing_status = overrides.get("listing_status", "draft")
    p.created_at = overrides.get("created_at", datetime(2026, 1, 1, 12, 0, 0))
    p.updated_at = overrides.get("updated_at", datetime(2026, 1, 2, 12, 0, 0))
    p.warehouse_shipping_cost = overrides.get("warehouse_shipping_cost", 1500.0)
    p.original_currency = overrides.get("original_currency", "EUR")
    p.exchange_rate = overrides.get("exchange_rate", 160.0)
    p.item_category = overrides.get("item_category", "bag")
    p.calculate_profit = MagicMock(return_value=2500.0)
    p.profit_rate = MagicMock(return_value=25.0)
    return p


class TestRowToProduct:
    @patch("app.services.product_csv_service._row_to_product")
    def test_basic_row_conversion(self, mock_fn):
        """Test that _row_to_product creates a Product with correct fields."""
        from app.models import Product

        row = {
            "name": "Jacket",
            "brand": "Moncler",
            "purchase_price": "50000",
            "selling_price": "80000",
        }
        product = _row_to_product(row)
        assert product.name == "Jacket"
        assert product.brand == "Moncler"
        assert product.purchase_price == 50000.0
        assert product.selling_price == 80000.0

    def test_stock_status_truthy_values(self):
        for val in ("1", "true", "True", "yes", "on"):
            row = {"name": "x", "purchase_price": "1", "selling_price": "2", "stock_status": val}
            p = _row_to_product(row)
            assert p.stock_status is True, f"Expected True for '{val}'"

    def test_stock_status_falsy_values(self):
        for val in ("0", "false", "no", ""):
            row = {"name": "x", "purchase_price": "1", "selling_price": "2", "stock_status": val}
            p = _row_to_product(row)
            assert p.stock_status is False, f"Expected False for '{val}'"

    def test_defaults_for_missing_fields(self):
        row = {"name": "x", "purchase_price": "1", "selling_price": "2"}
        p = _row_to_product(row)
        assert p.target_profit_rate == 0.10
        assert p.listing_status == "draft"
        assert p.original_currency == "JPY"
        assert p.exchange_rate == 1.0


class TestProductToRow:
    def test_converts_product_to_dict(self):
        p = _make_product()
        row = _product_to_row(p)
        assert row["id"] == 1
        assert row["name"] == "Test Product"
        assert row["brand"] == "TestBrand"
        assert row["purchase_price"] == 10000.0

    def test_stock_status_converted_to_int(self):
        p = _make_product(stock_status=True)
        assert _product_to_row(p)["stock_status"] == 1

        p2 = _make_product(stock_status=False)
        assert _product_to_row(p2)["stock_status"] == 0

    def test_target_profit_rate_converted_to_percentage(self):
        p = _make_product(target_profit_rate=0.15)
        assert _product_to_row(p)["target_profit_rate"] == 15.0

    def test_none_fields_use_defaults(self):
        p = _make_product(brand_tier=None, color=None)
        row = _product_to_row(p)
        assert row["brand_tier"] == ""
        assert row["color"] == ""

    def test_datetime_formatted(self):
        p = _make_product(created_at=datetime(2026, 5, 10, 14, 30, 0))
        assert _product_to_row(p)["created_at"] == "2026-05-10 14:30:00"

    def test_none_datetime_empty_string(self):
        p = _make_product(created_at=None, updated_at=None)
        row = _product_to_row(p)
        assert row["created_at"] == ""
        assert row["updated_at"] == ""


class TestCandidateToRow:
    def test_includes_profit_fields(self):
        p = _make_product()
        row = _candidate_to_row(p)
        assert row["profit"] == 2500.0
        assert row["profit_rate"] == 25.0

    def test_contains_expected_headers(self):
        p = _make_product()
        row = _candidate_to_row(p)
        for key in CANDIDATE_HEADERS:
            assert key in row


class TestImportProductsFromCsv:
    def test_imports_valid_rows(self):
        csv_content = "name,brand,purchase_price,selling_price\nJacket,Moncler,50000,80000\nShoes,Gucci,40000,60000"
        session = MagicMock()
        count, warnings = import_products_from_csv(csv_content, session)
        assert count == 2
        assert session.add.call_count == 2
        session.commit.assert_called_once()

    def test_skips_rows_without_name(self):
        csv_content = "name,brand,purchase_price,selling_price\n,Moncler,50000,80000\nShoes,Gucci,40000,60000"
        session = MagicMock()
        count, warnings = import_products_from_csv(csv_content, session)
        assert count == 1

    def test_skips_rows_without_prices(self):
        csv_content = "name,brand,purchase_price,selling_price\nJacket,Moncler,,\nShoes,Gucci,40000,60000"
        session = MagicMock()
        count, warnings = import_products_from_csv(csv_content, session)
        assert count == 1

    def test_empty_csv_returns_zero(self):
        csv_content = "name,brand,purchase_price,selling_price\n"
        session = MagicMock()
        count, warnings = import_products_from_csv(csv_content, session)
        assert count == 0

    def test_detects_double_currency_conversion(self):
        """purchase_price=180810 + EUR + rate=184.5 → 二重為替変換警告"""
        csv_content = (
            "name,brand,purchase_price,selling_price,original_currency,exchange_rate\n"
            "Shoes,Valentino,180810,322293,EUR,184.5"
        )
        session = MagicMock()
        count, warnings = import_products_from_csv(csv_content, session)
        assert count == 1
        assert len(warnings) == 1
        assert "異常に高値" in warnings[0]

    def test_detects_double_commission(self):
        """transaction_fee > 5% of selling_price → 手数料二重計上警告"""
        csv_content = (
            "name,brand,purchase_price,selling_price,transaction_fee\n"
            "Shoes,Valentino,950,312427,54243"
        )
        session = MagicMock()
        count, warnings = import_products_from_csv(csv_content, session)
        assert count == 1
        assert len(warnings) == 1
        assert "二重計上" in warnings[0]

    def test_no_warnings_for_clean_data(self):
        """正常データなら警告なし"""
        csv_content = (
            "name,brand,purchase_price,selling_price,original_currency,exchange_rate\n"
            "Shoes,Valentino,950,312427,EUR,184.5"
        )
        session = MagicMock()
        count, warnings = import_products_from_csv(csv_content, session)
        assert count == 1
        assert len(warnings) == 0


class TestExportProductsCsv:
    def test_generates_csv_with_headers(self):
        products = [_make_product()]
        csv_str = export_products_csv(products)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["name"] == "Test Product"

    def test_headers_match_export_headers(self):
        products = [_make_product()]
        csv_str = export_products_csv(products)
        reader = csv.DictReader(io.StringIO(csv_str))
        assert reader.fieldnames == EXPORT_HEADERS

    def test_empty_list_generates_headers_only(self):
        csv_str = export_products_csv([])
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 0


class TestExportCandidatesCsv:
    def test_generates_candidate_csv(self):
        products = [_make_product()]
        csv_str = export_candidates_csv(products)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["name"] == "Test Product"

    def test_headers_match_candidate_headers(self):
        products = [_make_product()]
        csv_str = export_candidates_csv(products)
        reader = csv.DictReader(io.StringIO(csv_str))
        assert reader.fieldnames == CANDIDATE_HEADERS
