"""Tests for csv_handler module."""

from unittest.mock import MagicMock, patch

import pytest

from app.utils.csv_handler import import_products_from_csv


@pytest.fixture
def mock_db():
    with patch("app.utils.csv_handler.db") as mock:
        mock.session.add = MagicMock()
        mock.session.commit = MagicMock()
        yield mock


@pytest.fixture
def mock_product_cls():
    with patch("app.utils.csv_handler.Product") as cls:
        mock_product = MagicMock()
        mock_product.calculate_profit.return_value = 0.5
        cls.return_value = mock_product
        yield cls, mock_product


def _make_csv_file(tmp_path, rows):
    csv_file = tmp_path / "products.csv"
    header = "商品名,ブランド,仕入価格,販売価格,仕入先URL,画像URL,在庫状況"
    lines = [header] + rows
    csv_file.write_text("\n".join(lines), encoding="utf-8")
    return csv_file


class TestImportProductsFromCsv:
    def test_import_single_product(self, tmp_path, mock_db, mock_product_cls):
        csv_file = _make_csv_file(tmp_path, ["バッグ,CHANEL,100000,150000,http://example.com,http://img.com,True"])
        with csv_file.open("rb") as f:
            import_products_from_csv(f)
        mock_db.session.add.assert_called_once()
        mock_db.session.commit.assert_called_once()

    def test_import_multiple_products(self, tmp_path, mock_db, mock_product_cls):
        csv_file = _make_csv_file(
            tmp_path,
            [
                "バッグ1,CHANEL,100000,150000,http://a.com,http://img.com,True",
                "バッグ2,LV,80000,120000,http://b.com,http://img2.com,False",
            ],
        )
        with csv_file.open("rb") as f:
            import_products_from_csv(f)
        assert mock_db.session.add.call_count == 2

    def test_invalid_price_skipped(self, tmp_path, mock_db, mock_product_cls):
        csv_file = _make_csv_file(tmp_path, ["バッグ,CHANEL,abc,150000,http://a.com,http://img.com,True"])
        with csv_file.open("rb") as f:
            import_products_from_csv(f)
        mock_db.session.add.assert_not_called()

    def test_missing_column_uses_defaults(self, tmp_path, mock_db, mock_product_cls):
        csv_file = tmp_path / "partial.csv"
        csv_file.write_text("商品名,ブランド,仕入価格,販売価格\n靴,Nike,5000,10000\n", encoding="utf-8")
        with csv_file.open("rb") as f:
            import_products_from_csv(f)
        mock_db.session.add.assert_called_once()

    def test_empty_csv(self, tmp_path, mock_db, mock_product_cls):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("商品名,ブランド,仕入価格,販売価格,仕入先URL,画像URL,在庫状況\n", encoding="utf-8")
        with csv_file.open("rb") as f:
            import_products_from_csv(f)
        mock_db.session.add.assert_not_called()

    def test_stock_status_parsing(self, tmp_path, mock_db, mock_product_cls):
        csv_file = _make_csv_file(tmp_path, ["バッグ,CHANEL,100,150,url,img,false"])
        with csv_file.open("rb") as f:
            import_products_from_csv(f)
        cls, product = mock_product_cls
        call_kwargs = cls.call_args[1]
        assert call_kwargs["stock_status"] is False
