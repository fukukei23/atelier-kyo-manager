"""template_service のテスト — DB依存はmock"""
import json
from unittest.mock import MagicMock, patch

from app.services.template_service import (
    _fallback_template,
    generate_buyma_csv,
    generate_listing_text,
)


class TestFallbackTemplate:
    def test_returns_template_with_text(self):
        t = _fallback_template()
        assert "{{brand}}" in t.template_text
        assert "{{productName}}" in t.template_text


class TestGenerateListingText:
    def test_with_explicit_template(self):
        mock_template = MagicMock()
        mock_template.render.return_value = "出品文テスト"
        product = MagicMock()
        result = generate_listing_text(product, template=mock_template)
        assert result == "出品文テスト"
        mock_template.render.assert_called_once_with(product)

    @patch("app.services.template_service.ListingTemplate")
    def test_no_template_uses_default(self, mock_lt_cls):
        mock_default = MagicMock()
        mock_default.render.return_value = "デフォルト出品文"
        mock_lt_cls.query.filter_by.return_value.first.return_value = mock_default
        product = MagicMock()
        result = generate_listing_text(product)
        assert result == "デフォルト出品文"

    def test_no_default_uses_fallback(self):
        with patch("app.services.template_service.ListingTemplate") as mock_lt_cls:
            mock_lt_cls.query.filter_by.return_value.first.return_value = None
            product = MagicMock()
            result = generate_listing_text(product)
            assert result is not None


class TestGenerateBuymaCsv:
    def _make_product(self, **kwargs):
        p = MagicMock()
        p.name = kwargs.get("name", "テスト商品")
        p.brand = kwargs.get("brand", "Nike")
        p.color = kwargs.get("color", "Red")
        p.size = kwargs.get("size", "M")
        p.material = kwargs.get("material", "Cotton")
        p.retail_price = kwargs.get("retail_price", 20000)
        p.selling_price = kwargs.get("selling_price", 25000)
        p.source_region = kwargs.get("source_region", "US")
        p.description = kwargs.get("description", "説明文")
        p.item_category = kwargs.get("item_category", "バッグ")
        p.image_url = kwargs.get("image_url", "img.jpg")
        p.processed_images = kwargs.get("processed_images", None)
        p.pipeline_status = kwargs.get("pipeline_status", "complete")
        return p

    def test_basic_csv_generation(self):
        products = [self._make_product()]
        csv_text, skipped = generate_buyma_csv(products)
        assert skipped == 0
        assert "\ufeff" in csv_text  # BOM
        assert "テスト商品" in csv_text
        assert "Nike" in csv_text

    def test_skip_no_description(self):
        products = [self._make_product(description="")]
        csv_text, skipped = generate_buyma_csv(products)
        assert skipped == 1
        assert "テスト商品" not in csv_text

    def test_processed_images_json(self):
        products = [self._make_product(
            processed_images=json.dumps(["img1.jpg", "img2.jpg"]),
            image_url="fallback.jpg"
        )]
        csv_text, skipped = generate_buyma_csv(products)
        assert "img1.jpg,img2.jpg" in csv_text

    def test_processed_images_invalid_json(self):
        products = [self._make_product(
            processed_images="not json",
            image_url="fallback.jpg"
        )]
        csv_text, skipped = generate_buyma_csv(products)
        assert "fallback.jpg" in csv_text

    def test_empty_products(self):
        csv_text, skipped = generate_buyma_csv([])
        assert skipped == 0
        assert "商品名" in csv_text  # headers only

    def test_mixed_products(self):
        products = [
            self._make_product(name="商品A", description="説明A"),
            self._make_product(name="商品B", description=""),
            self._make_product(name="商品C", description="説明C"),
        ]
        csv_text, skipped = generate_buyma_csv(products)
        assert skipped == 1
        assert "商品A" in csv_text
        assert "商品B" not in csv_text
        assert "商品C" in csv_text
