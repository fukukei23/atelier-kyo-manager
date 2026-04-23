"""Tests for ai_generate_descriptions module."""
from unittest.mock import MagicMock, patch

import pytest

from app.utils.ai_generate_descriptions import DescriptionGenerator


@pytest.fixture
def mock_controller():
    with patch("app.utils.ai_generate_descriptions.AILlmController") as MockClass:
        instance = MagicMock()
        instance.generate.return_value = "素晴らしい商品説明文です。"
        MockClass.return_value = instance
        yield instance


class TestDescriptionGenerator:
    def test_generate_returns_description(self, mock_controller):
        gen = DescriptionGenerator()
        result = gen.generate({"brand": "CHANEL", "name": "バッグ", "category": "バッグ", "features": "ラムスキン"})
        assert result == "素晴らしい商品説明文です。"
        mock_controller.generate.assert_called_once()

    def test_generate_uses_model_name(self, mock_controller):
        gen = DescriptionGenerator()
        gen.generate({"name": "test"}, model_name="glm")
        call_args = mock_controller.generate.call_args
        assert call_args[1]["model_name"] == "glm"

    def test_generate_handles_empty_product_info(self, mock_controller):
        gen = DescriptionGenerator()
        result = gen.generate({})
        assert result == "素晴らしい商品説明文です。"

    def test_batch_process_returns_list(self, mock_controller):
        gen = DescriptionGenerator()
        products = [
            {"brand": "A", "name": "item1"},
            {"brand": "B", "name": "item2"},
        ]
        results = gen.batch_process(products)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_batch_process_empty_list(self, mock_controller):
        gen = DescriptionGenerator()
        results = gen.batch_process([])
        assert results == []
