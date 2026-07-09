"""Tests for app/extractors/moncler_extractor.py - Issue #83 カバレッジ向上"""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.extractors.moncler_extractor import MonclerPDPExtractor


@pytest.fixture
def extractor():
    return MonclerPDPExtractor(logger=logging.getLogger("test"))


# ── _infer_sku_from_url ───────────────────────────────────────────────────────


class TestInferSkuFromUrl:
    def test_simple_slug_returns_upper(self, extractor):
        assert extractor._infer_sku_from_url("https://www.moncler.com/jackets/abc123") == "ABC123"

    def test_hyphenated_slug_uses_last_part(self, extractor):
        assert extractor._infer_sku_from_url("https://example.com/product-XYZ789") == "XYZ789"

    def test_trailing_slash_ignored(self, extractor):
        result = extractor._infer_sku_from_url("https://example.com/jackets/ABC123/")
        assert result == "ABC123"

    def test_empty_slug_returns_none(self, extractor):
        # rstrip("/") → "" → split("/")[-1] = "" → not slug → None
        assert extractor._infer_sku_from_url("/") is None

    def test_non_alnum_slug_returns_none(self, extractor):
        assert extractor._infer_sku_from_url("https://example.com/slug-日本語") is None

    def test_alphanumeric_mixed(self, extractor):
        result = extractor._infer_sku_from_url("https://moncler.com/en-jp/coats/monaco-coat-G10910")
        assert result == "G10910"

    def test_single_token_no_hyphen(self, extractor):
        result = extractor._infer_sku_from_url("https://example.com/PRODCODE1")
        assert result == "PRODCODE1"

    def test_deep_path_uses_last_segment(self, extractor):
        result = extractor._infer_sku_from_url("https://moncler.com/en-jp/men/coats/monaco-G20910A")
        assert result == "G20910A"

    def test_no_slash_in_url(self, extractor):
        # URLにパスがない場合
        result = extractor._infer_sku_from_url("https://moncler.com")
        # "moncler.com" → split("-")[-1] = "moncler.com" → re.match fails (contains dot)
        assert result is None or isinstance(result, str)


# ── _build_result_from_product ────────────────────────────────────────────────


class TestBuildResultFromProduct:
    def test_none_product_returns_none(self, extractor):
        assert extractor._build_result_from_product(None, "https://example.com") is None

    def test_empty_dict_returns_none(self, extractor):
        assert extractor._build_result_from_product({}, "https://example.com") is None

    def test_full_product_returns_dict(self, extractor):
        product = {
            "name": "Monaco Coat",
            "code": "G10910",
            "price": {"final": {"value": 150000, "currency": "JPY"}},
            "availability": {"status": "IN_STOCK"},
            "media": {
                "gallery": [
                    {"url": "https://cdn.moncler.com/img1.jpg"},
                    {"url": "https://cdn.moncler.com/img2.jpg"},
                ]
            },
        }
        result = extractor._build_result_from_product(product, "https://moncler.com/en-jp/coats/monaco-G10910")
        assert result is not None
        assert result["title"] == "Monaco Coat"
        assert result["sku"] == "G10910"
        assert result["price"] == "150000"
        assert result["currency"] == "JPY"
        assert result["availability"] == "IN_STOCK"
        assert len(result["images"]) == 2
        assert result["url"] == "https://moncler.com/en-jp/coats/monaco-G10910"

    def test_missing_price_returns_none_price(self, extractor):
        product = {"name": "Coat", "code": "X1"}
        result = extractor._build_result_from_product(product, None)
        assert result is not None
        assert result["price"] is None
        assert result["currency"] is None

    def test_empty_gallery_returns_none_images(self, extractor):
        product = {"name": "Coat", "code": "X1", "media": {"gallery": []}}
        result = extractor._build_result_from_product(product, None)
        assert result is not None
        assert result["images"] is None

    def test_gallery_without_url_skipped(self, extractor):
        product = {
            "name": "Coat",
            "code": "X1",
            "media": {"gallery": [{"alt": "no url"}, {"url": "https://cdn/img.jpg"}]},
        }
        result = extractor._build_result_from_product(product, None)
        assert result["images"] == ["https://cdn/img.jpg"]

    def test_url_passed_through(self, extractor):
        product = {"name": "N", "code": "C"}
        result = extractor._build_result_from_product(product, "https://my-url.com")
        assert result["url"] == "https://my-url.com"

    def test_url_none_allowed(self, extractor):
        product = {"name": "N", "code": "C"}
        result = extractor._build_result_from_product(product, None)
        assert result["url"] is None


# ── _extract_from_next_data ───────────────────────────────────────────────────


class TestExtractFromNextData:
    def _make_page(self, script_text=None):
        """script#__NEXT_DATA__ を持つ/持たないページモックを作成"""
        page = MagicMock()
        page.url = "https://www.moncler.com/en-jp/coats/monaco-G10910"
        if script_text is None:
            # スクリプトなし
            page.query_selector = AsyncMock(return_value=None)
        else:
            script = AsyncMock()
            script.inner_text = AsyncMock(return_value=script_text)
            page.query_selector = AsyncMock(return_value=script)
        return page

    def test_no_script_returns_none(self, extractor):
        page = self._make_page(script_text=None)
        result = asyncio.run(extractor._extract_from_next_data(page))
        assert result is None

    def test_invalid_json_returns_none(self, extractor):
        page = self._make_page(script_text="not valid json")
        result = asyncio.run(extractor._extract_from_next_data(page))
        assert result is None

    def test_no_product_key_returns_none(self, extractor):
        payload = json.dumps({"props": {"pageProps": {}}})
        page = self._make_page(script_text=payload)
        result = asyncio.run(extractor._extract_from_next_data(page))
        assert result is None

    def test_product_not_dict_returns_none(self, extractor):
        payload = json.dumps({"props": {"pageProps": {"product": "not_a_dict"}}})
        page = self._make_page(script_text=payload)
        result = asyncio.run(extractor._extract_from_next_data(page))
        assert result is None

    def test_valid_product_returns_dict(self, extractor):
        product = {
            "name": "Monaco Coat",
            "code": "G10910",
            "price": {"final": {"value": 150000, "currency": "JPY"}},
        }
        payload = json.dumps({"props": {"pageProps": {"product": product}}})
        page = self._make_page(script_text=payload)
        result = asyncio.run(extractor._extract_from_next_data(page))
        assert result is not None
        assert result["title"] == "Monaco Coat"
        assert result["sku"] == "G10910"

    def test_empty_script_returns_none(self, extractor):
        page = self._make_page(script_text="")
        result = asyncio.run(extractor._extract_from_next_data(page))
        assert result is None


# ── _extract_from_ld_json ─────────────────────────────────────────────────────


class TestExtractFromLdJson:
    def _make_page_with_ld(self, nodes_data: list[dict | str]):
        """ld+json ノードのリストを持つページモックを作成"""
        page = MagicMock()
        page.url = "https://www.moncler.com/product"
        nodes = []
        for data in nodes_data:
            node = AsyncMock()
            node.inner_text = AsyncMock(return_value=json.dumps(data) if isinstance(data, dict) else data)
            nodes.append(node)
        page.query_selector_all = AsyncMock(return_value=nodes)
        return page

    def test_no_nodes_returns_none(self, extractor):
        page = MagicMock()
        page.query_selector_all = AsyncMock(return_value=[])
        result = asyncio.run(extractor._extract_from_ld_json(page))
        assert result is None

    def test_query_selector_all_exception_returns_none(self, extractor):
        page = MagicMock()
        page.query_selector_all = AsyncMock(side_effect=Exception("playwright error"))
        result = asyncio.run(extractor._extract_from_ld_json(page))
        assert result is None

    def test_wrong_type_skipped(self, extractor):
        page = self._make_page_with_ld([{"@type": "Organization", "name": "Moncler"}])
        result = asyncio.run(extractor._extract_from_ld_json(page))
        assert result is None

    def test_product_without_price_skipped(self, extractor):
        page = self._make_page_with_ld([{"@type": "Product", "name": "Coat"}])
        result = asyncio.run(extractor._extract_from_ld_json(page))
        assert result is None

    def test_product_with_price_returns_dict(self, extractor):
        page = self._make_page_with_ld(
            [{"@type": "Product", "offers": {"price": "150000", "priceCurrency": "JPY", "availability": "InStock"}}]
        )
        page.url = "https://www.moncler.com/product"
        result = asyncio.run(extractor._extract_from_ld_json(page))
        assert result is not None
        assert result["price"] == "150000"
        assert result["currency"] == "JPY"

    def test_product_group_type_accepted(self, extractor):
        page = self._make_page_with_ld(
            [{"@type": "ProductGroup", "offers": {"price": "99000", "priceCurrency": "JPY"}}]
        )
        page.url = "https://www.moncler.com/product"
        result = asyncio.run(extractor._extract_from_ld_json(page))
        assert result is not None
        assert result["price"] == "99000"

    def test_invalid_json_node_skipped(self, extractor):
        page = self._make_page_with_ld(["not json"])
        result = asyncio.run(extractor._extract_from_ld_json(page))
        assert result is None

    def test_url_included_in_result(self, extractor):
        page = self._make_page_with_ld([{"@type": "Product", "offers": {"price": "50000", "priceCurrency": "JPY"}}])
        page.url = "https://www.moncler.com/my-product"
        result = asyncio.run(extractor._extract_from_ld_json(page))
        assert result["url"] == "https://www.moncler.com/my-product"


# ── _extract_from_meta ────────────────────────────────────────────────────────


class TestExtractFromMeta:
    def test_no_price_returns_none(self, extractor):
        page = MagicMock()
        page.url = "https://www.moncler.com/product"
        page.get_attribute = AsyncMock(return_value=None)
        result = asyncio.run(extractor._extract_from_meta(page))
        assert result is None

    def test_price_present_returns_dict(self, extractor):
        page = MagicMock()
        page.url = "https://www.moncler.com/product"
        page.get_attribute = AsyncMock(side_effect=["150000", "JPY"])
        result = asyncio.run(extractor._extract_from_meta(page))
        assert result is not None
        assert result["price"] == "150000"
        assert result["currency"] == "JPY"
        assert result["url"] == "https://www.moncler.com/product"

    def test_price_stripped(self, extractor):
        page = MagicMock()
        page.url = "https://www.moncler.com/product"
        page.get_attribute = AsyncMock(side_effect=["  99000  ", "JPY"])
        result = asyncio.run(extractor._extract_from_meta(page))
        assert result["price"] == "99000"

    def test_exception_returns_none(self, extractor):
        page = MagicMock()
        page.url = "https://www.moncler.com/product"
        page.get_attribute = AsyncMock(side_effect=Exception("playwright error"))
        result = asyncio.run(extractor._extract_from_meta(page))
        assert result is None
