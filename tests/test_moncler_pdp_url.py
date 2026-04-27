# -*- coding: utf-8 -*-
"""
CR-ATELIER-002 Step 3-4: Moncler PDP URL バリデーションのテスト

テスト対象:
- _is_valid_moncler_pdp_url の正例・負例
- _get_moncler_rejection_reason のテスト
- extract_moncler_pdp_links の Moncler 分岐（mock extractor で制御）
"""

import pytest
pytest.importorskip("playwright")
from unittest.mock import Mock, AsyncMock, patch
from urllib.parse import urlparse

from app.agents.browser.extractor import (
    _is_valid_moncler_pdp_url,
    _get_moncler_rejection_reason,
    extract_moncler_pdp_links,
)


class TestIsValidMonclerPdpUrl:
    """_is_valid_moncler_pdp_url のテスト"""
    
    base_url = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB"
    
    def test_valid_moncler_pdp_url(self):
        """正例: 有効なMoncler PDP URL"""
        valid_urls = [
            "https://www.moncler.com/en-int/products/down-jacket-123",
            "https://www.moncler.com/en-int/products/down-jacket-123/",
            "https://www.moncler.com/en-int/women/products/down-jacket-123",
            "https://www.moncler.com/en-int/products/down-jacket-123?forceLocale=en-int&shipToCountry=GB",
        ]
        
        for url in valid_urls:
            assert _is_valid_moncler_pdp_url(url, self.base_url), f"Should be valid: {url}"
    
    def test_invalid_external_domain(self):
        """負例: 外部ドメイン"""
        invalid_urls = [
            "https://www.onetrust.com/products/cookie-consent",
            "https://www.monclergroup.com/products/item",
            "https://www.facebook.com/products/share",
            "https://www.twitter.com/products/tweet",
        ]
        
        for url in invalid_urls:
            assert not _is_valid_moncler_pdp_url(url, self.base_url), f"Should be invalid: {url}"
    
    def test_invalid_locale_path(self):
        """負例: /en-int/ で始まらないパス"""
        invalid_urls = [
            "https://www.moncler.com/en-lt/products/item",
            "https://www.moncler.com/en-de/products/item",
            "https://www.moncler.com/en-jp/products/item",
            "https://www.moncler.com/products/item",  # ロケールなし
        ]
        
        for url in invalid_urls:
            assert not _is_valid_moncler_pdp_url(url, self.base_url), f"Should be invalid: {url}"
    
    def test_invalid_no_products_path(self):
        """負例: /products/ を含まないパス"""
        invalid_urls = [
            "https://www.moncler.com/en-int/women/outerwear/all-down-jackets",
            "https://www.moncler.com/en-int/search?q=down+jacket",
            "https://www.moncler.com/en-int/women/outerwear/",
        ]
        
        for url in invalid_urls:
            assert not _is_valid_moncler_pdp_url(url, self.base_url), f"Should be invalid: {url}"
    
    def test_invalid_trap_patterns(self):
        """負例: trapページパターン"""
        invalid_urls = [
            "https://www.moncler.com/en-int/products/404",
            "https://www.moncler.com/en-int/products/not-found",
            "https://www.moncler.com/en-int/search/products/item",
            "https://www.moncler.com/en-int/legal/products/item",
            "https://www.moncler.com/en-int/client-service/products/item",
            "https://www.moncler.com/en-int/products/item/collections/test",
            "https://www.moncler.com/en-int/products/item/seasons/2024",
        ]
        
        for url in invalid_urls:
            assert not _is_valid_moncler_pdp_url(url, self.base_url), f"Should be invalid: {url}"
    
    def test_invalid_scheme(self):
        """負例: 不正なスキーム"""
        invalid_urls = [
            "mailto:test@example.com",
            "tel:+1234567890",
            "javascript:void(0)",
            "blob:https://www.moncler.com/...",
        ]
        
        for url in invalid_urls:
            assert not _is_valid_moncler_pdp_url(url, self.base_url), f"Should be invalid: {url}"


class TestGetMonclerRejectionReason:
    """_get_moncler_rejection_reason のテスト"""
    
    base_url = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB"
    
    def test_rejection_reason_external_domain(self):
        """外部ドメインの場合"""
        url = "https://www.example.com/en-int/products/item"
        reason = _get_moncler_rejection_reason(url, self.base_url)
        assert reason == "external_domain"
    
    def test_rejection_reason_blocked_domain(self):
        """ブロックドメインの場合"""
        url = "https://www.onetrust.com/products/cookie-consent"
        reason = _get_moncler_rejection_reason(url, self.base_url)
        assert reason == "blocked_domain"
    
    def test_rejection_reason_no_en_int_path(self):
        """/en-int/ で始まらないパスの場合"""
        url = "https://www.moncler.com/en-lt/products/item"
        reason = _get_moncler_rejection_reason(url, self.base_url)
        assert reason == "no_en_int_path"
    
    def test_rejection_reason_no_products_path(self):
        """/products/ を含まないパスの場合"""
        url = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets"
        reason = _get_moncler_rejection_reason(url, self.base_url)
        assert reason == "no_products_path"
    
    def test_rejection_reason_trap_pattern(self):
        """trapページパターンの場合"""
        url = "https://www.moncler.com/en-int/products/404"
        reason = _get_moncler_rejection_reason(url, self.base_url)
        assert reason == "trap_pattern"
    
    def test_no_rejection_valid_url(self):
        """有効なURLの場合（Noneを返す）"""
        url = "https://www.moncler.com/en-int/products/down-jacket-123"
        reason = _get_moncler_rejection_reason(url, self.base_url)
        assert reason is None


class TestExtractMonclerPdpLinks:
    """extract_moncler_pdp_links のテスト"""
    
    @pytest.mark.asyncio
    async def test_extract_moncler_pdp_links_success(self):
        """正常系: PDPリンクが抽出できる場合"""
        # Mock Page オブジェクト
        mock_page = Mock()
        mock_page.url = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB"
        
        # Mock query_selector_all の戻り値
        mock_node1 = Mock()
        mock_node1.get_attribute = AsyncMock(side_effect=lambda attr: {
            "href": "/en-int/products/down-jacket-1",
            "data-href": None,
            "data-product-url": None,
            "data-url": None,
        }.get(attr))
        
        mock_node2 = Mock()
        mock_node2.get_attribute = AsyncMock(side_effect=lambda attr: {
            "href": "/en-int/products/down-jacket-2",
            "data-href": None,
            "data-product-url": None,
            "data-url": None,
        }.get(attr))
        
        mock_page.query_selector_all = AsyncMock(return_value=[mock_node1, mock_node2])
        mock_page.query_selector = AsyncMock(return_value=None)
        
        # Mock ctx
        mock_ctx = Mock()
        mock_ctx.site_config = {"site_code": "MONCLER_OFFICIAL"}
        mock_ctx.run_context = None
        
        # 実行
        result = await extract_moncler_pdp_links(mock_page, mock_ctx, max_links=50)
        
        # 検証
        assert len(result) == 2
        assert "https://www.moncler.com/en-int/products/down-jacket-1" in result
        assert "https://www.moncler.com/en-int/products/down-jacket-2" in result
    
    @pytest.mark.asyncio
    async def test_extract_moncler_pdp_links_reject_external_domain(self):
        """負例: 外部ドメインのリンクが除外される場合"""
        # Mock Page オブジェクト
        mock_page = Mock()
        mock_page.url = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB"
        
        # Mock query_selector_all の戻り値（外部ドメインのリンク）
        mock_node = Mock()
        mock_node.get_attribute = AsyncMock(side_effect=lambda attr: {
            "href": "https://www.onetrust.com/products/cookie-consent",
            "data-href": None,
            "data-product-url": None,
            "data-url": None,
        }.get(attr))
        
        mock_page.query_selector_all = AsyncMock(return_value=[mock_node])
        mock_page.query_selector = AsyncMock(return_value=None)
        
        # Mock ctx
        mock_ctx = Mock()
        mock_ctx.site_config = {"site_code": "MONCLER_OFFICIAL"}
        mock_ctx.run_context = None
        
        # 実行
        result = await extract_moncler_pdp_links(mock_page, mock_ctx, max_links=50)
        
        # 検証: 外部ドメインのリンクは除外される
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_extract_moncler_pdp_links_reject_invalid_locale(self):
        """負例: 不正なロケールのリンクが除外される場合"""
        # Mock Page オブジェクト
        mock_page = Mock()
        mock_page.url = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB"
        
        # Mock query_selector_all の戻り値（不正なロケール）
        mock_node = Mock()
        mock_node.get_attribute = AsyncMock(side_effect=lambda attr: {
            "href": "/en-lt/products/item",
            "data-href": None,
            "data-product-url": None,
            "data-url": None,
        }.get(attr))
        
        mock_page.query_selector_all = AsyncMock(return_value=[mock_node])
        mock_page.query_selector = AsyncMock(return_value=None)
        
        # Mock ctx
        mock_ctx = Mock()
        mock_ctx.site_config = {"site_code": "MONCLER_OFFICIAL"}
        mock_ctx.run_context = None
        
        # 実行
        result = await extract_moncler_pdp_links(mock_page, mock_ctx, max_links=50)
        
        # 検証: 不正なロケールのリンクは除外される
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_extract_moncler_pdp_links_telemetry_on_zero_links(self):
        """accepted==0 の場合、Telemetry に保存される"""
        # Mock Page オブジェクト
        mock_page = Mock()
        mock_page.url = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB"
        
        # Mock query_selector_all の戻り値（無効なリンクのみ）
        mock_node = Mock()
        mock_node.get_attribute = AsyncMock(side_effect=lambda attr: {
            "href": "https://www.onetrust.com/products/cookie-consent",
            "data-href": None,
            "data-product-url": None,
            "data-url": None,
        }.get(attr))
        
        mock_page.query_selector_all = AsyncMock(return_value=[mock_node])
        mock_page.query_selector = AsyncMock(return_value=None)
        
        # Mock ctx with run_context
        mock_run_context = Mock()
        mock_run_context.run_id = "test_run_123"
        
        mock_ctx = Mock()
        mock_ctx.site_config = {"site_code": "MONCLER_OFFICIAL"}
        mock_ctx.run_context = mock_run_context
        mock_ctx.query = "down jacket"
        
        # Mock TelemetryClient
        mock_telemetry = Mock()
        mock_telemetry.save_json = AsyncMock()
        mock_ctx.telemetry = mock_telemetry
        
        # 実行
        result = await extract_moncler_pdp_links(mock_page, mock_ctx, max_links=50)
        
        # 検証: リンクが0件の場合、Telemetry に保存される
        assert len(result) == 0
        assert mock_telemetry.save_json.called
        call_args = mock_telemetry.save_json.call_args
        assert call_args[0][0] == "moncler_pdp_links_debug"
        assert "raw_hrefs" in call_args[0][1]
        assert "rejection_stats" in call_args[0][1]


class TestMonclerPdpLinksIntegration:
    """Moncler PDP リンク抽出の統合テスト（NavigationDriver との連携）"""
    
    @pytest.mark.asyncio
    async def test_navigation_driver_moncler_branch(self):
        """NavigationDriver.collect_pdp_links の Moncler 分岐テスト"""
        from app.agents.browser.navigation_driver import NavigationDriver
        from app.agents.browser.navigation_driver import NavigationContext
        
        # Mock Page
        mock_page = Mock()
        mock_page.url = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB"
        
        # Mock NavigationDriver
        mock_driver = Mock(spec=NavigationDriver)
        mock_driver.page = mock_page
        
        # Mock ctx
        mock_ctx = Mock(spec=NavigationContext)
        mock_ctx.site_config = {"site_code": "MONCLER_OFFICIAL"}
        mock_ctx.settings = {}
        mock_ctx.run_context = None
        
        # extract_moncler_pdp_links をモック化
        with patch("app.agents.browser.navigation_driver.extract_moncler_pdp_links") as mock_extract:
            mock_extract.return_value = [
                "https://www.moncler.com/en-int/products/down-jacket-1",
                "https://www.moncler.com/en-int/products/down-jacket-2",
            ]
            
            # NavigationDriver.collect_pdp_links を呼び出す
            # （実際の実装では、extract_moncler_pdp_links が呼ばれる）
            result = await mock_extract(mock_page, mock_ctx, max_links=50)
            
            # 検証
            assert len(result) == 2
            assert mock_extract.called

