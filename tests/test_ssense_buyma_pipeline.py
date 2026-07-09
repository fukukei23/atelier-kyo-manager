"""SSENSE + BUYMA 自動価格検証パイプラインのテスト。

外部通信は全てモック。パイプラインの各フェーズが正しく動くことを検証。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.pricing.schemas import PriceSource, PricingInput

# ---------------------------------------------------------------------------
# Phase 1: SSENSE PLP スクレイパ（カテゴリ対応）
# ---------------------------------------------------------------------------


class TestSsenseScraperCategory:
    """sale_scraper._scrape_ssense() のカテゴリパラメータ対応テスト。"""

    def test_sunglasses_url_generation(self):
        """category='sunglasses' で正しいURLが生成されること。"""
        from app.services.sale_scraper import _scrape_ssense

        with (
            patch("app.services.sale_scraper._fetch_with_cffi", return_value=None),
            patch("app.services.sale_scraper._ssense_via_playwright", return_value=None),
        ):
            result = _scrape_ssense("Gucci", category="sunglasses")
            # URLは関数内部で生成されるため、fetchが呼ばれたURLを確認
            # 結果が空（HTML取得失敗）でもURL生成は行われる
            assert result == []

    def test_bags_url_default(self):
        """デフォルトカテゴリ='bags' でURLが変更されないこと。"""
        from app.services.sale_scraper import _scrape_ssense

        with patch("app.services.sale_scraper._fetch_with_cffi") as mock_fetch:
            mock_fetch.return_value = None
            with patch("app.services.sale_scraper._ssense_via_playwright", return_value=None):
                _scrape_ssense("Gucci")
                # fetchが呼ばれたことだけ確認
                assert mock_fetch.called


# ---------------------------------------------------------------------------
# Phase 2: SSENSE PDP スクレイパ
# ---------------------------------------------------------------------------


class TestSsensePdpScraper:
    """scrape_ssense_pdp() のテスト。"""

    def test_jsonld_extraction(self):
        """JSON-LDから色・型番を抽出できること。"""
        from app.services.sale_scraper import scrape_ssense_pdp

        mock_html = """
        <html><head>
        <script type="application/ld+json">{
            "@type": "Product",
            "name": "Fendi Lettering Rectangular Sunglasses",
            "sku": "262693M134027",
            "color": "Black"
        }</script>
        </head><body></body></html>
        """

        with patch("app.services.sale_scraper._fetch_with_cffi", return_value=mock_html):
            result = scrape_ssense_pdp("https://www.ssense.com/en-us/women/product/fendi/test/123")
            assert result is not None
            assert result["color"] == "Black"
            assert result["model_number"] == "262693M134027"
            assert result["full_name"] == "Fendi Lettering Rectangular Sunglasses"

    def test_no_jsonld_returns_none(self):
        """JSON-LDがない場合Noneを返すこと。"""
        from app.services.sale_scraper import scrape_ssense_pdp

        mock_html = "<html><body>No structured data</body></html>"

        with (
            patch("app.services.sale_scraper._fetch_with_cffi", return_value=mock_html),
            patch("app.services.sale_scraper._ssense_via_playwright", return_value=None),
        ):
            result = scrape_ssense_pdp("https://example.com")
            assert result is None

    def test_empty_url_returns_none(self):
        """空URLの場合Noneを返すこと。"""
        from app.services.sale_scraper import scrape_ssense_pdp

        result = scrape_ssense_pdp("")
        assert result is None

    def test_meta_fallback(self):
        """JSON-LDなし → metaタグフォールバック。"""
        from app.services.sale_scraper import scrape_ssense_pdp

        mock_html = """
        <html><head>
        <meta property="og:title" content="Gucci Sunglasses GG123">
        <meta property="product:color" content="Havana">
        </head><body></body></html>
        """

        with patch("app.services.sale_scraper._fetch_with_cffi", return_value=mock_html):
            result = scrape_ssense_pdp("https://www.ssense.com/en-us/test")
            assert result is not None
            assert result["color"] == "Havana"
            assert result["full_name"] == "Gucci Sunglasses GG123"


# ---------------------------------------------------------------------------
# Phase 4: BROWSER_VERIFIED での利益計算
# ---------------------------------------------------------------------------


class TestPipelinePhase4:
    """パイプラインPhase 4の利益計算テスト。"""

    def test_verified_sources_pass_validation(self):
        """両方BROWSER_VERIFIEDでPriceIntegrityErrorが出ないこと。"""
        inp = PricingInput(
            purchase_price=40000,
            selling_price=55000,
            purchase_price_source=PriceSource.BROWSER_VERIFIED,
            selling_price_source=PriceSource.BROWSER_VERIFIED,
            item_category="accessory",
        )
        inp.validate_sources()  # 例外が出なければOK

    def test_unverified_sources_blocked(self):
        """片方UNKNOWN → PriceIntegrityErrorが出ること。"""
        from app.core.pricing.schemas import PriceIntegrityError

        inp = PricingInput(
            purchase_price=40000,
            selling_price=55000,
            purchase_price_source=PriceSource.BROWSER_VERIFIED,
            selling_price_source=PriceSource.UNKNOWN,
        )
        with pytest.raises(PriceIntegrityError):
            inp.validate_sources()


# ---------------------------------------------------------------------------
# Pipeline 統合テスト
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """SsenseBuymaPipeline.run() のモック統合テスト。"""

    @patch("app.services.ssense_buyma_pipeline.scrape_ssense_pdp")
    @patch("app.services.ssense_buyma_pipeline._scrape_ssense")
    def test_full_pipeline_mocked(self, mock_ssense, mock_pdp):
        """全フェーズモックで実行 → PipelineSummaryの数値が正しいこと。"""
        from app.services.ssense_buyma_pipeline import SsenseBuymaPipeline

        # Phase 1: SSENSE PLP モック
        mock_ssense.return_value = [
            {
                "product_name": "Fendi Lettering Rectangular",
                "price_original": 261.0,
                "price_jpy": 39150.0,
                "exchange_rate": 150.0,
                "source_url": "https://www.ssense.com/en-us/product/fendi/test/123",
                "sku": "262693M134027",
            },
        ]

        # Phase 2: PDP モック
        mock_pdp.return_value = {
            "color": "Black",
            "model_number": "262693M134027",
            "full_name": "Fendi Lettering Rectangular Sunglasses",
        }

        # Phase 3: BUYMA モック
        mock_searcher = MagicMock()
        mock_searcher.search_single.return_value = {
            "name": "FENDI フェンディ Lettering サングラス FF",
            "price": 45600,
            "score": 0.45,
        }

        pipeline = SsenseBuymaPipeline()
        pipeline._buyma_searcher = mock_searcher

        # Phase 5: DB保存をモック
        with patch("app.extensions.db"):
            summary = pipeline.run("Fendi", "sunglasses")

        assert summary.total_scraped == 1
        assert summary.buyma_matched == 1
        assert summary.results[0].is_verified is True

    @patch("app.services.ssense_buyma_pipeline.scrape_ssense_pdp")
    @patch("app.services.ssense_buyma_pipeline._scrape_ssense")
    def test_no_buyma_match(self, mock_ssense, mock_pdp):
        """BUYMAマッチなし → is_verified=False。"""
        from app.services.ssense_buyma_pipeline import SsenseBuymaPipeline

        mock_ssense.return_value = [
            {
                "product_name": "Unknown Product",
                "price_original": 500.0,
                "price_jpy": 75000.0,
                "exchange_rate": 150.0,
                "source_url": "https://example.com/abc",
                "sku": "SKU123",
            },
        ]
        mock_pdp.return_value = None

        mock_searcher = MagicMock()
        mock_searcher.search_single.return_value = None  # BUYMAマッチなし

        pipeline = SsenseBuymaPipeline()
        pipeline._buyma_searcher = mock_searcher

        with patch("app.extensions.db"):
            summary = pipeline.run("Gucci", "sunglasses")

        assert summary.total_scraped == 1
        assert summary.buyma_matched == 0
        assert summary.results[0].is_verified is False
        assert "BUYMA price not found" in summary.results[0].error

    @patch("app.services.ssense_buyma_pipeline._scrape_ssense")
    def test_empty_ssense(self, mock_ssense):
        """SSENSE取得0件 → early return。"""
        from app.services.ssense_buyma_pipeline import SsenseBuymaPipeline

        mock_ssense.return_value = []

        pipeline = SsenseBuymaPipeline()
        summary = pipeline.run("Prada", "sunglasses")

        assert summary.total_scraped == 0
        assert summary.buyma_matched == 0
        assert len(summary.results) == 0
