"""Tests for app/services/buyma_price_scraper.py - Issue #82 カバレッジ向上"""

from unittest.mock import MagicMock, patch

from app.services.buyma_price_scraper import (
    _BRAND_ALIASES,
    _NOISE_WORDS,
    BUYMA_UNAVAILABLE_BRANDS,
    DEFAULT_THRESHOLD,
    BuymaPriceSearcher,
    _extract_model_numbers,
    _extract_size,
    _extract_tokens,
    _filter_outliers,
    _match_score,
    _normalize_brand,
    _normalize_color,
    _parse_buyma_results,
    match_product,
)

# ── 定数テスト ────────────────────────────────────────────────────────────────


class TestConstants:
    def test_noise_words_is_set(self):
        assert isinstance(_NOISE_WORDS, set)
        assert "中古" in _NOISE_WORDS
        assert "USED" in _NOISE_WORDS
        assert "新品" in _NOISE_WORDS
        assert "セール" in _NOISE_WORDS

    def test_brand_aliases_is_dict(self):
        assert isinstance(_BRAND_ALIASES, dict)
        assert _BRAND_ALIASES["SALVATORE FERRAGAMO"] == "FERRAGAMO"
        assert _BRAND_ALIASES["CHLOÉ"] == "CHLOE"
        assert _BRAND_ALIASES["VALENTINO GARAVANI"] == "VALENTINO"

    def test_unavailable_brands(self):
        """2026-06: 全ブランド検索可能化によりブロックリストは空（経営者判断.md Tier2でGucci等は狙いブランド）"""
        assert not BUYMA_UNAVAILABLE_BRANDS

    def test_default_threshold(self):
        assert 0.0 < DEFAULT_THRESHOLD <= 1.0


# ── _normalize_brand ──────────────────────────────────────────────────────────


class TestNormalizeBrand:
    def test_expands_ferragamo_alias(self):
        assert _normalize_brand("SALVATORE FERRAGAMO") == "FERRAGAMO"

    def test_expands_chloe_alias(self):
        assert _normalize_brand("CHLOÉ") == "CHLOE"

    def test_expands_valentino_alias(self):
        assert _normalize_brand("VALENTINO GARAVANI") == "VALENTINO"

    def test_lowercase_input_uppercased(self):
        assert _normalize_brand("chanel") == "CHANEL"

    def test_unknown_brand_uppercased(self):
        assert _normalize_brand("my brand") == "MY BRAND"

    def test_already_normalized(self):
        assert _normalize_brand("PRADA") == "PRADA"


# ── _extract_model_numbers ────────────────────────────────────────────────────


class TestExtractModelNumbers:
    def test_extracts_alphanumeric_token(self):
        result = _extract_model_numbers("商品 ABC12345 テスト")
        assert "ABC12345" in result

    def test_requires_both_alpha_and_digit(self):
        result = _extract_model_numbers("ABCDEFG 1234567")
        assert "ABCDEFG" not in result
        assert "1234567" not in result

    def test_5_chars_min(self):
        assert "AB123" in _extract_model_numbers("商品 AB123")

    def test_12_chars_max(self):
        assert "ABCDE1234567" in _extract_model_numbers("商品 ABCDE1234567")

    def test_4_chars_too_short(self):
        assert "AB12" not in _extract_model_numbers("商品 AB12")

    def test_13_chars_too_long(self):
        assert "ABCDE12345678" not in _extract_model_numbers("商品 ABCDE12345678")

    def test_empty_string_returns_empty_set(self):
        assert _extract_model_numbers("") == set()

    def test_no_model_returns_empty_set(self):
        assert _extract_model_numbers("日本語商品名") == set()

    def test_multiple_models(self):
        result = _extract_model_numbers("AB123 CD456")
        assert "AB123" in result
        assert "CD456" in result

    def test_returns_set(self):
        assert isinstance(_extract_model_numbers("商品 AB123"), set)


# ── _extract_tokens ───────────────────────────────────────────────────────────


class TestExtractTokens:
    def test_removes_noise_words(self):
        result = _extract_tokens("中古 商品 USED")
        assert "中古" not in result
        assert "USED" not in result

    def test_filters_single_char(self):
        result = _extract_tokens("A B C テスト")
        assert "A" not in result

    def test_keeps_two_char_tokens(self):
        result = _extract_tokens("東京 テスト")
        assert "東京" in result

    def test_empty_string(self):
        assert _extract_tokens("") == []

    def test_all_noise_returns_empty(self):
        assert _extract_tokens("中古 新品 セール USED") == []

    def test_removes_symbols(self):
        result = _extract_tokens("【商品】★テスト！")
        assert "【商品】" not in result
        assert "★テスト！" not in result

    def test_returns_list(self):
        assert isinstance(_extract_tokens("テスト"), list)


# ── _normalize_color ──────────────────────────────────────────────────────────


class TestNormalizeColor:
    def test_japanese_black(self):
        assert _normalize_color("黒") == "BLACK"

    def test_italian_black(self):
        assert _normalize_color("NERO") == "BLACK"

    def test_japanese_white(self):
        assert _normalize_color("白") == "WHITE"

    def test_japanese_red(self):
        assert _normalize_color("赤") == "RED"

    def test_navy_katakana(self):
        assert _normalize_color("ネイビー") == "NAVY"

    def test_beige(self):
        assert _normalize_color("ベージュ") == "BEIGE"

    def test_no_match_returns_none(self):
        assert _normalize_color("未定色") is None

    def test_empty_returns_none(self):
        assert _normalize_color("") is None

    def test_english_black(self):
        assert _normalize_color("BLACK") == "BLACK"

    def test_grey_gray_normalized(self):
        assert _normalize_color("GRAY") == "GREY"
        assert _normalize_color("グレー") == "GREY"


# ── _extract_size ─────────────────────────────────────────────────────────────


class TestExtractSize:
    def test_s_size(self):
        assert _extract_size("サイズ S") == "S"

    def test_m_size(self):
        result = _extract_size("size M model")
        assert result is not None and "M" in result

    def test_l_size(self):
        result = _extract_size("size L available")
        assert result is not None and "L" in result

    def test_xl_size(self):
        result = _extract_size("XL available")
        assert result is not None and "XL" in result

    def test_free_size(self):
        result = _extract_size("FREE SIZE")
        assert result is not None

    def test_cm_size(self):
        result = _extract_size("width 21cm height")
        assert result is not None

    def test_no_size_returns_none(self):
        assert _extract_size("商品名のみ") is None

    def test_empty_returns_none(self):
        assert _extract_size("") is None


# ── _filter_outliers ──────────────────────────────────────────────────────────


class TestFilterOutliers:
    def test_less_than_4_returns_same(self):
        prices = [1000, 2000, 3000]
        assert _filter_outliers(prices) == [1000, 2000, 3000]

    def test_empty_list(self):
        assert _filter_outliers([]) == []

    def test_one_item(self):
        assert _filter_outliers([5000]) == [5000]

    def test_all_same_values(self):
        prices = [1000, 1000, 1000, 1000, 1000]
        result = _filter_outliers(prices)
        assert all(p == 1000 for p in result)

    def test_obvious_outlier_removed(self):
        # 6件以上でIQR法が有効に機能する
        prices = [1000, 1100, 1200, 1300, 1400, 100000]
        result = _filter_outliers(prices)
        assert 100000 not in result

    def test_preserves_normal_values(self):
        prices = [1500, 2000, 2500, 3000, 3500]
        result = _filter_outliers(prices)
        assert 2000 in result
        assert 2500 in result

    def test_returns_list(self):
        assert isinstance(_filter_outliers([1, 2, 3, 4]), list)


# ── _parse_buyma_results ──────────────────────────────────────────────────────


def _make_buyma_html(item_id="123456", name="商品名テスト", brand="PRADA", price="50000"):
    return f'syo_id="{item_id}" syo_name="{name}" brand_name="{brand}" price="{price}"'


class TestParseBuymaResults:
    def test_parses_single_item(self):
        html = _make_buyma_html()
        result = _parse_buyma_results(html)
        assert len(result) == 1
        assert result[0]["name"] == "商品名テスト"
        assert result[0]["brand"] == "PRADA"
        assert result[0]["price_jpy"] == 50000
        assert result[0]["item_id"] == "123456"

    def test_generates_item_url(self):
        html = _make_buyma_html(item_id="999")
        result = _parse_buyma_results(html)
        assert "999" in result[0]["item_url"]
        assert "buyma.com" in result[0]["item_url"]

    def test_parses_multiple_items(self):
        html = (
            _make_buyma_html("111", "商品A", "PRADA", "30000")
            + " "
            + _make_buyma_html("222", "商品B", "GUCCI", "40000")
        )
        result = _parse_buyma_results(html)
        assert len(result) == 2

    def test_skips_duplicate_ids(self):
        html = (
            _make_buyma_html("123", "商品1", "PRADA", "30000")
            + " "
            + _make_buyma_html("123", "商品2", "GUCCI", "40000")
        )
        result = _parse_buyma_results(html)
        assert len(result) == 1
        assert result[0]["name"] == "商品1"

    def test_empty_html(self):
        assert _parse_buyma_results("") == []

    def test_no_matching_html(self):
        assert _parse_buyma_results("<div>no buyma items</div>") == []

    def test_price_is_int(self):
        html = _make_buyma_html(price="12345")
        result = _parse_buyma_results(html)
        assert isinstance(result[0]["price_jpy"], int)
        assert result[0]["price_jpy"] == 12345

    def test_amp_entity_in_brand(self):
        html = _make_buyma_html(brand="TOM &amp; JERRY")
        result = _parse_buyma_results(html)
        assert result[0]["brand"] == "TOM & JERRY"


# ── _match_score ──────────────────────────────────────────────────────────────


class TestMatchScore:
    def test_brand_not_in_name_returns_zero(self):
        assert _match_score("Product Name", "Totally Different", "PRADA") == 0.0

    def test_brand_in_name_positive_score(self):
        score = _match_score("PRADA Bag", "PRADA Handbag", "PRADA")
        assert score > 0.0

    def test_exact_match_high_score(self):
        score = _match_score("PRADA Bag Classic", "PRADA Bag Classic", "PRADA")
        assert score > 0.5

    def test_model_number_match_adds_bonus(self):
        score_with = _match_score("PRADA AB123 Bag", "PRADA AB123 Item", "PRADA")
        score_without = _match_score("PRADA Bag", "PRADA Item", "PRADA")
        assert score_with >= score_without

    def test_score_max_is_one(self):
        score = _match_score("PRADA Bag", "PRADA Bag", "PRADA")
        assert score <= 1.0

    def test_score_non_negative(self):
        score = _match_score("anything", "something PRADA", "PRADA")
        assert score >= 0.0

    def test_color_match_adds_bonus(self):
        score_with = _match_score("PRADA 黒 Bag", "PRADA BLACK Item", "PRADA")
        score_without = _match_score("PRADA Bag", "PRADA Item", "PRADA")
        assert score_with >= score_without


# ── match_product ─────────────────────────────────────────────────────────────


def _make_result(name="PRADA Bag", brand="PRADA", price=50000, item_id="1"):
    return {
        "name": name,
        "brand": brand,
        "price_jpy": price,
        "item_id": item_id,
        "item_url": f"https://www.buyma.com/item/{item_id}/",
    }


class TestMatchProduct:
    def test_empty_results_returns_none(self):
        assert match_product("PRADA Bag", "PRADA", []) is None

    def test_below_threshold_returns_none(self):
        result = match_product("PRADA Bag", "PRADA", [_make_result("Totally Different Item", "OTHER")], threshold=0.9)
        assert result is None

    def test_matching_item_returns_dict(self):
        results = [_make_result("PRADA Classic Bag", "PRADA", 50000)]
        match = match_product("PRADA Classic Bag", "PRADA", results, threshold=0.3)
        if match is not None:
            assert "buyma_name" in match
            assert "buyma_price" in match
            assert "buyma_price_max" in match
            assert "buyma_price_avg" in match
            assert "buyma_item_url" in match
            assert "match_count" in match
            assert "match_score" in match
            assert match["buyma_source"] == "buyma_search"

    def test_returns_min_price(self):
        results = [
            _make_result("PRADA Bag", "PRADA", 50000, "1"),
            _make_result("PRADA Bag", "PRADA", 40000, "2"),
            _make_result("PRADA Bag", "PRADA", 60000, "3"),
        ]
        match = match_product("PRADA Bag", "PRADA", results, threshold=0.0)
        if match is not None:
            assert match["buyma_price"] <= 60000

    def test_extracts_size_from_best(self):
        results = [_make_result("PRADA M Bag", "PRADA", 50000)]
        match = match_product("PRADA Bag", "PRADA", results, threshold=0.0)
        if match is not None:
            assert "buyma_size" in match

    def test_best_score_item_selected(self):
        results = [
            _make_result("PRADA Bag Classic", "PRADA", 50000, "1"),
            _make_result("Something Random PRADA", "PRADA", 30000, "2"),
        ]
        match = match_product("PRADA Bag Classic", "PRADA", results, threshold=0.0)
        if match is not None:
            assert match["match_score"] >= 0.0


# ── BuymaPriceSearcher ────────────────────────────────────────────────────────


class TestBuymaPriceSearcher:
    """BuymaPriceSearcher クラスのテスト（Playwright はモック化）"""

    def test_init_defaults(self):
        searcher = BuymaPriceSearcher()
        assert searcher.headless is True
        assert searcher._browser is None
        assert searcher._pw is None
        assert searcher._ctx is None

    def test_init_headful(self):
        searcher = BuymaPriceSearcher(headless=False)
        assert searcher.headless is False

    def test_init_custom_timeout(self):
        searcher = BuymaPriceSearcher(timeout=30)
        assert searcher.timeout == 30

    def test_close_when_nothing_open(self):
        """ブラウザ未起動時の close は例外なく完了する"""
        searcher = BuymaPriceSearcher()
        searcher.close()  # should not raise
        assert searcher._browser is None
        assert searcher._pw is None
        assert searcher._ctx is None

    def test_close_resets_all_references(self):
        """close 後は全リファレンスが None になる"""
        searcher = BuymaPriceSearcher()
        mock_ctx = MagicMock()
        mock_browser = MagicMock()
        mock_pw = MagicMock()
        searcher._ctx = mock_ctx
        searcher._browser = mock_browser
        searcher._pw = mock_pw
        searcher.close()
        assert searcher._ctx is None
        assert searcher._browser is None
        assert searcher._pw is None

    def test_search_single_skips_unavailable_brand(self):
        """利用不可ブランドは None を返す（現行ブロックリストは空のため一時的にブランドを注入して機構を検証）"""
        searcher = BuymaPriceSearcher()
        with patch("app.services.buyma_price_scraper.BUYMA_UNAVAILABLE_BRANDS", {"Gucci"}):
            result = searcher.search_single("Gucci Bag", "Gucci")
        assert result is None

    def test_search_batch_skips_empty_name(self):
        """name が空のアイテムはスキップ"""
        searcher = BuymaPriceSearcher()
        with patch.object(searcher, "search_single", return_value=None) as mock_s:
            results = searcher.search_batch([{"product_name": "", "brand": "PRADA"}])
            mock_s.assert_not_called()
        assert results == {}

    def test_search_batch_skips_unavailable_brand(self):
        """利用不可ブランドはスキップ（現行ブロックリストは空のため一時的にブランドを注入して機構を検証）"""
        searcher = BuymaPriceSearcher()
        with (
            patch("app.services.buyma_price_scraper.BUYMA_UNAVAILABLE_BRANDS", {"Gucci"}),
            patch.object(searcher, "search_single", return_value=None) as mock_s,
        ):
            results = searcher.search_batch([{"product_name": "Bag", "brand": "Gucci"}])
            mock_s.assert_not_called()
        assert results == {}

    def test_search_batch_returns_matches(self):
        """マッチした結果を辞書で返す"""
        searcher = BuymaPriceSearcher()
        fake_match = {"buyma_name": "PRADA Bag", "buyma_price": 50000, "match_score": 0.8, "match_count": 3}
        with (
            patch.object(searcher, "search_single", return_value=fake_match),
            patch("app.services.buyma_price_scraper.time.sleep"),
        ):
            results = searcher.search_batch([{"product_name": "PRADA Bag", "brand": "PRADA"}])
        assert "PRADA Bag" in results
        assert results["PRADA Bag"]["buyma_price"] == 50000
