"""product_matching ユーティリティのテスト"""

from __future__ import annotations

from app.utils.product_matching import (
    cross_site_match_score,
    extract_model_numbers,
    extract_tokens,
    match_score_buyma,
    normalize_color,
)


class TestExtractModelNumbers:
    def test_extracts_alphanumeric_codes(self):
        result = extract_model_numbers("Gucci Wallet 523153 GG Supreme")
        assert "523153" not in result  # pure digits, no alpha → excluded

    def test_extracts_mixed_codes(self):
        result = extract_model_numbers("Gucci Wallet ABC12345 Beige")
        assert "ABC12345" in result

    def test_empty_string(self):
        assert extract_model_numbers("") == set()


class TestExtractTokens:
    def test_basic_tokenization(self):
        tokens = extract_tokens("GG Supreme Continental Wallet")
        assert "GG" in tokens
        assert "Supreme" in tokens
        assert "Continental" in tokens

    def test_removes_noise(self):
        tokens = extract_tokens("GG Supreme 即発 中古 Wallet")
        assert "即発" not in tokens
        assert "中古" not in tokens
        assert "Wallet" in tokens


class TestNormalizeColor:
    def test_english_color(self):
        assert normalize_color("GG Supreme Wallet Black") == "BLACK"

    def test_italian_color(self):
        assert normalize_color("Portafoglio Nero") == "BLACK"

    def test_japanese_color(self):
        assert normalize_color("GGスプリーム ブラック") == "BLACK"

    def test_no_color(self):
        assert normalize_color("GG Supreme Wallet") is None


class TestCrossSiteMatchScore:
    def test_exact_match(self):
        score = cross_site_match_score(
            "GG Supreme Continental Wallet",
            "GG Supreme Continental Wallet",
        )
        assert score > 0.9

    def test_partial_match(self):
        score = cross_site_match_score(
            "GG Supreme Continental Wallet",
            "Gucci GG Supreme Continental Wallet Beige",
        )
        assert score > 0.5

    def test_no_match(self):
        score = cross_site_match_score(
            "GG Supreme Continental Wallet",
            "Prada Re-Nylon Shoulder Bag",
        )
        assert score < 0.3

    def test_brand_requirement(self):
        score = cross_site_match_score(
            "GG Supreme Wallet",
            "GG Supreme Wallet",
            brand="Gucci",
            require_brand_in_candidate=True,
        )
        assert score == 0.0  # "GUCCI" not in "GG Supreme Wallet"

    def test_brand_present(self):
        score = cross_site_match_score(
            "GG Supreme Wallet",
            "Gucci GG Supreme Wallet",
            brand="Gucci",
            require_brand_in_candidate=True,
        )
        assert score > 0.5


class TestMatchScoreBuyma:
    def test_brand_required(self):
        score = match_score_buyma(
            "GG Supreme Wallet",
            "GG Supreme Wallet",
            "Gucci",
        )
        assert score == 0.0  # brand not in buyma_name

    def test_brand_present_high_score(self):
        score = match_score_buyma(
            "GG Supreme Continental Wallet",
            "Gucci GG Supreme Continental Wallet Black",
            "Gucci",
        )
        assert score > 0.7

    def test_different_products_low_score(self):
        score = match_score_buyma(
            "GG Supreme Continental Wallet",
            "Gucci Horsebit 1955 Shoulder Bag",
            "Gucci",
        )
        assert score < 0.4
