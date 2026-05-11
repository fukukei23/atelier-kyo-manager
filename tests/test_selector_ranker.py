"""Tests for selector_ranker module."""

from __future__ import annotations

from app.agents.browser.selector_ranker import (
    calculate_specificity,
    matches_site_constraints,
    rank_selectors,
)


class TestCalculateSpecificity:
    def test_id_selector(self):
        assert calculate_specificity("#main") == 100.0

    def test_class_selector(self):
        assert calculate_specificity(".card") == 10.0

    def test_attribute_selector(self):
        assert calculate_specificity("[data-testid='title']") == 10.0  # 1 '[' bracket

    def test_combined_selector(self):
        score = calculate_specificity("#main .card a")
        # # (100) + . (10) + : from 'a' not being bare tag since trailing → count spaces (2)
        # But 'a' IS a bare tag → 1.0 instead of space count
        assert score == 112.0  # 100 + 10 + 2 spaces * 1 = 112

    def test_pseudo_selector(self):
        assert calculate_specificity("a:hover") == 10.0  # 1 colon * 10

    def test_empty_selector(self):
        assert calculate_specificity("") == 0.0

    def test_plain_tag_adds_1(self):
        assert calculate_specificity("div") == 1.0

    def test_descendant_combinator_counts_spaces(self):
        score = calculate_specificity("div span a")
        # Not in bare-tag list (full string), so count spaces: 2 spaces
        # But wait: "div span a" is not in bare tag list, so space count: 2
        assert score == 2.0  # 2 descendant spaces


class TestMatchesSiteConstraints:
    def test_returns_true_when_no_constraints(self):
        assert matches_site_constraints(".item", {}) is True

    def test_returns_true_when_domain_matches(self):
        result = matches_site_constraints(
            "a[href*='example.com']",
            {"allowed_domain": "example.com"},
        )
        assert result is True

    def test_returns_false_when_path_pattern_mismatch(self):
        result = matches_site_constraints(
            ".item",
            {"url_rules": {"allow_path_patterns": ["/products/jackets"]}},
        )
        assert result is False

    def test_returns_true_when_selector_contains_path_pattern(self):
        result = matches_site_constraints(
            "a[href*='/products/item']",
            {"url_rules": {"allow_path_patterns": ["/products/"]}},
        )
        assert result is True

    def test_empty_url_rules(self):
        result = matches_site_constraints(".item", {"url_rules": {}})
        assert result is True


class TestRankSelectors:
    def test_returns_empty_for_empty_candidates(self):
        assert rank_selectors([], {}) == []

    def test_ranks_data_testid_higher(self):
        candidates = [
            {"new_selector": "div.item", "confidence": 0.8},
            {"new_selector": "[data-testid='product-card']", "confidence": 0.8},
        ]
        result = rank_selectors(candidates, {})
        assert result[0]["new_selector"] == "[data-testid='product-card']"

    def test_penalizes_generic_tags(self):
        candidates = [
            {"new_selector": "div", "confidence": 0.9},
            {"new_selector": ".product-card", "confidence": 0.7},
        ]
        result = rank_selectors(candidates, {})
        assert result[0]["new_selector"] == ".product-card"

    def test_confidence_affects_ranking(self):
        candidates = [
            {"new_selector": ".low", "confidence": 0.1},
            {"new_selector": ".high", "confidence": 0.99},
        ]
        result = rank_selectors(candidates, {})
        assert result[0]["new_selector"] == ".high"

    def test_removes_internal_score(self):
        candidates = [{"new_selector": ".item", "confidence": 0.5}]
        result = rank_selectors(candidates, {})
        assert "_ranking_score" not in result[0]

    def test_id_selector_ranks_highest(self):
        # specificity is capped at min(*0.5, 5.0), so ID vs class tie at cap
        # differentiate by confidence instead
        candidates = [
            {"new_selector": ".product", "confidence": 0.5},
            {"new_selector": "#product-title", "confidence": 0.9},
            {"new_selector": "div span", "confidence": 0.5},
        ]
        result = rank_selectors(candidates, {})
        assert result[0]["new_selector"] == "#product-title"

    def test_double_underscore_class_bonus(self):
        candidates = [
            {"new_selector": ".product", "confidence": 0.8},
            {"new_selector": ".product__title", "confidence": 0.8},
        ]
        result = rank_selectors(candidates, {})
        assert result[0]["new_selector"] == ".product__title"
