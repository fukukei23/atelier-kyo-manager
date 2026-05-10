"""Selector ranking by site constraints and CSS specificity."""

from __future__ import annotations

from typing import Any


def rank_selectors(
    candidates: list[dict[str, Any]],
    site_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """LLM が返した複数セレクタをスコアリングしてランキングする."""
    if not candidates:
        return []

    scored_candidates = []
    for candidate in candidates:
        selector = candidate.get("new_selector", "")
        score = 0.0

        if matches_site_constraints(selector, site_config):
            score += 20.0

        if "[data-testid" in selector or "[data-" in selector:
            score += 15.0

        if "__" in selector and selector.startswith("."):
            score += 10.0

        specificity_score = calculate_specificity(selector)
        score += min(specificity_score * 0.5, 5.0)

        if selector.strip() in ["div", "span", "a", "p", "h1", "h2", "h3"]:
            score -= 10.0

        confidence = candidate.get("confidence", 0.0)
        score += confidence * 10.0

        scored_candidates.append({**candidate, "_ranking_score": score})

    ranked = sorted(scored_candidates, key=lambda x: x.get("_ranking_score", 0.0), reverse=True)

    for candidate in ranked:
        candidate.pop("_ranking_score", None)

    return ranked


def matches_site_constraints(selector: str, site_config: dict[str, Any]) -> bool:
    """セレクタがサイト固有制約に合致するかチェック."""
    url_rules = site_config.get("url_rules", {})
    allow_path_patterns = url_rules.get("allow_path_patterns", [])

    if allow_path_patterns:
        has_products_pattern = any("/products/" in p or "/p/" in p for p in allow_path_patterns)
        if has_products_pattern and "/products/" not in selector and "/p/" not in selector:
            return False

    allowed_domain = site_config.get("allowed_domain")
    if allowed_domain and allowed_domain in selector:
        return True

    return True


def calculate_specificity(selector: str) -> float:
    """CSS specificity を計算（簡易版）."""
    score = 0.0
    score += selector.count("#") * 100.0
    score += selector.count(".") * 10.0
    score += selector.count("[") * 10.0
    score += selector.count(":") * 10.0

    if selector.strip() in ["div", "span", "a", "p", "h1", "h2", "h3", "h4", "h5", "h6"]:
        score += 1.0
    else:
        score += selector.count(" ") * 1.0

    return score
