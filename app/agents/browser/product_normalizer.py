"""Price/text normalization utilities for product extraction."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse, urlunparse

# Default selectors (fallback when site_config has no definition)
DEFAULT_PRICE_SELECTORS = [
    "meta[property='product:price:amount']",
    "meta[itemprop='price']",
    "[itemprop=price]",
    "span:has(meta[itemprop='price'])",
    "[data-testid*=price]",
    "[class*=price]",
]

DEFAULT_TITLE_SELECTORS = [
    "h1[itemprop='name']",
    "h1.product-title",
    "h1",
    "meta[property='og:title']",
]

DEFAULT_SIZE_BUTTON_SELECTORS = [
    "button[aria-disabled='false'][data-size]",
    "button[aria-disabled='false'][data-testid*='size']",
    "button:not([disabled])[data-size]",
    "li[data-size] button:not([disabled])",
    "button[class*='size']:not([disabled])",
]


def normalize_price_text(price_text: str, price_rules: dict[str, Any]) -> str | None:
    """Normalize price text to a numeric string."""
    if not price_text:
        return None

    normalized = price_text
    for char in price_rules.get("strip_chars", []):
        normalized = normalized.replace(char, "")

    pattern = price_rules.get("price_pattern", r"[\d.,]+")
    match = re.search(pattern, normalized)
    if match:
        return match.group(0)

    return normalized.strip()


def normalize_price_to_float(price_text: str, price_rules: dict[str, Any]) -> float | None:
    """Normalize price text and convert to float."""
    if not price_text:
        return None

    try:
        normalized_str = normalize_price_text(price_text, price_rules)
        if not normalized_str:
            return None

        thousands_sep = price_rules.get("thousands_separator", ",")
        if thousands_sep:
            normalized_str = normalized_str.replace(thousands_sep, "")

        decimal_sep = price_rules.get("decimal_separator", ".")
        if decimal_sep != ".":
            normalized_str = normalized_str.replace(decimal_sep, ".")

        return float(normalized_str)
    except (ValueError, TypeError):
        return None


def normalize_image_url(
    src: str,
    page_url: str,
    base_url: str | None = None,
) -> str | None:
    """Normalize image URL to absolute form."""
    if not src:
        return None

    if src.startswith("//"):
        return f"https:{src}"

    if src.startswith("http://") or src.startswith("https://"):
        return src

    if src.startswith("/"):
        base = base_url or page_url
        try:
            parsed = urlparse(base)
            return urlunparse((parsed.scheme, parsed.netloc, src, "", "", ""))
        except Exception:
            return None

    return src
