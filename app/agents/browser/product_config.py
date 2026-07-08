"""PDP config resolution from site_config."""

from __future__ import annotations

from typing import Any

from app.agents.browser.product_normalizer import (
    DEFAULT_PRICE_SELECTORS,
    DEFAULT_SIZE_BUTTON_SELECTORS,
    DEFAULT_TITLE_SELECTORS,
)


class PdpConfigResolver:
    """Resolves PDP extraction config from site_config with fallback defaults."""

    def __init__(self, site_config: dict[str, Any]) -> None:
        self._site_config = site_config
        self._pdp_config: dict[str, Any] | None = None
        self._price_rules: dict[str, Any] | None = None

    def get_pdp_config(self) -> dict[str, Any]:
        if self._pdp_config is not None:
            return self._pdp_config

        selectors = self._site_config.get("selectors") or {}
        pdp_cfg = selectors.get("pdp") or {}

        self._pdp_config = {
            "title": pdp_cfg.get("title") or DEFAULT_TITLE_SELECTORS,
            "price": pdp_cfg.get("price") or DEFAULT_PRICE_SELECTORS,
            "list_price": pdp_cfg.get("list_price") or [],
            "currency": pdp_cfg.get("currency")
            or [
                "meta[property='product:price:currency']",
                "meta[itemprop='priceCurrency']",
            ],
            "images": _normalize_list_config(pdp_cfg.get("images"))
            or [
                ".product-images img",
                ".product-gallery img",
                "[data-testid*='image'] img",
                "img[itemprop='image']",
            ],
            "sizes": pdp_cfg.get("size")
            or pdp_cfg.get("sizes")
            or [
                ".size-selector option",
                "button[data-size]",
                "[role='radiogroup'] [role='radio']",
            ],
            "colors": pdp_cfg.get("color")
            or pdp_cfg.get("colors")
            or [
                ".color-selector .swatch",
                "button[data-color]",
                "[data-testid*='color']",
            ],
            "description": pdp_cfg.get("description")
            or [
                ".product-description",
                "[itemprop='description']",
                "meta[property='og:description']",
            ],
            "brand": pdp_cfg.get("brand")
            or [
                "meta[property='og:site_name']",
                "[itemprop='brand']",
                ".product-brand",
            ],
            "sku": pdp_cfg.get("sku") or [],
            "availability": _normalize_list_config(pdp_cfg.get("availability")) or [],
            "breadcrumbs": pdp_cfg.get("breadcrumbs") or [],
            "size_button": pdp_cfg.get("size_button") or DEFAULT_SIZE_BUTTON_SELECTORS,
            "size_select_policy": pdp_cfg.get("size_select_policy")
            or {
                "mode": "off",
                "prefer_labels": [],
                "price_wait_ms": 4000,
            },
            "visible_price_selectors": pdp_cfg.get("visible_price_selectors") or DEFAULT_PRICE_SELECTORS,
            "image_attr": _get_dict_field(pdp_cfg.get("images"), "image_attr") or pdp_cfg.get("image_attr", "src"),
            "image_base_url": _get_dict_field(pdp_cfg.get("images"), "base_url") or pdp_cfg.get("image_base_url"),
            "raw_html_capture": pdp_cfg.get(
                "raw_html_capture",
                {
                    "enabled": True,
                    "filename": "pdp_raw.html",
                },
            ),
            "json_ld": pdp_cfg.get(
                "json_ld",
                {
                    "enabled": True,
                    "paths": {
                        "price": ["offers.price", "offers[0].price"],
                        "currency": ["offers.priceCurrency", "offers[0].priceCurrency"],
                        "title": ["name"],
                        "description": ["description"],
                    },
                },
            ),
            "meta_fallback": pdp_cfg.get(
                "meta_fallback",
                {
                    "enabled": True,
                    "selectors": [
                        "meta[property='og:price:amount']",
                        "meta[name='twitter:data1']",
                    ],
                },
            ),
            "availability_patterns": _get_availability_patterns(
                pdp_cfg.get("availability"),
                pdp_cfg.get("availability_patterns"),
            )
            or ["out of stock", "在庫なし"],
        }

        return self._pdp_config

    def get_price_rules(self) -> dict[str, Any]:
        if self._price_rules is not None:
            return self._price_rules

        pdp_cfg = self.get_pdp_config()
        price_cfg = pdp_cfg.get("price")

        normalize_rules = {}
        if isinstance(price_cfg, dict) and "normalize_rules" in price_cfg:
            normalize_rules = price_cfg["normalize_rules"]
        else:
            normalize_rules = self._site_config.get("price_rules", {})

        self._price_rules = {
            "strip_chars": normalize_rules.get("strip_chars", ["¥", ",", " "]),
            "thousands_separator": normalize_rules.get("thousands_separator", ","),
            "decimal_separator": normalize_rules.get("decimal_separator", "."),
            "currency_fallback": normalize_rules.get("currency_fallback", "JPY"),
            "price_pattern": normalize_rules.get("price_pattern", r"[\d.,]+"),
            "currency_symbols": normalize_rules.get(
                "currency_symbols",
                {
                    "¥": "JPY",
                    "$": "USD",
                    "€": "EUR",
                    "£": "GBP",
                },
            ),
        }

        return self._price_rules


def _normalize_list_config(cfg: Any) -> list[str] | None:
    """Normalize a config value to a list of strings."""
    if not cfg:
        return None
    if isinstance(cfg, list):
        return cfg
    if isinstance(cfg, dict):
        return cfg.get("selectors")
    return None


def _get_dict_field(cfg: Any, field: str) -> str | None:
    """Get a field from a dict config value."""
    if isinstance(cfg, dict):
        return cfg.get(field)
    return None


def _get_availability_patterns(availability_cfg: Any, availability_patterns: Any) -> list[str] | None:
    """Resolve availability patterns from config."""
    if isinstance(availability_cfg, dict):
        patterns = availability_cfg.get("patterns")
        if patterns:
            return patterns
    if availability_patterns:
        return availability_patterns
    return None
