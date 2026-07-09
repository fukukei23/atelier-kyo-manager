"""PDP extraction: site_config-driven generic ProductExtractor."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from playwright.async_api import BrowserContext, Page

from app.agents.browser.product_config import PdpConfigResolver
from app.agents.browser.product_normalizer import (
    normalize_image_url,
    normalize_price_to_float,
)

if TYPE_CHECKING:
    from app.core.run_context import RunContext

logger = logging.getLogger(__name__)


@dataclass
class ProductInfo:
    """Extracted product information."""

    title: str | None = None
    price: float | None = None
    currency: str | None = None
    images: list[str] = field(default_factory=list)
    sizes: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    description: str | None = None
    raw_html_path: str | None = None
    url: str | None = None
    brand: str | None = None
    list_price: float | None = None
    discount_pct: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ProductExtractor:
    """
    Generic PDP extractor driven by site_config selectors.

    Config resolution is delegated to PdpConfigResolver.
    Price normalization is delegated to product_normalizer module.
    """

    def __init__(
        self,
        site_config: dict[str, Any],
        run_context: RunContext | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.site_config = site_config
        self.run_context = run_context
        self.logger = logger or logging.getLogger(__name__)
        self._config_resolver = PdpConfigResolver(site_config)

    def _get_pdp_config(self) -> dict[str, Any]:
        return self._config_resolver.get_pdp_config()

    def _get_price_rules(self) -> dict[str, Any]:
        return self._config_resolver.get_price_rules()

    async def extract(
        self,
        page: Page,
        *,
        context: BrowserContext | None = None,
        prepare_page: Any | None = None,
    ) -> ProductInfo:
        if prepare_page:
            try:
                await prepare_page(page)
            except Exception as prep_e:
                self.logger.debug(f"[ProductExtractor] prepare_page skipped: {prep_e}")

        pdp_config = self._get_pdp_config()
        price_rules = self._get_price_rules()
        product_info = ProductInfo(url=page.url)

        product_info.title = await self._extract_title(page, pdp_config)
        product_info.price = await self._extract_price_with_size_option(page, pdp_config, price_rules)
        product_info.currency = await self._extract_currency(page, pdp_config)
        product_info.images = await self._extract_images(page, pdp_config)
        product_info.sizes = await self._extract_sizes(page, pdp_config)
        product_info.colors = await self._extract_colors(page, pdp_config)
        product_info.description = await self._extract_description(page, pdp_config)
        product_info.brand = await self._extract_brand(page, pdp_config)

        list_price, discount_pct = await self._extract_list_price_and_discount(page, pdp_config, price_rules)
        product_info.list_price = list_price
        product_info.discount_pct = discount_pct

        if product_info.price is None:
            fallback_data = await self._extract_from_json_ld_or_meta(page, pdp_config)
            if fallback_data:
                if product_info.price is None and fallback_data.get("price"):
                    product_info.price = normalize_price_to_float(str(fallback_data["price"]), price_rules)
                if not product_info.currency and fallback_data.get("currency"):
                    product_info.currency = fallback_data["currency"]

        product_info.metadata = {
            "extraction_timestamp": time.time(),
            "url": product_info.url,
            "has_title": product_info.title is not None,
            "has_price": product_info.price is not None,
            "has_currency": product_info.currency is not None,
            "image_count": len(product_info.images),
            "size_count": len(product_info.sizes),
            "color_count": len(product_info.colors),
        }

        html_capture = pdp_config.get("raw_html_capture", {})
        if html_capture.get("enabled", True) and self.run_context:
            try:
                html_content = await page.content()
                filename = html_capture.get("filename", "pdp_raw.html")
                self.run_context.save_content(filename, html_content)
                product_info.raw_html_path = str(self.run_context.get_path(filename))
            except Exception as e:
                self.logger.warning(f"[ProductExtractor] Failed to save HTML: {e}")

        return product_info

    # ------------------------------------------------------------------ #
    # Field extractors
    # ------------------------------------------------------------------ #

    async def _extract_title(self, page: Page, pdp_config: dict[str, Any]) -> str | None:
        for selector in pdp_config.get("title", []):
            try:
                locator = page.locator(selector).first
                if await locator.count() == 0:
                    continue
                content = (
                    await locator.get_attribute("content")
                    if selector.startswith("meta[")
                    else await locator.inner_text()
                )
                title = (content or "").strip()
                if title:
                    self.logger.debug(f"[ProductExtractor] Title found via: {selector}")
                    return title
            except Exception:
                continue
        return None

    async def _extract_price_with_size_option(
        self,
        page: Page,
        pdp_config: dict[str, Any],
        price_rules: dict[str, Any],
    ) -> float | None:
        price = await self._extract_price(page, pdp_config, price_rules)
        if price is not None:
            return price

        size_select_policy = pdp_config.get("size_select_policy", {}) or {}
        if size_select_policy.get("mode") == "off":
            return None

        self.logger.debug("[ProductExtractor] Price not found initially, attempting size selection...")
        if await self._click_size_to_reveal_price(page, pdp_config, size_select_policy):
            price = await self._extract_price(page, pdp_config, price_rules)
            if price is not None:
                self.logger.debug("[ProductExtractor] Price found after size selection.")
                return price
        return None

    async def _extract_price(
        self,
        page: Page,
        pdp_config: dict[str, Any],
        price_rules: dict[str, Any],
    ) -> float | None:
        price_selectors = pdp_config.get("price", [])
        if isinstance(price_selectors, dict):
            price_selectors = price_selectors.get("selectors", [])

        for selector in price_selectors:
            try:
                locator = page.locator(selector).first
                if await locator.count() == 0:
                    continue
                content = (
                    await locator.get_attribute("content")
                    if selector.startswith("meta[")
                    else await locator.inner_text()
                )
                price_text = (content or "").strip()
                if price_text:
                    price_float = normalize_price_to_float(price_text, price_rules)
                    if price_float is not None:
                        self.logger.debug(f"[ProductExtractor] Price found via: {selector} = {price_float}")
                        return price_float
            except Exception:
                continue
        return None

    async def _extract_currency(self, page: Page, pdp_config: dict[str, Any]) -> str | None:
        for selector in pdp_config.get("currency", []):
            try:
                locator = page.locator(selector).first
                if await locator.count() == 0:
                    continue
                content = await locator.get_attribute("content")
                currency = (content or "").strip()
                if currency:
                    self.logger.debug(f"[ProductExtractor] Currency found via: {selector}")
                    return currency
            except Exception:
                continue
        return None

    async def _extract_images(self, page: Page, pdp_config: dict[str, Any]) -> list[str]:
        image_selectors = pdp_config.get("images", [])
        image_attr = pdp_config.get("image_attr", "src")
        image_base_url = pdp_config.get("image_base_url")

        images: list[str] = []
        for selector in image_selectors:
            try:
                locators = page.locator(selector)
                count = await locators.count()
                for i in range(count):
                    try:
                        src = await locators.nth(i).get_attribute(image_attr)
                        if src:
                            src = normalize_image_url(src, page.url, image_base_url)
                            if src:
                                images.append(src)
                    except Exception:
                        continue
            except Exception:
                continue
        return list(dict.fromkeys(images))

    async def _extract_sizes(self, page: Page, pdp_config: dict[str, Any]) -> list[str]:
        sizes: list[str] = []
        for selector in pdp_config.get("sizes", []):
            try:
                locators = page.locator(selector)
                count = await locators.count()
                for i in range(count):
                    try:
                        text = await locators.nth(i).inner_text()
                        if text:
                            sizes.append(text.strip())
                    except Exception:
                        continue
            except Exception:
                continue
        return list(dict.fromkeys(sizes))

    async def _extract_colors(self, page: Page, pdp_config: dict[str, Any]) -> list[str]:
        colors: list[str] = []
        for selector in pdp_config.get("colors", []):
            try:
                locators = page.locator(selector)
                count = await locators.count()
                for i in range(count):
                    try:
                        el = locators.nth(i)
                        text = await el.get_attribute("aria-label") or await el.inner_text()
                        if text:
                            colors.append(text.strip())
                    except Exception:
                        continue
            except Exception:
                continue
        return list(dict.fromkeys(colors))

    async def _extract_description(self, page: Page, pdp_config: dict[str, Any]) -> str | None:
        for selector in pdp_config.get("description", []):
            try:
                locator = page.locator(selector).first
                if await locator.count() == 0:
                    continue
                content = (
                    await locator.get_attribute("content")
                    if selector.startswith("meta[")
                    else await locator.inner_text()
                )
                desc = (content or "").strip()
                if desc:
                    self.logger.debug(f"[ProductExtractor] Description found via: {selector}")
                    return desc
            except Exception:
                continue
        return None

    async def _extract_brand(self, page: Page, pdp_config: dict[str, Any]) -> str | None:
        for selector in pdp_config.get("brand", []):
            try:
                locator = page.locator(selector).first
                if await locator.count() == 0:
                    continue
                content = (
                    await locator.get_attribute("content")
                    if selector.startswith("meta[")
                    else await locator.inner_text()
                )
                brand = (content or "").strip()
                if brand:
                    self.logger.debug(f"[ProductExtractor] Brand found via: {selector}")
                    return brand
            except Exception:
                continue
        return None

    async def _extract_list_price_and_discount(
        self,
        page: Page,
        pdp_config: dict[str, Any],
        price_rules: dict[str, Any],
    ) -> tuple[float | None, float | None]:
        list_price = None
        discount_pct = None

        for selector in pdp_config.get("list_price", []):
            try:
                locator = page.locator(selector).first
                if await locator.count() == 0:
                    continue
                content = await locator.inner_text()
                text = (content or "").strip()
                if text:
                    list_price = normalize_price_to_float(text, price_rules)
                    if list_price is not None:
                        break
            except Exception:
                continue

        if list_price is not None:
            current_price = await self._extract_price(page, pdp_config, price_rules)
            if current_price is not None and list_price > current_price:
                discount_pct = ((list_price - current_price) / list_price) * 100

        return list_price, discount_pct

    async def _click_size_to_reveal_price(
        self,
        page: Page,
        pdp_config: dict[str, Any],
        size_select_policy: dict[str, Any],
    ) -> bool:
        size_button_selectors = pdp_config.get("size_button", [])
        availability_patterns = pdp_config.get("availability_patterns", [])

        try:
            buttons = page.locator(", ".join(size_button_selectors))
            count = await buttons.count()
        except Exception:
            return False

        if count == 0:
            return False

        mode = size_select_policy.get("mode", "first_instock")
        prefer_labels = size_select_policy.get("prefer_labels", [])
        clicked = False

        if mode == "by_label" and prefer_labels:
            prefer = [s.strip().upper() for s in prefer_labels if s.strip()]
            for i in range(count):
                try:
                    btn = buttons.nth(i)
                    if not await btn.is_visible() or await btn.is_disabled():
                        continue
                    label = (
                        (
                            await btn.get_attribute("data-size")
                            or await btn.get_attribute("aria-label")
                            or await btn.inner_text()
                            or ""
                        )
                        .strip()
                        .upper()
                    )
                    if label and any(label == pref for pref in prefer):
                        await btn.click()
                        clicked = True
                        break
                except Exception:
                    continue

        if not clicked and mode in ("first_instock", "by_label"):
            for i in range(count):
                try:
                    btn = buttons.nth(i)
                    if not await btn.is_visible() or await btn.is_disabled():
                        continue
                    if await btn.get_attribute("aria-disabled") == "true":
                        continue
                    text = (await btn.inner_text() or "").lower()
                    if any(pattern.lower() in text for pattern in availability_patterns):
                        continue
                    await btn.click()
                    clicked = True
                    break
                except Exception:
                    continue

        if clicked:
            try:
                await page.wait_for_load_state("networkidle", timeout=2000)
            except Exception:
                self.logger.debug("[ProductExtractor] networkidle timeout after size click.")
            try:
                wait_ms = size_select_policy.get("price_wait_ms", 4000)
                visible_price_selectors = pdp_config.get("visible_price_selectors", [])
                sel = ", ".join(visible_price_selectors)
                await page.wait_for_selector(sel, state="visible", timeout=wait_ms)
            except Exception:
                self.logger.debug("[ProductExtractor] Price selector did not become visible after size click.")
            await page.wait_for_timeout(500)

        return clicked

    async def _extract_from_json_ld_or_meta(
        self,
        page: Page,
        pdp_config: dict[str, Any],
    ) -> dict[str, Any] | None:
        json_ld_cfg = pdp_config.get("json_ld", {})
        meta_fallback_cfg = pdp_config.get("meta_fallback", {})

        if json_ld_cfg.get("enabled", True):
            try:
                scripts = await page.query_selector_all("script[type='application/ld+json']")
                paths = json_ld_cfg.get("paths", {})

                for script in scripts:
                    try:
                        raw = await script.inner_text()
                        data = json.loads(raw or "{}")
                    except Exception:
                        continue

                    price_paths = paths.get("price", [])
                    currency_paths = paths.get("currency", [])

                    price = None
                    currency = None

                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        for path in price_paths:
                            price = _get_nested_value(item, path)
                            if price is not None:
                                break
                        for path in currency_paths:
                            currency = _get_nested_value(item, path)
                            if currency is not None:
                                break
                        if price is not None:
                            break

                    if price is not None:
                        return {"price": str(price), "currency": currency}
            except Exception:
                pass

        if meta_fallback_cfg.get("enabled", True):
            for selector in meta_fallback_cfg.get("selectors", []):
                try:
                    node = page.locator(selector).first
                    if await node.count() == 0:
                        continue
                    content = await node.get_attribute("content")
                    if content:
                        return {"price": content.strip()}
                except Exception:
                    continue

        return None


def _get_nested_value(obj: Any, path: str) -> Any:
    """Get a value from a nested dict using dot notation (e.g. 'offers.price')."""
    current = obj
    for part in path.split("."):
        if part.endswith("]") and "[" in part:
            key = part[: part.index("[")]
            idx = int(part[part.index("[") + 1 : part.index("]")])
            current = current.get(key) if isinstance(current, dict) else None
            if isinstance(current, list) and 0 <= idx < len(current):
                current = current[idx]
            else:
                return None
        else:
            current = current.get(part) if isinstance(current, dict) else None
        if current is None:
            return None
    return current
