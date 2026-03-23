# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from playwright.async_api import BrowserContext, Page


class MonclerPDPExtractor:
    GRAPHQL_ENDPOINT = "https://www.moncler.com/graphql"
    PRICE_API_ENDPOINT = "https://www.moncler.com/api/prices"

    def __init__(self, logger) -> None:
        self.logger = logger

    async def extract(
        self,
        *,
        page: Page,
        context: Optional[BrowserContext],
        locale: str = "en-jp",
        country: str = "JP",
    ) -> Optional[Dict[str, Any]]:
        data = await self._extract_from_next_data(page)
        if data:
            return data

        ld_json = await self._extract_from_ld_json(page)
        if ld_json:
            return ld_json

        sku = data["sku"] if data else self._infer_sku_from_url(page.url)
        if context and sku:
            gql = await self._fetch_graphql(context, sku=sku, locale=locale)
            if gql:
                return gql

            price = await self._fetch_price_api(context, sku=sku, country=country)
            if price:
                price["url"] = page.url
                return price

        meta = await self._extract_from_meta(page)
        if meta:
            return meta

        return None

    async def _extract_from_next_data(self, page: Page) -> Optional[Dict[str, Any]]:
        try:
            script = await page.query_selector("script#__NEXT_DATA__")
            if not script:
                return None
            raw = await script.inner_text()
            payload = json.loads(raw or "{}")
        except Exception as e:
            self.logger.debug(f"[MonclerExtractor] NEXT_DATA parse failed: {e}")
            return None

        product = (
            payload.get("props", {})
            .get("pageProps", {})
            .get("product")
        )
        if not isinstance(product, dict):
            return None

        return self._build_result_from_product(product, page.url)

    async def _extract_from_ld_json(self, page: Page) -> Optional[Dict[str, Any]]:
        try:
            nodes = await page.query_selector_all("script[type='application/ld+json']")
        except Exception:
            return None
        for node in nodes:
            try:
                data = json.loads(await node.inner_text() or "{}")
            except Exception:
                continue
            if isinstance(data, dict) and data.get("@type") in {"Product", "ProductGroup"}:
                offers = data.get("offers")
                if isinstance(offers, dict):
                    price = offers.get("price")
                    currency = offers.get("priceCurrency")
                    if price:
                        return {
                            "price": str(price),
                            "currency": currency,
                            "availability": offers.get("availability"),
                            "url": page.url,
                        }
        return None

    async def _extract_from_meta(self, page: Page) -> Optional[Dict[str, Any]]:
        try:
            price = await page.get_attribute("meta[property='og:price:amount']", "content")
        except Exception:
            price = None
        if price:
            currency = await page.get_attribute("meta[property='og:price:currency']", "content")
            return {"price": price.strip(), "currency": currency, "url": page.url}
        return None

    async def _fetch_graphql(self, context: BrowserContext, *, sku: str, locale: str) -> Optional[Dict[str, Any]]:
        try:
            payload = {
                "operationName": "ProductDetail",
                "variables": {"code": sku, "language": locale},
                "query": """
                query ProductDetail($code: String!, $language: String!) {
                  product(code: $code, language: $language) {
                    code
                    name
                    availability { status }
                    price { final { value currency } }
                    media { gallery { url alt } }
                  }
                }
                """,
            }
            response = await context.request.post(
                self.GRAPHQL_ENDPOINT,
                data=json.dumps(payload),
                headers={
                    "content-type": "application/json",
                    "x-mon-language": locale,
                },
                timeout=15000,
            )
            if response.ok:
                data = await response.json()
                product = (
                    data.get("data", {})
                    .get("product")
                )
                if isinstance(product, dict):
                    return self._build_result_from_product(product, None)
        except Exception as e:
            self.logger.debug(f"[MonclerExtractor] GraphQL fallback failed: {e}")
        return None

    async def _fetch_price_api(self, context: BrowserContext, *, sku: str, country: str) -> Optional[Dict[str, Any]]:
        try:
            response = await context.request.get(
                f"{self.PRICE_API_ENDPOINT}?sku={sku}&country={country}",
                timeout=10000,
            )
            if not response.ok:
                return None
            data = await response.json()
            amount = (data.get("prices") or {}).get("final")
            currency = data.get("currency")
            if amount:
                return {"price": str(amount), "currency": currency}
        except Exception as e:
            self.logger.debug(f"[MonclerExtractor] Price API failed: {e}")
        return None

    def _build_result_from_product(self, product: Dict[str, Any], url: Optional[str]) -> Optional[Dict[str, Any]]:
        if not product:
            return None
        price_info = (product.get("price") or {}).get("final") or {}
        gallery = ((product.get("media") or {}).get("gallery") or [])
        return {
            "title": product.get("name"),
            "sku": product.get("code"),
            "price": str(price_info.get("value")) if price_info.get("value") is not None else None,
            "currency": price_info.get("currency"),
            "availability": (product.get("availability") or {}).get("status"),
            "images": [img.get("url") for img in gallery if isinstance(img, dict) and img.get("url")] or None,
            "url": url,
        }

    def _infer_sku_from_url(self, url: str) -> Optional[str]:
        slug = url.rstrip("/").split("/")[-1]
        if not slug:
            return None
        candidate = slug.split("-")[-1]
        if candidate and re.match(r"^[A-Za-z0-9]+$", candidate):
            return candidate.upper()
        return None

