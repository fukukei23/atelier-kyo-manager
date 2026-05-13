"""Moncler-specific locale handling Mixin."""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from playwright.async_api import Page

_LOCALE_SEG_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.IGNORECASE)


class MonclerLocaleMixin:
    """Moncler 固有のロケール処理を担当。"""

    _page: Page | None
    logger: logging.Logger

    def _normalize_to_en_int_url(self, url: str) -> str:
        u = urlparse(url)
        path = (u.path or "/").replace("//", "/")
        path = path.replace("/en-gb/", "/en-int/")
        seg = [s for s in path.split("/") if s]
        i = 0
        while i < len(seg) and _LOCALE_SEG_RE.match(seg[i] or ""):
            i += 1
        seg = [s for s in seg[i:] if s.lower() != "en-int"]
        norm = "/en-int/" + "/".join(seg)
        if not norm.endswith("/"):
            norm += "/"
        q = dict(parse_qsl(u.query))
        q["forceLocale"] = "en-int"
        q.setdefault("shipToCountry", "GB")
        return urlunparse((u.scheme, u.netloc, norm, u.params, urlencode(q), u.fragment))

    async def _force_en_int(self, page: Page) -> None:
        try:
            if page.context:
                await page.context.add_cookies(
                    [
                        {"name": "moncler-shipping-country", "value": "GB", "domain": ".moncler.com", "path": "/"},
                        {"name": "moncler-shipping-language", "value": "en", "domain": ".moncler.com", "path": "/"},
                    ]
                )
        except Exception:
            pass
        try:
            fixed = self._normalize_to_en_int_url(page.url)
            if fixed != page.url:
                await page.goto(url=fixed, wait_until="domcontentloaded")
                with contextlib.suppress(Exception):
                    await page.wait_for_load_state("networkidle", timeout=1500)
        except Exception:
            pass

    async def _force_plp_recover(self, page, site_config: dict, target_url: str | None) -> None:
        try:
            plp = (
                target_url
                or site_config.get("plp_hard_nav")
                or site_config.get("seed_plp_url")
                or site_config.get("fallback_url")
                or site_config.get("home_url")
            )
            if not plp:
                self.logger.debug("[recover] no PLP candidate found; skip")
                return
            plp = self._normalize_to_en_int_url(plp)
            self.logger.info("[recover] Forcing PLP navigation: %s", plp)
            await page.goto(url=plp, wait_until="domcontentloaded")
        except Exception as e:
            self.logger.debug("[recover] force PLP failed: %r", e)

    def _looks_like_trap_or_legal(self, url: str) -> bool:
        from app.agents.browser.locale_manager import LocaleMixin
        return LocaleMixin._looks_like_trap_or_legal(self, url)
