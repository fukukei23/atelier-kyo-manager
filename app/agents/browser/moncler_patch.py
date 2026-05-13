"""Moncler URL normalization and trap/legal page detection."""
from __future__ import annotations

import contextlib
import logging
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunparse, urlunsplit

from playwright.async_api import Page

logger = logging.getLogger(__name__)

_LOCALE_SEG_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.IGNORECASE)

_LEGAL_KEYWORDS = (
    "/cookie-policy", "/cookies", "/privacy", "/legal", "/help",
    "/customer-service", "/customer_service", "/support", "/account",
    "/login", "/accessibility-statement", "/client-service/",
)

_LOCALE_GATE_PATHS = frozenset({"/en-int", "/en-int/", "/en-gb", "/en-gb/", "/en-us", "/en-us/"})

_LOCALE_COOKIES = [
    {"name": "moncler-shipping-country", "value": "GB", "domain": ".moncler.com", "path": "/"},
    {"name": "moncler-shipping-language", "value": "en", "domain": ".moncler.com", "path": "/"},
]


def normalize_to_en_int_url(url: str) -> str:
    u = urlparse(url)
    path = (u.path or "/").replace("//", "/").replace("/en-gb/", "/en-int/")
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


async def force_en_int(page: Page) -> None:
    try:
        if page.context:
            await page.context.add_cookies(_LOCALE_COOKIES)
    except Exception:
        pass
    try:
        fixed = normalize_to_en_int_url(page.url)
        if fixed != page.url:
            await page.goto(url=fixed, wait_until="domcontentloaded")
            with contextlib.suppress(Exception):
                await page.wait_for_load_state("networkidle", timeout=1500)
    except Exception:
        pass


async def force_plp_recover(page: Page, site_config: dict[str, Any], target_url: str | None, log: Any) -> None:
    plp = target_url or site_config.get("plp_hard_nav") or site_config.get("seed_plp_url") or site_config.get("fallback_url") or site_config.get("home_url")
    if not plp:
        log.debug("[recover] no PLP candidate; skip")
        return
    plp = normalize_to_en_int_url(plp)
    log.info("[recover] Forcing PLP: %s", plp)
    try:
        await page.goto(url=plp, wait_until="domcontentloaded")
    except Exception as e:
        log.debug("[recover] force PLP failed: %r", e)


def _normalize_locale_url(url: str) -> str:
    try:
        sp = urlsplit(url)
        path = (sp.path or "").replace("/en-jp/en-int/", "/en-int/").replace("/en-jp/", "/en-int/")
        return urlunsplit(sp._replace(path=path, fragment=""))
    except Exception:
        return url


def _detect_trap_flags(url: str) -> dict[str, bool]:
    u = urlparse(url)
    path_lower = (u.path or "").lower()
    host = (u.netloc or "").lower()
    full_lower = url.lower()
    return {
        "jp_locale": "moncler.com" in full_lower and "/en-jp" in path_lower,
        "corporate": "monclergroup.com" in host or "/brands/moncler" in path_lower,
        "legal": any(k in path_lower for k in _LEGAL_KEYWORDS),
        "locale_gate": "moncler.com" in host and path_lower in _LOCALE_GATE_PATHS,
    }


def looks_like_trap_or_legal(url: str, log: Any) -> bool:
    url = _normalize_locale_url(url)
    flags = _detect_trap_flags(url)
    for name, matched in flags.items():
        if matched:
            log.warning(f"[_looks_like_trap] Detected {name}: {url}")
    return any(flags.values())
