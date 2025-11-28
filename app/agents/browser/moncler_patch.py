# -*- coding: utf-8 -*-
"""
moncler_patch.py - Moncler-specific URL normalization and recovery logic

This module handles:
- URL normalization to en-int locale
- Force en-int locale
- PLP recovery navigation
- Trap/legal page detection
"""

from __future__ import annotations
import re
import logging
from typing import Dict, Optional
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl, urlsplit, urlunsplit
from playwright.async_api import Page

logger = logging.getLogger(__name__)
_LOCALE_SEG_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.IGNORECASE)


def normalize_to_en_int_url(url: str) -> str:
    """
    Normalize URL to en-int locale.
    
    - Replace /en-gb/ with /en-int/
    - Remove duplicate locale segments
    - Add forceLocale and shipToCountry query params
    """
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


async def force_en_int(page: Page) -> None:
    """Force page to en-int locale by setting cookies and navigating."""
    try:
        if page.context:
            await page.context.add_cookies([
                {"name": "moncler-shipping-country", "value": "GB", "domain": ".moncler.com", "path": "/"},
                {"name": "moncler-shipping-language", "value": "en", "domain": ".moncler.com", "path": "/"}
            ])
    except Exception:
        pass
    try:
        fixed = normalize_to_en_int_url(page.url)
        if fixed != page.url:
            await page.goto(url=fixed, wait_until="domcontentloaded")
            try:
                await page.wait_for_load_state("networkidle", timeout=1500)
            except Exception:
                pass
    except Exception:
        pass


async def force_plp_recover(
    page: Page,
    site_config: Dict[str, Any],
    target_url: Optional[str],
    logger: Any,
) -> None:
    """
    Force navigation to PLP URL.
    
    Tries in order:
    1. target_url
    2. plp_hard_nav
    3. seed_plp_url
    4. fallback_url
    5. home_url
    """
    try:
        plp = (
            target_url
            or site_config.get("plp_hard_nav")
            or site_config.get("seed_plp_url")
            or site_config.get("fallback_url")
            or site_config.get("home_url")
        )
        if not plp:
            logger.debug("[recover] no PLP candidate found; skip")
            return
        # Normalize locale
        plp = normalize_to_en_int_url(plp)
        logger.info("[recover] Forcing PLP navigation: %s", plp)
        await page.goto(url=plp, wait_until="domcontentloaded")
    except Exception as e:
        logger.debug("[recover] force PLP failed: %r", e)


def looks_like_trap_or_legal(url: str, logger: Any) -> bool:
    """
    Detect if URL is a trap/legal page (not a product page).
    
    Returns True if URL looks like:
    - Legal/help pages (/cookie-policy, /privacy, etc.)
    - Corporate site redirects
    - Japanese locale pages (/en-jp)
    - Moncler locale gate/home pages
    """
    try:
        # Normalize URL first
        sp = urlsplit(url)
        path = sp.path or ""
        # Fix double locale
        path = path.replace("/en-jp/en-int/", "/en-int/").replace("/en-jp/", "/en-int/")
        # Remove fragment
        sp = sp._replace(path=path, fragment="")
        url = urlunsplit(sp)
    except Exception:
        pass  # Continue with original URL if normalization fails

    try:
        u = urlparse(url)
        full_lower = url.lower()
        path_lower = (u.path or "").lower()
        host = (u.netloc or "").lower()

        # /en-jp (Japanese locale)
        jp_locale = "moncler.com" in full_lower and "/en-jp" in path_lower

        # Corporate site
        corporate = "monclergroup.com" in host or "/brands/moncler" in path_lower

        # Moncler locale gate/home
        moncler_locale_gate = (
            "moncler.com" in host
            and path_lower in ("/en-int", "/en-int/", "/en-gb", "/en-gb/", "/en-us", "/en-us/")
        )

        # Legal keywords
        legal_kw = any(k in path_lower for k in (
            "/cookie-policy", "/cookies", "/privacy", "/legal", "/help",
            "/customer-service", "/customer_service", "/support",
            "/account", "/login", "/accessibility-statement", "/client-service/"
        ))

        # Log warnings
        if jp_locale:
            logger.warning(f"[_looks_like_trap] Detected /en-jp locale trap: {url}")
        if corporate:
            logger.warning(f"[_looks_like_trap] Detected corporate site redirect/path: {url}")
        if legal_kw:
            logger.warning(f"[_looks_like_trap] Detected legal/help keyword trap: {url}")
        if moncler_locale_gate:
            logger.warning(f"[_looks_like_trap] Detected Moncler locale gate/home: {url}")

        return jp_locale or corporate or legal_kw or moncler_locale_gate

    except Exception:
        return False

