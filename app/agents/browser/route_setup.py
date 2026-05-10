"""Browser context route setup (ad-blocking, locale normalization)."""

from __future__ import annotations

import contextlib
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import BrowserContext, Route

from app.agents.browser.session_manager import EXTERNAL_BLOCKLIST_HOSTS


async def setup_routes(
    context: BrowserContext,
    settings: dict[str, Any],
    normalize_url_fn: Any,  # callable(url) -> str
    logger: Any,
) -> None:
    base_routes = ["**/*onetrust.com/**", "**/*cookielaw.org/**", "**/*cookiepro.com/**"]
    extra = settings.get("extra_block_routes") or []
    extra_hosts, extra_globs = [], []
    for x in extra:
        x = (x or "").strip()
        if not x:
            continue
        if "*" in x or "/" in x:
            extra_globs.append(x)
        else:
            extra_hosts.append(x.lstrip("."))

    block_hosts = tuple(h.lower().lstrip(".") for h in EXTERNAL_BLOCKLIST_HOSTS) + tuple(h.lower() for h in extra_hosts)

    async def _abort(route: Route):
        await route.abort()

    async def _locale_rewrite(route: Route):
        try:
            req = route.request
            url = req.url
            if req.resource_type == "document" and "moncler.com" in url:
                pu = urlparse(url)
                path_lower = (pu.path or "/").lower()
                skip_paths = ("/search", "/account", "/customer", "/help", "/privacy", "/legal")
                is_skippable = path_lower == "/" or any(
                    path_lower.startswith(s) or path_lower.startswith(f"{s}/") for s in skip_paths
                )
                if not is_skippable:
                    fixed = normalize_url_fn(url)
                    if fixed != url:
                        logger.info(f"[Route] URL normalized: {url} -> {fixed}")
                        await route.continue_(url=fixed)
                        return
                else:
                    logger.debug(f"[Route] Skipping locale normalization for path: {path_lower}")
            host = urlparse(url).netloc.lower().strip(".")
            if any(host == b or host.endswith("." + b) for b in block_hosts):
                await route.abort()
                return
            try:
                await route.fallback()
            except Exception:
                await route.continue_()
        except Exception as e:
            logger.warning(f"[Route] handler error: {e}")
            with contextlib.suppress(Exception):
                await route.continue_()

    await context.route("**/*", _locale_rewrite)
    for pat in base_routes + extra_globs:
        await context.route(pat, _abort)
