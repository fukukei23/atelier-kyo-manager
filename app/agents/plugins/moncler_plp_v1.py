import logging
import re
from urllib.parse import urlparse

from .base import StrategyPlugin

logger = logging.getLogger(__name__)

# Moncler PLP抽出用定数（browser/extractor.py から参照）
_MONCLER_PLP_TILE_TUPLE = (
    "a[href*='/products/']",
    "a[href*='/product/']",
    "a[href*='/p/']",
    "a[href*='/p-']",
    "[data-testid='product-card']",
    "[data-test='product-card']",
    "div[data-testid='product-card']",
    "div[data-testid='product-tile']",
    "div[class*='product-card' i]",
    "div[class*='product-tile' i]",
    "div.product-tile",
    "div.c-product-tile",
    "div[data-component*='ProductCard']",
    "li:has(a[href*='/products/'])",
    "article:has(a[href*='/products/'])",
    "div:has(a[href*='/products/'])",
    "section[role='region'] .product-list a[href*='/products/']",
)

MONCLER_PLP_CONTAINER_SELECTORS = [
    "div[class*='product-tile']",
    "div[class*='product-card']",
    "div[data-testid='product-tile']",
    "div[data-testid='product-card']",
    "li[class*='product-item']",
    "article[class*='product']",
]

MONCLER_PLP_TILE_SELECTORS = _MONCLER_PLP_TILE_TUPLE

MONCLER_PLP_PDP_LINK_SELECTORS = [
    "a[href*='/products/']",
    "a[href*='/product/']",
    "a[href*='/p/']",
]

MONCLER_PLP_PDP_LINK_SELECTORS_PRIMARY = [
    "a[href*='/products/']",
]

MONCLER_PLP_PDP_LINK_SELECTORS_SECONDARY = [
    "a[href*='/product/']",
    "a[href*='/p/']",
]

MONCLER_PLP_PDP_LINK_SELECTORS_TERTIARY = [
    "a[href*='/p-']",
]


class MonclerPLPStrategy(StrategyPlugin):
    site = "MONCLER_OFFICIAL"
    _DEFAULT_LOCALE = "en-int"
    _DEFAULT_COUNTRY = "GB"
    _HARD_PLP_URL = (
        "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB"
    )
    _PLP_TILE_SELECTORS = _MONCLER_PLP_TILE_TUPLE

    def before_navigate(self, url: str, ctx) -> str:
        url = self.strip_fragment(url)
        locale, country = self._preferred_locale(ctx)
        url = self.force_query(url, {"forceLocale": locale, "shipToCountry": country})
        host = self.hostname(url)
        if "monclergroup.com" in host:
            return self._HARD_PLP_URL
        path = self._path(url)
        if not path or path == "/" or re.match(r"^/en-[a-z]{2}/?$", path or "", re.IGNORECASE):
            return self._HARD_PLP_URL
        if host and host.endswith("moncler.com"):
            if not re.search(r"/(outerwear|search|products|p[-/])", path, re.IGNORECASE):
                return self._HARD_PLP_URL
        return url

    async def after_navigate(self, page, ctx):
        if isinstance(ctx, dict) and not ctx.get("_moncler_route_patched"):
            ctx["_moncler_route_patched"] = True

            async def _route_enforce(route, request):
                try:
                    if request.resource_type == "document":
                        url = request.url
                        if "moncler.com" in url:
                            if ("/en-jp/" in url) or ("/en-de/" in url) or ("/en-int/" not in url):
                                new_url = re.sub(r"/en-[a-z]{2}/", "/en-int/", url, flags=re.IGNORECASE)
                                if new_url == url:
                                    new_url = "https://www.moncler.com/en-int/"
                                try:
                                    await route.continue_(url=new_url)
                                    return
                                except Exception:
                                    pass
                except Exception:
                    logger.debug("[MonclerPLPStrategy] route enforce failed", exc_info=True)
                await route.continue_()

            try:
                await page.route("**/*", _route_enforce)
            except Exception:
                logger.debug("[MonclerPLPStrategy] page.route setup failed", exc_info=True)

        await self._pin_locale(page, ctx)
        await self.dismiss_consent(page)
        await self._handle_locale_modal(page)
        try:
            for sel in [
                "#onetrust-accept-btn-handler",
                "button:has-text('Accept All')",
                "button:has-text('Alle akzeptieren')",
                "button:has-text('Continue')",
                "button:has-text('Continue to site')",
                "button:has-text('Save')",
                "button:has-text('English')",
            ]:
                locator = page.locator(sel).first
                if await locator.count() > 0 and await locator.is_visible():
                    await locator.click(timeout=2000)
                    await page.wait_for_timeout(300)
        except Exception:
            logger.debug("[MonclerPLPStrategy] Continue/cookie fallback click skipped", exc_info=True)

    async def assert_plp(self, page, ctx) -> bool:
        min_cards = int(ctx.get("plp_min_cards", 8))
        return await self._count_tiles(page) >= min_cards

    async def materialize(self, page, ctx) -> bool:
        min_cards = int(ctx.get("plp_min_cards", 8))
        max_passes = int(ctx.get("scroll_max_passes", 10))
        last = 0
        for _ in range(max_passes):
            try:
                if ("/en-int/" not in page.url) or self._is_locale_root(page.url):
                    try:
                        await page.goto(self._HARD_PLP_URL, wait_until="domcontentloaded")
                        await self._pin_locale(page, ctx)
                        await page.wait_for_timeout(1500)
                    except Exception:
                        logger.debug("[MonclerPLPStrategy] force return to PLP failed", exc_info=True)
                await page.mouse.wheel(0, 3000)
                await page.wait_for_timeout(800)
                curr = await self._count_tiles(page)
                if curr >= min_cards:
                    return True
                if curr == last:
                    await page.wait_for_timeout(500)
                last = curr
            except Exception:
                pass
        return False

    def _site_config(self, ctx) -> dict:
        if isinstance(ctx, dict):
            return (ctx.get("site_config") or ctx.get("site") or {}) or {}
        return getattr(ctx, "site_config", {}) or {}

    def _preferred_locale(self, ctx):
        cfg = self._site_config(ctx)
        discovery = cfg.get("discovery_settings") or {}
        locale = cfg.get("forceLocale") or discovery.get("forceLocale") or self._DEFAULT_LOCALE
        country = cfg.get("shipToCountry") or discovery.get("shipToCountry") or self._DEFAULT_COUNTRY
        return locale, country

    def _is_locale_root(self, url: str) -> bool:
        try:
            path = urlparse(url).path or "/"
            return (not path or path == "/") or bool(re.match(r"^/en-[a-z]{2}/?$", path, re.IGNORECASE))
        except Exception:
            return False

    async def _pin_locale(self, page, ctx) -> None:
        locale, country = self._preferred_locale(ctx)
        payload = {"locale": locale, "country": country}
        init_script = """
            ({locale, country}) => {
                try {
                    localStorage.setItem("akm.forceLocale", locale);
                    localStorage.setItem("akm.shipToCountry", country);
                    localStorage.setItem("moncler-shipping-country", country);
                    localStorage.setItem("moncler-force-locale", locale);
                    document.cookie = `akm.forceLocale=${locale}; path=/; domain=.moncler.com; max-age=31536000`;
                    document.cookie = `akm.shipToCountry=${country}; path=/; domain=.moncler.com; max-age=31536000`;
                    document.cookie = `moncler-force-locale=${locale}; path=/; domain=.moncler.com; max-age=31536000`;
                    document.cookie = `moncler-shipping-country=${country}; path=/; domain=.moncler.com; max-age=31536000`;
                } catch (e) {}
            }
        """
        try:
            await page.add_init_script(init_script, payload)
        except Exception:
            logger.debug("[MonclerPLPStrategy] add_init_script failed", exc_info=True)
        try:
            await page.evaluate(init_script, payload)
        except Exception:
            logger.debug("[MonclerPLPStrategy] evaluate locale pin failed", exc_info=True)

        current = page.url
        desired = self.force_query(current, {"forceLocale": locale, "shipToCountry": country})
        if desired != current:
            try:
                await page.goto(desired, wait_until="domcontentloaded")
            except Exception:
                logger.debug("[MonclerPLPStrategy] Reload with forced locale failed", exc_info=True)

    async def _handle_locale_modal(self, page) -> None:
        try:
            modal = page.locator("text=Select your location").first
            if await modal.count() == 0 or not await modal.is_visible():
                return
            selectors = [
                "[data-country-code='GB']",
                "[data-country='GB']",
                "button[data-value='GB']",
                "button[data-locale*='en-gb' i]",
                "a[data-locale*='en-gb' i]",
                "button:has-text('United Kingdom')",
                "a:has-text('United Kingdom')",
                "button:has-text('United Kingdom EN')",
            ]
            for sel in selectors:
                locator = page.locator(sel).first
                if await locator.count() > 0 and await locator.is_visible():
                    await locator.click(timeout=2000)
                    await page.wait_for_timeout(400)
                    logger.info("[MonclerPLPStrategy] Locale modal: selected United Kingdom / EN")
                    return

            clicked = await page.evaluate(
                """() => {
                    const labels = ["united kingdom", "united kingdom en", "english (uk)", "gb en"];
                    const targets = Array.from(document.querySelectorAll("button, a, [role='button']"));
                    for (const target of targets) {
                        const text = (target.innerText || "").trim().toLowerCase();
                        if (!text) continue;
                        if (labels.some(label => text.includes(label))) {
                            target.click();
                            return true;
                        }
                    }
                    return false;
                }"""
            )
            if clicked:
                await page.wait_for_timeout(400)
                logger.info("[MonclerPLPStrategy] Locale modal: selected country via JS fallback")
        except Exception:
            logger.debug("[MonclerPLPStrategy] Locale modal handling failed", exc_info=True)

    async def _count_tiles(self, page) -> int:
        total = 0
        counts = {}
        for sel in self._PLP_TILE_SELECTORS:
            try:
                cnt = await page.locator(sel).count()
                counts[sel] = cnt
                total += cnt
            except Exception:
                continue
        logger.info("[MonclerPLPStrategy] Tile counts (total=%s): %s", total, counts)
        return total

    def _path(self, url: str) -> str:
        try:
            return urlparse(url).path or "/"
        except Exception:
            return "/"
