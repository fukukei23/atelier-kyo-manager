import logging
import re
from urllib.parse import urlparse

from .base import StrategyPlugin

logger = logging.getLogger(__name__)

# ==============================================================================
# CR-ATELIER-002 Step 4: Moncler PLP→PDP 抽出ロジック実装（実ブラウザ検証版）
# ==============================================================================
#
# 【Moncler PLP→PDP 抽出のデータフロー図】
#
# 1. BrowserUseAgent / NavigationDriver.run_plp_flow()
#    └─> NavigationDriver.collect_pdp_links(ctx)
#         └─> [MONCLER_OFFICIAL判定]
#              ├─> extract_moncler_pdp_links(page, ctx)  [Moncler専用]
#              │    ├─> Moncler専用セレクタで抽出（site_config.selectors.plp.pdp_link_selectors）
#              │    │    ├─> article[data-component*='ProductCard'] a[href*='/products/']
#              │    │    ├─> [data-testid*='product-card'] a[href*='/products/']
#              │    │    └─> その他Moncler専用セレクタ（MONCLER_PLP_PDP_LINK_SELECTORS）
#              │    ├─> _is_valid_moncler_pdp_url() でバリデーション
#              │    │    ├─> origin == "https://www.moncler.com"
#              │    │    ├─> path contains "/products/"
#              │    │    ├─> locale path == "/en-int/"（二重ロケールパターンはreject）
#              │    │    └─> 外部ドメイン除外（onetrust.com等）
#              │    └─> Telemetry保存（accepted==0の場合）
#              │
#              └─> [汎用ロジック]  [Moncler専用が失敗した場合のフォールバック]
#                   ├─> Phase 1a: Global <a href> sweep + Regex Filter
#                   ├─> Phase 1b: Selector-based補完（site_configから取得）
#                   ├─> Phase 2: Deep Extraction Fallback
#                   └─> Phase 3: Noise Filtering & Saving
#
# 2. MonclerPLPStrategy（StrategyPlugin）
#    └─> site_config.selectors.plp.* を正として、コード側はそれに合わせる
#         └─> .html パターンは削除済み、/products/ パターンのみを使用
#
# 3. URLバリデーション（CR-ATELIER-002 Step 4-2）
#    └─> extractor.py: _is_valid_moncler_pdp_url()  [Moncler専用]
#         └─> Accept: origin == moncler.com, path == /en-int/.../products/...
#         └─> Reject: 外部ドメイン、trapページパターン、二重ロケールパターン
#
# 4. ロケール制御（CR-ATELIER-002 Step 4-2）
#    └─> NavigationDriver._ensure_expected_locale()
#         └─> 現在のページ自体を /en-int/...&shipToCountry=GB に揃える
#         └─> 二重ロケールパターン（/en-lt/en-int/ 等）を検出して修正
#
# ==============================================================================
#
# 【Moncler PLP 想定 DOM 構造（実ブラウザ検証ベース）】
#
# CR-ATELIER-002 Step 4-1: 実DOM（plp_dom_initial_materialized.html）を前提とした構造
#
# - Main listing container:
#   <main role="main"> または <div data-component="ProductListing"> など
#   - セレクタ候補（site_config.selectors.plp.container_selectors）:
#     * main[role='main']
#     * div[data-component='ProductListing']
#     * section[role='region']
#     * .product-grid
#     * [data-testid='product-grid']
#
# - Product tile:
#   <article data-component="ProductCard"> または <div data-testid="product-card">
#   - セレクタ候補（site_config.selectors.plp.tile_selectors）:
#     * article[data-component*='ProductCard']
#     * [data-testid='product-card']
#     * [data-testid='product-tile']
#     * div[class*='product-card' i]
#     * div[class*='product-tile' i]
#
# - PDP link:
#   <a href="/en-int/.../products/..."> 各product tile内に存在
#   - セレクタ候補（site_config.selectors.plp.pdp_link_selectors）:
#     * article[data-component*='ProductCard'] a[href*='/products/']
#     * [data-testid*='product-card'] a[href*='/products/']
#     * a[href*='/en-int/products/']
#     * a[href*='/products/']
#
# 【注意】
# - site_config.selectors.plp.* を正として、コード側はそれに合わせる
# - .html パターンは削除済み、/products/ パターンのみを使用
# - 実DOMに基づいてセレクタを調整する場合は、site_configを更新すること
#
# ==============================================================================
# CR-ATELIER-002 Step 3: Moncler専用 PLP→PDP 抽出セレクタ
# ==============================================================================
# CR-ATELIER-002 Step3:
#   - Moncler 専用 PLP→PDP 抽出ロジックの実装
#   - 詳細は docs/spec/CR-ATELIER-002_MONCLER_PLP_PDP_EXTRACTION_FIX.md を参照
# ==============================================================================

# Moncler専用: PLPメインコンテナのセレクタ
MONCLER_PLP_CONTAINER_SELECTORS = [
    "main[role='main']",
    "div[data-component='ProductListing']",
    "section[role='region']",
    ".product-grid",
    "[data-testid='product-grid']",
]

# Moncler専用: Product tile（カード要素）のセレクタ
MONCLER_PLP_TILE_SELECTORS = [
    "article[data-component*='ProductCard']",
    "[data-testid='product-card']",
    "[data-testid='product-tile']",
    "[data-test='product-card']",
    "div[data-testid='product-card']",
    "div[data-testid='product-tile']",
    "div[class*='product-card' i]",
    "div[class*='product-tile' i]",
    "div.product-tile",
    "div.c-product-tile",
    "div[data-component*='ProductCard']",
]

# ==============================================================================
# CR-ATELIER-002 Step 5: セレクタ戦略のレイヤリング設計
# ==============================================================================
#
# 【セレクタレイヤの優先順位】
#
# Primary Layer（site_config準拠）:
#   - site_config.selectors.plp.pdp_link_selectors から取得
#   - /products/ パターンを前提としたセレクタ
#   - 優先度: 最高（site_configが正とされる）
#
# Secondary Layer（DOM構造ベース）:
#   - DOM構造から判明したセレクタ
#   - data-component / data-testid ベースのセレクタ
#   - 優先度: 中（Primaryが失敗した場合に使用）
#
# Tertiary Layer（汎用フォールバック）:
#   - 汎用的なセレクタ
#   - 全ページスイープ
#   - 優先度: 低（Primary/Secondaryが失敗した場合に使用）
#
# ==============================================================================

# Moncler専用: PDP link（実際にクリックしたい <a>）のセレクタ
# CR-ATELIER-002 Step 5-2: Primary Layer（site_config準拠）
# site_config.selectors.plp.pdp_link_selectors を優先的に使用
# このリストは、site_config が存在しない場合のフォールバックとして使用
MONCLER_PLP_PDP_LINK_SELECTORS_PRIMARY = [
    "article[data-component*='ProductCard'] a[href*='/products/']",
    "[data-testid*='product-card'] a[href*='/products/']",
    "[data-testid*='product-tile'] a[href*='/products/']",
    "a[href*='/en-int/products/']",
    "a[href*='/products/']",
]

# CR-ATELIER-002 Step 5-2: Secondary Layer（DOM構造ベース）
# DOM構造から判明したセレクタ（data-component / data-testid ベース）
MONCLER_PLP_PDP_LINK_SELECTORS_SECONDARY = [
    "[data-test*='product-card'] a[href*='/products/']",
    ".product-card a[href*='/products/']",
    ".c-product-card a[href*='/products/']",
    ".product-tile a[href*='/products/']",
    ".c-product-tile a[href*='/products/']",
    "[data-qa='product-tile'] a[href*='/products/']",
    "[data-qa*='product'] a[href*='/products/']",
]

# CR-ATELIER-002 Step 5-2: Tertiary Layer（汎用フォールバック）
# 汎用的なセレクタ（全ページスイープ）
MONCLER_PLP_PDP_LINK_SELECTORS_TERTIARY = [
    "div:has(a[href*='/products/']) a[href*='/products/']",
    "a[href*='/products/']",
]

# 後方互換性のため、既存の定数も残す
MONCLER_PLP_PDP_LINK_SELECTORS = (
    MONCLER_PLP_PDP_LINK_SELECTORS_PRIMARY +
    MONCLER_PLP_PDP_LINK_SELECTORS_SECONDARY +
    MONCLER_PLP_PDP_LINK_SELECTORS_TERTIARY
)

# ==============================================================================


class MonclerPLPStrategy(StrategyPlugin):
    site = "MONCLER_OFFICIAL"
    _DEFAULT_LOCALE = "en-int"
    _DEFAULT_COUNTRY = "GB"
    _HARD_PLP_URL = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB"
    
    # CR-ATELIER-002 Step 3: Moncler専用セレクタをクラス属性として公開
    # site_config.selectors.plp.* を正として、コード側はそれに合わせる
    PLP_CONTAINER_SELECTORS = MONCLER_PLP_CONTAINER_SELECTORS
    PLP_TILE_SELECTORS = MONCLER_PLP_TILE_SELECTORS
    PLP_PDP_LINK_SELECTORS = MONCLER_PLP_PDP_LINK_SELECTORS
    
    # CR-ATELIER-002 Step 3: .htmlベースのセレクタは削除
    # Monclerは /products/ パターンのみを使用するため、.htmlベースのセレクタは不要
    # site_config.selectors.plp.tile_selectors と site_config.selectors.plp.pdp_link_selectors を使用
    _PLP_TILE_SELECTORS = MONCLER_PLP_TILE_SELECTORS

    def before_navigate(self, url: str, ctx) -> str:
        # 1) フラグメント除去
        url = self.strip_fragment(url)
        # 2) ロケール/配送国を固定（overridesの有無にかかわらず安全側で）
        locale, country = self._preferred_locale(ctx)
        url = self.force_query(url, {
            "forceLocale": locale,
            "shipToCountry": country
        })
        # 3) ホーム/モンクラーグループ/ロケールルートに落ちたら正規PLPへ強制戻し
        host = self.hostname(url)
        if "monclergroup.com" in host:
            return self._HARD_PLP_URL
        path = self._path(url)
        if not path or path == "/" or re.match(r"^/en-[a-z]{2}/?$", path or "", re.IGNORECASE):
            return self._HARD_PLP_URL
        # 4) PLPらしさが無いURLは正規PLPへ戻す
        if host and host.endswith("moncler.com"):
            if not re.search(r"/(outerwear|search|products|p[-/])", path, re.IGNORECASE):
                return self._HARD_PLP_URL
        return url

    async def after_navigate(self, page, ctx):
        # ルート書き換えを一度だけ仕込む（documentナビを en-int に強制）
        if isinstance(ctx, dict) and not ctx.get("_moncler_route_patched"):
            ctx["_moncler_route_patched"] = True
            async def _route_enforce(route, request):
                try:
                    if request.resource_type == "document":
                        url = request.url
                        if "moncler.com" in url:
                            # ロケールが en-int 以外なら置換して継続
                            if ("/en-jp/" in url) or ("/en-de/" in url) or ("/en-int/" not in url):
                                new_url = re.sub(r"/en-[a-z]{2}/", "/en-int/", url, flags=re.IGNORECASE)
                                if new_url == url:
                                    # en-xx が無いケースは先頭に付与
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

        # 0) 強制ロケールを localStorage / cookie / URL に焼き付ける
        await self._pin_locale(page, ctx)
        # 1) 同意処理
        await self.dismiss_consent(page)
        # 2) ロケーションモーダルでUK/ENを選択する
        await self._handle_locale_modal(page)
        # 3) Cookie/ロケールボタンを順番に叩く
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
        # カード候補の合計が一定数以上でPLPとみなす
        min_cards = int(ctx.get("plp_min_cards", 8))
        return await self._count_tiles(page) >= min_cards

    async def materialize(self, page, ctx) -> bool:
        """スクロールでカードを出し切る。成功ならTrue。"""
        min_cards = int(ctx.get("plp_min_cards", 8))
        max_passes = int(ctx.get("scroll_max_passes", 10))
        last = 0
        for _ in range(max_passes):
            try:
                # パスごとにロケールホーム/他ロケールへ戻されていないか監視し、戻されたら即PLPへ復帰
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
                if curr == last:  # 伸びてない
                    await page.wait_for_timeout(500)
                last = curr
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------
    def _site_config(self, ctx) -> dict:
        if isinstance(ctx, dict):
            return (ctx.get("site_config") or ctx.get("site") or {}) or {}
        return getattr(ctx, "site_config", {}) or {}

    def _preferred_locale(self, ctx):
        cfg = self._site_config(ctx)
        discovery = cfg.get("discovery_settings") or {}
        locale = (
            cfg.get("forceLocale")
            or discovery.get("forceLocale")
            or self._DEFAULT_LOCALE
        )
        country = (
            cfg.get("shipToCountry")
            or discovery.get("shipToCountry")
            or self._DEFAULT_COUNTRY
        )
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
            # add_init_script は一部環境で失敗する場合があるため警告のみ
            logger.debug("[MonclerPLPStrategy] add_init_script failed", exc_info=True)
        try:
            await page.evaluate(init_script, payload)
        except Exception:
            logger.debug("[MonclerPLPStrategy] evaluate locale pin failed", exc_info=True)

        current = page.url
        desired = self.force_query(current, {
            "forceLocale": locale,
            "shipToCountry": country
        })
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
