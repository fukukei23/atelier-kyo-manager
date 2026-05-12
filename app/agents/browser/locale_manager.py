# ==============================================================================
# File: app/agents/browser/locale_manager.py
# P1-1 Phase 4: LocaleMixin - ロケール関連メソッドを抽出
# ==============================================================================
"""
LocaleMixin: NavigationDriver から抽出したロケール管理系メソッド群

LocaleGuardMixin を継承し、_ensure_expected_locale は nav_locale_guard.py に委譲。

- is_expected_locale_path: パスが期待ロケールで始まるか判定
- _is_locale_stable: ロケールが安定しているか判定
- _ensure_expected_locale: (LocaleGuardMixin に委譲)
- _looks_like_trap_or_legal: trap/legal ページ判定
- _normalize_abs_url: URL を絶対 URL に正規化
- _normalize_url: URL を汎用的に正規化
- _dismiss_geo_modal: ジオ/ロケールモーダルを閉じる
- _force_plp_recover: PLP 回復
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlparse, urlsplit, urlunsplit

from playwright.async_api import Locator, Page

from app.agents.browser.nav_types import _LOCALE_SEG_RE

logger = logging.getLogger(__name__)

from app.agents.browser.nav_locale_guard import LocaleGuardMixin
from app.agents.browser.navigation_helpers import (
    normalize_abs_url as nav_normalize_abs_url,
)


class LocaleMixin(LocaleGuardMixin):
    """NavigationDriver から抽出したロケール管理系メソッド群

    LocaleGuardMixin を継承し、_ensure_expected_locale は nav_locale_guard.py に委譲。
    self.page, self.strategy 等は実行時に MRO 経由で解決される。
    """

    page: Page  # mixin pattern: resolved at runtime

    def is_expected_locale_path(self, path: str, expected_locale: str) -> bool:
        """
        CR-E2E-003B拡張: Locale判定の共通化

        パスが期待ロケールで始まるか判定する。
        - ^/{expected_locale}(/|$) をTrueとする
        - ^/en-[a-z]{2}/{expected_locale}(/|$) をTrueとする（国コードprefix付き）

        Args:
            path: 検証対象パス
            expected_locale: 期待ロケール（例: "en-int"）

        Returns:
            bool: 期待ロケールで始まる場合True
        """
        if not path:
            return False
        path_lower = path.lower()
        expected_lower = expected_locale.lower()

        # パターン1: ^/{expected_locale}(/|$)
        if path_lower.startswith(f"/{expected_lower}/") or path_lower == f"/{expected_lower}":
            return True

        # パターン2: ^/en-[a-z]{2}/{expected_locale}(/|$)
        locale_prefix_pattern = re.compile(rf"^/en-[a-z]{{2}}/{expected_lower}(/|$)", re.I)
        return bool(locale_prefix_pattern.match(path_lower))

    def _is_locale_stable(
        self,
        url: str,
        site_config: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        CR-E2E-003B拡張: ロケールが安定しているか判定

        Args:
            url: 検証対象URL
            site_config: サイト設定（任意）

        Returns:
            (is_stable, diagnostics): 安定している場合True、診断情報
        """
        diagnostics = {
            "url": url,
            "path_ok": False,
            "country_ok": False,
            "reject_path_matched": False,
            "trap_pattern_matched": False,
            "stable": False,
        }

        try:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(url)
            path = parsed.path or ""
            query_params = parse_qs(parsed.query)

            # locale_policyから設定を取得
            locale_policy = {}
            if site_config:
                nav_cfg = site_config.get("navigation", {}) or {}
                locale_policy = nav_cfg.get("locale_policy", {}) or {}

            target_locale = locale_policy.get("target_locale", "en-int")
            target_country = locale_policy.get("target_country", "GB")
            reject_path_prefixes = locale_policy.get("reject_path_prefixes", [])
            trap_url_patterns = (nav_cfg.get("trap_url_patterns", []) if site_config else []) or []

            # パスチェック（CR-E2E-003B拡張: 共通関数を使用）
            path_ok = self.is_expected_locale_path(path, target_locale)
            diagnostics["path_ok"] = path_ok

            # 国チェック（CR-E2E-003B拡張: パスが/en-jp/で始まる場合はcountry_okを緩和）
            ship_to_country = query_params.get("shipToCountry", [])
            # パスが/en-jp/で始まる場合は、shipToCountryがJPでも許容
            path_starts_with_jp = path.lower().startswith("/en-jp/")
            if path_starts_with_jp:
                # /en-jp/の場合は、shipToCountryがJPまたはGBのどちらでもOK
                country_ok = (target_country in ship_to_country or "JP" in ship_to_country) if ship_to_country else True
            else:
                country_ok = target_country in ship_to_country if ship_to_country else False
            diagnostics["country_ok"] = country_ok

            # reject_path_prefixesチェック
            reject_path_matched = False
            for prefix in reject_path_prefixes:
                if path.lower().startswith(prefix.lower()):
                    reject_path_matched = True
                    break
            diagnostics["reject_path_matched"] = reject_path_matched

            # trap_url_patternsチェック（CR-E2E-003B拡張: /en-xx/{expected_locale}/ を例外化）
            trap_pattern_matched = False
            for pattern in trap_url_patterns:
                if pattern in path.lower():
                    # CR-E2E-003B拡張: /en-int/search は PLP として許容
                    if pattern == "/search" and path.lower().startswith(f"/{target_locale}/search"):
                        continue
                    # CR-E2E-003B拡張: /en-xx/{expected_locale}/ パターンをtrap判定から除外
                    if self.is_expected_locale_path(path, target_locale):
                        continue
                    trap_pattern_matched = True
                    break
            diagnostics["trap_pattern_matched"] = trap_pattern_matched

            # 安定判定
            stable = path_ok and country_ok and not reject_path_matched and not trap_pattern_matched
            diagnostics["stable"] = stable

            return stable, diagnostics
        except Exception as e:
            logger.warning(f"[LocaleGuard] _is_locale_stable failed: {e}", exc_info=True)
            diagnostics["error"] = str(e)
            return False, diagnostics

    def _looks_like_trap_or_legal(self, url: str, site_config: dict[str, Any] | None = None) -> bool:
        """
        Stage 3A-2-3:
        旧 BrowserUseAgent._looks_like_trap_or_legal のロジックをここに移行。
        挙動・ログ・例外の流れはそのまま維持すること。

        明らかに商品一覧ではなく、法務/クッキー/ヘルプ系に飛ばされてると判断したら True。
        こういうページに張り付いてもPDPは取れないので、早期abortさせる。

        ★ V88.5.9: 先に軽量正規化を行ってから判定する。
        - /en-jp/en-int/ を /en-int/ に置換
        - #product-information-panel 等のハッシュを除去

        Stage 3A-2-5: site_config から trap_url_patterns と legal_url_patterns を取得
        """
        try:
            # Stage 4: 二重ロケールの早期修正（site_configに基づく）
            locale_cfg = (site_config or {}).get("locale", {}) or {}
            normalize_double_locale = locale_cfg.get("normalize_double_locale", False)

            if normalize_double_locale:
                sp = urlsplit(url)
                path = sp.path or ""
                double_locale_patterns = locale_cfg.get("double_locale_patterns", [])
                for pattern in double_locale_patterns:
                    from_pattern = pattern.get("from", "")
                    to_pattern = pattern.get("to", "")
                    if from_pattern and to_pattern:
                        path = path.replace(from_pattern, to_pattern)
                # "PDPアンカー"などのハッシュは評価前に捨てる
                sp = sp._replace(path=path, fragment="")
                url = urlunsplit(sp)
        except Exception:
            pass  # 正規化に失敗しても、元のURLで判定を続行

        try:
            u = urlparse(url)
            full_lower = url.lower()
            path_lower = (u.path or "").lower()
            host = (u.netloc or "").lower()

            # Stage 3A-2-5: site_config から trap パターンを取得
            nav_cfg = (site_config or {}).get("navigation", {}) or {}
            trap_patterns = nav_cfg.get("trap_url_patterns", [])
            legal_patterns = nav_cfg.get("legal_url_patterns", [])

            # Stage 4: site_config から trap パターンをチェック（Moncler固有ロジックを削除）
            if trap_patterns and any(pattern.lower() in full_lower for pattern in trap_patterns):
                logger.warning(f"[_looks_like_trap] Detected trap pattern: {url}")
                return True

            if legal_patterns and any(pattern.lower() in path_lower for pattern in legal_patterns):
                logger.warning(f"[_looks_like_trap] Detected legal pattern: {url}")
                return True

            # trap_domains をチェック
            trap_domains = nav_cfg.get("trap_domains", [])
            if trap_domains and any(domain.lower() in host for domain in trap_domains):
                logger.warning(f"[_looks_like_trap] Detected trap domain: {url}")
                return True

            # locale_gate_detection をチェック
            locale_gate_cfg = nav_cfg.get("locale_gate_detection", {}) or {}
            if locale_gate_cfg.get("enabled", False):
                target_locale = locale_gate_cfg.get("target_locale", "")
                gate_paths = locale_gate_cfg.get("gate_paths", [])
                if target_locale and gate_paths:
                    # ホストがallowed_domainに一致し、パスがgate_pathsに一致する場合
                    allowed_domain = site_config.get("allowed_domain", "")
                    if (
                        allowed_domain
                        and allowed_domain.lower() in host
                        and path_lower in [p.lower() for p in gate_paths]
                    ):
                        logger.warning(f"[_looks_like_trap] Detected locale gate: {url}")
                        return True

            # デフォルトのリーガルキーワード（site_configに定義がない場合のフォールバック）
            # ただし、これは最小限に抑える
            default_legal_keywords = ["/cookie-policy", "/privacy", "/legal", "/help", "/account", "/login"]
            # site_configにlegal_patternsが定義されていない場合のみフォールバック
            if not legal_patterns and any(kw in path_lower for kw in default_legal_keywords):
                logger.warning(f"[_looks_like_trap] Detected default legal keyword: {url}")
                return True

            return False

        except Exception:
            return False

    # Stage 3A-2-1: ヘルパーメソッド（BrowserUseAgent から移植）
    def _normalize_abs_url(self, base_url: str, href: str) -> str:
        """
        CR-E2E-003A拡張: URLを絶対URLに正規化。navigation_helpers に委譲可能なら委譲。
        プロトコル相対・スキーム除外は自前で処理。
        """
        if not href:
            return ""
        try:
            out = nav_normalize_abs_url(base_url, href)
            if out:
                parsed = urlparse(out)
                if parsed.scheme and parsed.scheme.lower() not in ("http", "https"):
                    pass  # スキーム除外は下の自前処理へ
                else:
                    return out
        except Exception:
            pass
        try:
            if href.startswith("//"):
                base_parsed = urlparse(base_url)
                href = f"{base_parsed.scheme}:{href}"
            absu = urljoin(base_url, href)
            parsed = urlparse(absu)
            if parsed.scheme and parsed.scheme.lower() not in ("http", "https"):
                return href
            parts = list(urlsplit(absu))
            if parts[2].endswith("/"):
                parts[2] = parts[2].rstrip("/")
            parts[3] = ""
            parts[4] = ""
            return urlunsplit(parts)
        except Exception:
            return href

    def _normalize_url(self, url: str, site_config: dict[str, Any]) -> str:
        """
        Stage 4: URLを汎用的に正規化する

        site_config["locale"]["normalize_rules"] と force_query_params を使用して
        ロケール正規化とクエリパラメータの追加を行う。
        /en-int/ などのハードコードを排除。
        """
        u = urlparse(url)
        path = (u.path or "/").replace("//", "/")

        # site_configからロケール設定を取得（既存設定との互換性を確保）
        locale_cfg = site_config.get("locale", {}) or {}

        # normalize_rules: locale.normalize_rules を優先、なければルートレベルの normalize_rules を参照
        normalize_rules = locale_cfg.get("normalize_rules", [])
        if not normalize_rules:
            normalize_rules = site_config.get("normalize_rules", [])

        # replace_rules も normalize_rules として扱う（既存設定との互換性）
        if not normalize_rules:
            replace_rules = locale_cfg.get("replace_rules", [])
            if replace_rules:
                normalize_rules = [{"from": r.get("from", ""), "to": r.get("to", "")} for r in replace_rules]

        prefer_locale = locale_cfg.get("prefer", None)

        # normalize_double_locale: フラグがない場合は replace_rules の存在で判断
        normalize_double_locale = locale_cfg.get("normalize_double_locale", False)
        if not normalize_double_locale and locale_cfg.get("replace_rules"):
            normalize_double_locale = True

        # 二重ロケールの正規化（例: /en-jp/en-int/ → /en-int/）
        if normalize_double_locale:
            double_locale_patterns = locale_cfg.get("double_locale_patterns", [])
            # replace_rules を double_locale_patterns として扱う（既存設定との互換性）
            if not double_locale_patterns:
                replace_rules = locale_cfg.get("replace_rules", [])
                if replace_rules:
                    double_locale_patterns = [{"from": r.get("from", ""), "to": r.get("to", "")} for r in replace_rules]

            for pattern in double_locale_patterns:
                from_pattern = pattern.get("from", "")
                to_pattern = pattern.get("to", "")
                if from_pattern and to_pattern:
                    path = path.replace(from_pattern, to_pattern)

        # normalize_rules を適用
        for rule in normalize_rules:
            # 既存の normalize_rules 形式（if_url_contains/replace）にも対応
            if "if_url_contains" in rule and "replace" in rule:
                if_url_contains = rule.get("if_url_contains", "")
                replace_dict = rule.get("replace", {})
                if if_url_contains in path:
                    for from_pattern, to_pattern in replace_dict.items():
                        path = path.replace(from_pattern, to_pattern)
            else:
                # 標準形式（from/to）
                from_pattern = rule.get("from", "")
                to_pattern = rule.get("to", "")
                if from_pattern and to_pattern:
                    path = path.replace(from_pattern, to_pattern)

        # ロケールセグメントの処理
        if prefer_locale:
            seg = [s for s in path.split("/") if s]
            i = 0
            # 先頭のロケールセグメントをスキップ
            while i < len(seg) and _LOCALE_SEG_RE.match(seg[i] or ""):
                i += 1
            # 既存のprefer_localeを削除してから追加
            seg = [s for s in seg[i:] if s.lower() != prefer_locale.lower()]
            norm = f"/{prefer_locale}/" + "/".join(seg)
        else:
            norm = path

        if not norm.endswith("/") and norm != "/":
            norm += "/"

        # クエリパラメータの処理
        q = dict(parse_qsl(u.query))

        # force_query_params を追加（既存設定との互換性を確保）
        force_params = locale_cfg.get("force_query_params", {})
        # discovery_settings.force_query_params も参照（既存設定との互換性）
        if not force_params:
            ds = site_config.get("discovery_settings", {}) or {}
            force_params = ds.get("force_query_params", {})
        if force_params:
            q.update(force_params)

        # ensure_params の処理（normalize_rules内のensure_params）
        for rule in normalize_rules:
            ensure_params = rule.get("ensure_params", {})
            if ensure_params:
                q.update(ensure_params)

        # URLを再構築
        if q:
            from urllib.parse import urlencode

            norm += "?" + urlencode(q)

        return f"{u.scheme}://{u.netloc}{norm}"

    async def _dismiss_geo_modal(self, page: Page, site_config: dict[str, Any] | None = None) -> None:
        """ジオ / ロケール関係のモーダルを潰す"""
        # Stage 3A-2-5: site_config["navigation"]["overlays"]["geo_modal_selectors"] から取得
        geo_selectors = []
        if site_config:
            nav_cfg = site_config.get("navigation", {}) or {}
            overlays_cfg = nav_cfg.get("overlays", {}) or {}
            geo_selectors = overlays_cfg.get("geo_modal_selectors", [])

        # フォールバック: 空の場合はデフォルトセレクタを使用
        if not geo_selectors:
            geo_selectors = [
                "text=STAY HERE",
                "text=REMAIN HERE",
                "text=REMAIN IN ENGLISH",
                "text=CONTINUE SHOPPING",
                "text=ショッピングを続ける",
            ]

        for sel in geo_selectors:
            try:
                el = page.locator(sel).first
                if await el.count() > 0 and await el.is_visible():
                    await el.click(timeout=3000)
                    break
            except Exception:
                continue

        async def _click_first(loc: Locator, desc: str) -> bool:
            try:
                if await loc.count() == 0:
                    return False
                target = loc.first
                try:
                    if not await target.is_visible(timeout=2000):
                        await target.scroll_into_view_if_needed(timeout=3000)
                except Exception as scroll_e:
                    logger.debug(f"[GeoModal] scroll_into_view failed ({desc}): {scroll_e}, skipping")
                try:
                    await target.click(timeout=5000)
                    logger.info(f"[GeoModal] Clicked {desc}")
                    await page.wait_for_timeout(300)
                    return True
                except Exception as click_e:
                    logger.debug(f"[GeoModal] Click failed ({desc}): {click_e}")
                    return False
            except Exception as e:
                logger.debug(f"[GeoModal] Click failed ({desc}): {e}")
                return False

        async def _wait_for_target_locale(timeout_ms: int = 4000) -> bool:
            """Stage 4: ターゲットロケールへの遷移を待つ（汎用化）"""
            locale_cfg = (site_config or {}).get("locale", {}) or {}
            prefer_locale = locale_cfg.get("prefer", "")

            if not prefer_locale:
                # ロケール設定がない場合は常にTrueを返す
                return True

            try:
                # ターゲットロケールがURLに含まれているかチェック
                locale_path = f"/{prefer_locale}/"
                await page.wait_for_function(
                    f"() => location.href.includes('{locale_path}')",
                    timeout=timeout_ms,
                )
                return True
            except Exception:
                return locale_path in (page.url or "").lower()

        try:
            # Stage 4: 汎用的なロケールゲートヘッダーの検出
            header = page.locator("text=Select your location").first
            header_visible = await header.count() > 0
            if header_visible:
                logger.info("[GeoModal] Locale gate header detected.")

            # Stage 4: site_configから優先ロケールを取得（汎用化）
            geo_modal_preferred_locale = overlays_cfg.get("geo_modal_preferred_locale", "")
            locale_cfg = (site_config or {}).get("locale", {}) or {}
            prefer_locale = locale_cfg.get("prefer", geo_modal_preferred_locale)

            # 優先ロケールに基づく候補セレクタ（デフォルトはen-gb）
            if prefer_locale and "gb" in prefer_locale.lower():
                # United Kingdom / English の候補
                preferred_candidates = [
                    page.get_by_text(re.compile(r"UNITED\s+KINGDOM\s*\|\s*ENGLISH", re.I)),
                    page.get_by_role("link", name=re.compile(r"UNITED\s+KINGDOM\s*\|\s*ENGLISH", re.I)),
                    page.get_by_role("button", name=re.compile(r"UNITED\s+KINGDOM\s*\|\s*ENGLISH", re.I)),
                    page.get_by_role("button", name=re.compile(r"United\s+Kingdom.*English", re.I)),
                    page.get_by_role("link", name=re.compile(r"United\s+Kingdom.*English", re.I)),
                    page.locator("[data-testid*='locale' i] button:has-text('United Kingdom')"),
                    page.locator("[data-component*='locale' i] button:has-text('United Kingdom')"),
                    page.locator("button:has-text('United Kingdom EN')"),
                    page.locator("text=/United\\s+Kingdom\\s*\\|\\s*English/i"),
                ]
            else:
                # その他のロケールの場合は、geo_modal_selectorsを使用
                preferred_candidates = []
                for sel in geo_selectors:
                    try:
                        preferred_candidates.append(page.locator(sel).first)
                    except Exception:
                        continue

            for loc in preferred_candidates:
                if await _click_first(loc, f"Preferred locale selector ({prefer_locale})"):
                    if await _wait_for_target_locale():
                        return
                    break

            close_candidates = [
                page.locator("button[aria-label*='close' i]"),
                page.locator("button:has-text('Close')"),
                page.locator(".modal__close, .c-modal__close"),
                page.locator("[data-testid*='close' i]"),
                page.locator("div[data-editorial-component='ticker-top-banner'] button[aria-label*='close' i]"),
            ]
            for loc in close_candidates:
                if await _click_first(loc, "locale gate close button"):
                    if await _wait_for_target_locale():
                        return
                    break
        except Exception as e:
            logger.warning(f"[GeoModal] Locale gate handling failed: {e}")

    async def _force_plp_recover(self, page: Page, site_config: dict[str, Any], target_url: str | None) -> None:
        """
        Stage 4: PLP 回復（汎用化）
        site_config["navigation"]["plp_recovery"] と discovery_settings.fallback_url を使用
        """
        try:
            # site_configからPLP回復設定を取得
            nav_cfg = site_config.get("navigation", {}) or {}
            plp_recovery_cfg = nav_cfg.get("plp_recovery", {}) or {}
            recovery_enabled = plp_recovery_cfg.get("enabled", True)

            if not recovery_enabled:
                logger.debug("[recover] PLP recovery is disabled in site_config")
                return

            # PLP URL候補の取得（優先順位: target_url > plp_hard_nav > seed_plp_url > fallback_url > discovery_settings.fallback_url > home_url）
            plp = (
                target_url
                or site_config.get("plp_hard_nav")
                or site_config.get("seed_plp_url")
                or site_config.get("fallback_url")
                or (site_config.get("discovery_settings", {}) or {}).get("fallback_url")
                or plp_recovery_cfg.get("fallback_url")
                or site_config.get("home_url")
            )

            if not plp:
                logger.debug("[recover] no PLP candidate found; skip")
                return

            # ロケール正規化（site_configに基づく）
            normalize_locale = plp_recovery_cfg.get("normalize_locale", True)
            if normalize_locale:
                plp = self._normalize_url(plp, site_config)

            logger.info("[recover] Forcing PLP navigation: %s", plp)
            await page.goto(url=plp, wait_until="domcontentloaded")
        except Exception as e:
            logger.debug("[recover] force PLP failed: %r", e)
