# ==============================================================================
# File: app/agents/browser/nav_locale_guard.py
# Purpose: LocaleGuardMixin - ロケール一貫性チェックと自動修正
# ==============================================================================
"""
LocaleMixin から _ensure_expected_locale を抽出した Mixin。

CR-ATELIER-002 Step 2: Locale Guard
CR-ATELIER-002 Step 5-3: Redirect / Locale 挙動の扱い整理
CR-E2E-003B拡張: モーダル検出→国/言語選択→再矯正→再判定（最大3回）
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from playwright.async_api import Page

from app.agents.browser.nav_types import _LOCALE_SEG_RE, NavigationContext

logger = logging.getLogger(__name__)


class LocaleGuardMixin:
    """
    ロケール一貫性チェックと自動修正を担当する Mixin。

    self.page, self.strategy 等は実行時に MRO 経由で解決される。
    """

    page: Page  # mixin pattern: resolved at runtime

    async def _ensure_expected_locale(self, ctx: NavigationContext) -> None:
        """
        ロケール一貫性チェックと自動修正。

        Pre-condition: PLP/検索 URL
        Post-condition:
          - page.url が /en-int/... で始まる
          - 「明らかな Trap」でないこと
          - 二重ロケールパターンを検出して修正
        """
        page = self.page
        site_config = ctx.site_config
        run_context = ctx.run_context

        diagnostics: dict[str, Any] = {
            "attempts": [],
            "http_errors": [],
            "final_url": None,
            "final_stable": False,
        }
        http_errors: list[dict[str, Any]] = []

        async def on_response(response):
            if response.status in [404, 410]:
                http_errors.append({"url": response.url, "status": response.status, "timestamp": time.time()})

        page.on("response", on_response)
        current_url = page.url or ""

        site_code = site_config.get("site_code") or site_config.get("site") or ctx.site or ""
        if site_code != "MONCLER_OFFICIAL":
            logger.debug(f"[LocaleGuard] Skipping locale check for site: {site_code}")
            return

        nav_cfg = (site_config.get("navigation", {}) or {}) if site_config else {}
        locale_policy = nav_cfg.get("locale_policy", {}) or {}
        location_modal_cfg = nav_cfg.get("location_modal", {}) or {}

        target_locale = locale_policy.get("target_locale", "en-int")
        target_country = locale_policy.get("target_country", "GB")
        max_attempts = locale_policy.get("max_correction_attempts", 3)
        stability_check_delay_ms = locale_policy.get("stability_check_delay_ms", 2000)
        require_stable = locale_policy.get("require_stable_before_proceed", True)

        parsed = urlparse(current_url)
        path = parsed.path or ""
        query_params = parse_qs(parsed.query)

        path_ok = self.is_expected_locale_path(path, target_locale)  # type: ignore[attr-defined]
        ship_to_country = query_params.get("shipToCountry", [])
        path_starts_with_jp = path.lower().startswith("/en-jp/")
        country_ok = (
            (target_country in ship_to_country or "JP" in ship_to_country)
            if path_starts_with_jp and ship_to_country
            else (target_country in ship_to_country if ship_to_country else not path_starts_with_jp)
        )

        if path_ok and country_ok:
            logger.info(f"[LocaleGuard] Checked locale, no change: {current_url}")
            final_stable, final_diag = self._is_locale_stable(current_url, site_config)  # type: ignore[attr-defined]
            diagnostics["final_url"] = current_url
            diagnostics["final_stable"] = final_stable
            diagnostics["http_errors"] = http_errors
            diagnostics["attempts"] = [
                {
                    "attempt": 0,
                    "url_before": current_url,
                    "url_after": current_url,
                    "stable_after": final_stable,
                    "stability_diagnostics": final_diag,
                    "note": "Locale already stable, no correction needed",
                }
            ]
            self._save_diagnostics(diagnostics, run_context)
            return

        logger.warning(f"[LocaleGuard] Locale mismatch: path_ok={path_ok}, country_ok={country_ok}, URL={current_url}")

        corrected_url = self._resolve_corrected_url(current_url, site_config, target_locale, target_country)

        try:
            for attempt_count in range(1, max_attempts + 1):
                attempt_info = await self._attempt_locale_correction(
                    page,
                    ctx,
                    location_modal_cfg,
                    run_context,
                    attempt_count,
                    corrected_url,
                    target_locale,
                    target_country,
                    stability_check_delay_ms,
                )
                diagnostics["attempts"].append(attempt_info)

                current_check = page.url or ""
                is_stable, stability_diag = self._is_locale_stable(current_check, site_config)  # type: ignore[attr-defined]
                attempt_info["url_after"] = current_check
                attempt_info["stable_after"] = is_stable
                attempt_info["stability_diagnostics"] = stability_diag

                if is_stable and require_stable:
                    logger.info(f"[LocaleGuard] Locale stable after attempt {attempt_count}: {current_check}")
                    diagnostics["final_url"] = current_check
                    diagnostics["final_stable"] = True
                    break

                if attempt_count < max_attempts:
                    corrected_url = self._build_corrected_url(current_check, target_locale, target_country)
                    if corrected_url != current_check:
                        logger.warning(f"[LocaleGuard] Attempt {attempt_count}: Navigating to: {corrected_url}")
                        try:
                            await page.goto(corrected_url, wait_until="domcontentloaded", timeout=30000)
                            await page.wait_for_timeout(stability_check_delay_ms)
                            attempt_info["navigation_performed"] = True
                        except Exception as e:
                            logger.warning(f"[LocaleGuard] Navigation failed attempt {attempt_count}: {e}")
                            attempt_info["navigation_error"] = str(e)

            final_url = page.url or ""
            final_stable, _ = self._is_locale_stable(final_url, site_config)  # type: ignore[attr-defined]
            diagnostics["final_url"] = final_url
            diagnostics["final_stable"] = final_stable
            diagnostics["http_errors"] = http_errors

            if not final_stable and require_stable:
                logger.warning(f"[LocaleGuard] Locale not stable after {max_attempts} attempts: {final_url}")

        except (asyncio.CancelledError, Exception) as e:
            if isinstance(e, asyncio.CancelledError):
                logger.warning(f"[LocaleGuard] Locale correction cancelled: {e}")
            else:
                logger.warning(f"[LocaleGuard] Failed to normalize locale: {e}", exc_info=True)
            diagnostics["final_url"] = page.url if hasattr(self, "page") and self.page else current_url
            diagnostics["final_stable"] = False
            diagnostics["error"] = str(e)
        finally:
            self._save_diagnostics(diagnostics, run_context)

    # ------------------------------------------------------------------ #
    # Helper methods (extracted from God Method)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _save_diagnostics(diagnostics: dict[str, Any], run_context: Any) -> None:
        if not run_context:
            logger.warning("[LocaleGuard] run_context is None, cannot save locale diagnostics")
            return
        try:
            diagnostics_path = run_context.run_path / "locale_diagnostics.json"
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diagnostics_path, "w", encoding="utf-8") as f:
                json.dump(diagnostics, f, indent=2, ensure_ascii=False)
            logger.info(f"[LocaleGuard] Saved locale diagnostics to: {diagnostics_path}")
        except Exception as e:
            logger.warning(f"[LocaleGuard] Failed to save locale diagnostics: {e}", exc_info=True)

    @staticmethod
    async def _save_screenshot(page: Page, run_context: Any, stage: str) -> str | None:
        if not run_context:
            return None
        try:
            timestamp = int(time.time() * 1000)
            filename = f"locale_{stage}_{timestamp}.png"
            path = run_context.screenshots_path / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(path))
            return str(path.relative_to(run_context.run_path))
        except Exception as e:
            logger.debug(f"[LocaleGuard] Failed to save screenshot {stage}: {e}")
            return None

    def _resolve_corrected_url(
        self,
        current_url: str,
        site_config: dict,
        target_locale: str,
        target_country: str,
    ) -> str | None:
        corrected_url = None
        if self.strategy and hasattr(self.strategy, "_HARD_PLP_URL"):  # type: ignore[attr-defined]
            corrected_url = self.strategy._HARD_PLP_URL  # type: ignore[attr-defined]
            logger.debug(f"[LocaleGuard] Using strategy _HARD_PLP_URL: {corrected_url}")

        if not corrected_url:
            nav_cfg = site_config.get("navigation", {}) or {}
            plp_recovery_cfg = nav_cfg.get("plp_recovery", {}) or {}
            corrected_url = (
                plp_recovery_cfg.get("plp_hard_nav")
                or site_config.get("seed_plp_url")
                or site_config.get("fallback_url")
                or (site_config.get("discovery_settings", {}) or {}).get("fallback_url")
                or site_config.get("home_url")
            )

        if not corrected_url:
            corrected_url = self._build_corrected_url(current_url, target_locale, target_country)

        return corrected_url

    @staticmethod
    def _build_corrected_url(current_url: str, target_locale: str, target_country: str) -> str | None:
        parsed = urlparse(current_url)
        path = parsed.path or ""
        query_params = parse_qs(parsed.query)

        if _LOCALE_SEG_RE.match(path.strip("/").split("/")[0] if path.strip("/") else ""):
            path_parts = [p for p in path.split("/") if p]
        else:
            path_parts = [p for p in path.split("/") if p]

        while path_parts and _LOCALE_SEG_RE.match(path_parts[0] or ""):
            path_parts.pop(0)
        if not any(target_locale.lower() in p.lower() for p in path_parts):
            normalized_path = f"/{target_locale}/" + "/".join(path_parts)
        else:
            normalized_path = "/" + "/".join(path_parts)
        if not normalized_path.endswith("/") and normalized_path != "/":
            normalized_path += "/"

        if "forceLocale" not in query_params or target_locale not in query_params.get("forceLocale", []):
            query_params["forceLocale"] = [target_locale]
        if "shipToCountry" not in query_params or target_country not in query_params.get("shipToCountry", []):
            query_params["shipToCountry"] = [target_country]
        normalized_query = urlencode(query_params, doseq=True)

        return urlunparse(
            (parsed.scheme, parsed.netloc, normalized_path, parsed.params, normalized_query, parsed.fragment)
        )

    async def _attempt_locale_correction(
        self,
        page: Page,
        ctx: NavigationContext,
        location_modal_cfg: dict,
        run_context: Any,
        attempt_count: int,
        corrected_url: str | None,
        target_locale: str,
        target_country: str,
        stability_check_delay_ms: int,
    ) -> dict[str, Any]:
        attempt_info: dict[str, Any] = {
            "attempt": attempt_count,
            "url_before": page.url or "",
            "modal_detected": False,
            "modal_handled": False,
            "navigation_performed": False,
            "url_after": None,
            "stable_after": False,
        }

        screenshot_before = await self._save_screenshot(page, run_context, f"before_attempt_{attempt_count}")
        if screenshot_before:
            attempt_info["screenshot_before"] = screenshot_before

        if location_modal_cfg.get("enabled", True):
            modal_result = await self._detect_and_handle_modal(page, location_modal_cfg, run_context, attempt_count)
            attempt_info.update(modal_result)

        return attempt_info

    async def _detect_and_handle_modal(
        self,
        page: Page,
        location_modal_cfg: dict,
        run_context: Any,
        attempt_count: int,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"modal_detected": False, "modal_handled": False}
        detection_selectors = location_modal_cfg.get("detection_selectors", [])

        for sel in detection_selectors:
            try:
                locator = page.locator(sel).first
                try:
                    is_visible = await locator.is_visible(timeout=1000)
                except (asyncio.CancelledError, Exception) as e:
                    if isinstance(e, asyncio.CancelledError):
                        logger.debug(f"[LocaleGuard] Modal detection cancelled for selector '{sel}'")
                    continue

                if is_visible:
                    result["modal_detected"] = True
                    logger.info(f"[LocaleGuard] Location modal detected (attempt {attempt_count})")

                    screenshot_modal = await self._save_screenshot(page, run_context, f"modal_attempt_{attempt_count}")
                    if screenshot_modal:
                        result["screenshot_modal"] = screenshot_modal

                    location_selected = await self._select_country(page, location_modal_cfg)
                    if location_selected:
                        result["modal_handled"] = True
                    else:
                        await self._close_modal(page, location_modal_cfg)
                    break
            except Exception as e:
                logger.debug(f"[LocaleGuard] Modal detection failed for selector '{sel}': {e}")
                continue

        return result

    @staticmethod
    async def _select_country(page: Page, location_modal_cfg: dict) -> bool:
        country_selectors = location_modal_cfg.get("country_selectors", [])
        for country_sel in country_selectors:
            try:
                country_locator = page.locator(country_sel).first
                try:
                    country_visible = await country_locator.is_visible(timeout=1000)
                except (asyncio.CancelledError, Exception):
                    continue
                if country_visible:
                    await country_locator.click(timeout=3000)
                    wait_after = location_modal_cfg.get("wait_after_selection_ms", 2000)
                    await page.wait_for_timeout(wait_after)
                    logger.info(f"[LocaleGuard] Selected location using selector: {country_sel}")
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    async def _close_modal(page: Page, location_modal_cfg: dict) -> None:
        close_selectors = location_modal_cfg.get("close_selectors", [])
        for close_sel in close_selectors:
            try:
                close_locator = page.locator(close_sel).first
                try:
                    close_visible = await close_locator.is_visible(timeout=1000)
                except (asyncio.CancelledError, Exception):
                    continue
                if close_visible:
                    await close_locator.click(timeout=2000)
                    await page.wait_for_timeout(1000)
                    break
            except Exception:
                continue
