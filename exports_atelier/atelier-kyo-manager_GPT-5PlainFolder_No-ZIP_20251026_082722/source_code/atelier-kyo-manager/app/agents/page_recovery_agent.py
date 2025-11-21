# ======================================================================
# File: app/agents/page_recovery_agent.py
# Version: 32.1.0J (Stable-but-empty → 非回復扱い)
# Note:
#  - Moncler の geo モーダル対応は維持
#  - 回復後に商品カード 0 件なら success=False を返し、上位で知的修復へ進ませる
# ======================================================================

from __future__ import annotations
import logging
import asyncio
import json
from typing import Any, Dict, Optional, Tuple, List
import sys
from pathlib import Path
from urllib.parse import urljoin

try:
    APP_ROOT = Path(__file__).resolve().parents[2]
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
except NameError:
    if Path.cwd() not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError
from app.core.run_context import RunContext

logger = logging.getLogger(__name__)


class PageRecoveryAgent:
    """ページの構造的な問題から回復を試みるエージェント。"""

    # --- Geo/Cookie overlays ---
    async def _dismiss_geo_modal(self, page: Page) -> bool:
        candidates = [
            "button:has-text('Remain in Italy')",
            "text=REMAIN IN ITALY",
            "text=REMAIN IN ENGLISH",
            "text=STAY HERE",
            "text=CONTINUE SHOPPING",
            "text=REMAIN HERE",
            "text=STAY ON THIS SITE",
            "button[data-testid='close-button-modal']",
        ]
        try:
            close_button = page.locator("button[aria-label*='close' i], [data-testid*='close' i]").first
            if await close_button.count() > 0 and await close_button.is_visible():
                try:
                    await close_button.click(timeout=1500)
                    await page.wait_for_timeout(500)
                    logger.info("[Recovery] ジオモーダルを閉じました (汎用Closeボタン)")
                    return True
                except Exception:
                    logger.debug("汎用Closeボタンのクリックに失敗。候補を試します。")

            for sel in candidates:
                try:
                    btn = page.locator(sel).first
                    if await btn.count() > 0 and await btn.is_visible():
                        try:
                            await btn.click(timeout=1500)
                        except Exception:
                            await btn.click(timeout=1500, force=True)
                        await page.wait_for_timeout(500)
                        logger.info(f"[Recovery] ジオモーダルを操作しました (候補: {sel})")
                        return True
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"[Recovery] _dismiss_geo_modalでエラー: {e}")
        return False

    async def _accept_cookies_if_present(self, page: Page) -> bool:
        candidates = [
            "#onetrust-accept-btn-handler", "text=ACCEPT ALL", "text=ALLOW ALL",
            "text=ACCEPT", "text=CONTINUE WITH", "button[aria-label*='Accept' i]",
            "button[id*='accept' i]"
        ]
        for sel in candidates:
            try:
                el = page.locator(sel).first
                if await el.count() > 0 and await el.is_visible():
                    try:
                        await el.click(timeout=1500)
                    except Exception:
                        await el.click(timeout=1500, force=True)
                    await page.wait_for_timeout(300)
                    logger.info(f"[Recovery] Cookieバナーを承諾しました: {sel}")
                    return True
            except Exception:
                continue
        return False

    async def _kill_overlays(self, page: Page) -> bool:
        removed = False
        try:
            if await self._accept_cookies_if_present(page):
                removed = True
                await page.wait_for_timeout(200)
            if await self._dismiss_geo_modal(page):
                removed = True
        except Exception as e:
            logger.error(f"[Recovery] _kill_overlaysでエラー: {e}")
        return removed

    # --- Helpers ---
    def _determine_fallback_url(self, run_context: RunContext, settings: dict) -> Optional[str]:
        nav_plan_path = run_context.get_path("navigation_plan.json")
        if nav_plan_path.exists():
            try:
                nav_plan = json.loads(nav_plan_path.read_text(encoding="utf-8"))
                chosen = nav_plan.get("chosen")
                if chosen and chosen.get("url"):
                    return chosen["url"]
            except Exception:
                pass
        discovery_settings = settings.get("discovery_settings", {})
        fb = discovery_settings.get("fallback_url")
        if fb:
            return urljoin(settings.get("home_url", ""), fb)
        return settings.get("home_url")

    async def _wait_for_stable_plp(
        self, *, page: Page, plp_container_selectors: List[str], timeout_ms: int
    ) -> Tuple[bool, Optional[str]]:
        if not plp_container_selectors:
            return True, "body"
        timeout_per = max(1000, int(timeout_ms / max(1, len(plp_container_selectors))))
        for selector in plp_container_selectors:
            try:
                await page.wait_for_selector(selector, state="visible", timeout=timeout_per)
                return True, selector
            except Exception:
                continue
        # 最後の手段: networkidle で安定化
        try:
            await page.wait_for_load_state("networkidle", timeout=3000)
            logger.warning("PLPコンテナセレクタが見つかりませんでした。networkidleでの安定化を試みます。")
            return True, "body (networkidle fallback)"
        except Exception:
            return False, None

    # --- Main ---
    async def execute(self, *, page: Page, run_context: RunContext, settings: dict, attempt: int) -> dict:
        logging.info(f"[RecoveryAgent] {attempt}回目の回復処理を開始...")
        await run_context.take_screenshot(page, f"50_recovery_attempt_{attempt}_start")

        try:
            # 1) 自己位置回復
            initial_url = page.url
            try:
                await page.go_back(wait_until="domcontentloaded", timeout=10000)
            except Exception:
                logger.debug("go_back が失敗またはタイムアウト。")
                if page.url == initial_url or page.url.startswith("about:blank"):
                    fallback_url = self._determine_fallback_url(run_context, settings)
                    if not fallback_url:
                        raise RuntimeError("復帰用のURLを特定できませんでした。")
                    await page.goto(fallback_url, wait_until="domcontentloaded")

            # Overlays
            await self._kill_overlays(page)

            # 2) PLP安定化待機
            pdp_settings = settings.get("selectors", {}).get("pdp", {})
            plp_selectors: List[str] = pdp_settings.get("plp_container_selectors", ["body"])
            total_timeout_ms = int(settings.get("discovery_settings", {}).get("timeout_sec", 30)) * 1000

            ok, found_selector = await self._wait_for_stable_plp(
                page=page, plp_container_selectors=plp_selectors, timeout_ms=total_timeout_ms
            )
            if not ok:
                raise PlaywrightTimeoutError(f"PLPコンテナの表示待機に失敗しました: {plp_selectors}")

            # 3) 安定化検証
            card_sel_list: List[str] = pdp_settings.get("pdp_link_selectors", ["a[href]"])
            card_selectors = ", ".join(card_sel_list)
            card_count = await page.locator(card_selectors).count()

            logging.info(f"回復後の状況分析: {card_count}件の商品カードを検出。")

            # ★★★ 重要ロジック: 0カードは“未回復”扱いにして知的修復へ誘導 ★★★
            if card_count == 0:
                message = f"Stable with '{found_selector}' after recovery, but found 0 cards (treat as not recovered)."
                logging.warning(message)
                await run_context.take_screenshot(page, f"51_recovery_stable_but_empty_{attempt}")
                return {
                    "success": False,
                    "message": message,
                    "card_count": card_count,
                    "reason": "stable_but_empty",
                }

            message = f"Recovery successful. Stable with '{found_selector}', found {card_count} cards."
            return {"success": True, "message": message, "card_count": card_count}

        except Exception as e:
            logging.error(f"ページ回復シーケンス中にエラー: {e}", exc_info=True)
            await run_context.take_screenshot(page, f"98_recovery_attempt_{attempt}_exception")
            return {"success": False, "message": str(e)}
