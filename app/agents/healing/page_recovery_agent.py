# ==============================================================================
# ファイル名 (File Name): page_recovery_agent.py
# レジストリ (Registry): app/agents/page_recovery_agent.py
# 更新日時 (Date & Time JST): 2025-09-22 23:15:00
# バージョン (Version): 8.1.0J (Survival Protocol + UI Fallback + PLP Hard-Land)
#
# 変更要約
# - [Cookie即応] goBack直後 / goto直後に Cookie同意 ('ACCEPT AND CONTINUE' 等) を
#   2秒タイムアウトで即時試行（失敗しても続行）
# - [UI導線フォールバック] PLP待機が不成立の場合、
#   A) 検索アイコン→検索→Enter→PLP再待機
#   B) ハンバーガー→(Women|Men)→カテゴリ→PLP再待機
# - [URL安定化] fallback_url が相対URLでも home_url と合成して絶対URL化
# - [設定外出し] UIセレクタ群は overrides.local.json の selectors.ui.* を優先
#
# 既存互換
# - 復帰URL解決: ナビプラン → discovery_settings.fallback_url → home_url
# - PLP安定化: attached→visible 待機 + マイクロスクロール + カード数カウント
# ==============================================================================

# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import urljoin

from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from app.core.run_context import RunContext

logger = logging.getLogger(__name__)


class PageRecoveryAgent:
    """物理的なページ回復を試みる工兵部隊。自己位置回復とサバイバル能力を持つ。"""

    # -----------------------------
    # 設定・URLヘルパ
    # -----------------------------
    def _determine_fallback_url(self, run_context: RunContext, settings: dict) -> str | None:
        """ナビプラン → fallback_url → home_url の順で復帰先URLを決定。"""
        # 1) 実行時ナビプラン（任意・一時ファイル）
        nav_plan_path = run_context.get_path("navigation_plan.json")
        if nav_plan_path.exists():
            try:
                nav_plan = json.loads(nav_plan_path.read_text(encoding="utf-8"))
                urls = nav_plan.get("candidate_urls")
                if urls and isinstance(urls, list) and urls[0]:
                    logger.info("復帰先URLをナビゲーションプランから特定しました。")
                    return urls[0]
            except Exception as e:
                logger.debug(f"navigation_plan.json 読み込み失敗（継続）: {e}")

        # 2) 恒久設定：PLP直リンク（相対指定も可）
        discovery_settings = settings.get("discovery_settings", {})
        fb = discovery_settings.get("fallback_url")
        if fb:
            logger.info("復帰先URLをサイト設定のfallback_urlから特定しました。")
            # 相対なら home_url と合成
            home = settings.get("home_url", "")
            return urljoin(home, fb)

        # 3) 恒久設定：home_url
        home = settings.get("home_url")
        if home:
            logger.info("復帰先URLをサイト設定のhome_urlから特定しました。")
            return home

        return None

    # -----------------------------
    # UI操作ユーティリティ
    # -----------------------------
    async def _handle_cookie_consent(self, page: Page, selectors_override: list[str] | None = None):
        """Cookie同意バナーを処理（短期待機分岐）。"""
        selectors = selectors_override or [
            'button:has-text("ACCEPT AND CONTINUE")',
            'button:has-text("ACCEPT")',
            '[aria-label*="Accept"]',
            '[data-testid*="accept"]',
            "text=ACCEPT AND CONTINUE",
            "text=ACCEPT",
        ]
        for selector in selectors:
            try:
                await page.locator(selector).first.click(timeout=2000)
                logger.info(f"Cookie同意バナー '{selector}' をクリックしました。")
                await asyncio.sleep(0.8)
                return
            except PlaywrightTimeoutError:
                continue
            except Exception as e:
                logger.debug(f"Cookie同意クリック失敗（継続）: {selector}: {e}")

    async def _wait_for_stable_plp(
        self, *, page: Page, plp_container_selectors: list[str], timeout_ms: int
    ) -> tuple[bool, str | None]:
        """PLP（商品一覧コンテナ）の安定表示を待機。"""
        if not plp_container_selectors:
            return True, "body"

        timeout_per = max(500, int(timeout_ms / max(1, len(plp_container_selectors))))
        for selector in plp_container_selectors:
            try:
                await page.wait_for_selector(selector, state="attached", timeout=timeout_per // 2)
                await page.wait_for_selector(selector, state="visible", timeout=timeout_per // 2)
                return True, selector
            except PlaywrightTimeoutError:
                continue
            except Exception as e:
                logger.debug(f"PLP待機中の例外（継続）: {selector}: {e}")
        return False, None

    # -----------------------------
    # UI導線フォールバック
    # -----------------------------
    async def _ui_fallback_to_plp(self, *, page: Page, settings: dict, run_context: RunContext, attempt: int) -> str:
        """
        PLP待機が失敗した際の導線フォールバック:
          A) 検索アイコン→検索→Enter
          B) ハンバーガー→(Women|Men)→カテゴリ
        成功時は 'search_flow' or 'menu_flow'、失敗時は 'none' を返す。
        """
        ui = settings.get("selectors", {}).get("ui", {})

        # --- A) 検索導線 ---
        search_icon_selectors = ui.get(
            "search_icon_selectors",
            [
                '[data-testid="search-icon"]',
                'button[aria-label*="Search"]',
                'svg[aria-label*="Search"]',
                'a[href*="search"]',
            ],
        )
        search_input_selectors = ui.get(
            "search_input_selectors",
            [
                'input[type="search"]',
                'input[name="q"]',
                'input[placeholder*="Search"]',
            ],
        )
        search_keyword = (
            settings.get("discovery_settings", {}).get("search_keyword") or ui.get("default_search_keyword") or "bag"
        )

        try:
            clicked = False
            for sel in search_icon_selectors:
                try:
                    await page.locator(sel).first.click(timeout=2000)
                    logger.info(f"[UI Fallback A] 検索アイコン '{sel}' をクリック。")
                    await asyncio.sleep(0.6)
                    clicked = True
                    break
                except Exception:
                    continue

            if clicked:
                entered = False
                for sel in search_input_selectors:
                    try:
                        loc = page.locator(sel).first
                        await loc.fill(search_keyword, timeout=2000)
                        await loc.press("Enter")
                        logger.info(f"[UI Fallback A] 検索ワード '{search_keyword}' を送信。")
                        entered = True
                        break
                    except Exception:
                        continue

                if entered:
                    await run_context.take_screenshot(page, f"60_fallbackA_search_{attempt}_after_enter")
                    await asyncio.sleep(2.0)
                    return "search_flow"
        except Exception as e:
            logger.debug(f"[UI Fallback A] 例外（継続）: {e}")

        # --- B) メニュー導線 ---
        hamburger_selectors = ui.get(
            "hamburger_selectors",
            [
                'button[aria-label*="Menu"]',
                'button[aria-label*="menu"]',
                '[data-testid="hamburger"]',
                ".hamburger, .burger",
            ],
        )
        gender_priority: list[str] = ui.get("gender_priority", ["Women", "Men"])
        category_path: list[str] = ui.get("category_path", ["Women", "Bags"])

        async def click_text_any(candidates: list[str], timeout=2000) -> bool:
            for txt in candidates:
                try:
                    await page.get_by_text(txt, exact=False).first.click(timeout=timeout)
                    logger.info(f"[UI Fallback B] '{txt}' をクリック。")
                    await asyncio.sleep(0.5)
                    return True
                except Exception:
                    continue
            return False

        try:
            opened = False
            for sel in hamburger_selectors:
                try:
                    await page.locator(sel).first.click(timeout=2000)
                    logger.info(f"[UI Fallback B] ハンバーガー '{sel}' をクリック。")
                    await asyncio.sleep(0.6)
                    opened = True
                    break
                except Exception:
                    continue

            if opened:
                await click_text_any(gender_priority)
                for node in category_path:
                    ok = await click_text_any([node])
                    if not ok:
                        logger.debug(f"[UI Fallback B] カテゴリ '{node}' クリック失敗（継続）。")
                        break

                await run_context.take_screenshot(page, f"61_fallbackB_menu_{attempt}_after_clicks")
                await asyncio.sleep(2.0)
                return "menu_flow"
        except Exception as e:
            logger.debug(f"[UI Fallback B] 例外（継続）: {e}")

        return "none"

    # -----------------------------
    # メイン実行
    # -----------------------------
    async def execute(self, *, page: Page, run_context: RunContext, settings: dict, attempt: int) -> dict:
        logging.info(f"PageRecoveryAgent: {attempt}回目の回復処理を開始...")
        await run_context.take_screenshot(page, f"50_recovery_attempt_{attempt}_start")

        recovery_method = "unknown"
        try:
            # 1) goBack 試行
            initial_url = page.url
            try:
                await page.go_back(wait_until="domcontentloaded", timeout=10000)
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.debug(f"go_back 例外（継続）: {e}")
            # goBack直後のCookie即応
            await self._handle_cookie_consent(page)

            # 2) 自己位置回復（goBack無効／about:blank等）
            current_url = page.url or ""
            if current_url == initial_url or current_url.startswith("about:blank"):
                logger.warning("goBackが機能しませんでした。代替復帰戦略を開始します。")
                fallback_url = self._determine_fallback_url(run_context, settings)
                if not fallback_url:
                    raise RuntimeError("復帰用のURLを特定できませんでした。")

                await page.goto(fallback_url, wait_until="domcontentloaded")
                # goto直後のCookie即応
                await self._handle_cookie_consent(page)
                recovery_method = "fallback_goto"
                await run_context.take_screenshot(page, f"51_recovery_attempt_{attempt}_{recovery_method}")
            else:
                recovery_method = "goback"
                await run_context.take_screenshot(page, f"51_recovery_attempt_{attempt}_{recovery_method}")

            # 3) PLP安定化待機
            pdp_settings = settings.get("selectors", {}).get("pdp", {})
            plp_selectors: list[str] = pdp_settings.get("plp_container_selectors", ["body"])
            total_timeout_ms = int(settings.get("discovery_settings", {}).get("timeout_sec", 30)) * 1000

            ok, found_selector = await self._wait_for_stable_plp(
                page=page, plp_container_selectors=plp_selectors, timeout_ms=total_timeout_ms
            )

            # 3.5) UI導線フォールバック（PLP未成立時のみ）
            if not ok:
                logger.warning(f"PLP待機に失敗。UI導線フォールバックを試行します: {plp_selectors}")
                ui_flow = await self._ui_fallback_to_plp(
                    page=page, settings=settings, run_context=run_context, attempt=attempt
                )

                # 導線後に再待機（やや短め＋再短縮）
                ok2, found_selector2 = await self._wait_for_stable_plp(
                    page=page, plp_container_selectors=plp_selectors, timeout_ms=max(8000, total_timeout_ms // 2)
                )
                if not ok2:
                    # 最後にCookie即応→短い再待機
                    await self._handle_cookie_consent(page)
                    ok2, found_selector2 = await self._wait_for_stable_plp(
                        page=page, plp_container_selectors=plp_selectors, timeout_ms=5000
                    )

                if not ok2:
                    raise PlaywrightTimeoutError(f"PLPコンテナの表示待機に失敗しました（UI導線後）: {plp_selectors}")

                recovery_method += f"+{ui_flow}"
                found_selector = found_selector2

            # 4) 安定化ルーチン（観測スクロール＆カード数）
            try:
                await page.mouse.wheel(0, 400)
                await asyncio.sleep(0.5)
            except Exception:
                pass

            card_sel_list: list[str] = pdp_settings.get("pdp_link_selectors", ["article"])
            card_selectors = ", ".join(card_sel_list)
            card_count = 0
            try:
                card_count = await page.locator(card_selectors).count()
            except Exception as e:
                logger.debug(f"カード数カウント例外（継続）: {e}")

            logging.info(f"回復後の状況分析: {card_count}件の商品カードを検出。")

            message = (
                f"Recovery successful via {recovery_method}. Stable with '{found_selector}', found {card_count} cards."
            )
            return {"success": True, "message": message, "card_count": card_count}

        except Exception as e:
            logging.error(f"ページ回復シーケンス中にエラー: {e}", exc_info=True)
            await run_context.take_screenshot(page, f"98_recovery_attempt_{attempt}_exception")
            return {"success": False, "message": str(e)}
