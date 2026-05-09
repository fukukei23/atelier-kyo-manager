# ==============================================================================
# ファイル名 (File Name): browser_utils.py
# レジストリ (Registry): app/utils/browser_utils.py
# 更新日時 (Date & Time JST): 2025-09-20 17:55:00
# バージョン (Version): 3.1.0J (Unified Screenshot Utility)
#
# --- v3.1.0Jでの主な変更点 ---
# - [機能統一] どのエージェントからも呼び出せる共通のスクリーンショット保存関数
#   `save_screenshot` を新設。ImportErrorを解決し、証拠保全ロジックを一元化。
# - [堅牢性] Playwrightのエラーを捕捉するtry/exceptブロックを備え、
#   撮影失敗時でもシステム全体が停止しないように設計されています。
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page

logger = logging.getLogger(__name__)
APP_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_DIR = APP_ROOT / "instance" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def save_screenshot(page: Page, file_prefix: str, full_page: bool = True) -> str:
    """
    タイムスタンプ付きでスクリーンショットを安全に保存する。
    Returns: 保存パス。失敗した場合は空文字列。
    """
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    path = SCREENSHOT_DIR / f"{file_prefix}_{ts}.png"
    try:
        page.screenshot(path=str(path), full_page=full_page)
        logger.info(f"Saved screenshot to: {path}")
        return str(path)
    except PlaywrightError as e:
        logger.error(f"Failed to save screenshot to {path}: {e}")
        return ""
    except Exception as e:
        logger.error(f"An unexpected critical error occurred during screenshot saving: {e}")
        return ""


def _click_first_visible_element(page: Page, selectors: list[str], purpose: str) -> bool:
    """
    提供されたセレクタのリストを順番に試し、最初に見つかった可視要素を1回だけクリックする。
    """
    if not selectors:
        logger.debug(f"No selectors provided for purpose: '{purpose}'. Skipping.")
        return False

    for selector in selectors:
        try:
            loc = page.locator(selector).first
            loc.wait_for(state="visible", timeout=2500)

            box = loc.bounding_box()
            if box and box["width"] > 5 and box["height"] > 5:
                logger.info(f"'{purpose}' element found with selector: '{selector}'. Clicking.")
                loc.click(delay=random.randint(80, 200))
                page.wait_for_timeout(random.randint(200, 400))
                logger.info(f"Successfully clicked the '{purpose}' element.")
                return True
        except PlaywrightError:
            logger.debug(f"'{purpose}' selector '{selector}' not found or not visible. Trying next.")
        except Exception as e:
            logger.warning(f"Unexpected error for selector '{selector}' during '{purpose}': {e}")

    logger.info(f"Finished checking all selectors for '{purpose}'. No clickable element found.")
    return False


def handle_cookie_consent(page: Page, site_config: dict[str, Any]) -> None:
    """サイト設定に基づき、Cookie同意モーダルを閉じる。"""
    selectors: list[str] = site_config.get("selectors", {}).get("pdp", {}).get("cookie_accept_selectors", [])
    _click_first_visible_element(page, selectors, "Cookie Consent")


def handle_dismissible_popups(page: Page, site_config: dict[str, Any]) -> None:
    """サイト設定に基づき、一般的なポップアップ（メール登録など）を閉じる。"""
    selectors: list[str] = site_config.get("selectors", {}).get("pdp", {}).get("dismiss_popups", [])
    _click_first_visible_element(page, selectors, "Dismissible Popup")


def handle_captcha(page: Page, site_config: dict[str, Any]) -> None:
    """CAPTCHAを検知し、設定に基づいて対処する。（実装詳細は省略）"""
    pass
