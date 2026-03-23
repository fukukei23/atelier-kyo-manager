# -*- coding: utf-8 -*-
# File: app/agents/plugins/base.py
# Version: 0.1.0
# Purpose: スクレイピング戦略プラグインの基底クラス

import logging
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl

logger = logging.getLogger(__name__)

# Bot回避Stealth設定（全pluginで共通）
STEALTH_CONFIG = {
    "navigator_webdriver": True,
    "navigator_vendor": True,
    "navigator_platform": True,
    "navigator_languages": True,
    "navigator_plugins": True,
    "navigator_hardware_concurrency": True,
    "chrome_load_times": True,
    "chrome_csi": True,
    "iframe_content_window": True,
    "media_codecs": True,
    "hairline": True,
    "error_prototype": True,
    "webgl_vendor": True,
    "sec_ch_ua": True,
}


def _apply_stealth(page_or_context) -> None:
    """Stealth設定を適用（遅延インポートで高速化）"""
    try:
        from playwright_stealth.stealth import Stealth
        stealth = Stealth(**STEALTH_CONFIG)
        stealth.apply_stealth_async(page_or_context)
    except ImportError:
        logger.debug("[Base] playwright-stealth not installed, skipping stealth")
    except Exception as e:
        logger.warning(f"[Base] Stealth apply failed: {e}")


class StrategyPlugin:
    """サイト別のスクレイピング戦略を差し替えるための基底クラス。"""
    site: str = ""  # 例: "MONCLER_OFFICIAL"

    # ---- フック群（必要なものだけオーバーライド） ----
    def before_navigate(self, url: str, ctx) -> str:
        """navigate前にURLを安全に補正するフック。"""
        return url

    async def after_navigate(self, page, ctx) -> None:
        """navigate直後にゲート/クッキー許諾などを処理するフック。"""
        return None

    async def assert_plp(self, page, ctx) -> bool:
        """ここがPLPかどうかの判定を強化したい場合。"""
        return True

    async def materialize(self, page, ctx) -> bool:
        """PLPのカードを表示させるためのスクロール等。成功ならTrue。"""
        return False

    # ---- ユーティリティ（派生クラスから安全に利用） ----
    def strip_fragment(self, url: str) -> str:
        p = urlparse(url); return urlunparse(p._replace(fragment=""))

    def force_query(self, url: str, fixed: dict) -> str:
        p = urlparse(url)
        q = dict(parse_qsl(p.query))
        q.update({k: v for k, v in fixed.items() if v is not None})
        return urlunparse(p._replace(query=urlencode(q)))

    def hostname(self, url: str) -> str:
        return urlparse(url).hostname or ""

    async def dismiss_consent(self, page) -> None:
        """同意ダイアログの簡易処理（必要に応じて上書き）。"""
        try:
            # よくあるパターンの軽量対応（失敗しても無視）
            for sel in [
                "button#onetrust-accept-btn-handler",
                "button[aria-label='Accept all']",
                "button:has-text('Accept All')",
                "button:has-text('同意')",
            ]:
                if await page.locator(sel).first.is_visible():
                    await page.locator(sel).first.click(timeout=2000)
                    break
        except Exception as e:
            logger.debug(f"[Base] dismiss_consent: {e}")
