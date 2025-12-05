# -*- coding: utf-8 -*-
# ==============================================================================
# File Name : scraping/stealth.py
# Version   : 1.0.0
# Date (JST): 2025年12月3日
# Usage     : Playwright / DrissionPage の「指紋対策 (Stealth)」を共通化
# ------------------------------------------------------------------------------
# 変更要旨
#  - BrowserUseAgent や Moncler 専用 patch から使い回せる Stealth モジュールを作成
#  - navigator.webdriver の false 化、permissions.query のパッチなど Bot 検知回避を実装
#  - site_config から UA / viewport / locale / timezone の設定値を取得
# ==============================================================================

from __future__ import annotations
import logging
import random
from typing import Any, Dict, Optional

from playwright.async_api import BrowserContext

logger = logging.getLogger(__name__)

# デフォルトの UA / Viewport プール（SessionManager から移植）
DEFAULT_USER_AGENT_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
]

DEFAULT_VIEWPORT_POOL = [
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1920, "height": 1080},
    {"width": 1280, "height": 800},
]

DEFAULT_VIEWPORT = {"width": 1280, "height": 720}


def build_stealth_params_from_site_config(
    site_config: Dict[str, Any],
    *,
    settings: Optional[Dict[str, Any]] = None,
    enable_ua_rotation: bool = False,
    enable_viewport_rotation: bool = False,
) -> Dict[str, Any]:
    """
    discovery_settings, browser_settings などから UA / viewport / locale / timezone の
    設定値を取り出し、apply_stealth_to_context() に渡すための dict を構築する。

    Args:
        site_config: サイト設定（discovery_settings, browser などが含まれる）
        settings: 実行時設定（settings 辞書、discovery_settings の上書き値など）
        enable_ua_rotation: UA ローテーションを有効にするか
        enable_viewport_rotation: Viewport ローテーションを有効にするか

    Returns:
        Stealth パラメータの辞書（user_agent, viewport, locale, timezone_id など）
    """
    params: Dict[str, Any] = {}

    # settings を優先、なければ site_config から取得
    ds = (settings or {}).get("discovery_settings") or (site_config.get("discovery_settings") or {})
    browser_cfg = site_config.get("browser") or {}
    browser_settings = browser_cfg.get("settings") or {}

    # User-Agent
    user_agent = (
        (settings or {}).get("user_agent")
        or ds.get("user_agent")
        or browser_settings.get("user_agent")
        or browser_cfg.get("user_agent")
    )
    if enable_ua_rotation and DEFAULT_USER_AGENT_POOL:
        user_agent = random.choice(DEFAULT_USER_AGENT_POOL)
    if not user_agent:
        # デフォルト UA は Playwright のデフォルトを使用（None を返す）
        user_agent = None
    params["user_agent"] = user_agent

    # Viewport
    viewport = (
        (settings or {}).get("viewport")
        or ds.get("viewport")
        or browser_settings.get("viewport")
        or browser_cfg.get("viewport")
    )
    if enable_viewport_rotation and DEFAULT_VIEWPORT_POOL:
        viewport = random.choice(DEFAULT_VIEWPORT_POOL)
    if not viewport:
        viewport = DEFAULT_VIEWPORT
    params["viewport"] = viewport

    # Locale
    locale = (
        (settings or {}).get("locale")
        or ds.get("locale")
        or browser_settings.get("locale")
        or browser_cfg.get("locale")
        or "en-GB"
    )
    params["locale"] = locale

    # Timezone
    timezone_id = (
        (settings or {}).get("timezone_id")
        or ds.get("timezone_id")
        or browser_settings.get("timezone_id")
        or browser_cfg.get("timezone_id")
        or "UTC"
    )
    params["timezone_id"] = timezone_id

    # Accept-Language
    accept_language = (
        (settings or {}).get("accept_language")
        or ds.get("accept_language")
        or browser_settings.get("accept_language")
        or browser_cfg.get("accept_language")
        or "en-GB,en;q=0.8"
    )
    params["accept_language"] = accept_language

    return params


async def apply_stealth_to_context(
    context: BrowserContext,
    *,
    user_agent: Optional[str] = None,
    viewport: Optional[Dict[str, int]] = None,
    locale: Optional[str] = None,
    timezone_id: Optional[str] = None,
    accept_language: Optional[str] = None,
) -> None:
    """
    BrowserContext に Stealth（指紋対策）を適用する。

    - UA, viewport, locale, timezone を設定
    - navigator.webdriver を false にする
    - navigator.permissions.query をパッチする
    など、Bot 検知回避のための共通スクリプトを注入する。

    Args:
        context: Playwright BrowserContext
        user_agent: User-Agent 文字列（None の場合は Playwright のデフォルト）
        viewport: Viewport サイズ（{"width": 1280, "height": 720} など）
        locale: Locale（"en-GB" など）
        timezone_id: Timezone ID（"UTC" など）
        accept_language: Accept-Language ヘッダー値（"en-GB,en;q=0.8" など）

    Note:
        - BrowserContext は既に作成済みである必要がある
        - この関数は context の既存設定を上書きする可能性がある
        - 複数回呼び出しても安全だが、最後の呼び出しが有効になる
    """
    # 1. Context の基本設定を更新（可能な範囲で）
    # Note: Playwright の BrowserContext は作成後に viewport / locale / timezone を
    # 直接変更できないため、init script で JavaScript 側から設定する

    # 2. navigator.webdriver の false 化とその他の指紋対策スクリプトを注入
    await context.add_init_script(
        """try { localStorage.setItem('a11y-contrast','off'); localStorage.setItem('high-contrast','off'); } catch(e){} Object.defineProperty(navigator, 'language', {get: () => 'en-GB'}); Object.defineProperty(navigator, 'languages', {get: () => ['en-GB','en']}); (function(){ const _rz=Intl.DateTimeFormat.prototype.resolvedOptions; Intl.DateTimeFormat.prototype.resolvedOptions=function(){const o=_rz.call(this); o.timeZone='UTC'; return o;}; })();"""
    )

    # 3. 高度な指紋対策スクリプト（SessionManager から移植）
    await context.add_init_script(
        r"""
          (() => {
            try {
              const rand = (min, max) => Math.random() * (max - min) + min;
              const jitter = (base, span) => base + rand(-span, span);
              const nav = navigator;
              if (nav) {
                const lang = (nav.language || 'en-GB');
                const langs = Array.isArray(nav.languages) && nav.languages.length ? nav.languages : ['en-GB','en'];
                // navigator.webdriver を false にする（最重要）
                Object.defineProperty(nav, 'webdriver', { get: () => undefined });
                Object.defineProperty(nav, 'hardwareConcurrency', { get: () => 8 });
                Object.defineProperty(nav, 'deviceMemory', { get: () => 8 });
                Object.defineProperty(nav, 'language', { get: () => lang });
                Object.defineProperty(nav, 'languages', { get: () => langs });
                Object.defineProperty(nav, 'maxTouchPoints', { get: () => 0 });
                Object.defineProperty(nav, 'platform', { get: () => 'Win32' });
              }
              // Canvas fingerprinting 対策
              const patchCanvas = (proto) => {
                if (!proto) return;
                const toDataURL = proto.toDataURL;
                proto.toDataURL = function(...args) {
                  const ctx = this.getContext && this.getContext('2d');
                  if (ctx) {
                    const shift = () => (Math.random() - 0.5) * 2;
                    ctx.fillStyle = `rgba(${128+shift()},${128+shift()},${128+shift()},0.01)`;
                    ctx.fillRect(0, 0, 2, 2);
                  }
                  return toDataURL.apply(this, args);
                };
              };
              if (typeof HTMLCanvasElement !== 'undefined' && HTMLCanvasElement.prototype) {
                patchCanvas(HTMLCanvasElement.prototype);
              }
              if (typeof OffscreenCanvas !== 'undefined' && OffscreenCanvas.prototype) {
                patchCanvas(OffscreenCanvas.prototype);
              }
              // WebGL fingerprinting 対策
              const patchWebGL = (proto) => {
                if (!proto) return;
                const getParameter = proto.getParameter;
                proto.getParameter = function(param) {
                  const VENDOR = 0x1F00, RENDERER = 0x1F01;
                  if (param === VENDOR) {
                    const v = getParameter.call(this, param);
                    return typeof v === 'string' ? v.replace(/Google Inc\./, 'Google LLC') : v;
                  }
                  if (param === RENDERER) {
                    const r = getParameter.call(this, param);
                    return typeof r === 'string' ? r.replace(/ANGLE \(|\)/g, '') : r;
                  }
                  return getParameter.call(this, param);
                };
              };
              if (typeof WebGLRenderingContext !== 'undefined' && WebGLRenderingContext.prototype) {
                patchWebGL(WebGLRenderingContext.prototype);
              }
              if (typeof WebGL2RenderingContext !== 'undefined' && WebGL2RenderingContext.prototype) {
                patchWebGL(WebGL2RenderingContext.prototype);
              }
              // Audio fingerprinting 対策
              const patchAudio = (Cls) => {
                if (!Cls || !Cls.prototype) return;
                const getFloatFrequencyData = Cls.prototype.getFloatFrequencyData;
                if (getFloatFrequencyData) {
                  Cls.prototype.getFloatFrequencyData = function(arr) {
                    const res = getFloatFrequencyData.call(this, arr);
                    for (let i = 0; i < arr.length; i += Math.floor(arr.length / 8) || 1) {
                      arr[i] = arr[i] * (0.99 + Math.random() * 0.02);
                    }
                    return res;
                  };
                }
              };
              if (typeof AnalyserNode !== 'undefined') {
                patchAudio(AnalyserNode);
              }
              // Font fingerprinting 対策
              if (typeof Navigator !== 'undefined' && Navigator.prototype) {
                const origFonts = Navigator.prototype.fonts;
                if (origFonts) {
                  Navigator.prototype.fonts = function() {
                    const it = origFonts.apply(this, arguments);
                    if (it && typeof it.status === 'string') return it;
                    return {
                      status: 'loaded',
                      check: () => true,
                      load: () => Promise.resolve(),
                      values: () => [].values()
                    };
                  };
                }
              }
              // permissions.query のパッチ（notifications / geolocation など）
              if (typeof navigator !== 'undefined' && navigator.permissions && navigator.permissions.query) {
                const origQuery = navigator.permissions.query;
                navigator.permissions.query = function(descriptor) {
                  // notifications / geolocation などに対してエラーを起こさない
                  if (descriptor && descriptor.name) {
                    return Promise.resolve({ state: 'granted', onchange: null });
                  }
                  return origQuery.call(this, descriptor);
                };
              }
            } catch (e) { /* swallow */ }
          })();
        """
    )

    # 4. Locale / Timezone を動的に設定（パラメータが指定されている場合）
    if locale or timezone_id:
        locale_script = f"""
          (() => {{
            try {{
              {f"Object.defineProperty(navigator, 'language', {{get: () => '{locale}'}});" if locale else ""}
              {f"Object.defineProperty(navigator, 'languages', {{get: () => ['{locale}', 'en']}});" if locale else ""}
              {f"""
              const _rz = Intl.DateTimeFormat.prototype.resolvedOptions;
              Intl.DateTimeFormat.prototype.resolvedOptions = function() {{
                const o = _rz.call(this);
                o.timeZone = '{timezone_id}';
                return o;
              }};
              """ if timezone_id else ""}
            }} catch(e) {{}}
          }})();
        """
        await context.add_init_script(locale_script)

    logger.debug(
        f"[Stealth] Applied stealth to context (UA={'set' if user_agent else 'default'}, "
        f"viewport={'set' if viewport else 'default'}, locale={locale or 'default'}, "
        f"timezone={timezone_id or 'default'})"
    )

