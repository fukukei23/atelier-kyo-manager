from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    Route,
    async_playwright,
)

from app.agents.browser.session_proxy import (
    MAX_PROXY_RETRIES_PER_RUN,
    ProxyEntry,
    ProxyManager,
)
from app.core.run_context import RunContext

# Stealth モジュールをインポート
try:
    from scraping.stealth import apply_stealth_to_context, build_stealth_params_from_site_config
except ImportError:
    build_stealth_params_from_site_config = None
    apply_stealth_to_context = None

logger = logging.getLogger(__name__)

USER_AGENT_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
]
VIEWPORT_POOL = [
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1920, "height": 1080},
    {"width": 1280, "height": 800},
]
SESSION_DIR = Path("instance/sessions")
EXTERNAL_BLOCKLIST_HOSTS = (
    "line.me",
    "instagram.com",
    "facebook.com",
    "youtube.com",
    "x.com",
    "tiktok.com",
    "wa.me",
    "wechat.com",
    "weixin.qq.com",
    "wx.qq.com",
    "doubleclick.net",
    "criteo.com",
    "googletagmanager.com",
    "googleadservices.com",
    "google-analytics.com",
    "googlesyndication.com",
)

UrlNormalizer = Callable[[str], str] | None


@dataclass
class _SessionHandles:
    playwright: Playwright
    browser: Browser
    context: BrowserContext
    page: Page


class SessionManager:
    """
    Playwright の起動/終了や Proxy, UA, Viewport ローテーションといった
    セッション管理ロジックを一元化するクラス。
    """

    def __init__(
        self,
        *,
        site: str,
        site_config: dict[str, Any],
        run_context: RunContext,
        settings: dict[str, Any],
        target_url: str,
        timeout_ms: int,
        likely_plp: bool = False,
        runtime_kwargs: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
        url_normalizer: UrlNormalizer = None,
    ) -> None:
        self.site = site
        self.site_config = site_config
        self.run_context = run_context
        self.settings = settings
        self.target_url = target_url
        self.timeout_ms = timeout_ms
        self.likely_plp = likely_plp
        self.runtime_kwargs = runtime_kwargs or {}
        self.logger = logger or logging.getLogger(__name__)
        self._url_normalizer = url_normalizer

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._handles: _SessionHandles | None = None

        self._proxy_mgr = ProxyManager()

    @property
    def context(self) -> BrowserContext | None:
        return self._handles.context if self._handles else None

    @property
    def page(self) -> Page | None:
        return self._handles.page if self._handles else None

    async def __aenter__(self) -> SessionManager:
        await self._ensure_open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def open(self) -> SessionManager:
        await self.__aenter__()
        return self

    async def close(self) -> None:
        try:
            if self._context and self.settings.get("enable_trace"):
                await self._context.tracing.stop(path=self.run_context.get_path("trace.zip"))
        except asyncio.CancelledError:
            # タイムアウトキャンセル時はCancelledErrorを握ってクリーンアップを継続
            self.logger.warning("[SessionManager] Tracing stop cancelled (timeout). Continuing cleanup.")
        except Exception as e:
            self.logger.warning(f"[SessionManager] Failed to stop tracing: {e}")

        try:
            if self._context:
                if self.settings.get("enable_video"):
                    for page in list(self._context.pages):
                        try:
                            video = getattr(page, "video", None)
                            if video:
                                video_path = await video.path()
                                self.logger.info(f"[SessionManager] Playwright video saved: {video_path}")
                        except Exception as ve:
                            self.logger.warning(f"[SessionManager] Failed to save video: {ve}")
                await self._context.close()
        except asyncio.CancelledError:
            # タイムアウトキャンセル時はCancelledErrorを握ってクリーンアップを継続
            self.logger.warning("[SessionManager] Context close cancelled (timeout). Continuing cleanup.")
        except Exception as e:
            self.logger.warning(f"[SessionManager] Failed to close context: {e}")

        try:
            if self._browser:
                await self._browser.close()
        except Exception as e:
            self.logger.warning(f"[SessionManager] Failed to close browser: {e}")

        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            self.logger.warning(f"[SessionManager] Failed to stop Playwright: {e}")

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._handles = None

    async def _ensure_open(self) -> Page:
        if self._handles and self._handles.page and not self._handles.page.is_closed():
            return self._handles.page

        if not self.target_url:
            raise ValueError("SessionManager: target_url が指定されていません。")

        pw = await async_playwright().start()
        self._playwright = pw

        # 三値（None/True/False）でproxy判定を行う
        # 優先順位: runtime_kwargs > site_config > discovery_settings > default
        use_proxy_value: bool | None = None
        source: str = "default"

        # 1. runtime_kwargsをチェック（keyが存在するかで判定、Falseも有効）
        if "use_proxy" in self.runtime_kwargs:
            use_proxy_value = bool(self.runtime_kwargs["use_proxy"])
            source = "runtime"
        # 2. site_configをチェック（keyが存在するかで判定、Falseも有効）
        elif "use_proxy" in self.site_config:
            use_proxy_value = bool(self.site_config["use_proxy"])
            source = "site_config"
        # 3. discovery_settingsをチェック（keyが存在するかで判定、Falseも有効）
        elif "use_proxy" in (self.site_config.get("discovery_settings") or {}):
            use_proxy_value = bool((self.site_config.get("discovery_settings") or {})["use_proxy"])
            source = "discovery_settings"
        # 4. 明示指定が無い場合のみ、proxy_poolの存在でデフォルト決定
        else:
            if self._proxy_mgr.get_list_for_site(self.site):
                use_proxy_value = True
                source = "default (proxies configured)"
            else:
                use_proxy_value = False
                source = "default (no proxies)"

        use_proxy_flag = use_proxy_value

        # ログ出力（use_proxy_flagとsourceを明示）
        self.logger.info(
            "[SessionManager] use_proxy=%s (source=%s, site=%s)",
            use_proxy_flag,
            source,
            self.site,
        )

        # proxyが有効だがproxy_poolにproxyが無い場合の警告
        if use_proxy_flag and not self._proxy_mgr.get_list_for_site(self.site):
            self.logger.warning(
                "[SessionManager] use_proxy=True but no proxies configured for %s. Falling back to direct connection.",
                self.site,
            )
            use_proxy_flag = False

        max_attempts = MAX_PROXY_RETRIES_PER_RUN if use_proxy_flag else 1
        last_error: Exception | None = None
        browser: Browser | None = None
        context: BrowserContext | None = None

        for attempt in range(1, max_attempts + 1):
            proxy_arg = None
            proxy_entry: ProxyEntry | None = None
            if use_proxy_flag:
                proxy_entry = self._proxy_mgr.get_for_site(self.site, self.site_config, self.runtime_kwargs)
                if proxy_entry:
                    proxy_arg = ProxyManager.build_playwright_arg(proxy_entry)
                    self.logger.info(
                        "[SessionManager] attempt %d/%d using proxy=%s (%s)",
                        attempt,
                        max_attempts,
                        proxy_entry.label,
                        proxy_entry.server,
                    )
                else:
                    self.logger.warning("[SessionManager] use_proxy=True but no available proxy for %s.", self.site)

            try:
                browser = await pw.chromium.launch(
                    headless=self.settings.get("headless", True),
                    slow_mo=self.settings.get("slow_mo", 0),
                    proxy=proxy_arg,
                )
                if browser is None:
                    raise RuntimeError("Failed to launch browser")

                context_opts = self._build_context_options()
                context = await browser.new_context(**context_opts)
                if context is None:
                    raise RuntimeError("Failed to create browser context")

                # Stealth を適用（context 作成後）
                if apply_stealth_to_context is not None and build_stealth_params_from_site_config is not None:
                    stealth_params = build_stealth_params_from_site_config(
                        self.site_config,
                        settings=self.settings,
                        enable_ua_rotation=self.settings.get("enable_ua_rotation", False),
                        enable_viewport_rotation=self.settings.get("enable_viewport_rotation", False),
                    )
                    if stealth_params is not None:
                        await apply_stealth_to_context(
                            context,
                            user_agent=stealth_params.get("user_agent"),
                            viewport=stealth_params.get("viewport"),
                            locale=stealth_params.get("locale"),
                            timezone_id=stealth_params.get("timezone_id"),
                            accept_language=stealth_params.get("accept_language"),
                        )
                    else:
                        # Stealth params の構築に失敗した場合はフォールバック
                        self.logger.warning("[SessionManager] Failed to build stealth params, using fallback")
                        await self._setup_init_scripts(context)
                else:
                    # フォールバック: 既存の _setup_init_scripts を使用
                    await self._setup_init_scripts(context)

                self._browser = browser
                self._context = context
                break
            except Exception as e:
                last_error = e
                self.logger.warning(
                    "[SessionManager] Playwright launch failed on attempt %d/%d: %s",
                    attempt,
                    max_attempts,
                    e,
                )
                if use_proxy_flag and proxy_entry is not None:
                    proxy_entry.active = False
                continue

        if browser is None or context is None:
            raise RuntimeError(
                f"Failed to launch Playwright browser after {max_attempts} attempts "
                f"(site={self.site}, use_proxy={use_proxy_flag}). Last error: {last_error}"
            )

        await self._setup_routes(context)
        # Stealth は既に context 作成時に適用済み（apply_stealth_to_context が None の場合は既存の _setup_init_scripts を使用）
        if apply_stealth_to_context is None:
            await self._setup_init_scripts(context)

        if self.settings.get("enable_trace"):
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        if self.site.upper() == "MONCLER_OFFICIAL":
            await context.add_cookies(
                [
                    {"name": "moncler-shipping-country", "value": "GB", "domain": ".moncler.com", "path": "/"},
                    {"name": "moncler-shipping-language", "value": "en", "domain": ".moncler.com", "path": "/"},
                ]
            )
            self.logger.info("[SessionManager] Injected Moncler locale cookies.")

        page = await context.new_page()
        page.set_default_timeout(self.timeout_ms)
        self._page = page

        try:
            await self._apply_saved_session(context)
        except Exception as sess_e:
            self.logger.warning(f"[SessionManager] Session restore skipped: {sess_e}")

        self._handles = _SessionHandles(
            playwright=pw,
            browser=browser,
            context=context,
            page=page,
        )
        return page

    def _build_context_options(self) -> dict[str, Any]:
        """
        BrowserContext 作成時のオプションを構築する。
        Stealth パラメータは build_stealth_params_from_site_config() から取得し、
        context 作成後に apply_stealth_to_context() で適用する。
        """
        ctx_opts: dict[str, Any] = {}

        # Stealth パラメータを取得（context 作成時の基本設定として使用）
        if build_stealth_params_from_site_config is not None:
            stealth_params = build_stealth_params_from_site_config(
                self.site_config,
                settings=self.settings,
                enable_ua_rotation=self.settings.get("enable_ua_rotation", False),
                enable_viewport_rotation=self.settings.get("enable_viewport_rotation", False),
            )
            # context 作成時に設定可能な項目を設定
            if stealth_params.get("viewport"):
                ctx_opts["viewport"] = stealth_params["viewport"]
            if stealth_params.get("locale"):
                ctx_opts["locale"] = stealth_params["locale"]
            if stealth_params.get("timezone_id"):
                ctx_opts["timezone_id"] = stealth_params["timezone_id"]
            if stealth_params.get("user_agent"):
                ctx_opts["user_agent"] = stealth_params["user_agent"]

            # Accept-Language ヘッダー
            headers = (self.settings.get("extra_http_headers") or {}).copy()
            headers["Accept-Language"] = stealth_params.get("accept_language") or "en-GB,en;q=0.8"
            ctx_opts["extra_http_headers"] = headers
        else:
            # フォールバック: 既存のロジックを使用
            viewport = self.settings.get("viewport")
            if self.settings.get("enable_viewport_rotation"):
                viewport = random.choice(VIEWPORT_POOL)
            if viewport:
                ctx_opts["viewport"] = viewport

            ctx_opts["locale"] = "en-GB"
            ctx_opts["timezone_id"] = "UTC"

            headers = (self.settings.get("extra_http_headers") or {}).copy()
            headers["Accept-Language"] = self.settings.get("accept_language") or "en-GB,en;q=0.8"
            ctx_opts["extra_http_headers"] = headers

            user_agent = self.settings.get("user_agent")
            if self.settings.get("enable_ua_rotation") and USER_AGENT_POOL:
                user_agent = random.choice(USER_AGENT_POOL)
            if user_agent:
                ctx_opts["user_agent"] = user_agent

        if self.settings.get("enable_har"):
            ctx_opts["record_har_path"] = str(self.run_context.get_path("network.har"))
        # Stage 4: record_trace_dir は Playwright のバージョンによってはサポートされていない
        # 代わりに、context.tracing.start() を使用する（_ensure_open() 内で既に実装済み）
        # そのため、ここでは record_trace_dir を設定しない
        if self.settings.get("enable_trace"):
            trace_dir = self.run_context.get_path("trace")
            Path(trace_dir).mkdir(parents=True, exist_ok=True)
            # record_trace_dir は削除（context.tracing.start() で代用）
        if self.settings.get("enable_video"):
            videos_dir = self.run_context.get_path("videos")
            Path(videos_dir).mkdir(parents=True, exist_ok=True)
            ctx_opts["record_video_dir"] = str(videos_dir)
            ctx_opts["record_video_size"] = {"width": 1280, "height": 720}

        return ctx_opts

    async def _setup_routes(self, context: BrowserContext) -> None:
        base_routes = ["**/*onetrust.com/**", "**/*cookielaw.org/**", "**/*cookiepro.com/**"]
        extra = self.settings.get("extra_block_routes") or []
        extra_hosts, extra_globs = [], []
        for x in extra:
            x = (x or "").strip()
            if not x:
                continue
            if "*" in x or "/" in x:
                extra_globs.append(x)
            else:
                extra_hosts.append(x.lstrip("."))
        block_hosts = tuple(h.lower().lstrip(".") for h in EXTERNAL_BLOCKLIST_HOSTS) + tuple(
            h.lower() for h in extra_hosts
        )

        async def _abort(route: Route) -> None:
            await route.abort()

        async def _locale_rewrite(route: Route) -> None:
            try:
                req = route.request
                url = req.url
                if req.resource_type == "document" and "moncler.com" in url:
                    parsed = urlparse(url)
                    path_lower = (parsed.path or "/").lower()
                    skip_paths = (
                        "/search",
                        "/account",
                        "/customer",
                        "/help",
                        "/privacy",
                        "/legal",
                    )
                    is_skippable = path_lower == "/" or any(
                        path_lower.startswith(s) or path_lower.startswith(f"{s}/") for s in skip_paths
                    )
                    if not is_skippable:
                        fixed = self._normalize_url(url)
                        if fixed != url:
                            self.logger.info(f"[SessionManager] URL normalized: {url} -> {fixed}")
                            await route.continue_(url=fixed)
                            return
                    else:
                        self.logger.debug(f"[SessionManager] Skipping locale normalization for path: {path_lower}")
                host = urlparse(url).netloc.lower().strip(".")
                if any(host == b or host.endswith("." + b) for b in block_hosts):
                    await route.abort()
                    return
                try:
                    await route.fallback()
                except Exception:
                    await route.continue_()
            except Exception as e:
                self.logger.warning(f"[SessionManager] route handler error: {e}")
                with contextlib.suppress(Exception):
                    await route.continue_()

        await context.route("**/*", _locale_rewrite)
        for pat in base_routes + extra_globs:
            await context.route(pat, _abort)

    def _normalize_url(self, url: str) -> str:
        if self._url_normalizer:
            try:
                return self._url_normalizer(url)
            except Exception as e:
                self.logger.debug(f"[SessionManager] URL normalizer failed: {e}")
        return url

    async def _setup_init_scripts(self, context: BrowserContext) -> None:
        await context.add_init_script(
            """try { localStorage.setItem('a11y-contrast','off'); localStorage.setItem('high-contrast','off'); } catch(e){} Object.defineProperty(navigator, 'language', {get: () => 'en-GB'}); Object.defineProperty(navigator, 'languages', {get: () => ['en-GB','en']}); (function(){ const _rz=Intl.DateTimeFormat.prototype.resolvedOptions; Intl.DateTimeFormat.prototype.resolvedOptions=function(){const o=_rz.call(this); o.timeZone='UTC'; return o;}; })();"""
        )
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
                Object.defineProperty(nav, 'webdriver', { get: () => undefined });
                Object.defineProperty(nav, 'hardwareConcurrency', { get: () => 8 });
                Object.defineProperty(nav, 'deviceMemory', { get: () => 8 });
                Object.defineProperty(nav, 'language', { get: () => lang });
                Object.defineProperty(nav, 'languages', { get: () => langs });
                Object.defineProperty(nav, 'maxTouchPoints', { get: () => 0 });
                Object.defineProperty(nav, 'platform', { get: () => 'Win32' });
              }
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
            } catch (e) { /* swallow */ }
          })();
        """
        )

    def _get_session_file(self) -> Path:
        ds = self.site_config.get("discovery_settings") or {}
        sess_path = ds.get("session_file")
        if sess_path:
            return Path(sess_path)
        safe_site = "".join(c for c in self.site.lower() if c.isalnum() or c in ("_", "-")) or "default"
        return SESSION_DIR / f"{safe_site}.json"

    async def _apply_saved_session(self, context: BrowserContext) -> None:
        sess_file = self._get_session_file()
        if not sess_file.exists():
            return
        try:
            data = json.loads(sess_file.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.warning(f"[SessionManager] Failed to read session file {sess_file}: {e}")
            return

        cookies = data.get("cookies") or []
        if cookies:
            try:
                await context.add_cookies(cookies)
                self.logger.info(f"[SessionManager] Restored {len(cookies)} cookies from {sess_file}")
            except Exception as e:
                self.logger.warning(f"[SessionManager] add_cookies failed: {e}")

        local_storage = data.get("localStorage") or {}
        if local_storage:
            try:
                payload = json.dumps(local_storage)
                script = f"""
                  (() => {{
                    try {{
                      const items = {payload};
                      for (const [k,v] of Object.entries(items || {{}})) {{
                        localStorage.setItem(k, v);
                      }}
                    }} catch (e) {{}}
                  }})();
                """
                await context.add_init_script(script)
                self.logger.info(
                    f"[SessionManager] Restored localStorage keys={list(local_storage.keys())[:5]} from {sess_file}"
                )
            except Exception as e:
                self.logger.warning(f"[SessionManager] localStorage restore failed: {e}")
