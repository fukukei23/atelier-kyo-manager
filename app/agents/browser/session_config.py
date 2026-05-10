"""Browser context option builder, session file helpers, and session restore."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext

from app.agents.browser.settings import SESSION_DIR, USER_AGENT_POOL, VIEWPORT_POOL


def build_context_options(
    settings: dict[str, Any],
    run_context_get_path: Any,  # RunContext.get_path callable
    logger: Any,
) -> dict[str, Any]:
    ctx_opts: dict[str, Any] = {}

    viewport = settings.get("viewport")
    if settings.get("enable_viewport_rotation"):
        viewport = random.choice(VIEWPORT_POOL)
    if viewport:
        ctx_opts["viewport"] = viewport

    ctx_opts["locale"] = "en-GB"
    ctx_opts["timezone_id"] = "UTC"

    headers = (settings.get("extra_http_headers") or {}).copy()
    headers["Accept-Language"] = settings.get("accept_language") or "en-GB,en;q=0.8"
    ctx_opts["extra_http_headers"] = headers

    user_agent = settings.get("user_agent")
    if settings.get("enable_ua_rotation") and USER_AGENT_POOL:
        user_agent = random.choice(USER_AGENT_POOL)
    if user_agent:
        ctx_opts["user_agent"] = user_agent

    if settings.get("enable_har"):
        ctx_opts["record_har_path"] = str(run_context_get_path("network.har"))
    if settings.get("enable_trace"):
        trace_dir = run_context_get_path("trace")
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        ctx_opts["record_trace_dir"] = str(trace_dir)
        logger.debug(f"[Playwright] record_trace_dir={trace_dir}")
    if settings.get("enable_video"):
        videos_dir = run_context_get_path("videos")
        Path(videos_dir).mkdir(parents=True, exist_ok=True)
        ctx_opts["record_video_dir"] = str(videos_dir)
        ctx_opts["record_video_size"] = {"width": 1280, "height": 720}

    return ctx_opts


def get_session_file(site: str, site_config: dict[str, Any]) -> Path:
    ds = site_config.get("discovery_settings") or {}
    sess_path = ds.get("session_file")
    if sess_path:
        return Path(sess_path)
    safe_site = "".join(c for c in site.lower() if c.isalnum() or c in ("_", "-")) or "default"
    return SESSION_DIR / f"{safe_site}.json"


async def apply_saved_session(
    context: BrowserContext,
    sess_file: Path,
    logger: Any,
) -> None:
    if not sess_file.exists():
        return
    try:
        data = json.loads(sess_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[Session] Failed to read session file {sess_file}: {e}")
        return

    cookies = data.get("cookies") or []
    if cookies:
        try:
            await context.add_cookies(cookies)
            logger.info(f"[Session] Restored {len(cookies)} cookies from {sess_file}")
        except Exception as e:
            logger.warning(f"[Session] add_cookies failed: {e}")

    ls = data.get("localStorage") or {}
    if ls:
        try:
            payload = json.dumps(ls)
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
            logger.info(f"[Session] Restored localStorage keys={list(ls.keys())[:5]} from {sess_file}")
        except Exception as e:
            logger.warning(f"[Session] localStorage restore failed: {e}")
