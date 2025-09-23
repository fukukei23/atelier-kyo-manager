"""Windows event loop policy patch (neutral) — 2025-08-28
本ファイルはループポリシーを変更しません。現在のポリシー名だけログ出力します。
"""

from __future__ import annotations
import asyncio, sys, logging

log = logging.getLogger("WinEventLoopPatch")

def log_policy():
    try:
        name = type(asyncio.get_event_loop_policy()).__name__
        log.debug(f"[WinEL] current policy = {name}")
        print(f"[WinEL] current policy = {name}")
    except Exception:
        pass

# インポート時に一回だけ表示
if sys.platform.startswith("win"):
    log_policy()
