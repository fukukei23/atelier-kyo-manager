#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Headful Playwright session dumper.
1) opens a browser at target URL
2) you manually pass any gates/login
3) press Enter in the terminal to dump cookies + localStorage to JSON

Usage:
  python tools/dump_session.py --site MONCLER_OFFICIAL \
    --url "https://www.moncler.com/en-int/" \
    --out instance/sessions/moncler_official.json
"""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump cookies/localStorage for manual session reuse.")
    p.add_argument("--site", required=True, help="Site key (used only for logging).")
    p.add_argument("--url", required=True, help="Start URL to open headfully.")
    p.add_argument("--out", type=Path, default=Path("instance/sessions/session_dump.json"),
                   help="Output JSON path (default: instance/sessions/session_dump.json)")
    return p.parse_args()


async def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        print(f"[dump_session] Opening URL: {args.url}")
        await page.goto(args.url, wait_until="domcontentloaded")
        print("[dump_session] Please complete manual steps (gate/login/etc.).")
        input(">> Press Enter when ready to dump cookies/localStorage...")

        cookies = await context.cookies()
        ls = await page.evaluate("() => { const out = {}; for (const k in localStorage) { out[k] = localStorage.getItem(k); } return out; }")

        payload = {"site": args.site, "url": page.url, "cookies": cookies, "localStorage": ls}
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[dump_session] Saved to {args.out} (cookies={len(cookies)}, localStorage_keys={len(ls)})")

        await browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
