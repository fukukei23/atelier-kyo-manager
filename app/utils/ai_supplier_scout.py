# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# File: app/utils/ai_supplier_scout.py
# Version: 1.6-clean (2025-08-26)
# ------------------------------------------------------------
from __future__ import annotations

import os
import re
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Windows loop policy (Event loop is closed 対策)
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError
from playwright.async_api import Error as PWError

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = Path(__file__).resolve().parents[2]
INST = ROOT / "instance"
DBG_DIR = INST / "supplier_debug"
LOG_DIR = INST / "logs"
DBG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _clean_price(s: str) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"([\d,]+)", s)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except Exception:
        return None


class SupplierScout:
    """
    海外ECサイトを横断して最安仕入れ候補を探索。
    連続エラー時はサイト単位で打ち切る（サーキットブレーカー）。
    """

    def __init__(self, headless: bool = True):
        self.headless = headless and os.getenv("PLAYWRIGHT_HEADFUL", "") != "1"
        self.debug = os.getenv("SUPPLIER_DEBUG", "") == "1"

        self.max_same_error = int(os.getenv("SUPPLIER_MAX_SAME_ERROR", "3"))
        self.site_timeout = int(os.getenv("SUPPLIER_SITE_TIMEOUT", "45"))
        self.query_timeout = int(os.getenv("SUPPLIER_QUERY_TIMEOUT", "120"))

        self.error_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.error_log: List[Dict] = []
        self.human_gate_used: Dict[str, bool] = defaultdict(bool)

        self.last_error_path = LOG_DIR / "supplier_scout_last_error.json"

    def find_cheapest_supplier(self, product_name: str, sites: Optional[List[str]] = None) -> Optional[Dict]:
        sites = sites or ["farfetch", "ssense"]
        try:
            return asyncio.run(self._run_async(product_name, sites))
        except KeyboardInterrupt:
            LOG.warning("SupplierScout interrupted by user.")
            return None
        except Exception as e:
            LOG.exception(f"SupplierScout failed on '{product_name}': {e}")
            self._write_last_error(product_name, aborted=True, reason=str(e))
            return None

    async def _run_async(self, product_name: str, sites: List[str]) -> Optional[Dict]:
        deadline = time.monotonic() + self.query_timeout
        best: Optional[Dict] = None

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    channel="chrome",
                    headless=self.headless,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                    ],
                )
                context = await browser.new_context(
                    locale="ja-JP",
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"
                    ),
                )

                for site in sites:
                    if time.monotonic() > deadline:
                        LOG.warning("Query timed out (overall).")
                        break

                    try:
                        remain = max(5, int(deadline - time.monotonic()))
                        site_budget = min(self.site_timeout, remain)
                        res = await asyncio.wait_for(
                            self._search_site(context, site, product_name),
                            timeout=site_budget,
                        )
                    except (PWTimeoutError, asyncio.TimeoutError) as e:
                        if self._record_error(site, "timeout", str(e)):
                            LOG.warning(f"[{site}] timeout threshold exceeded. give up.")
                            break
                        else:
                            continue
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        if self._record_error(site, type(e).__name__, str(e)):
                            LOG.warning(f"[{site}] error threshold exceeded. give up.")
                            break
                        else:
                            continue

                    for cand in res:
                        if best is None or cand["price"] < best["price"]:
                            best = cand

                await context.close()
                await browser.close()

        except KeyboardInterrupt:
            self._write_last_error(product_name, aborted=True, reason="KeyboardInterrupt")
            return None
        except Exception as e:
            LOG.exception(f"Fatal in SupplierScout: {e}")
            self._write_last_error(product_name, aborted=True, reason=str(e))
            return None

        if best is None:
            self._write_last_error(product_name, aborted=False, reason="no candidates")
        return best

    async def _search_site(self, context, site: str, product_name: str) -> List[Dict]:
        site = site.lower().strip()
        if site == "farfetch":
            return await self._search_farfetch(context, product_name)
        if site == "ssense":
            return await self._search_ssense(context, product_name)
        return []

    async def _search_farfetch(self, context, query: str) -> List[Dict]:
        page = await context.new_page()
        items: List[Dict] = []
        ts = _now_ts()

        async def dbg_dump(name: str):
            if not self.debug:
                return
            base = DBG_DIR / f"{ts}_farfetch_{re.sub(r'[^A-Za-z0-9]+','_',query)[:40]}"
            base.mkdir(parents=True, exist_ok=True)
            try:
                with open(base / f"{name}.html", "w", encoding="utf-8") as f:
                    f.write(await page.content())
                await page.screenshot(path=str(base / f"{name}.png"))
            except Exception:
                pass

        try:
            search_url = f"https://www.farfetch.com/jp/shopping/all/items.aspx?q={self._enc(query)}"
            LOG.debug(f"[farfetch] GET {search_url}")
            resp = await page.goto(search_url, wait_until="domcontentloaded")
            status = (resp.status if resp else 0) or 0

            if status in (403, 404):
                await dbg_dump(f"resp_{status}")
                if status == 403:
                    if await self._maybe_human_gate("farfetch", page):
                        resp = await page.reload(wait_until="domcontentloaded")
                        status = (resp.status if resp else 0) or 0
                        if status == 403:
                            raise PermissionError("farfetch 403 after human gate")
                    else:
                        raise PermissionError("farfetch 403")

            items = await self._eval_extract_list(page, site="farfetch")
            if not items and status == 200:
                for seg in ("men", "women"):
                    url = f"https://www.farfetch.com/jp/shopping/{seg}/items.aspx?q={self._enc(query)}"
                    LOG.debug(f"[farfetch] fallback GET {url}")
                    resp = await page.goto(url, wait_until="domcontentloaded")
                    status = (resp.status if resp else 0) or 0
                    if status == 200:
                        items = await self._eval_extract_list(page, site="farfetch")
                        if items:
                            break

            if not items:
                await dbg_dump("no_candidates")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            await dbg_dump("exception")
            raise e
        finally:
            try:
                await page.close()
            except Exception:
                pass

        return items

    async def _search_ssense(self, context, query: str) -> List[Dict]:
        page = await context.new_page()
        items: List[Dict] = []
        ts = _now_ts()

        async def dbg_dump(name: str):
            if not self.debug:
                return
            base = DBG_DIR / f"{ts}_ssense_{re.sub(r'[^A-Za-z0-9]+','_',query)[:40]}"
            base.mkdir(parents=True, exist_ok=True)
            try:
                with open(base / f"{name}.html", "w", encoding="utf-8") as f:
                    f.write(await page.content())
                await page.screenshot(path=str(base / f"{name}.png"))
            except Exception:
                pass

        try:
            url = f"https://www.ssense.com/ja-jp/shop?q={self._enc(query)}"
            LOG.debug(f"[ssense] GET {url}")
            resp = await page.goto(url, wait_until="domcontentloaded")
            status = (resp.status if resp else 0) or 0

            if status == 403:
                await dbg_dump("403")
                if await self._maybe_human_gate("ssense", page):
                    resp = await page.reload(wait_until="domcontentloaded")
                    status = (resp.status if resp else 0) or 0
                    if status == 403:
                        raise PermissionError("ssense 403 after human gate")
                else:
                    raise PermissionError("ssense 403")

            items = await self._eval_extract_list(page, site="ssense")
            if not items:
                await dbg_dump("no_candidates")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            await dbg_dump("exception")
            raise e
        finally:
            try:
                await page.close()
            except Exception:
                pass

        return items

    async def _eval_extract_list(self, page, site: str) -> List[Dict]:
        js = """
        (() => {
          const out = [];
          const yenRe = /[¥\\u00A5]\\s*([\\d,]+)|([\\d,]+)\\s*円/;
          const anchors = Array.from(document.querySelectorAll("a[href*='/item/'], a[href*='/shop/']"));
          function text(e){return (e && (e.innerText || e.textContent) || '').replace(/\\s+/g,' ').trim();}
          function findPrice(el){
            if(!el) return '';
            const nodes = [el, ...el.querySelectorAll('*')];
            for(const n of nodes){
              const t = text(n);
              if(yenRe.test(t)) return t;
              if(/^[\\d,.]+$/.test(t)) return t;
            }
            return '';
          }
          const ld = Array.from(document.querySelectorAll("script[type='application/ld+json']"))
              .map(s=>{try{ return JSON.parse(s.textContent); }catch(e){ return null; }})
              .filter(Boolean);
          const scanPrice = (node) => {
            if(!node) return null;
            if(Array.isArray(node)) { for(const n of node){ const r=scanPrice(n); if(r) return r; } return null; }
            if(typeof node==='object'){
              if(node['@type']==='Product'){
                const offers=node['offers'];
                const pick=o=> (o && (o.price || (o.priceSpecification && o.priceSpecification.price)));
                if(Array.isArray(offers)){
                  for(const o of offers){ const p = pick(o); if(p) return String(p); }
                }else{
                  const p = pick(offers); if(p) return String(p);
                }
              }
              for(const k in node){ const r=scanPrice(node[k]); if(r) return r; }
            }
            return null;
          };
          for(const a of anchors){
            try{
              const url = a.href || '';
              if(!url) continue;
              const card = a.closest('li, article, div[class*="product"], div[class*="card"], div');
              const name = text(a) || text(card).slice(0,180);
              const priceT = findPrice(card);
              let priceNum = '';
              if(priceT){
                const m = priceT.match(/[\\d,]+/);
                if(m) priceNum = m[0];
              }
              if(!priceNum){
                for(const root of ld){
                  const p = scanPrice(root);
                  if(p){ priceNum = (p+'').match(/[\\d,]+/); priceNum = priceNum?priceNum[0]:''; if(priceNum) break; }
                }
              }
              if(priceNum){
                out.push({name, price: priceNum, url});
              }
            }catch(e){}
          }
          return out;
        })();
        """
        nodes = await page.evaluate(js)
        items: List[Dict] = []
        for n in nodes:
            price = _clean_price(n.get("price", ""))
            if not price:
                continue
            items.append({
                "site": site,
                "name": n.get("name") or "",
                "price": price,
                "currency": "JPY",
                "url": n.get("url") or "",
            })
        return items

    def _enc(self, s: str) -> str:
        try:
            from urllib.parse import quote_plus
            return quote_plus(s)
        except Exception:
            return s.replace(" ", "+")

    def _record_error(self, site: str, etype: str, message: str) -> bool:
        key = (site, etype)
        self.error_counts[key] += 1
        self.error_log.append({
            "ts": _now_ts(),
            "site": site,
            "etype": etype,
            "message": message[:500],
        })
        LOG.debug(f"[{site}] error {etype}: count={self.error_counts[key]}")
        return self.error_counts[key] >= self.max_same_error

    async def _maybe_human_gate(self, site: str, page) -> bool:
        if self.human_gate_used.get(site, False):
            return False
        self.human_gate_used[site] = True
        try:
            if os.getenv("PLAYWRIGHT_HEADFUL", "") == "1":
                print("\n" + "=" * 50)
                print(f"[{site}] 人間判定/403 の解除が必要です。解除後に Enter を押してください。")
                input("Enter to continue...")
                print("=" * 50 + "\n")
                return True
        except Exception:
            pass
        return False

    def _write_last_error(self, query: str, aborted: bool, reason: str):
        data = {
            "query": query,
            "aborted": aborted,
            "reason": reason,
            "max_same_error": self.max_same_error,
            "site_timeout": self.site_timeout,
            "query_timeout": self.query_timeout,
            "errors": self.error_log,
            "counts": [{"site": s, "etype": e, "count": c}
                       for (s, e), c in sorted(self.error_counts.items())],
            "ts": _now_ts(),
        }
        try:
            self.last_error_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            LOG.info(f"[SupplierScout] wrote error report: {self.last_error_path}")
        except Exception as e:
            LOG.error(f"failed to write error report: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, nargs="+", help="product query (e.g. 'Gucci GG0637SK')")
    parser.add_argument("--sites", type=str, default="farfetch,ssense", help="comma separated site names")
    parser.add_argument("--headful", action="store_true", help="run with visible browser window")
    args = parser.parse_args()
    if args.headful:
        os.environ["PLAYWRIGHT_HEADFUL"] = "1"

    scout = SupplierScout(headless=not args.headful)
    q = " ".join(args.query).strip()
    best = scout.find_cheapest_supplier(q, sites=[s.strip() for s in args.sites.split(",") if s.strip()])
    print("BEST:", best)
