"""Deep extraction Phase 2: JSON-LD, onclick, data-* attribute link extraction."""

from __future__ import annotations

import logging
from typing import Any

from playwright.async_api import ElementHandle, Page

logger = logging.getLogger(__name__)


async def run_deep_extraction_phase2(
    page: Page,
    site_config: dict[str, Any],
    safe_wait_selector_fn: Any,  # async callable(page, selector, timeout_ms, state) -> bool
) -> list[str]:
    logger.debug("[Phase 2] Running deep extraction (JSON-LD, onclick, data-*, ...)")
    container_sels: list[str] = ((site_config.get("selectors") or {}).get("pdp") or {}).get(
        "plp_container_selectors", []
    ) or []
    for cont in container_sels or []:
        await safe_wait_selector_fn(page, cont, timeout_ms=1000, state="visible")
    try:
        for _ in range(2):
            await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            await page.wait_for_timeout(200)
    except Exception:
        pass

    scope = page.locator("main, [role='main'], #main, #app")
    handle: ElementHandle | None = None
    try:
        await scope.first.wait_for(state="attached", timeout=4000)
        handle = await scope.first.element_handle(timeout=4000)
    except Exception as e_handle:
        logger.warning(f"[Phase 2] Could not get element handle for scope: {e_handle}. Falling back to page evaluate.")
        handle = None

    _js_script = """
      (node) => {
        const area = node || document;
        const out = [];
        const push = (u) => { if (u && typeof u === 'string' && !u.startsWith('javascript:')) out.push(u); };
        area.querySelectorAll("[data-product-url],[data-product-href],[data-href],a[href]").forEach(el => {
          const a = el.closest("a") || el;
          const cand = a.getAttribute("href") || a.getAttribute("data-href") || a.getAttribute("data-product-url") || a.getAttribute("data-product-href");
          if (cand) push(cand);
        });
        area.querySelectorAll("[onclick]").forEach(el => {
          const oc = el.getAttribute("onclick") || "";
          const m1 = oc.match(/(?:location\\.(?:href|assign)|window\\.location|document\\.location)\\s*=\\s*['"]([^'"]+)['"]/i);
          if (m1 && m1[1]) push(m1[1]);
          const m2 = oc.match(/history\\.pushState\\s*\\(\\s*[^,]*,\\s*[^,]*,\\s*['"]([^'"]+)['"]\\s*\\)/i);
          if (m2 && m2[1]) push(m2[1]);
        });
        area.querySelectorAll("script[type='application/ld+json']").forEach(s => {
          try {
            const data = JSON.parse(s.textContent || "null");
            const arr = Array.isArray(data) ? data : [data];
            const pushAny = (v) => { if (v && typeof v === "string") push(v); };
            arr.forEach(d => {
              if (!d || typeof d !== "object") return;
              pushAny(d.url || d['@id']);
              if (Array.isArray(d.offers)) {
                d.offers.forEach(o => { pushAny(o && (o.url || o['@id'])); });
              } else if (d.offers && typeof d.offers === "object") {
                pushAny(d.offers.url || d.offers['@id']);
              }
              if (d.itemListElement && Array.isArray(d.itemListElement)) {
                d.itemListElement.forEach(it => {
                  if (it && it.item && (it.item.url || it.item['@id'])) {
                    pushAny(it.item.url || it.item['@id']);
                  }
                });
              }
            });
          } catch(e) {}
        });
        return out.filter(Boolean);
      }
    """

    hrefs: list[str] = []
    try:
        if handle:
            hrefs = await handle.evaluate(_js_script)
            logger.debug("[Phase 2] Deep extraction performed using element handle.")
        else:
            hrefs = await page.evaluate("""
              () => {
                const out = [];
                const push = (u) => { if (u && typeof u === 'string' && !u.startsWith('javascript:')) out.push(u); };
                document.querySelectorAll("[data-product-url],[data-product-href],[data-href],a[href]").forEach(el => {
                  const a = el.closest("a") || el;
                  const cand = a.getAttribute("href") || a.getAttribute("data-href") || a.getAttribute("data-product-url") || a.getAttribute("data-product-href");
                  if (cand) push(cand);
                });
                document.querySelectorAll("[onclick]").forEach(el => {
                  const oc = el.getAttribute("onclick") || "";
                  const m1 = oc.match(/(?:location\\.(?:href|assign)|window\\.location|document\\.location)\\s*=\\s*['"]([^'"]+)['"]/i);
                  if (m1 && m1[1]) push(m1[1]);
                  const m2 = oc.match(/history\\.pushState\\s*\\(\\s*[^,]*,\\s*[^,]*,\\s*['"]([^'"]+)['"]\\s*\\)/i);
                  if (m2 && m2[1]) push(m2[1]);
                });
                document.querySelectorAll("script[type='application/ld+json']").forEach(s => {
                  try {
                    const data = JSON.parse(s.textContent || "null");
                    const arr = Array.isArray(data) ? data : [data];
                    const pushAny = (v) => { if (v && typeof v === "string") push(v); };
                    arr.forEach(d => {
                      if (!d || typeof d !== "object") return;
                      pushAny(d.url || d['@id']);
                      if (Array.isArray(d.offers)) {
                        d.offers.forEach(o => { pushAny(o && (o.url || o['@id'])); });
                      } else if (d.offers && typeof d.offers === "object") {
                        pushAny(d.offers.url || d.offers['@id']);
                      }
                      if (d.itemListElement && Array.isArray(d.itemListElement)) {
                        d.itemListElement.forEach(it => {
                          if (it && it.item && (it.item.url || it.item['@id'])) {
                            pushAny(it.item.url || it.item['@id']);
                        }
                        });
                      }
                    });
                  } catch(e) {}
                });
                return out.filter(Boolean);
              }
            """)
            logger.debug("[Phase 2] Deep extraction performed using page evaluate (fallback).")
    except Exception as e:
        logger.warning(f"[Phase 2] Deep extraction evaluate failed: {e}")
        hrefs = []

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for h in hrefs:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    return unique
