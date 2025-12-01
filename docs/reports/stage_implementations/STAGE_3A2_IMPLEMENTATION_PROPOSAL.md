# Stage 3A-2 実装提案

## 概要

Stage 3A-2-1 で `_collect_pdp_links` のロジックを NavigationDriver に移行しましたが、
シグネチャを `NavigationContext` を使う形に統一し、より一貫性のある API にします。

## 実装案

### 1. NavigationDriver.collect_pdp_links のシグネチャ変更

**現在の実装:**
```python
async def collect_pdp_links(
    self,
    page: Page,
    site_config: Dict[str, Any],
    settings: Dict[str, Any],
    run_context: Any,
) -> List[str]:
```

**提案する実装:**
```python
async def collect_pdp_links(
    self,
    ctx: NavigationContext,
) -> List[str]:
```

### 2. BrowserUseAgent 側の変更

`_collect_pdp_links` を NavigationContext を構築して NavigationDriver を呼ぶ形に変更します。

---

## 差分パッチ

### パッチ 1: NavigationDriver.collect_pdp_links のシグネチャ変更

```diff
--- a/app/agents/browser/navigation_driver.py
+++ b/app/agents/browser/navigation_driver.py
@@ -152,15 +152,12 @@ class NavigationDriver:
         logger.debug(f"[NavigationDriver] run_plp_flow (stub): entry_url={entry}, trap_detected={outcome.trap_detected}")
         return outcome
 
     async def collect_pdp_links(
         self,
-        page: Page,
-        site_config: Dict[str, Any],
-        settings: Dict[str, Any],
-        run_context: Any,
+        ctx: NavigationContext,
     ) -> List[str]:
         """
         Stage 3A-2-1:
         旧 BrowserUseAgent._collect_pdp_links のロジックをここに移行。
         挙動・ログ・例外の流れはそのまま維持されている。
         
         Phase 1a: Global <a href> sweep + Regex Filter
         Phase 1b: Selector-based補完
         Phase 2: Deep Extraction Fallback (only if Phase 1 failed)
         Phase 3: Noise Filtering & Saving
         
         Args:
-            page: Playwright Page オブジェクト
-            site_config: サイト設定
-            settings: 実行設定
-            run_context: 実行コンテキスト
+            ctx: ナビゲーションコンテキスト
             
         Returns:
             List[str]: PDP リンクのリスト
         """
+        page = self.page
+        site_config = ctx.site_config
+        settings = ctx.settings
+        run_context = ctx.run_context
         target_url = page.url
         found_links: Set[str] = set()
 
         # Phase 1a: Global <a href> sweep + Regex Filter
         try:
             raw_hrefs: List[str] = await page.evaluate("() => Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href')).filter(Boolean)")
         except Exception as e:
             logger.warning(f"[PLP→PDP][1a] Sweep failed: {e}")
             raw_hrefs = []
         pdp_rx = re.compile(r"/(products?|p)/", re.I)
         for href in raw_hrefs:
             if pdp_rx.search(href):
                 norm_url = self._normalize_abs_url(target_url, href)
                 if is_same_origin(norm_url, target_url) and looks_like_product_url(norm_url):
                     found_links.add(norm_url)
         if found_links:
             logger.info(f"[PLP→PDP][1a] Sweep found {len(found_links)} links.")
 
         # Phase 1b: Selector-based補完
         selectors_cfg = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}
         PLP_PDP_LINK_SELECTORS = _dedupe_keep_order(
             (selectors_cfg.get("pdp_link_selectors", []) or []) + [
                 "a[href*='/products/']",
                 "a[href*='/product/']",
                 "a[href*='/p/']",
                 "[data-component*='ProductCard'] a[href]",
                 "[class*='product-card'] a[href]",
                 "article [data-testid*='product']:is(a, * a)",
                 "[data-testid*='card'] a[href]",
                 "[data-testid*='product-card'] a[href]",
                 "a[data-product-url]",
                 "[data-qa='product-tile'] a[href]",
             ]
         )
         for sel in PLP_PDP_LINK_SELECTORS:
             try:
                 nodes = await page.query_selector_all(sel)
                 if not nodes:
                     continue
                 matched_count = 0
                 for n in nodes:
                     href = await n.get_attribute("href") or await n.get_attribute("data-href") or await n.get_attribute("data-product-url") or await n.get_attribute("data-url")
                     if not href:
                         continue
                     norm_url = self._normalize_abs_url(target_url, href)
                     if is_same_origin(norm_url, target_url) and looks_like_product_url(norm_url):
                         found_links.add(norm_url)
                         matched_count += 1
                 if matched_count > 0:
                     logger.info(f"[PLP→PDP][1b] selector='{sel}' added {matched_count} links.")
             except Exception as e:
                 logger.warning(f"[PLP→PDP][1b] selector='{sel}' failed: {e}")
 
         # Phase 2: Deep Extraction Fallback (only if Phase 1 failed)
         if not found_links:
             logger.warning("[PLP→PDP] Phase 1a/1b found no links. Falling back to Phase 2 (Deep Extraction)...")
             try:
                 deep_hrefs = await self._run_deep_extraction_phase2(page, site_config)
                 for href in deep_hrefs:
                     norm_url = self._normalize_abs_url(target_url, href)
                     if is_same_origin(norm_url, target_url) and looks_like_product_url(norm_url):
                         found_links.add(norm_url)
                 if found_links:
                     logger.info(f"[PLP→PDP][2] Deep Extraction found {len(found_links)} links.")
             except Exception as e:
                 logger.error(f"[PLP→PDP][2] Deep Extraction failed: {e}")
 
         links = sorted(list(found_links))
         if not links:
             logger.warning("[PLP→PDP] No PDP hrefs found after all phases.")
             return []
 
         # Phase 3: Noise Filtering & Saving
         cleaned: List[str] = []
         noise_rx = re.compile(r"/(collections?|seasons?|client-service|login|legal|cart|wishlist|search)/", re.I)
         for u in links:
             if not noise_rx.search(u):
                 cleaned.append(u)
         logger.info(f"[PLP→PDP] collected {len(cleaned)} PDP-like links (raw={len(links)})")
         try:
             sample = cleaned[:20]
             logger.debug(f"[PLP→PDP] sample={sample}")
             if run_context and hasattr(run_context, "save_json"):
                 run_context.save_json("raw_pdp_links_v85.5.json", {"links": cleaned, "sample": sample})
             # Stage 3B: TelemetryService を使用
             try:
                 if run_context:
                     if self.telemetry:
                         await self.telemetry.save_raw_hrefs(cleaned, name="raw_hrefs_final_cleaned")
                     else:
                         # フォールバック: 既存のobservability.py関数を使用
                         from app.utils.observability import save_raw_hrefs
                         if callable(save_raw_hrefs):
                             res = save_raw_hrefs(run_context, cleaned, name="raw_hrefs_final_cleaned")
                             if asyncio.iscoroutine(res):
                                 await res
             except Exception as e:
                 logger.debug(f"[PLP→PDP] TelemetryService.save_raw_hrefs failed: {e}")
         except Exception:
             pass
         return cleaned
```

### パッチ 2: BrowserUseAgent._collect_pdp_links の変更

```diff
--- a/app/agents/browser_use_agent.py
+++ b/app/agents/browser_use_agent.py
@@ -1175,15 +1175,20 @@ class BrowserUseAgent:
     async def _collect_pdp_links(self, page: Page, site_config: Dict, settings: Dict, run_context: RunContext) -> List[str]:
         """
         Stage 3A-2-1:
         PDPリンク収集の本体は NavigationDriver.collect_pdp_links に移譲。
         互換性維持のため、既存シグネチャは残しておく。
         """
-        # Stage 3A-2-1: NavigationDriver を使用して PDP リンクを収集
-        # TelemetryService は既存のものを取得
-        telemetry = self._ensure_telemetry()
+        # Stage 3A-2-1: NavigationContext を構築して NavigationDriver を呼ぶ
+        from app.agents.browser.navigation_driver import NavigationContext, NavigationDriver
+        
+        # NavigationContext を構築（site, query は _run_plp_flow から取得する必要があるが、
+        # このメソッドでは直接取得できないため、最小限の値で構築）
+        # 注意: このメソッドは _run_plp_flow 内から呼ばれるため、
+        # 将来的には _run_plp_flow から直接 NavigationDriver を呼ぶ形に変更する
+        nav_ctx = NavigationContext(
+            site=self.runtime_kwargs.get("site", "UNKNOWN"),
+            query=self.runtime_kwargs.get("query", ""),
+            site_config=site_config,
+            settings=settings,
+            run_context=run_context,
+            start_t=time.time(),
+            budget_ms=int(settings.get("timeout_sec", 60)) * 1000,
+        )
         
         navigation_driver = NavigationDriver(
             page=page,
-            trap_checker=None,  # collect_pdp_links では trap_checker は不要
             telemetry=self._ensure_telemetry(),
-            strategy=None,  # collect_pdp_links では strategy は不要
+            trap_checker=None,
+            strategy=None,
         )
-        return await navigation_driver.collect_pdp_links(page, site_config, settings, run_context)
+        return await navigation_driver.collect_pdp_links(nav_ctx)
```

### パッチ 3: _run_plp_flow からの直接呼び出し（推奨）

より良いアプローチとして、`_run_plp_flow` 内で既に構築されている `NavigationContext` を再利用し、
`_collect_pdp_links` を経由せずに直接 `NavigationDriver.collect_pdp_links` を呼ぶ形に変更します。

```diff
--- a/app/agents/browser_use_agent.py
+++ b/app/agents/browser_use_agent.py
@@ -1694,7 +1694,11 @@ class BrowserUseAgent:
             except Exception as e2:
                 logger.warning(f"[Hook A1] Fallback also failed: {e2}")
 
-        pdp_links = await self._collect_pdp_links(page, site_config, settings, run_context)
+        # Stage 3A-2-1: NavigationDriver を直接使用
+        # 既に _run_plp_flow 内で NavigationContext が構築されているため、それを再利用
+        navigation_driver = NavigationDriver(
+            page=page,
+            telemetry=self._ensure_telemetry(),
+            trap_checker=None,
+            strategy=None,
+        )
+        pdp_links = await navigation_driver.collect_pdp_links(nav_ctx)
 
         # Fallback logic (header search, click first card)
         # V88.5.7: このブロックは ok_materialized=True (タイル1枚以上) だが
@@ -1742,7 +1746,7 @@ class BrowserUseAgent:
                         except Exception as e2:
                             logger.warning(f"[Hook A3] Fallback also failed: {e2}")
                     # Stage 3A-2-1: NavigationDriver を直接使用
-                    pdp_links = await self._collect_pdp_links(page, site_config, settings, run_context)
+                    pdp_links = await navigation_driver.collect_pdp_links(nav_ctx)
 
                     # --- V88.5.5: 早期失敗ロジック ---
                     if not pdp_links:
```

---

## 実装の優先順位

### オプション A: 段階的移行（推奨）

1. **ステップ 1**: NavigationDriver.collect_pdp_links のシグネチャを NavigationContext に変更（パッチ 1）
2. **ステップ 2**: BrowserUseAgent._collect_pdp_links を NavigationContext を使う形に変更（パッチ 2）
3. **ステップ 3**: _run_plp_flow から直接 NavigationDriver を呼ぶ形に変更（パッチ 3）
4. **ステップ 4**: _collect_pdp_links を削除（将来的に）

### オプション B: 一気に移行

パッチ 1 + パッチ 3 を同時に適用し、`_collect_pdp_links` を経由せずに直接 NavigationDriver を呼ぶ。

---

## 注意事項

1. **NavigationContext の構築**: `_collect_pdp_links` 内では `site` と `query` を直接取得できないため、
   `self.runtime_kwargs` から取得する必要があります。より良い方法は、`_run_plp_flow` から直接呼ぶことです。

2. **互換性**: 既存の `_collect_pdp_links` のシグネチャは残しておくことで、他の箇所からの呼び出しがあっても動作します。

3. **将来の拡張**: Stage 3A-2-4 で Phase 3（ヘッダ検索 / カードクリック fallback）を NavigationDriver に移行する際も、
   NavigationContext を使うことで一貫性が保たれます。

