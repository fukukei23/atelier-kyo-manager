# Moncler 用 site_config チューニング - JSON Diff

**作成日時**: 2025-01-28  
**目的**: `overrides.local.json` の `MONCLER_OFFICIAL` 設定を Stage 5 新スキーマに完全対応させる

---

## 変更サマリー

### 削除対象
- `selectors.pdp.plp_container_selectors` (357-369行目) - PLP用
- `selectors.pdp.pdp_link_selectors` (370-386行目) - PLP用  
- `selectors.pdp.plp_verification_selectors` (387-391行目) - PLP用
- `selectors.pdp.pdp_click_fallback_selectors` (392-400行目) - PLP用
- `selectors_patch.*` (528-584行目) - 旧スキーマ
- `overrides_patch.*` (585-643行目) - 旧スキーマ（一部は残す可能性あり）

### 統合対象
- `title_selectors` (401-406行目) → `title` に統合
- `price_selectors` (407-414行目) → `price` に統合

### 追加対象
- `selectors.pdp.images`
- `selectors.pdp.colors`
- `selectors.pdp.sizes`
- `selectors.pdp.description`
- `selectors.pdp.breadcrumbs`
- `selectors.pdp.sku`
- `selectors.pdp.availability`
- `selectors.pdp.json_ld`
- `selectors.pdp.meta_fallback`
- `selectors.pdp.raw_html_capture`
- `price_rules` (トップレベル)

---

## 詳細な JSON Diff

### 変更位置: `app/config/sites/overrides.local.json` の `MONCLER_OFFICIAL` ブロック内

```diff
  "MONCLER_OFFICIAL": {
    // ... 既存の設定（navigation, locale, behavior, discovery_settings など）は変更なし ...
    
    "selectors": {
      // ... plp, ui は変更なし ...
      
      "pdp": {
-        // ===== 削除: PLP 用セレクタ（PDP では不要） =====
-        "plp_container_selectors": [
-          "[data-testid='product-grid']",
-          "[data-test='product-grid']",
-          "section[id*='product'] ul[role='list']",
-          "[data-testid='plp-grid']",
-          "[data-qa='product-tile']",
-          "[data-test*='product' i]",
-          "[data-component*='ProductGrid']",
-          "section[role='region'] .product-list",
-          ".plp-grid",
-          ".product-grid",
-          ".c-product-grid"
-        ],
-        "pdp_link_selectors": [
-          "[data-testid='product-card'] a[href*='/products/']",
-          "[data-test='product-card'] a[href*='/products/']",
-          "ul[role='list'] li a[href*='/products/']",
-          "a[href*='/en-de/'][href*='/products/']",
-          "a[href*='/en-int/'][href*='/products/']",
-          "a[href*='/en-int/'][href*='/p/']",
-          "[data-qa='product-tile'] a[href]",
-          "[data-testid*='product' i] a[href]",
-          "a[data-qa='product-card-link']",
-          "a[href*='/products/']",
-          "a[href*='/p/']",
-          ".product-card a[href*='/products/']",
-          ".c-product-card__link",
-          ".c-product-tile a[href*='/products/']",
-          ".c-product-tile a[href*='/p/']"
-        ],
-        "plp_verification_selectors": [
-          "[data-testid='plp-grid']",
-          ".plp-grid",
-          ".product-grid"
-        ],
-        "pdp_click_fallback_selectors": [
-          "[data-qa='product-tile']",
-          "[data-testid='product-card']",
-          "[data-test*='product' i]",
-          ".c-product-card",
-          ".c-product-tile",
-          ".product-card",
-          ".product-item"
-        ],
-        
-        // ===== 統合: 旧キー名を新キー名に変更 =====
-        "title_selectors": [
+        // ===== 基本情報 =====
+        "title": [
           "[data-testid='pdp-title']",
+          "h1[data-test='product-name']",
+          "h1[itemprop='name']",
           "h1",
           "[data-testid='product-name']",
-          ".c-product-title"
+          ".c-product-title",
+          "h1.product-name",
+          "meta[property='og:title']"
         ],
-        "price_selectors": [
+        "price": [
           "[data-testid='pdp-price']",
+          "[data-test='pdp-price'] [itemprop='price']",
           "[itemprop='price']",
           "[data-testid*='price' i]",
           ".c-price__value",
           ".price",
-          "[class*='price' i]"
+          "[class*='price' i]",
+          "meta[itemprop='price'][content]"
         ],
+        "list_price": [
+          "[data-testid='pdp-list-price']",
+          "[data-test='pdp-list-price']",
+          ".c-price__original",
+          "[class*='list-price' i]",
+          "[class*='original-price' i]"
+        ],
+        "currency": [
+          "meta[property='product:price:currency']",
+          "meta[itemprop='priceCurrency'][content]",
+          "[itemprop='priceCurrency']",
+          "[data-testid*='currency' i]"
+        ],
+        
+        // ===== メディア =====
+        "images": {
+          "selectors": [
+            ".product-gallery img",
+            "[data-testid='product-image'] img",
+            "[data-testid='pdp-image'] img",
+            "img[itemprop='image']",
+            ".c-product-image img",
+            ".product-images img"
+          ],
+          "image_attr": "src",
+          "base_url": null
+        },
+        
+        // ===== 説明・詳細 =====
+        "description": [
+          "[data-testid='product-description']",
+          "[data-test='product-description']",
+          "[itemprop='description']",
+          ".c-product-description",
+          ".product-description",
+          "meta[property='og:description']"
+        ],
+        "breadcrumbs": [
+          ".breadcrumb li",
+          "nav[aria-label='Breadcrumb'] a",
+          "[data-testid='breadcrumb'] a",
+          "ol.breadcrumb li"
+        ],
+        
+        // ===== 属性 =====
+        "colors": [
+          "button[data-testid*='color' i]",
+          "button[data-test='color-swatch']",
+          "[data-color]",
+          ".color-selector button",
+          "[role='radiogroup'][aria-label*='Color' i] button"
+        ],
+        "sizes": [
+          "button[data-testid*='size' i]:not([disabled])",
+          "button[data-test='size-option']:not([disabled])",
+          "[role='radiogroup'][aria-label*='Size' i] button:not([disabled])",
+          ".size-selector button:not([disabled])",
+          "button[aria-pressed='false'][data-testid*='size' i]:not([disabled])"
+        ],
+        "brand": [
+          "[data-testid='product-brand']",
+          "[itemprop='brand']",
+          "meta[property='product:brand']",
+          "meta[property='og:site_name']"
+        ],
+        "sku": [
+          "[data-testid='product-sku']",
+          "[data-test='product-sku']",
+          "[itemprop='sku']",
+          ".product-code",
+          ".sku"
+        ],
+        "availability": {
+          "selectors": [
+            "[data-testid='stock-status']",
+            "[itemprop='availability']",
+            "meta[property='product:availability']",
+            "[data-test='availability']"
+          ],
+          "patterns": [
+            "out of stock",
+            "in stock",
+            "pre-order",
+            "在庫なし",
+            "在庫あり"
+          ]
+        },
+        
         // ===== サイズ選択関連（価格表示のため）- 既存の設定は残す =====
         "price_requires_size": true,
         "auto_select_size": true,
+        "size_button": [
+          "button[aria-pressed='false'][data-testid*='size' i]:not([disabled])",
+          "button[role='radio']:not([aria-disabled='true']):not([disabled])",
+          "[role='option']:not([aria-disabled='true']) button:not([disabled])",
+          "li:not([aria-disabled='true']) button:not([disabled])"
+        ],
+        "size_select_policy": {
+          "mode": "first_instock",
+          "prefer_labels": ["M", "L"],
+          "price_wait_ms": 4000
+        },
         "size_container_selectors": [
           "[data-testid*='size' i]",
           "[class*='size' i]",
           "fieldset[aria-label*='Size' i]"
         ],
-        "size_option_selectors": [
-          "button[aria-pressed='false'][data-testid*='size' i]:not([disabled])",
-          "button[role='radio']:not([aria-disabled='true']):not([disabled])",
-          "[role='option']:not([aria-disabled='true']) button:not([disabled])",
-          "li:not([aria-disabled='true']) button:not([disabled])"
-        ],
         "size_selected_check_selectors": [
           "button[aria-pressed='true'][data-testid*='size' i]",
           "[role='option'][aria-selected='true']",
           "[aria-live] .selected-size"
         ],
-        "price_after_size_wait_selectors": [
+        "visible_price_selectors": [
           "[itemprop='price']",
           "[data-testid*='price' i]",
           ".c-price__value",
           ".price",
           "[class*='price' i]"
         ],
+        
+        // ===== JSON-LD フォールバック =====
+        "json_ld": {
+          "enabled": true,
+          "paths": {
+            "price": ["offers.price", "offers[0].price"],
+            "currency": ["offers.priceCurrency", "offers[0].priceCurrency"],
+            "title": ["name"],
+            "description": ["description"],
+            "availability": ["offers.availability"]
+          }
+        },
+        
+        // ===== Meta タグフォールバック =====
+        "meta_fallback": {
+          "enabled": true,
+          "selectors": {
+            "price": [
+              "meta[property='og:price:amount']",
+              "meta[property='product:price:amount']"
+            ],
+            "currency": [
+              "meta[property='og:price:currency']",
+              "meta[property='product:price:currency']"
+            ],
+            "title": ["meta[property='og:title']"],
+            "description": ["meta[property='og:description']"]
+          }
+        },
+        
+        // ===== HTML 保存設定 =====
+        "raw_html_capture": {
+          "enabled": true,
+          "filename": "pdp_raw.html"
+        },
+        
+        // ===== ブロックリスト（既存の設定は残す） =====
         "blocklist_href_substrings": [
           "/en-jp/",
           "/legal/",
           "/client-service/",
           "/children/",
           "/brands/moncler",
           "monclergroup.com",
           "onetrust",
           "cookie",
           "/help",
           "/account",
           "/stores",
           "/contact",
           "doubleclick.net",
           "criteo.com",
           "tracking",
           "service_worker",
           "mailto:",
           "tel:",
           "instagram.com",
           "facebook.com",
           "youtube.com",
           "x.com",
           "line.me",
           "#",
           "#product-information-panel",
           "#search-field"
         ]
       },
       
       // ... url_rules は変更なし ...
       
-      // ===== 削除: 旧スキーマ =====
-      "selectors_patch": {
-        "plp": {
-          "item": [
-            "a[href*='/p-']",
-            "a[href*='/products/']",
-            "li[data-testid='product-card']",
-            "article[data-testid='product-card']",
-            "div.product-tile, div.c-product-tile"
-          ],
-          "link": [
-            "a[href*='/products/']",
-            "a[href*='/p-']"
-          ],
-          "price": "[data-testid='product-card'] [data-testid*='price'], [data-testid='product-card'] [class*='price']",
-          "container": "main [data-testid='product-list'], main [data-component='ProductGrid'], main .product-grid",
-          "items": "li [data-testid='product-card'] a[href*='/products/'], article [data-testid='product-card'] a[href*='/products/'], a[href*='/products/'][data-testid*='product']",
-          "title": "[data-testid='product-card'] [data-testid*='title'], [data-testid='product-card'] [class*='title']",
-          "tile_alt": [
-            "[data-test='product-grid'] a[href$='.html']",
-            "[data-testid='product-grid'] a[href$='.html']",
-            "#product-grid a[href$='.html']",
-            "main a[href$='.html']"
-          ],
-          "link_within_tile_alt": "a[href$='.html']"
-        },
-        "pdp": {
-          "price": "meta[itemprop='price']@content, [data-testid='pdp-price'] [data-testid*='value'], [class*='Price'] [class*='value']",
-          "title": "h1[data-testid='pdp-title'], h1[itemprop='name'], h1",
-          "title_alt": [
-            "h1[data-test='product-name']",
-            "h1[class*='product']",
-            "meta[property='og:title']::attr(content)"
-          ],
-          "price_alt": [
-            "script[type='application/ld+json']$json:$.offers.price",
-            "[itemprop='price']::attr(content)",
-            "[data-test*='price']",
-            ".price, .ProductPrice"
-          ]
-        },
-        "modals": {
-          "accept_cookies": [
-            "#onetrust-accept-btn-handler",
-            "button:has-text('Accept All')",
-            "button:has-text('Alle akzeptieren')"
-          ],
-          "geo_continue": [
-            "button:has-text('Continue')",
-            "button:has-text('Save')",
-            "button:has-text('Fortfahren')"
-          ]
-        },
-        "dismiss": {
-          "cookie": "#onetrust-accept-btn-handler, button[aria-label*='Accept all cookies']",
-          "geo": "[data-testid='geolocation-continue'], button[data-testid*='dialog-accept'], button[data-test*='continue']"
-        }
-      },
-      "overrides_patch": {
-        "wait_for_selectors": [
-          "#onetrust-accept-btn-handler",
-          "[data-test='product-grid'], [data-testid='product-grid'], #product-grid",
-          "[data-testid='geolocation-continue'], button[data-testid='dialog-accept']",
-          "main [data-component='ProductGrid'], main [data-testid='product-list']",
-          "a[href*='/products/']",
-          "a[href*='/p-']"
-        ],
-        "pre_actions": [
-          {
-            "action": "click_if_present",
-            "selector": "#onetrust-accept-btn-handler"
-          },
-          {
-            "action": "click_if_present",
-            "selector": "button:has-text('Accept All')"
-          },
-          {
-            "action": "click_if_present",
-            "selector": "button:has-text('Alle akzeptieren')"
-          },
-          {
-            "action": "click_if_present",
-            "selector": "button:has-text('Continue')"
-          },
-          {
-            "action": "click_if_present",
-            "selector": "button:has-text('Save')"
-          }
-        ],
-        "headers": {
-          "Accept-Language": "en-DE,en;q=0.9,de;q=0.8"
-        },
-        "scroll": {
-          "enabled": true,
-          "until_min_items": 24,
-          "stall_rounds": 2
-        },
-        "start_url": "https://www.moncler.com/en-int/men/ready-to-wear/jackets",
-        "actions": [
-          {
-            "click": "#onetrust-accept-btn-handler",
-            "optional": true
-          },
-          {
-            "click": "[data-testid='geolocation-continue'], button[data-testid='dialog-accept']",
-            "optional": true
-          }
-        ],
-        "pagination": {
-          "next": "a[rel='next'], button[aria-label='Next']",
-          "load_more": "button[aria-label*='Load more'], button[data-testid*='load-more']"
-        },
-        "click_selectors_once": [
-          "#onetrust-accept-btn-handler",
-          "button[aria-label*='Accept all']"
-        ]
-      },
-      "rationale": "The failure occurred on a PDP (.html) while expecting a PLP; Moncler often shows a cookie consent overlay that can block rendering. We add cookie-accept handling and broaden PLP grid/tile selectors to robust container/link patterns commonly used on moncler.com. We also add PDP fallbacks using JSON-LD to reliably read price/title if we land on a PDP. URL normalization rules convert accidental PDP URLs back to their parent PLP and enforce trailing slashes while avoiding locale drift and off-domain redirects (e.g., monclergroup.com).",
-      "code_hints": [
-        "If no PLP tiles materialize and the current URL ends with .html on moncler.com, treat the page as a PDP and attempt PDP extraction instead of failing immediately.",
-        "Implement JSON-LD fallback on PDP: parse the first script[type='application/ld+json'] block; read Product.name for title and Offers.price (or offers[0].price) for price. Currency is typically Offers.priceCurrency.",
-        "After navigation, auto-click cookie consent if #onetrust-accept-btn-handler exists before querying PLP/PDP selectors.",
-        "For PLP extraction, first wait for [data-test='product-grid'], [data-testid='product-grid'], or #product-grid; then collect product links via a[href$='.html'] within that container.",
-        "When normalizing URLs for PLP intent, apply regex s#/[^/]+\\.html(\\?.*)?$#/# to back out of PDPs to the parent category and ensure a trailing slash.",
-        "Guard against locale drift: if a redirect changes /en-de/ to another locale, rewrite back to the original locale before continuing.",
-        "Set Accept-Language to en to encourage en-int routing.",
-        "After navigation, attempt optional clicks on selectors.dismiss.cookie and selectors.dismiss.geo, then wait for selectors.plp.container.",
-        "If still on homepage after normalization, open the mega-menu and click a category link (e.g., a[href*='/en-int/men/ready-to-wear/jackets']).",
-        "On PDP, if CSS price is missing, parse application/ld+json for offers.price and currency.",
-        "Before waiting for PLP items, attempt to dismiss OneTrust via #onetrust-accept-btn-handler; fall back to button:has-text('Accept All') or 'Alle akzeptieren'.",
-        "Detect a locale/region modal (generic dialog) and click the first visible button matching 'Continue'/'Save'/'Fortfahren' if present.",
-        "Wait for PLP anchors a[href*='/p-'] as the product item/link selector; treat the anchor as the card when extracting href/title/price.",
-        "Implement auto-scroll until at least 24 unique a[href*='/p-'] are collected or two scroll cycles yield no new items.",
-        "If navigation to PLP gets redirected to /en-de/ (homepage), re-navigate to the canonical_plp and re-run pre_actions.",
-        "On PDP, read price from [data-testid='price'] first, then fall back to span[itemprop='price'] or span[class*='price']."
-      ],
-      "risk": "medium"
     }
   },
+  // ===== 追加: 価格正規化ルール（MONCLER_OFFICIAL ブロック内のトップレベル） =====
+  "price_rules": {
+    "strip_chars": ["€", ",", " ", "EUR", "GBP", "USD"],
+    "thousands_separator": ",",
+    "decimal_separator": ".",
+    "currency_fallback": "EUR",
+    "price_pattern": "[\\d.,]+",
+    "currency_symbols": {
+      "€": "EUR",
+      "£": "GBP",
+      "$": "USD",
+      "¥": "JPY"
+    }
+  },
```

---

## 適用方法

### 方法1: 手動で JSON を編集

上記の diff に従って、`app/config/sites/overrides.local.json` の `MONCLER_OFFICIAL` ブロックを編集してください。

### 方法2: パッチファイルを適用（推奨）

完全な JSON ブロックを作成して、置き換える方式です（後続で作成）。

---

## 注意事項

1. **既存の設定は保持**: `navigation.*`, `locale.*`, `behavior.*`, `discovery_settings.*`, `selectors.plp.*`, `selectors.ui.*`, `url_rules.*` は変更しません。

2. **サイズ選択関連**: `price_requires_size`, `auto_select_size`, `size_container_selectors`, `size_selected_check_selectors` は既存の設定を保持します。

3. **後方互換性**: 旧キー（`title_selectors`, `price_selectors`）を削除する前に、新キー（`title`, `price`）で動作確認してください。

4. **`selectors_patch` と `overrides_patch`**: これらは旧スキーマですが、削除する前に実際の動作を確認してください。特に `overrides_patch.pre_actions` などは有用な可能性があります。

---

次: 完全な JSON ブロックを作成します。

