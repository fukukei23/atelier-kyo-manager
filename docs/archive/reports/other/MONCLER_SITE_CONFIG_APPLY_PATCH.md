# Moncler 用 site_config 適用パッチ

**作成日時**: 2025-01-28  
**適用先**: `app/config/sites/overrides.local.json`  
**対象**: `MONCLER_OFFICIAL` ブロック内の `selectors.pdp.*` セクション

---

## 適用手順

このパッチは、`overrides.local.json` の `MONCLER_OFFICIAL` ブロック内の `selectors.pdp.*` セクションを Stage 5 新スキーマに完全対応させるためのものです。

### 変更内容

1. **削除**: PLP 用セレクタ（`plp_container_selectors`, `pdp_link_selectors`, `plp_verification_selectors`, `pdp_click_fallback_selectors`）
2. **統合**: `title_selectors` → `title`, `price_selectors` → `price`
3. **追加**: `images`, `colors`, `sizes`, `description`, `breadcrumbs`, `sku`, `availability`, `json_ld`, `meta_fallback`, `raw_html_capture`
4. **追加**: トップレベルの `price_rules`
5. **削除**: `selectors_patch.*`, `overrides_patch.*`, `rationale`, `code_hints`, `risk`

---

## 完全な JSON ブロック（置き換え用）

`app/config/sites/overrides.local.json` の 356行目（`"pdp": {` から 468行目（`}` まで）を以下の内容に置き換えてください：

```json
      "pdp": {
        "title": [
          "[data-testid='pdp-title']",
          "h1[data-test='product-name']",
          "h1[itemprop='name']",
          "h1",
          "[data-testid='product-name']",
          ".c-product-title",
          "h1.product-name",
          "meta[property='og:title']"
        ],
        "price": [
          "[data-testid='pdp-price']",
          "[data-test='pdp-price'] [itemprop='price']",
          "[itemprop='price']",
          "[data-testid*='price' i]",
          ".c-price__value",
          ".price",
          "[class*='price' i]",
          "meta[itemprop='price'][content]"
        ],
        "list_price": [
          "[data-testid='pdp-list-price']",
          "[data-test='pdp-list-price']",
          ".c-price__original",
          "[class*='list-price' i]",
          "[class*='original-price' i]"
        ],
        "currency": [
          "meta[property='product:price:currency']",
          "meta[itemprop='priceCurrency'][content]",
          "[itemprop='priceCurrency']",
          "[data-testid*='currency' i]"
        ],
        "images": {
          "selectors": [
            ".product-gallery img",
            "[data-testid='product-image'] img",
            "[data-testid='pdp-image'] img",
            "img[itemprop='image']",
            ".c-product-image img",
            ".product-images img"
          ],
          "image_attr": "src",
          "base_url": null
        },
        "description": [
          "[data-testid='product-description']",
          "[data-test='product-description']",
          "[itemprop='description']",
          ".c-product-description",
          ".product-description",
          "meta[property='og:description']"
        ],
        "breadcrumbs": [
          ".breadcrumb li",
          "nav[aria-label='Breadcrumb'] a",
          "[data-testid='breadcrumb'] a",
          "ol.breadcrumb li"
        ],
        "colors": [
          "button[data-testid*='color' i]",
          "button[data-test='color-swatch']",
          "[data-color]",
          ".color-selector button",
          "[role='radiogroup'][aria-label*='Color' i] button"
        ],
        "sizes": [
          "button[data-testid*='size' i]:not([disabled])",
          "button[data-test='size-option']:not([disabled])",
          "[role='radiogroup'][aria-label*='Size' i] button:not([disabled])",
          ".size-selector button:not([disabled])",
          "button[aria-pressed='false'][data-testid*='size' i]:not([disabled])"
        ],
        "brand": [
          "[data-testid='product-brand']",
          "[itemprop='brand']",
          "meta[property='product:brand']",
          "meta[property='og:site_name']"
        ],
        "sku": [
          "[data-testid='product-sku']",
          "[data-test='product-sku']",
          "[itemprop='sku']",
          ".product-code",
          ".sku"
        ],
        "availability": {
          "selectors": [
            "[data-testid='stock-status']",
            "[itemprop='availability']",
            "meta[property='product:availability']",
            "[data-test='availability']"
          ],
          "patterns": [
            "out of stock",
            "in stock",
            "pre-order",
            "在庫なし",
            "在庫あり"
          ]
        },
        "price_requires_size": true,
        "auto_select_size": true,
        "size_button": [
          "button[aria-pressed='false'][data-testid*='size' i]:not([disabled])",
          "button[role='radio']:not([aria-disabled='true']):not([disabled])",
          "[role='option']:not([aria-disabled='true']) button:not([disabled])",
          "li:not([aria-disabled='true']) button:not([disabled])"
        ],
        "size_select_policy": {
          "mode": "first_instock",
          "prefer_labels": ["M", "L"],
          "price_wait_ms": 4000
        },
        "size_container_selectors": [
          "[data-testid*='size' i]",
          "[class*='size' i]",
          "fieldset[aria-label*='Size' i]"
        ],
        "size_selected_check_selectors": [
          "button[aria-pressed='true'][data-testid*='size' i]",
          "[role='option'][aria-selected='true']",
          "[aria-live] .selected-size"
        ],
        "visible_price_selectors": [
          "[itemprop='price']",
          "[data-testid*='price' i]",
          ".c-price__value",
          ".price",
          "[class*='price' i]"
        ],
        "json_ld": {
          "enabled": true,
          "paths": {
            "price": ["offers.price", "offers[0].price"],
            "currency": ["offers.priceCurrency", "offers[0].priceCurrency"],
            "title": ["name"],
            "description": ["description"],
            "availability": ["offers.availability"]
          }
        },
        "meta_fallback": {
          "enabled": true,
          "selectors": {
            "price": [
              "meta[property='og:price:amount']",
              "meta[property='product:price:amount']"
            ],
            "currency": [
              "meta[property='og:price:currency']",
              "meta[property='product:price:currency']"
            ],
            "title": ["meta[property='og:title']"],
            "description": ["meta[property='og:description']"]
          }
        },
        "raw_html_capture": {
          "enabled": true,
          "filename": "pdp_raw.html"
        },
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
      }
```

---

## 追加: `price_rules`（トップレベル）

`MONCLER_OFFICIAL` ブロック内の `selectors` の後（`url_rules` の前）に以下を追加してください：

```json
      "url_rules": {
        // ... 既存の設定 ...
      },
      "price_rules": {
        "strip_chars": ["€", ",", " ", "EUR", "GBP", "USD"],
        "thousands_separator": ",",
        "decimal_separator": ".",
        "currency_fallback": "EUR",
        "price_pattern": "[\\d.,]+",
        "currency_symbols": {
          "€": "EUR",
          "£": "GBP",
          "$": "USD",
          "¥": "JPY"
        }
      }
```

**注意**: `price_rules` は `selectors_patch` の前、`url_rules` の後に追加してください。

---

## 削除対象（手動で削除）

以下のセクションを削除してください：

- `"selectors_patch": { ... }` (528行目〜584行目)
- `"overrides_patch": { ... }` (585行目〜643行目)
- `"rationale": "..."` (644行目)
- `"code_hints": [...]` (645行目〜662行目)
- `"risk": "medium"` (663行目)

---

## 適用後の確認

1. JSON 構文チェック: `python -m json.tool app/config/sites/overrides.local.json`
2. テスト実行: `pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -v`
3. E2E 確認: `python run_orchestrator.py --site MONCLER_OFFICIAL --query "down jacket" --headless`

---

次: 実際のファイルに適用する diff を作成します。

