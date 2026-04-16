# MONCLER_OFFICIAL.json に追加すべき構造

## 概要

テスト結果から、以下の構造が site_config に存在しないため、フォールバック（既存のハードコード値）が使用されています。これらの構造を追加することで、site_config から直接取得できるようになります。

## 追加すべき構造

### 1. selectors.plp

```json
{
  "MONCLER_OFFICIAL": {
    "selectors": {
      "plp": {
        "pdp_link_selectors": [
          "a[data-testid='product-card'] a[href*='/products/']",
          "a[data-test='product-card'] a[href*='/products/']",
          "ul[role='list'] li a[href*='/products/']"
        ],
        "card_selectors": [
          "[data-qa='product-tile']",
          ".c-product-tile",
          ".product-card"
        ],
        "container_selectors": [
          "[data-qa='product-grid']",
          ".product-grid",
          ".search-grid"
        ]
      }
    }
  }
}
```

### 2. navigation.header_search

```json
{
  "MONCLER_OFFICIAL": {
    "navigation": {
      "header_search": {
        "search_open_selector": [
          "button[aria-label='Search']",
          "[aria-label*='Search' i]"
        ],
        "search_input_selector": [
          "form[role='search'] input",
          "input[type='search']",
          "input[name='q']"
        ],
        "submit_selector": [
          "form[role='search'] button[type='submit']"
        ],
        "clear_before_type": true
      }
    }
  }
}
```

### 3. navigation.overlays

```json
{
  "MONCLER_OFFICIAL": {
    "navigation": {
      "overlays": {
        "cookie_banner_selectors": [
          "#onetrust-accept-btn-handler",
          "button:has-text('ACCEPT ALL')",
          "button[aria-label*='Accept' i]"
        ],
        "geo_modal_selectors": [
          "text=STAY HERE",
          "text=REMAIN HERE",
          "text=CONTINUE SHOPPING"
        ],
        "generic_close_buttons": [
          ".overlay",
          ".backdrop",
          ".modal-backdrop",
          "#onetrust-banner-sdk",
          ".cookie-banner",
          "[aria-modal=\"true\"]"
        ]
      }
    }
  }
}
```

### 4. navigation.fallback.click_first_card

```json
{
  "MONCLER_OFFICIAL": {
    "navigation": {
      "fallback": {
        "click_first_card": {
          "enabled": true,
          "card_selectors": [
            "[data-qa='product-tile']",
            ".c-product-tile",
            ".product-card"
          ],
          "blocklist_href_substrings": [
            "/cart",
            "/wishlist",
            "javascript:void"
          ]
        }
      }
    }
  }
}
```

### 5. navigation.trap_url_patterns と legal_url_patterns

```json
{
  "MONCLER_OFFICIAL": {
    "navigation": {
      "trap_url_patterns": [
        "monclergroup.com",
        "/en-jp/",
        "/brands/moncler"
      ],
      "legal_url_patterns": [
        "/cookie-policy",
        "/cookies",
        "/privacy",
        "/legal",
        "/help",
        "/customer-service",
        "/client-service/"
      ]
    }
  }
}
```

## 実装方法

### 方法1: overrides.local.json に追加

`app/config/sites/overrides.local.json` の `MONCLER_OFFICIAL` セクションに上記の構造を追加します。

### 方法2: base.json に追加

`app/config/sites/base.json` の `MONCLER_OFFICIAL` セクションに上記の構造を追加します（他のサイトでも使用する場合）。

## 確認方法

追加後、以下のコマンドで確認できます：

```bash
python test_site_config_connection.py
```

警告が消え、すべてのセレクタが site_config から取得できるようになります。

