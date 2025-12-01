# Stage 5 Task B: PDP 抽出用 site_config スキーマ案

## 1. selectors.pdp.* のキー設計案

```jsonc
{
  "selectors": {
    "pdp": {
      // 基本情報
      "title": [
        "css=h1.product-title",
        "css=h1[itemprop='name']",
        "css=meta[property='og:title']"
      ],
      "price": [
        "css=.price-current",
        "css=[itemprop='price']",
        "css=meta[property='product:price:amount']"
      ],
      "list_price": [
        "css=.price-original",
        "css=.price-before",
        "css=[data-testid='list-price']"
      ],
      "currency": [
        "css=.price-currency",
        "css=meta[property='product:price:currency']",
        "css=meta[itemprop='priceCurrency']"
      ],
      
      // メディア
      "images": [
        "css=.product-image img",
        "css=.product-gallery img",
        "css=img[itemprop='image']"
      ],
      
      // 説明・詳細
      "description": [
        "css=.product-description",
        "css=[itemprop='description']",
        "css=meta[property='og:description']"
      ],
      "breadcrumbs": [
        "css=.breadcrumb li",
        "css=nav[aria-label='Breadcrumb'] a"
      ],
      
      // 属性
      "colors": [
        "css=.color-swatch",
        "css=button[data-color]",
        "css=[data-testid*='color']"
      ],
      "sizes": [
        "css=.size-selector option",
        "css=button[data-size]",
        "css=[role='radiogroup'] [role='radio']"
      ],
      "brand": [
        "css=.product-brand",
        "css=[itemprop='brand']",
        "css=meta[property='og:site_name']"
      ],
      "sku": [
        "css=[itemprop='sku']",
        "css=.sku",
        "css=[data-testid='sku']"
      ],
      "availability": [
        "css=.stock-status",
        "css=[itemprop='availability']",
        "css=meta[property='product:availability']"
      ],
      
      // サイズ選択関連（価格表示のため）
      "size_button": [
        "css=button[aria-disabled='false'][data-size]",
        "css=button:not([disabled])[data-size]"
      ],
      "size_select_policy": {
        "mode": "first_instock",  // "off" | "first_instock" | "by_label"
        "prefer_labels": ["M", "L"],
        "price_wait_ms": 4000
      },
      "visible_price_selectors": [
        "css=.price-current",
        "css=[itemprop='price']"
      ],
      
      // 画像 URL 抽出の設定
      "image_attr": "src",  // "src" | "data-src" | "data-lazy-src"
      "image_base_url": null,  // 相対 URL を絶対 URL に変換する際のベース URL（null の場合は page.url を使用）
      
      // HTML 保存設定
      "raw_html_capture": {
        "enabled": true,
        "filename": "pdp_raw.html"  // カスタムファイル名（オプション）
      }
    }
  }
}
```

## 2. 価格正規化ルール（price_rules / price.normalize_rules）の案

```jsonc
{
  "price_rules": {
    // 削除する文字
    "strip_chars": ["¥", "$", "€", ",", "円", " ", "USD", "JPY", "EUR"],
    
    // 区切り文字
    "thousands_separator": ",",  // 千の位区切り（例: "1,234"）
    "decimal_separator": ".",    // 小数点（例: "12.34"）
    
    // 通貨フォールバック（価格が見つかったが通貨が見つからない場合）
    "currency_fallback": "JPY",
    
    // 価格抽出の正規表現パターン（オプション、デフォルトは "[\d.,]+"）
    "price_pattern": "[\\d.,]+",
    
    // 通貨記号から通貨コードへのマッピング（オプション）
    "currency_symbols": {
      "¥": "JPY",
      "$": "USD",
      "€": "EUR",
      "£": "GBP"
    }
  }
}
```

または、`selectors.pdp.price` の下にネストする形式：

```jsonc
{
  "selectors": {
    "pdp": {
      "price": {
        "selectors": [
          "css=.price-current",
          "css=[itemprop='price']"
        ],
        "normalize_rules": {
          "strip_chars": ["¥", ",", " "],
          "thousands_separator": ",",
          "decimal_separator": ".",
          "currency_fallback": "JPY"
        }
      }
    }
  }
}
```

## 3. メタデータ用のスキーマ案

```jsonc
{
  "selectors": {
    "pdp": {
      // メタデータ収集の設定
      "metadata": {
        // 収集するメタデータのキー（オプション、デフォルトはすべて収集）
        "include_keys": [
          "extraction_timestamp",
          "url",
          "has_title",
          "has_price",
          "has_currency",
          "image_count",
          "size_count",
          "color_count",
          "has_description",
          "has_brand",
          "has_sku"
        ],
        // カスタムメタデータの計算ロジック（オプション）
        "custom": {
          "discount_available": "list_price != null && price != null && list_price > price",
          "has_multiple_images": "image_count > 1"
        }
      }
    }
  }
}
```

## 4. JSON-LD / Meta タグフォールバックの設定

```jsonc
{
  "selectors": {
    "pdp": {
      // JSON-LD フォールバック設定
      "json_ld": {
        "enabled": true,
        "paths": {
          "price": ["offers.price", "offers[0].price"],
          "currency": ["offers.priceCurrency", "offers[0].priceCurrency"],
          "title": ["name"],
          "description": ["description"]
        }
      },
      
      // Meta タグフォールバック設定
      "meta_fallback": {
        "enabled": true,
        "selectors": [
          "meta[property='og:price:amount']",
          "meta[name='twitter:data1']"
        ]
      }
    }
  }
}
```

## 5. Moncler 用のサンプル site_config 断片

```jsonc
{
  "MONCLER_OFFICIAL": {
    "selectors": {
      "pdp": {
        "title": [
          "css=h1[data-test='product-title']",
          "css=h1.product-title",
          "css=meta[property='og:title']"
        ],
        "price": [
          "css=[data-test='price-current']",
          "css=.price-final",
          "css=meta[property='product:price:amount']"
        ],
        "list_price": [
          "css=[data-test='price-original']",
          "css=.price-before"
        ],
        "currency": [
          "css=meta[property='product:price:currency']",
          "css=meta[itemprop='priceCurrency']"
        ],
        "images": [
          "css=.product-gallery img[data-test='product-image']",
          "css=img[itemprop='image']"
        ],
        "description": [
          "css=[data-test='product-description']",
          "css=.product-details",
          "css=meta[property='og:description']"
        ],
        "colors": [
          "css=button[data-test='color-swatch']",
          "css=.color-selector button"
        ],
        "sizes": [
          "css=button[data-test='size-option']:not([disabled])",
          "css=.size-selector button[aria-disabled='false']"
        ],
        "brand": [
          "css=meta[property='og:site_name']"
        ],
        "sku": [
          "css=[data-test='product-sku']",
          "css=.product-code"
        ],
        "size_button": [
          "css=button[data-test='size-option']:not([disabled])",
          "css=button[aria-disabled='false'][data-size]"
        ],
        "size_select_policy": {
          "mode": "first_instock",
          "prefer_labels": ["M", "L"],
          "price_wait_ms": 4000
        },
        "visible_price_selectors": [
          "css=[data-test='price-current']",
          "css=.price-final"
        ],
        "image_attr": "src",
        "raw_html_capture": {
          "enabled": true,
          "filename": "pdp_raw.html"
        },
        "json_ld": {
          "enabled": true,
          "paths": {
            "price": ["offers.price", "offers[0].price"],
            "currency": ["offers.priceCurrency", "offers[0].priceCurrency"]
          }
        },
        "metadata": {
          "include_keys": [
            "extraction_timestamp",
            "url",
            "has_title",
            "has_price",
            "has_currency",
            "image_count",
            "size_count",
            "color_count"
          ]
        }
      }
    },
    "price_rules": {
      "strip_chars": ["¥", "$", "€", ",", " ", "円"],
      "thousands_separator": ",",
      "decimal_separator": ".",
      "currency_fallback": "JPY",
      "currency_symbols": {
        "¥": "JPY",
        "$": "USD",
        "€": "EUR"
      }
    }
  }
}
```

## 6. 後方互換性の考慮

既存の `selectors.pdp.*` スキーマとの互換性を保つため、以下のフォールバックを実装：

1. **旧スキーマ**: `selectors.pdp.pdp_link_selectors`, `selectors.pdp.plp_container_selectors` など
   - これらは PLP 用なので、PDP 抽出では使用しないが、エラーを出さない

2. **旧 price_rules**: トップレベルの `price_rules` キー
   - 新スキーマの `selectors.pdp.price.normalize_rules` が優先され、なければ `price_rules` を参照

3. **デフォルトセレクタ**: site_config に定義がない場合
   - ProductExtractor 内の `DEFAULT_*_SELECTORS` を使用

## 7. スキーマの拡張性

将来的に追加可能な設定：

- **画像の品質フィルタリング**: `images.min_width`, `images.min_height`
- **価格の検証ルール**: `price.min_value`, `price.max_value`
- **カスタム抽出ロジック**: `custom_extractors` で JavaScript 関数を指定
- **抽出順序の制御**: `extraction_order` でフィールドの抽出順を指定

