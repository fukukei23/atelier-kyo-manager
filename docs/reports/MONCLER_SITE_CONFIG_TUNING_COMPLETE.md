# Moncler 用 site_config チューニング - 完全版

**作成日時**: 2025-01-28  
**目的**: MONCLER_OFFICIAL 向けの PDP 抽出が Stage 4 / Stage 5 の新スキーマに完全対応するよう site_config を調整

---

## Step 1: 現状 Moncler の site_config 分析結果

### ✅ PLP 関連 - Stage 4 対応済み

- `selectors.plp.*` - ✅ 新スキーマ化済み
- `navigation.plp.*` - ✅ 新スキーマ化済み  
- `navigation.overlays.*` - ✅ 新スキーマ化済み
- `navigation.trap_url_patterns` - ✅ 新スキーマ化済み

### ⚠️ PDP 関連 - Stage 5 部分対応（要改善）

**現状の問題点**:

1. **旧スキーマキーが残存**: `selectors.pdp.plp_container_selectors`, `selectors.pdp.pdp_link_selectors` など（PLP 用なので削除対象）

2. **Stage 5 新スキーマで必要なキーが不足**:
   - `selectors.pdp.images` - 未定義
   - `selectors.pdp.colors` - 未定義
   - `selectors.pdp.sizes` - 抽出用が未定義
   - `selectors.pdp.description` - 未定義
   - `selectors.pdp.breadcrumbs` - 未定義
   - `selectors.pdp.sku` - 未定義
   - `selectors.pdp.availability` - 未定義
   - `selectors.pdp.json_ld.*` - 未定義
   - `price_rules` - 未定義

3. **旧キー名が残存**: `title_selectors`, `price_selectors` → `title`, `price` に統合すべき

---

## Step 2: Moncler 実 HTML / コードベースからのギャップ分析

### 既存の Moncler セレクタ情報

#### `instance/sites/MONCLER_OFFICIAL/learned_selectors.json` から:

```json
{
  "price_selectors": [
    "[itemprop='price']",
    "meta[itemprop='price'][content]",
    "[data-testid*='price' i]",
    "[data-qa*='price' i]",
    "[class*='price' i]"
  ],
  "title_selectors": [
    "h1",
    "h1[itemprop='name']",
    "[data-testid='product-name']",
    "[data-testid*='product' i][data-testid*='name' i]"
  ]
}
```

#### `app/extractors/moncler_extractor.py` から:

- **JSON-LD 抽出**: `script[type='application/ld+json']` から `offers.price`, `offers.priceCurrency` を取得
- **NEXT_DATA 抽出**: `script#__NEXT_DATA__` から商品情報を取得
- **Meta タグ抽出**: `meta[property='og:price:amount']`, `meta[property='og:price:currency']`

#### `app/config/sites_pinned/20_moncler.pinned.json` から:

```json
{
  "pdp_title": ["h1[data-test='product-name']", "h1.product-name"],
  "pdp_price": ["[data-test='pdp-price'] [itemprop='price']", "meta[itemprop='price']"],
  "pdp_json_ld": "script[type='application/ld+json']"
}
```

#### `overrides.local.json` の現状設定から:

```jsonc
"selectors": {
  "pdp": {
    "title_selectors": [
      "[data-testid='pdp-title']",
      "h1",
      "[data-testid='product-name']",
      ".c-product-title"
    ],
    "price_selectors": [
      "[data-testid='pdp-price']",
      "[itemprop='price']",
      "[data-testid*='price' i]",
      ".c-price__value",
      ".price",
      "[class*='price' i]"
    ]
  }
}
```

### 理想的な selectors.pdp.*（Moncler 実サイト構造に基づく提案）

以下は、既存のコードと設定から推測される Moncler の DOM 構造に基づく提案です:

---

## Step 3: Moncler 用 site_config の具体的チューニング案（JSON スニペット）

### 3.1. Stage 5 新スキーマに準拠した完全な `selectors.pdp.*` ブロック

```jsonc
"selectors": {
  "pdp": {
    // ===== 基本情報 =====
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
    
    // ===== メディア =====
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
      "base_url": null  // page.url を使用
    },
    
    // ===== 説明・詳細 =====
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
    
    // ===== 属性 =====
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
    
    // ===== サイズ選択関連（価格表示のため） =====
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
    
    "visible_price_selectors": [
      "[itemprop='price']",
      "[data-testid*='price' i]",
      ".c-price__value",
      ".price",
      "[class*='price' i]"
    ],
    
    // ===== JSON-LD フォールバック =====
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
    
    // ===== Meta タグフォールバック =====
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
    
    // ===== HTML 保存設定 =====
    "raw_html_capture": {
      "enabled": true,
      "filename": "pdp_raw.html"
    }
  }
}
```

### 3.2. 価格正規化ルール（`price_rules`）

```jsonc
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

または、ネスト形式（`selectors.pdp.price.normalize_rules`）:

```jsonc
"selectors": {
  "pdp": {
    "price": {
      "selectors": [...],
      "normalize_rules": {
        "strip_chars": ["€", ",", " "],
        "thousands_separator": ",",
        "decimal_separator": ".",
        "currency_fallback": "EUR"
      }
    }
  }
}
```

---

## Step 4: 差分パッチ（追加／上書き／削除）

### 4.1. 削除すべきキー

以下のキーは削除または別の場所に移動してください:

```jsonc
// ❌ 削除: PLP 用なので PDP から削除
"selectors": {
  "pdp": {
    "plp_container_selectors": [...],      // ❌ 削除（PLP 用）
    "pdp_link_selectors": [...],           // ❌ 削除（PLP 用）
    "plp_verification_selectors": [...],   // ❌ 削除（PLP 用）
    "pdp_click_fallback_selectors": [...]  // ❌ 削除（PLP 用）
  }
}

// ❌ 削除: 旧スキーマ、新スキーマに統合済み
"selectors_patch": {...}  // ❌ 全体削除
"overrides_patch": {...}  // ❌ 全体削除（一部は残す可能性あり）

// ⚠️ 統合: 旧キー名を新キー名に統合
"selectors": {
  "pdp": {
    "title_selectors": [...]  // ⚠️ 削除して "title": [...] に統合
    "price_selectors": [...]  // ⚠️ 削除して "price": [...] に統合
  }
}
```

### 4.2. 追加すべきキー

以下のキーを追加してください:

```jsonc
"MONCLER_OFFICIAL": {
  // ... 既存の設定 ...
  
  "selectors": {
    "pdp": {
      // ===== 追加: 上記の完全な selectors.pdp.* ブロック =====
      "images": {...},
      "colors": [...],
      "sizes": [...],
      "description": [...],
      "breadcrumbs": [...],
      "sku": [...],
      "availability": {...},
      "json_ld": {...},
      "meta_fallback": {...},
      "raw_html_capture": {...}
    }
  },
  
  // ===== 追加: 価格正規化ルール =====
  "price_rules": {...}
}
```

### 4.3. 上書きすべきキー

以下のキーは現状の値を新スキーマ形式に上書きしてください:

```jsonc
// ⚠️ 上書き: 旧キー名を新キー名に変更
"selectors": {
  "pdp": {
    // 旧: "title_selectors": [...]
    // 新: "title": [...]
    
    // 旧: "price_selectors": [...]
    // 新: "price": [...]
  }
}
```

---

## Step 5: 動作確認用のテスト・手順提案

### 5.1. テストファイル追加案

`tests/test_product_extractor.py` に以下を追加:

```python
@pytest.mark.asyncio
async def test_product_extractor_moncler_pdp_sample(
    mock_page: AsyncMock,
    run_context: MagicMock,
):
    """Moncler 用 PDP fixture を使ったサイト固有テスト"""
    
    # Moncler 用 site_config を準備
    site_config = {
        "MONCLER_OFFICIAL": {
            "selectors": {
                "pdp": {
                    "title": ["h1[data-testid='pdp-title']", "h1"],
                    "price": ["[data-testid='pdp-price']", "[itemprop='price']"],
                    "images": {
                        "selectors": [".product-gallery img"],
                        "image_attr": "src"
                    },
                    "json_ld": {
                        "enabled": True,
                        "paths": {
                            "price": ["offers.price"],
                            "currency": ["offers.priceCurrency"]
                        }
                    }
                }
            },
            "price_rules": {
                "strip_chars": ["€", ",", " "],
                "currency_fallback": "EUR"
            }
        }
    }
    
    # HTML fixture を読み込み（実際の Moncler PDP HTML を使用）
    html_content = load_fixture("moncler_pdp_sample.html")
    mock_page.set_content(html_content)
    
    # ProductExtractor を初期化
    extractor = ProductExtractor(
        site_config=site_config.get("MONCLER_OFFICIAL", {}),
        run_context=run_context
    )
    
    # 抽出実行
    result = await extractor.extract(mock_page)
    
    # アサーション
    assert result is not None
    assert result.title is not None
    assert result.price is not None
    assert result.currency is not None
    assert len(result.images) > 0
    assert result.metadata.get("has_title") is True
    assert result.metadata.get("has_price") is True
    assert result.metadata.get("image_count") > 0
```

### 5.2. 実行手順

```bash
# 単体テスト実行
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -v

# または、Moncler 関連のテストをすべて実行
pytest tests/test_product_extractor.py -k moncler -v
```

### 5.3. E2E 確認手順

```bash
# run_orchestrator.py を使った E2E 確認
python run_orchestrator.py \
    --site MONCLER_OFFICIAL \
    --query "down jacket" \
    --headless

# 結果確認
# - instance/runs/<RUN_ID>/pdp_extracted_data.json を確認
# - instance/runs/<RUN_ID>/pdp_raw.html を確認（raw_html_capture が有効な場合）
```

---

## まとめ

### ✅ 実施すべき変更

1. **削除**: `selectors.pdp.plp_container_selectors`, `selectors.pdp.pdp_link_selectors` など（PLP 用）
2. **削除**: `selectors_patch.*`, `overrides_patch.*`（旧スキーマ）
3. **統合**: `title_selectors` → `title`, `price_selectors` → `price`
4. **追加**: `images`, `colors`, `sizes`, `description`, `breadcrumbs`, `sku`, `availability`
5. **追加**: `json_ld.*`, `meta_fallback.*`, `raw_html_capture.*`
6. **追加**: `price_rules` または `selectors.pdp.price.normalize_rules`

### ⚠️ 注意事項

- 既存の PLP 設定（`selectors.plp.*`）は変更不要（Stage 4 で対応済み）
- サイズ選択関連の設定（`price_requires_size`, `auto_select_size` など）は残す
- 後方互換性のため、旧キーを削除する前に新キーで動作確認すること

---

**次**: 実際の JSON diff を作成して、`overrides.local.json` への適用パッチを生成します。

