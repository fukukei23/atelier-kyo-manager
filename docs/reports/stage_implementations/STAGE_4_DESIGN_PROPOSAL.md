# Stage 4: 汎用PLP Driver化 - 設計方針

**作成日時**: 2025-11-28  
**目的**: 汎用PLP Driver（NavigationDriver）の設計とsite_config標準スキーマの提案

---

## 1. 想定インターフェース案

### 1.1 NavigationDriver の責務

NavigationDriverは以下の3段階を、**site_configに基づき汎用的に処理**するクラスとして設計します。

```python
class NavigationDriver:
    """
    任意サイトの PLP に入り、商品カードを十分にロードし、
    PDP 候補 URL のリストを返す汎用ドライバ。
    site_config に強く依存し、コード側はロジックだけ持つ。
    """

    async def build_entry_url(
        self, 
        *, 
        site_config: dict, 
        query: str,
        tactic: Optional[str] = None
    ) -> str:
        """
        エントリURLを生成する
        
        Args:
            site_config: サイト設定
            query: 検索クエリ
            tactic: ナビゲーション戦略（"official_catalog", "search", "fallback_url"等）
            
        Returns:
            str: エントリURL
            
        処理内容:
            - site_config["discovery_settings"]["navigation_tactics"] から戦略を選択
            - site_config["discovery_settings"]["url_templates"][tactic] からURLテンプレートを取得
            - {query}, {brand} などのプレースホルダを置換
            - site_config["locale"] の設定に基づいてロケールを正規化
        """
        ...

    async def ensure_plp_ready(
        self, 
        *, 
        page: Page, 
        site_config: dict, 
        run_context: RunContext
    ) -> bool:
        """
        PLPを安定化させる（スクロール、タイル検出、オーバーレイ処理）
        
        Args:
            page: Playwright Page
            site_config: サイト設定
            run_context: 実行コンテキスト
            
        Returns:
            bool: 安定化成功
            
        処理内容:
            - Cookieバナー、ジオモーダルなどのオーバーレイを処理
            - site_config["selectors"]["plp"]["tile_selectors"] でタイルを検出
            - site_config["discovery_settings"]["plp_scroll_rounds"] に基づいてスクロール
            - 最小タイル数（site_config["navigation"]["plp"]["min_pdp_links"]）を確認
        """
        ...

    async def collect_pdp_links(
        self, 
        *, 
        page: Page, 
        site_config: dict, 
        run_context: RunContext
    ) -> list[str]:
        """
        PDPリンクを収集する
        
        Args:
            page: Playwright Page
            site_config: サイト設定
            run_context: 実行コンテキスト
            
        Returns:
            list[str]: PDPリンクのリスト
            
        処理内容:
            - Phase 1a: Global <a href> sweep + Regex Filter
            - Phase 1b: site_config["selectors"]["plp"]["pdp_link_selectors"] を使用
            - Phase 2: Deep Extraction Fallback（JSON-LD, onclick等）
            - Phase 3: ノイズフィルタリング（site_config["selectors"]["pdp"]["blocklist_href_substrings"]）
        """
        ...
```

### 1.2 既存メソッドとの対応関係

| 新インターフェース | 既存メソッド | 変更内容 |
|------------------|------------|---------|
| `build_entry_url()` | `_build_entry_url()` (BrowserUseAgent内) | site_configからURLテンプレートを取得 |
| `ensure_plp_ready()` | `ensure_plp_materialized()` | 既存実装を維持、site_config依存を強化 |
| `collect_pdp_links()` | `collect_pdp_links()` | 既存実装を維持、site_config依存を強化 |

---

## 2. site_config標準スキーマ案

### 2.1 必須キー一覧

```json
{
  "site_name": {
    "home_url": "https://example.com",
    "allowed_domain": "example.com",
    
    "locale": {
      "prefer": "en-int",
      "normalize_rules": [
        {"from": "/en-jp/", "to": "/en-int/"},
        {"from": "/en-gb/", "to": "/en-int/"}
      ],
      "normalize_double_locale": true,
      "force_query_params": {
        "forceLocale": "en-int",
        "shipToCountry": "GB"
      }
    },
    
    "navigation": {
      "trap_url_patterns": ["/legal", "/privacy", "/cookies"],
      "legal_url_patterns": ["/legal", "/privacy"],
      "trap_domains": ["corporate.example.com"],
      "locale_gate_detection": {
        "enabled": true,
        "target_locale": "en-int",
        "gate_paths": ["/en-int", "/en-gb"]
      },
      
      "overlays": {
        "cookie_banner_selectors": [
          "button#onetrust-accept-btn-handler",
          "button:has-text('ACCEPT ALL')"
        ],
        "geo_modal_selectors": [
          "button:has-text('United Kingdom')",
          "button[data-country-code='GB']"
        ],
        "geo_modal_preferred_locale": "en-gb",
        "generic_close_buttons": [
          "button[aria-label='Close']",
          ".modal button.close"
        ]
      },
      
      "header_search": {
        "enabled": true,
        "search_open_selector": ["button[aria-label='Search']"],
        "search_input_selector": ["input[type='search']"],
        "submit_selector": ["form[role='search'] button[type='submit']"],
        "clear_before_type": true,
        "url_template": "/search?q={query}&locale={locale}",
        "base_url": "home_url"
      },
      
      "plp_recovery": {
        "enabled": true,
        "fallback_url": "seed_plp_url",
        "normalize_locale": true
      },
      
      "plp": {
        "supports_header_search": true,
        "supports_card_click_fallback": true,
        "supports_locale_normalization": true,
        "min_pdp_links": 12,
        "max_scroll_iterations": 12,
        "scroll_pause_ms": 350,
        "plp_timeout_ms": 35000
      },
      
      "fallback": {
        "click_first_card": {
          "enabled": true,
          "card_selectors": [
            "div[data-test='product-card'] a",
            ".product-card a"
          ],
          "blocklist_href_substrings": ["/cart", "/wishlist"]
        }
      }
    },
    
    "discovery_settings": {
      "navigation_tactics": [
        "fallback_url",
        "official_catalog",
        "search"
      ],
      "url_templates": {
        "fallback_url": "/en-int/women/outerwear/all-down-jackets",
        "official_catalog": "/en-int/women/outerwear/all-down-jackets",
        "search": "/en-int/search?q={query}&forceLocale=en-int&shipToCountry=GB"
      },
      "force_query_params": {
        "forceLocale": "en-int",
        "shipToCountry": "GB"
      },
      "plp_scroll_rounds": 8,
      "plp_scroll_wait_ms": 400,
      "overall_plp_budget_ms": 60000,
      "enable_locale_escape": true,
      "enable_moncler_patch": false
    },
    
    "selectors": {
      "plp": {
        "container_selectors": [
          "div[data-test='product-grid']",
          ".product-grid"
        ],
        "card_selectors": [
          "div[data-test='product-card']",
          ".product-card"
        ],
        "tile_selectors": [
          "a[href*='/products/']",
          "div:has(a[href*='/products/'])"
        ],
        "pdp_link_selectors": [
          "a[href*='/products/']",
          "a[href*='/product/']",
          "a[href*='/p/']"
        ]
      },
      "pdp": {
        "plp_container_selectors": [
          "[data-testid='product-grid']",
          ".product-grid"
        ],
        "pdp_link_selectors": [
          "a[href*='/products/']",
          "a[href*='/product/']"
        ],
        "blocklist_href_substrings": [
          "/cart",
          "/wishlist",
          "/legal"
        ]
      },
      "ui": {
        "cookie_accept": [
          "#onetrust-accept-btn-handler",
          "button:has-text('ACCEPT ALL')"
        ],
        "continue_shopping": [
          "a:has-text('CONTINUE SHOPPING')"
        ],
        "search_open": [
          "button[aria-label='Search']"
        ],
        "search_input": [
          "input[type='search']"
        ],
        "search_submit": [
          "form[role='search'] button[type='submit']"
        ]
      }
    }
  }
}
```

### 2.2 各キーの役割とMoncler例

| キー | 役割 | Moncler例 | 他ブランドでの想定 |
|------|------|-----------|------------------|
| `locale.prefer` | 優先ロケール | `"en-int"` | `"en-gb"`, `"en-us"`, `"ja-jp"` |
| `locale.normalize_rules` | ロケール正規化ルール | `[{"from": "/en-jp/", "to": "/en-int/"}]` | サイトごとのロケールマッピング |
| `locale.force_query_params` | 強制クエリパラメータ | `{"forceLocale": "en-int", "shipToCountry": "GB"}` | サイトごとの必須パラメータ |
| `navigation.trap_url_patterns` | Trap URLパターン | `["/legal", "/privacy"]` | サイトごとのtrapパターン |
| `navigation.overlays.cookie_banner_selectors` | Cookieバナーセレクタ | `["#onetrust-accept-btn-handler"]` | サイトごとのCookieバナー |
| `navigation.overlays.geo_modal_selectors` | ジオモーダルセレクタ | `["button:has-text('United Kingdom')"]` | サイトごとのジオモーダル |
| `navigation.header_search.url_template` | 検索URLテンプレート | `"/en-int/search?q={query}&forceLocale=en-int"` | サイトごとの検索URL形式 |
| `discovery_settings.navigation_tactics` | ナビゲーション戦略 | `["fallback_url", "official_catalog", "search"]` | サイトごとの戦略順序 |
| `discovery_settings.url_templates` | URLテンプレート | `{"official_catalog": "/en-int/women/..."}` | サイトごとのURL形式 |
| `selectors.plp.tile_selectors` | PLPタイルセレクタ | `["a[href*='/products/']"]` | サイトごとのタイル構造 |
| `selectors.plp.pdp_link_selectors` | PDPリンクセレクタ | `["a[href*='/products/']"]` | サイトごとのリンク構造 |

### 2.3 他サイト（SSENSE, MATCHESFASHION等）との整合性

**現状の問題**:
- SSENSE, MATCHESFASHION等の設定が `overrides.local.json` に存在するが、スキーマが統一されていない
- Moncler固有のキー（`force_locale`, `ship_to`等）が他サイトには存在しない

**解決策**:
1. **標準スキーマの確立**: 上記スキーマを全サイトに適用
2. **後方互換性の維持**: 既存のMoncler設定を段階的に移行
3. **デフォルト値の提供**: サイト固有設定がない場合はデフォルト値を使用

---

## 3. 実装方針

### 3.1 Phase 1: NavigationDriverの汎用化

1. **URL正規化の汎用化**
   - `_normalize_to_en_int_url()` → `_normalize_url(url, site_config)` に変更
   - `site_config["locale"]["normalize_rules"]` を使用

2. **Trap判定の汎用化**
   - `_looks_like_trap_or_legal()` を `site_config["navigation"]["trap_url_patterns"]` に完全依存

3. **ヘッダ検索の汎用化**
   - `header_search_fallback()` のMoncler URLハードコードを削除
   - `site_config["navigation"]["header_search"]["url_template"]` を使用

### 3.2 Phase 2: BrowserUseAgentの責務整理

1. **NavigationDriverへの完全委譲**
   - `_run_plp_flow()` から重複ロジックを削除
   - NavigationDriverの結果をそのまま使用

2. **重複メソッドの削除**
   - `_normalize_to_en_int_url()` を削除（NavigationDriverに統一）
   - `_looks_like_trap_or_legal()` を削除（NavigationDriverに統一）
   - `_force_plp_recover()` を削除（NavigationDriverに統一）

### 3.3 Phase 3: site_configの標準化

1. **MONCLER_OFFICIAL設定の移行**
   - 既存のMoncler設定を標準スキーマに移行
   - 後方互換性を維持

2. **他サイト設定の確認**
   - SSENSE, MATCHESFASHION等の設定を標準スキーマに合わせる
   - 不足しているキーを追加

---

## 4. 受け入れ条件（Acceptance Criteria）

### 4.1 構造面

- [ ] NavigationDriverが `build_entry_url()`, `ensure_plp_ready()`, `collect_pdp_links()` の3段階を汎用的に処理
- [ ] BrowserUseAgentからMoncler固有のif文やハードコードがほぼ消えている
- [ ] 重複メソッド（`_normalize_to_en_int_url()`等）が削除されている

### 4.2 設定面

- [ ] MONCLER_OFFICIALのsite_configが標準スキーマに沿っている
- [ ] 他サイト（SSENSE等）も標準スキーマに沿っている
- [ ] 新しいPLP Driverの標準スキーマだけで最低限リンク抽出が試せる

### 4.3 観測性

- [ ] PLPナビゲーションの各段階でRunContextにログやスクリーンショットを残している
- [ ] 失敗時にSelectorDiscoveryAgentやSelfHealingAgentに渡せる情報が整理されている

---

**次のタスク**: NavigationDriverからMoncler固有ロジックの排除

