# Stage 4: 汎用PLP Driver化 - 現状把握レポート

**作成日時**: 2025-11-28  
**目的**: Moncler固有ロジックの棚卸しと汎用化方針の策定

---

## 1. 関連ファイルの特定と全体構造

### 1.1 主要ファイル一覧

| ファイル | 行数 | 役割 | ステータス |
|---------|------|------|-----------|
| `app/agents/browser_use_agent.py` | 2259行 | PLP/PDPナビゲーションの中核 | ⚠️ Moncler固有ロジック多数 |
| `app/agents/browser/navigation_driver.py` | 1449行 | NavigationDriver（Stage 3Aで抽出済み） | ⚠️ Moncler固有ロジック残存 |
| `app/agents/browser_use_moncler_patch.py` | 561行 | Moncler専用パッチ | 🔴 Moncler専用 |
| `app/agents/page_recovery_agent.py` | - | ページ回復エージェント | ✅ 汎用的 |
| `app/utils/ai_research_orchestrator.py` | - | オーケストレータ | ✅ 汎用的 |
| `app/agents/supplier_scout_agent.py` | - | エージェント選択 | ✅ 汎用的 |
| `app/core/run_context.py` | - | 実行コンテキスト | ✅ 汎用的 |

### 1.2 NavigationDriverの現状

**Stage 3Aで既に抽出済み**:
- `run_plp_flow()`: PLPフロー全体の実行
- `ensure_plp_materialized()`: PLP安定化（スクロール、タイル検出）
- `collect_pdp_links()`: PDPリンク抽出
- `recover_plp()`: PLP回復
- `header_search_fallback()`: ヘッダ検索フォールバック
- `click_first_card_or_link()`: カードクリックフォールバック

**問題点**: NavigationDriver内にもMoncler固有ロジックが残存

---

## 2. Moncler固有ロジックの棚卸し

### 2.1 NavigationDriver / BrowserUseAgent の役割整理

| モジュール | メソッド | 役割 | site_config依存キー | Moncler固有度 |
|-----------|---------|------|-------------------|--------------|
| `NavigationDriver` | `_normalize_to_en_int_url()` | URLを`/en-int/`に正規化 | ❌ ハードコード | 🔴 **高** |
| `NavigationDriver` | `_looks_like_trap_or_legal()` | trap判定（Moncler特化） | `navigation.trap_url_patterns` | 🟡 **中** |
| `NavigationDriver` | `_dismiss_geo_modal()` | ジオモーダル処理 | `navigation.overlays.geo_modal_selectors` | 🟡 **中** |
| `NavigationDriver` | `_force_plp_recover()` | PLP回復（`/en-int/`強制） | `seed_plp_url`, `fallback_url` | 🔴 **高** |
| `NavigationDriver` | `ensure_plp_materialized()` | PLP安定化（`en-gb`検出） | `selectors.plp.tile_selectors` | 🟡 **中** |
| `NavigationDriver` | `header_search_fallback()` | ヘッダ検索（Moncler URLハードコード） | `navigation.header_search.*` | 🔴 **高** |
| `BrowserUseAgent` | `_normalize_to_en_int_url()` | URL正規化（重複実装） | ❌ ハードコード | 🔴 **高** |
| `BrowserUseAgent` | `_looks_like_trap_or_legal()` | trap判定（重複実装） | `navigation.trap_url_patterns` | 🟡 **中** |
| `BrowserUseAgent` | `_force_plp_recover()` | PLP回復（重複実装） | `seed_plp_url` | 🔴 **高** |
| `BrowserUseAgent` | `_plp_header_search_fallback()` | ヘッダ検索（Moncler URLハードコード） | `navigation.header_search.*` | 🔴 **高** |
| `browser_use_moncler_patch.py` | `moncler_plp_recovery()` | Moncler専用回復 | `discovery_settings.force_locale` | 🔴 **高** |

### 2.2 Moncler固有ロジックの詳細分類

#### A. site_configに完全に押し込めるべきもの

1. **URL正規化ルール**
   - **現状**: `_normalize_to_en_int_url()` で `/en-int/` をハードコード
   - **対応**: `site_config.locale.normalize_rules` に移行（既に `overrides.local.json` に定義あり）
   - **例**:
     ```json
     "locale": {
       "prefer": "en-int",
       "normalize_rules": [
         {"from": "/en-jp/", "to": "/en-int/"},
         {"from": "/en-gb/", "to": "/en-int/"}
       ]
     }
     ```

2. **ロケールパラメータ**
   - **現状**: `forceLocale=en-int&shipToCountry=GB` をハードコード
   - **対応**: `discovery_settings.force_query_params` に移行（既に定義あり）
   - **例**:
     ```json
     "discovery_settings": {
       "force_query_params": {
         "forceLocale": "en-int",
         "shipToCountry": "GB"
       }
     }
     ```

3. **セレクタ定義**
   - **現状**: Moncler固有セレクタがコードに散在
   - **対応**: `selectors.plp.*`, `selectors.pdp.*` に移行済み（要確認）

4. **Trap URLパターン**
   - **現状**: `_looks_like_trap_or_legal()` でMoncler固有判定
   - **対応**: `navigation.trap_url_patterns` に移行（既に定義あり）

#### B. 汎用化できるが、少し抽象化（フラグやパラメータ化）が必要なもの

1. **ロケールゲート検出**
   - **現状**: `moncler.com` と `/en-int/` をハードコード
   - **対応**: `navigation.locale_gate_detection` フラグ + `locale.prefer` を使用
   - **例**:
     ```json
     "navigation": {
       "locale_gate_detection": {
         "enabled": true,
         "target_locale": "en-int",
         "gate_paths": ["/en-int", "/en-gb", "/en-us"]
       }
     }
     ```

2. **ジオモーダル処理**
   - **現状**: Moncler固有の「United Kingdom / English」選択ロジック
   - **対応**: `navigation.overlays.geo_modal_selectors` に移行済み（要拡張）
   - **例**:
     ```json
     "navigation": {
       "overlays": {
         "geo_modal_selectors": [
           "button:has-text('United Kingdom')",
           "button[data-country-code='GB']"
         ],
         "geo_modal_preferred_locale": "en-gb"
       }
     }
     ```

3. **PLP回復ロジック**
   - **現状**: `_force_plp_recover()` で `/en-int/` を強制
   - **対応**: `navigation.plp_recovery` 設定を使用
   - **例**:
     ```json
     "navigation": {
       "plp_recovery": {
         "enabled": true,
         "fallback_url": "seed_plp_url",
         "normalize_locale": true
       }
     }
     ```

4. **ヘッダ検索フォールバック**
   - **現状**: Moncler URLをハードコード（`https://www.moncler.com/en-int/search?q=...`）
   - **対応**: `navigation.header_search.url_template` を使用
   - **例**:
     ```json
     "navigation": {
       "header_search": {
         "url_template": "/en-int/search?q={query}&forceLocale=en-int&shipToCountry=GB",
         "base_url": "home_url"
       }
     }
     ```

#### C. どうしてもコード側に残さざるを得ないが、site_configのフラグで有効化/無効化したいもの

1. **Moncler専用パッチ（`browser_use_moncler_patch.py`）**
   - **現状**: Cookie注入、ロケーションモーダル処理など
   - **対応**: `discovery_settings.enable_moncler_patch` フラグで制御
   - **例**:
     ```json
     "discovery_settings": {
       "enable_moncler_patch": true,
       "moncler_patch_config": {
         "cookie_injection": true,
         "locale_modal_handling": true
       }
     }
     ```

2. **ロケール正規化の特殊処理**
   - **現状**: `/en-jp/en-int/` の二重ロケール検出
   - **対応**: `locale.normalize_double_locale` フラグで制御
   - **例**:
     ```json
     "locale": {
       "normalize_double_locale": true,
       "double_locale_patterns": [
         {"from": "/en-jp/en-int/", "to": "/en-int/"}
       ]
     }
     ```

3. **コーポレートサイトリダイレクト検出**
   - **現状**: `monclergroup.com` をハードコード
   - **対応**: `navigation.trap_domains` に移行（既に定義あり）

---

## 3. 現状の問題点

### 3.1 コード重複
- `_normalize_to_en_int_url()` が `NavigationDriver` と `BrowserUseAgent` の両方に存在
- `_looks_like_trap_or_legal()` が両方に存在
- `_force_plp_recover()` が両方に存在

### 3.2 Moncler固有ロジックの散在
- URL正規化: `_normalize_to_en_int_url()` で `/en-int/` をハードコード
- Trap判定: `moncler.com` と `/en-jp/` をハードコード
- ヘッダ検索: Moncler URLをハードコード
- ジオモーダル: 「United Kingdom / English」をハードコード

### 3.3 site_configの未活用
- `overrides.local.json` に既に多くの設定があるが、コード側で参照されていない
- 例: `locale.normalize_rules`, `navigation.trap_url_patterns` など

---

## 4. 次のステップ

1. **NavigationDriverからMoncler固有ロジックの排除**
   - `_normalize_to_en_int_url()` → `site_config.locale.normalize_rules` を使用
   - `_looks_like_trap_or_legal()` → `site_config.navigation.trap_url_patterns` を使用
   - `header_search_fallback()` → `site_config.navigation.header_search.url_template` を使用

2. **BrowserUseAgentの責務整理**
   - NavigationDriverへの完全委譲
   - 重複メソッドの削除

3. **site_config標準スキーマの確立**
   - 他サイト（SSENSE, MATCHESFASHION等）との整合性確認
   - 必須キーの定義

---

**次のタスク**: 汎用PLP Driverの設計方針提示

