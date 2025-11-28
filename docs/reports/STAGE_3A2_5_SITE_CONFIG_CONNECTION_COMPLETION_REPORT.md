# Stage 3A-2-5: MONCLER_OFFICIAL.json を NavigationDriver & Extractor に接続 - 完了レポート

## 実装日時
2025-11-27

## 概要

### 目的
手書きの site_config（MONCLER_OFFICIAL.json）を、NavigationDriver / Extractor から実際に使うように配線する。ハードコードされた CSS セレクタ・trap パターン・fallback セレクタなどを、site_config 経由で取得する状態にする。

### ゴール
- PLP / PDP で使う CSS セレクタ・trap パターン・fallback セレクタなどを、site_config 経由で取得する
- ハードコードされた値を減らし、site_config に寄せる
- 挙動を変えないこと（既存の Moncler 用セレクタや trap 判定ロジックと挙動が一致するように）

### 原則
- 挙動不変: 既存の Moncler 用セレクタや trap 判定ロジックと挙動が一致するように、既存ハードコード値を site_config に反映させている前提で進める
- 差分は小さく: PLP セレクタ → fallback → trap → PDP セレクタ の順で小さく進める
- フォールバック: site_config が空の場合、既存のハードコード値をフォールバックとして使用

## 実装ステップ

### Step 1: PLP 用セレクタを NavigationDriver.collect_pdp_links で site_config から参照する

**変更内容:**
- `collect_pdp_links` メソッドで、`site_config["selectors"]["plp"]["pdp_link_selectors"]` を優先的に使用
- `site_config["selectors"]["pdp"]["pdp_link_selectors"]` をフォールバックとして使用
- 両方とも空の場合は、既存のハードコードされたセレクタを使用

**変更ファイル:**
- `app/agents/browser/navigation_driver.py` (324-339行目)

**コード例:**
```python
# Before:
selectors_cfg = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}
PLP_PDP_LINK_SELECTORS = _dedupe_keep_order(
    (selectors_cfg.get("pdp_link_selectors", []) or []) + [
        "a[href*='/products/']",
        # ... ハードコードされたセレクタ
    ]
)

# After:
plp_selectors = (site_config.get("selectors", {}) or {}).get("plp", {}) or {}
pdp_selectors = (site_config.get("selectors", {}) or {}).get("pdp", {}) or {}

pdp_link_selectors = _dedupe_keep_order(
    (plp_selectors.get("pdp_link_selectors", []) or []) +
    (pdp_selectors.get("pdp_link_selectors", []) or [])
)

if not pdp_link_selectors:
    pdp_link_selectors = [
        "a[href*='/products/']",
        # ... デフォルトセレクタ
    ]
```

### Step 2: NavigationDriver の fallback（header_search / click_first_card）を site_config から取る

**変更内容:**
- `header_search_fallback` メソッドで、`site_config["navigation"]["header_search"]` からセレクタを取得
- `click_first_card_or_link` メソッドで、`site_config["navigation"]["fallback"]["click_first_card"]` からセレクタを取得
- 既存の `selectors.ui` 構造もフォールバックとしてサポート

**変更ファイル:**
- `app/agents/browser/navigation_driver.py` (1173-1268行目)

**コード例:**
```python
# header_search_fallback
nav_cfg = (site_config.get("navigation", {}) or {})
hs_cfg = nav_cfg.get("header_search", {}) or {}

sel_input = _dedupe_keep_order(
    (hs_cfg.get("search_input_selector", []) or []) +
    (ui.get("search_input", []) or []) + [
        # ... デフォルトセレクタ
    ]
)
clear_before_type = hs_cfg.get("clear_before_type", True)

# click_first_card_or_link
fb_cfg = nav_cfg.get("fallback", {}) or {}
click_cfg = fb_cfg.get("click_first_card", {}) or {}

if not click_cfg.get("enabled", True):
    return None

link_sel = _dedupe_keep_order(
    (click_cfg.get("card_selectors", []) or []) +
    (plp_selectors.get("card_selectors", []) or []) +
    (pdp.get("pdp_link_selectors", []) or [])
)
```

### Step 3: trap 判定と overlay 除去も site_config に寄せる

**変更内容:**
- `_looks_like_trap_or_legal` メソッドで、`site_config["navigation"]["trap_url_patterns"]` と `site_config["navigation"]["legal_url_patterns"]` からパターンを取得
- `_accept_cookies_if_present` メソッドで、`site_config["navigation"]["overlays"]["cookie_banner_selectors"]` からセレクタを取得
- `_dismiss_geo_modal` メソッドで、`site_config["navigation"]["overlays"]["geo_modal_selectors"]` からセレクタを取得
- `_kill_overlays` メソッドで、`site_config["navigation"]["overlays"]["generic_close_buttons"]` からセレクタを取得

**変更ファイル:**
- `app/agents/browser/navigation_driver.py` (495-580行目, 788-808行目, 810-888行目, 890-900行目)

**コード例:**
```python
# _looks_like_trap_or_legal
nav_cfg = (site_config or {}).get("navigation", {})
trap_patterns = nav_cfg.get("trap_url_patterns", [])
legal_patterns = nav_cfg.get("legal_url_patterns", [])

if trap_patterns:
    if any(pattern.lower() in full_lower for pattern in trap_patterns):
        return True

# _accept_cookies_if_present
nav_cfg = (site_config.get("navigation", {}) or {})
overlays_cfg = nav_cfg.get("overlays", {}) or {}
cookie_selectors = overlays_cfg.get("cookie_banner_selectors", [])

candidates = _dedupe_keep_order(
    (cookie_selectors or []) +
    (ui.get("cookie_accept", []) or []) + [
        # ... デフォルトセレクタ
    ]
)
```

### Step 4: Extractor（PDP 抽出）側で selectors.pdp を使う

**変更内容:**
- `_read_price_or_none` メソッドで、`site_config["selectors"]["pdp"]["price"]` からセレクタを取得
- `_click_size_to_reveal_price` メソッドで、`site_config["selectors"]["pdp"]["size_button"]` からセレクタを取得
- `_extract_price_with_size_option` メソッドで、`site_config["selectors"]["pdp"]["visible_price_selectors"]` からセレクタを取得

**変更ファイル:**
- `app/agents/browser/extractor.py` (265-280行目, 282-350行目)

**コード例:**
```python
# _read_price_or_none
pdp_selectors = (site_config or {}).get("selectors", {}).get("pdp", {}) or {}
price_selectors = pdp_selectors.get("price", [])

if not price_selectors:
    price_selectors = PRICE_SELECTORS

# _click_size_to_reveal_price
pdp_selectors = (site_config or {}).get("selectors", {}).get("pdp", {}) or {}
size_button_selectors = pdp_selectors.get("size_button", [])

if not size_button_selectors:
    size_button_selectors = SIZE_BUTTON_SELECTORS
```

### Step 5: _run_plp_flow 側で site_config を正しく渡すことを確認

**確認内容:**
- `browser_use_agent.py` で `NavigationDriver` が初期化される際、`NavigationContext` に `site_config` が含まれていることを確認
- `NavigationContext` は既に `site_config` フィールドを持っているため、追加の変更は不要

**確認ファイル:**
- `app/agents/browser_use_agent.py` (1562-1567行目, 1647-1656行目)

## 変更ファイル一覧

### 新規作成ファイル
なし

### 変更ファイル

1. **app/agents/browser/navigation_driver.py**
   - `collect_pdp_links`: PLP セレクタを `site_config["selectors"]["plp"]` から取得
   - `_looks_like_trap_or_legal`: trap パターンを `site_config["navigation"]` から取得
   - `header_search_fallback`: 検索セレクタを `site_config["navigation"]["header_search"]` から取得
   - `click_first_card_or_link`: カードセレクタを `site_config["navigation"]["fallback"]["click_first_card"]` から取得
   - `_accept_cookies_if_present`: cookie セレクタを `site_config["navigation"]["overlays"]` から取得
   - `_dismiss_geo_modal`: geo モーダルセレクタを `site_config["navigation"]["overlays"]` から取得
   - `_kill_overlays`: overlay セレクタを `site_config["navigation"]["overlays"]` から取得

2. **app/agents/browser/extractor.py**
   - `_read_price_or_none`: 価格セレクタを `site_config["selectors"]["pdp"]["price"]` から取得
   - `_extract_price_with_size_option`: site_config を渡すように修正
   - `_click_size_to_reveal_price`: サイズボタンセレクタを `site_config["selectors"]["pdp"]["size_button"]` から取得

## 動作確認結果

### 静的解析結果
- リンター: `playwright.async_api` のインポート警告のみ（環境の問題、コードの問題ではない）
- 型チェッカー: エラーなし

### コードレビュー結果
- すべてのメソッドで site_config からの取得が実装されている
- フォールバック処理が適切に実装されている
- 既存の挙動を維持するためのフォールバックが実装されている

### テスト結果
- テストは未実行（実ブラウザテストが必要）

## 設計上の改善点

### アーキテクチャの改善
- ハードコードされたセレクタを減らし、site_config に集約することで、サイト固有の設定を一元管理できるようになった
- フォールバック処理により、既存の挙動を維持しながら、新しい構造に対応できるようになった

### 将来の拡張性への配慮
- `site_config["selectors"]["plp"]` と `site_config["selectors"]["pdp"]` の両方に対応することで、将来の構造変更にも対応可能
- `site_config["navigation"]` 配下に新しい設定を追加することで、ナビゲーション関連の設定を一元管理できる

### コード品質の向上
- セレクタの取得ロジックを統一することで、コードの可読性が向上
- フォールバック処理により、site_config が不完全でも動作する

## 既知の制約・注意事項

### 既存コードとの互換性
- 既存の `selectors.ui` 構造もフォールバックとしてサポートしているため、既存の site_config でも動作する
- 既存のハードコードされたセレクタもフォールバックとして使用されるため、site_config が空でも動作する

### 制限事項やトレードオフ
- `_looks_like_trap_or_legal` メソッドは、`trap_checker` が外部から渡される場合、site_config を使用しない（既存の挙動を維持するため）
- `_dismiss_geo_modal` と `_kill_overlays` は、site_config が None の場合でも動作する（既存の挙動を維持するため）

### 移行時の注意点
- MONCLER_OFFICIAL.json に以下の構造を追加する必要がある:
  - `selectors.plp.pdp_link_selectors`
  - `navigation.header_search.search_input_selector`, `submit_selector`, `clear_before_type`
  - `navigation.fallback.click_first_card.enabled`, `card_selectors`
  - `navigation.trap_url_patterns`, `legal_url_patterns`
  - `navigation.overlays.cookie_banner_selectors`, `geo_modal_selectors`, `generic_close_buttons`
  - `selectors.pdp.price`, `size_button`, `visible_price_selectors`

## 次のステップ

1. **MONCLER_OFFICIAL.json の更新**
   - 上記の構造を追加して、site_config からセレクタを取得できるようにする

2. **実ブラウザテスト**
   - Moncler サイトで実際に動作確認を行い、site_config から正しくセレクタが取得されていることを確認

3. **他のサイトへの展開**
   - 他のサイト（GIVENCHY_OFFICIAL など）にも同様の構造を追加し、汎用的に使用できるようにする

4. **ドキュメント化**
   - site_config の構造をドキュメント化し、新しいサイトを追加する際のガイドラインを作成

