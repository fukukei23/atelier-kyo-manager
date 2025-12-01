# Stage 4: 汎用 PLP Driver 設計書

**作成日時**: 2025-12-01  
**目的**: ブランド固有・サイト固有のPLP → PDPナビゲーションロジックを、site_configベースの汎用PLP Driverに統合

---

## Task A: 現状コードの分析結果

### A.1 現状の問題点

#### 1. **責務の混在**
- `BrowserUseAgent` にPLPロジックが残存（`_run_plp_flow`, `_ensure_plp_materialized`, `_collect_pdp_links`）
- `PlpDriver` は存在するが、不完全（`navigate_to_pdp` のみ）
- Moncler専用パッチ（`browser_use_moncler_patch.py`）が分離しているが、汎用化されていない

#### 2. **site_configの不統一**
- PLP設定が複数のキーに散在：
  - `selectors.plp.*` （Moncler専用）
  - `selectors.pdp.plp_container_selectors` （汎用）
  - `selectors.pdp.pdp_link_selectors` （PDP抽出用だがPLPでも使用）
  - `navigation.plp.*` （Moncler専用）
  - `discovery_settings.plp_*` （汎用）

#### 3. **Trap検出ロジックの分散**
- `NavigationDriver._looks_like_trap_or_legal()` に実装済み
- `PlpDriver._looks_like_trap_or_legal()` が `NavigationDriver` を呼び出す（重複）
- `BrowserUseAgent._looks_like_trap_or_legal()` も存在（重複）

#### 4. **Overlay処理の分散**
- Cookie: `BrowserUseAgent._accept_cookies_if_present()`, `PlpDriver._accept_cookies_if_present()`
- Geo Modal: `BrowserUseAgent._dismiss_geo_modal()`, `PlpDriver._dismiss_geo_modal()`
- Overlay削除: `BrowserUseAgent._kill_overlays()`, `PlpDriver._kill_overlays()`

### A.2 PLP/PDP判定ロジック

**現状の実装箇所**:
- `NavigationDriver._looks_like_trap_or_legal()`: URLパターンマッチング
- `PlpDriver._looks_like_trap_or_legal()`: `NavigationDriver`を呼び出すラッパー

**判定条件**:
- URLに `/legal`, `/privacy`, `/terms` などのパスが含まれる
- `site_config.navigation.trap_url_patterns` で定義されたパターン
- `site_config.navigation.legal_url_patterns` で定義されたパターン

### A.3 タイル取得ロジック

**現状の実装箇所**:
- `PlpDriver._materialize_plp_tiles()`: スクロールしながらタイルをカウント
- `NavigationDriver.ensure_plp_materialized()`: 同様の処理
- `BrowserUseAgent._ensure_plp_materialized()`: フォールバック用

**取得方法**:
1. site_configからセレクタリストを構築
2. スクロールを繰り返し実行
3. タイル数をカウント（目標: 8個以上）
4. タイムアウトまたは目標数に達するまで繰り返す

### A.4 クリックの実装

**現状の実装箇所**:
- `PlpDriver._click_tile_and_navigate_to_pdp()`: タイルクリックとPDP遷移
- `PlpDriver._click_and_capture_navigation()`: 新タブ/同タブ/SPA遷移のレース処理

**クリック戦略**:
1. リンクセレクタから試行（`pdp_link_selectors`）
2. タイルセレクタから試行（`plp_container_selectors` + デフォルトタイルセレクタ）
3. ブロックリストで除外（`blocklist_href_substrings`）

### A.5 Trap検出条件

**検出方法**:
1. URLパターンマッチング（`trap_url_patterns`, `legal_url_patterns`）
2. Selectorマッチング（`trap_selector_patterns`） - 未実装

**リカバリアクション**:
- `go_back`: `page.go_back()`
- `reload`: `page.reload()`
- `click_close`: クローズボタンをクリック
- `goto_target`: `target_url` に遷移

### A.6 Overlay処理

**処理対象**:
1. Cookieバナー: `selectors.ui.cookie_accept`
2. Geo Modal: `navigation.overlays.geo_modal_selectors`
3. その他のオーバーレイ: JavaScriptで削除（`.overlay`, `.backdrop`, `.modal-backdrop` など）

**処理順序**:
1. Cookieバナーを処理
2. Geo Modalを処理
3. その他のオーバーレイを削除

### A.7 navigate_to_pdpの例外処理フロー

**現状の実装**:
```python
async def navigate_to_pdp(...) -> PlpNavigationResult:
    # 1. PLPタイルのマテリアライズ
    tiles_seen = await self._materialize_plp_tiles(...)
    if tiles_seen == 0:
        raise ValueError("PLP did not materialize")
    
    # 2. Trapページ検出
    if self._looks_like_trap_or_legal(self.page.url):
        # リカバリ試行
        await self._recover_from_trap(target_url)
        # 再チェック
        if self._looks_like_trap_or_legal(self.page.url):
            raise ValueError("Still on trap/legal page after recovery")
    
    # 3. タイルクリック → PDP遷移
    pdp_page = await self._click_tile_and_navigate_to_pdp()
    if pdp_page is None:
        raise ValueError("Failed to navigate to PDP")
    
    return PlpNavigationResult(...)
```

**問題点**:
- 例外処理が不十分（一部のエラーがキャッチされない）
- リカバリ失敗時の処理が不統一

### A.8 BrowserUseAgentが持っているPLPロジック（混在箇所）

**混在しているメソッド**:
1. `_run_plp_flow()`: PLPフローのメインエントリーポイント（1,752行目）
2. `_ensure_plp_materialized()`: タイルマテリアライズ（1,142行目）
3. `_collect_pdp_links()`: PDPリンク収集（1,288行目）- 既に`NavigationDriver`に移行済み
4. `_force_plp_recover()`: PLP回復（未確認、おそらく存在）
5. `_accept_cookies_if_present()`: Cookie処理（922行目）
6. `_dismiss_geo_modal()`: Geo Modal処理（933行目）
7. `_kill_overlays()`: オーバーレイ削除（873行目）

---

## Task B: 「理想の汎用 PLP Driver」設計案

### B.1 クラス構造案

```python
class PlpDriver:
    """
    汎用 PLP Driver - site_configベースで動作
    """
    
    # === Public API ===
    async def navigate_to_pdp(
        self,
        page: Page,
        site_config: Dict[str, Any],
        run_context: RunContext,
        *,
        target_url: Optional[str] = None,
        timeout_ms: int = 60000,
    ) -> PlpNavigationResult:
        """PLP → PDP ナビゲーションのメインエントリーポイント"""
    
    # === Private Methods ===
    
    # 1. Materialization (タイルマテリアライズ)
    async def _materialize_tiles(...) -> int:
        """PLPタイルをスクロールしながらマテリアライズ"""
    
    # 2. Overlay Handling
    async def _handle_overlays(...) -> None:
        """Cookie/Geo/その他のオーバーレイを処理"""
    
    async def _handle_cookie_banner(...) -> bool:
        """Cookieバナーを処理"""
    
    async def _handle_geo_modal(...) -> bool:
        """Geo Modalを処理"""
    
    async def _remove_generic_overlays(...) -> None:
        """その他のオーバーレイを削除"""
    
    # 3. Trap Detection & Recovery
    async def _detect_and_recover_from_trap(...) -> bool:
        """Trapページを検出し、リカバリを試みる"""
    
    def _is_trap_page(...) -> bool:
        """Trapページかどうかを判定"""
    
    async def _recover_from_trap(...) -> bool:
        """Trapページから回復する"""
    
    # 4. Tile Clicking
    async def _click_tile_and_navigate(...) -> Optional[Page]:
        """タイルをクリックしてPDPに遷移"""
    
    async def _click_and_wait_for_navigation(...) -> Optional[Page]:
        """クリック後のナビゲーションを待機（新タブ/同タブ/SPA）"""
    
    # 5. URL Normalization
    def _normalize_pdp_url(...) -> str:
        """PDP URLを正規化"""
    
    # 6. Configuration Helpers
    def _get_plp_config(...) -> Dict[str, Any]:
        """site_configからPLP設定を取得"""
    
    def _get_overlay_config(...) -> Dict[str, Any]:
        """site_configからOverlay設定を取得"""
    
    def _get_trap_config(...) -> Dict[str, Any]:
        """site_configからTrap設定を取得"""
```

### B.2 site_config スキーマ案（PLP部分）

```json
{
  "selectors": {
    "plp": {
      "product_tiles": [
        "article[data-test='product-tile']",
        "li[data-test='product-tile']",
        "a[data-test='product-tile-link']",
        ".product-card",
        ".c-product-card"
      ],
      "product_link": [
        "a[data-test='product-tile-link']",
        ".product-card a",
        "a[href*='/products/']",
        "a[href*='/product/']",
        "a[href*='/p/']"
      ],
      "container": [
        "main [data-test='product-grid']",
        "ul.products-grid",
        ".plp-grid",
        ".product-grid"
      ],
      "click_strategy": "link",
      "wait_for_navigation": true,
      "min_tiles": 8,
      "max_scroll_rounds": 10,
      "scroll_pause_ms": 160,
      "target_load_state": "networkidle"
    }
  },
  "navigation": {
    "plp": {
      "url_patterns": {
        "plp": [
          ".*/category/.*",
          ".*/collection/.*",
          ".*/products?/.*"
        ],
        "pdp": [
          ".*/product[s]?/[^/]+",
          ".*/p/[^/]+",
          ".*/pp/[^/]+"
        ]
      },
      "normalize_url": {
        "strip_fragments": true,
        "strip_query_params": ["utm_*", "ref", "source"],
        "ensure_trailing_slash": false
      }
    },
    "overlays": {
      "cookie_banner": {
        "selectors": [
          "#onetrust-accept-btn-handler",
          "button:has-text('ACCEPT ALL')",
          "button[aria-label*='Accept' i]"
        ],
        "wait_after_click_ms": 500
      },
      "geo_popup": {
        "selectors": [
          "button:has-text('STAY HERE')",
          "button:has-text('REMAIN HERE')",
          "button[aria-label*='close' i]"
        ],
        "wait_after_click_ms": 500
      },
      "other_overlays": {
        "remove_selectors": [
          ".overlay",
          ".backdrop",
          ".modal-backdrop",
          "[aria-modal='true']"
        ],
        "remove_body_classes": ["modal-open", "locked", "no-scroll"]
      }
    },
    "trap": {
      "detect_by_url": {
        "patterns": [
          ".*/legal/.*",
          ".*/privacy/.*",
          ".*/terms/.*",
          ".*/forbidden/.*"
        ],
        "exact_matches": []
      },
      "detect_by_selector": [
        "h1:has-text('Page not found')",
        "h1:has-text('404')",
        "[role='alert']:has-text('Access Denied')"
      ],
      "recovery_actions": [
        {
          "action": "go_back",
          "max_attempts": 1
        },
        {
          "action": "goto_target",
          "target_url_key": "seed_plp_url",
          "max_attempts": 1
        },
        {
          "action": "reload",
          "max_attempts": 1
        }
      ]
    }
  },
  "discovery_settings": {
    "plp": {
      "budget_ms": 60000,
      "scroll_rounds": 10,
      "scroll_pause_ms": 160,
      "wait_for_selectors": [
        "[data-test='product-grid']",
        ".product-grid"
      ],
      "wait_until": "networkidle"
    }
  }
}
```

### B.3 責務分離マップ

#### BrowserUseAgent が担う責務:
1. **ページを開く**: `_bootstrap_session_page()` で初期ナビゲーション
2. **RunContext の準備**: 実行コンテキストの初期化
3. **PlpDriver を呼ぶ**: `_run_plp_flow()` 内で `PlpDriver.navigate_to_pdp()` を呼び出し
4. **結果を Extractor に渡す**: `PlpNavigationResult` を受け取り、`Extractor` に渡す

#### PlpDriver が担う責務:
1. **PLP上の商品タイル解析**: `_materialize_tiles()` でタイルをカウント
2. **タイルクリック or リンククリック**: `_click_tile_and_navigate()` でクリック
3. **PDP着地の判定**: `_click_and_wait_for_navigation()` で遷移を検知
4. **Trap / Overlay の処理**: `_detect_and_recover_from_trap()`, `_handle_overlays()`
5. **NavigationResult の返却**: `PlpNavigationResult` を返す

#### NavigationDriver との関係:
- `NavigationDriver`: PLPへの初期ナビゲーション、URL正規化、trap判定（URLベース）
- `PlpDriver`: PLP上でのタイル操作、PDP遷移、overlay処理、trap回復

### B.4 API 入出力の定義

#### PlpNavigationResult (拡張版)

```python
@dataclass
class PlpNavigationResult:
    """PLP → PDP ナビゲーションの結果"""
    pdp_url: str
    pdp_opened_in_new_tab: bool
    plp_url: str
    tiles_seen: int
    trap_detected: bool
    trap_reason: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    overlays_handled: List[str] = field(default_factory=list)  # ["cookie", "geo"]
    navigation_method: Optional[str] = None  # "new_tab", "same_tab", "spa"
    errors: List[str] = field(default_factory=list)  # 発生したエラーメッセージ
```

### B.5 リファクタリング戦略（段階的移行）

#### Phase 1: site_config スキーマの統一
- 既存の `selectors.pdp.*` を `selectors.plp.*` に移行
- `navigation.plp.*` スキーマを追加
- 後方互換性を維持（既存のキーも読み込む）

#### Phase 2: PlpDriver の拡張
- 既存の `PlpDriver` を拡張（破壊的変更を避ける）
- `navigate_to_pdp()` を完全実装
- Overlay処理、Trap検出・回復を統合

#### Phase 3: BrowserUseAgent からの移行
- `_run_plp_flow()` を `PlpDriver.navigate_to_pdp()` に置き換え
- `_ensure_plp_materialized()` を削除
- Overlay処理メソッドを削除

#### Phase 4: テストと検証
- 既存テストが通ることを確認
- 新規テストを追加
- Moncler設定の動作確認

#### Phase 5: クリーンアップ
- 未使用コードの削除
- ドキュメント更新

---

## Task C: 具体的コード修正案

### C.1 修正前コードの問題点

1. **責務の混在**: BrowserUseAgent にPLPロジックが残存
2. **設定の散在**: PLP設定が複数のキーに分かれている
3. **重複コード**: Overlay処理が複数箇所に存在
4. **例外処理の不統一**: エラーハンドリングが一貫していない

### C.2 修正後のコード構造

```
app/agents/browser/
├── plp_driver.py          # 汎用PLP Driver（拡張版）
├── navigation_driver.py   # 既存（変更なし）
├── extractor.py          # 既存（変更なし）
└── product_extractor.py  # 既存（変更なし）

app/agents/
└── browser_use_agent.py  # PLPロジックを削除、PlpDriverを呼び出すだけ
```

### C.3 新しい PlpDriver のコード（主要部分）

```python
# app/agents/browser/plp_driver.py (拡張版)

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, TYPE_CHECKING
from urllib.parse import urlparse, urljoin

from playwright.async_api import Page, BrowserContext, Locator

if TYPE_CHECKING:
    from app.core.run_context import RunContext
    from app.agents.browser.telemetry import TelemetryClient

logger = logging.getLogger(__name__)


@dataclass
class PlpNavigationResult:
    """PLP → PDP ナビゲーションの結果（拡張版）"""
    pdp_url: str
    pdp_opened_in_new_tab: bool
    plp_url: str
    tiles_seen: int
    trap_detected: bool
    trap_reason: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    overlays_handled: List[str] = field(default_factory=list)
    navigation_method: Optional[str] = None
    errors: List[str] = field(default_factory=list)


class PlpDriver:
    """
    汎用 PLP Driver - site_configベースで動作
    
    Stage 4: 完全汎用化
    - ブランド固有ロジックをすべて site_config に移行
    - コードは汎用的なナビゲーションロジックのみ
    """
    
    def __init__(
        self,
        page: Page,
        context: BrowserContext,
        *,
        site_config: Dict[str, Any],
        run_context: "RunContext",
        logger: Optional[logging.Logger] = None,
        telemetry: Optional["TelemetryClient"] = None,
    ) -> None:
        self.page = page
        self.context = context
        self.site_config = site_config
        self.run_context = run_context
        self.logger = logger or logging.getLogger(__name__)
        self.telemetry = telemetry
    
    async def navigate_to_pdp(
        self,
        *,
        target_url: Optional[str] = None,
        timeout_ms: int = 60000,
    ) -> PlpNavigationResult:
        """
        PLP → PDP ナビゲーションを実行する（メインエントリーポイント）
        
        Args:
            target_url: ターゲットPLP URL（リカバリ用）
            timeout_ms: タイムアウト（ミリ秒）
            
        Returns:
            PlpNavigationResult: ナビゲーション結果
        """
        start_t = time.time()
        plp_url = self.page.url
        errors: List[str] = []
        overlays_handled: List[str] = []
        
        try:
            # 1. Overlay処理（Cookie、Geo、その他）
            await self._handle_overlays(overlays_handled)
            
            # 2. PLPタイルのマテリアライズ
            plp_config = self._get_plp_config()
            tiles_seen = await self._materialize_tiles(
                plp_config=plp_config,
                start_t=start_t,
                budget_ms=timeout_ms,
                target_url=target_url,
            )
            
            if tiles_seen == 0:
                raise ValueError(f"PLP did not materialize (no product tiles). URL={plp_url}")
            
            # 3. Trap検出と回復
            trap_config = self._get_trap_config()
            trap_detected = False
            trap_reason = None
            recovery_attempted = False
            recovery_successful = False
            
            if self._is_trap_page(self.page.url, trap_config):
                trap_detected = True
                trap_reason = f"Trap/legal page detected: {self.page.url}"
                self.logger.warning(f"[PlpDriver] {trap_reason}")
                
                if target_url:
                    recovery_attempted = True
                    recovery_successful = await self._recover_from_trap(
                        target_url=target_url,
                        trap_config=trap_config,
                    )
                    
                    if recovery_successful:
                        # 回復後に再度trap判定
                        if self._is_trap_page(self.page.url, trap_config):
                            raise ValueError(
                                f"Still on trap/legal page after recovery: {self.page.url}"
                            )
                    else:
                        raise ValueError(
                            f"Trap recovery failed. URL={self.page.url}"
                        )
            
            # 4. タイルクリック → PDP遷移
            pdp_page = await self._click_tile_and_navigate(
                plp_config=plp_config,
                timeout_ms=min(10000, timeout_ms // 6),
            )
            
            if pdp_page is None:
                raise ValueError("Failed to navigate to PDP from PLP tile")
            
            pdp_url = pdp_page.url
            pdp_opened_in_new_tab = (pdp_page is not self.page)
            navigation_method = (
                "new_tab" if pdp_opened_in_new_tab else
                "same_tab" if pdp_url != plp_url else
                "spa"
            )
            
            # 新タブが開かれた場合、self.page を更新
            if pdp_opened_in_new_tab:
                self.page = pdp_page
            
            return PlpNavigationResult(
                pdp_url=pdp_url,
                pdp_opened_in_new_tab=pdp_opened_in_new_tab,
                plp_url=plp_url,
                tiles_seen=tiles_seen,
                trap_detected=trap_detected,
                trap_reason=trap_reason,
                recovery_attempted=recovery_attempted,
                recovery_successful=recovery_successful,
                overlays_handled=overlays_handled,
                navigation_method=navigation_method,
                errors=errors,
            )
            
        except Exception as e:
            errors.append(str(e))
            self.logger.error(f"[PlpDriver] Navigation failed: {e}", exc_info=True)
            raise
    
    # === Private Methods ===
    
    def _get_plp_config(self) -> Dict[str, Any]:
        """site_configからPLP設定を取得（後方互換性維持）"""
        selectors = (self.site_config.get("selectors") or {})
        plp_cfg = selectors.get("plp") or {}
        pdp_cfg = selectors.get("pdp") or {}
        
        # 後方互換性: pdp.* から plp.* にフォールバック
        return {
            "product_tiles": (
                plp_cfg.get("product_tiles") or
                pdp_cfg.get("plp_container_selectors") or
                []
            ),
            "product_link": (
                plp_cfg.get("product_link") or
                pdp_cfg.get("pdp_link_selectors") or
                []
            ),
            "container": (
                plp_cfg.get("container") or
                pdp_cfg.get("plp_container_selectors") or
                ["main", "[role='main']"]
            ),
            "click_strategy": plp_cfg.get("click_strategy", "link"),
            "wait_for_navigation": plp_cfg.get("wait_for_navigation", True),
            "min_tiles": plp_cfg.get("min_tiles", 8),
            "max_scroll_rounds": (
                plp_cfg.get("max_scroll_rounds") or
                self.site_config.get("discovery_settings", {}).get("plp_scroll_rounds", 10) or
                10
            ),
            "scroll_pause_ms": plp_cfg.get("scroll_pause_ms", 160),
            "target_load_state": plp_cfg.get("target_load_state", "networkidle"),
        }
    
    def _get_overlay_config(self) -> Dict[str, Any]:
        """site_configからOverlay設定を取得"""
        nav_cfg = (self.site_config.get("navigation") or {})
        overlays_cfg = nav_cfg.get("overlays") or {}
        selectors = (self.site_config.get("selectors") or {})
        ui_cfg = selectors.get("ui") or {}
        
        return {
            "cookie": {
                "selectors": (
                    overlays_cfg.get("cookie_banner", {}).get("selectors") or
                    ui_cfg.get("cookie_accept") or
                    []
                ),
                "wait_after_click_ms": (
                    overlays_cfg.get("cookie_banner", {}).get("wait_after_click_ms") or
                    500
                ),
            },
            "geo": {
                "selectors": (
                    overlays_cfg.get("geo_popup", {}).get("selectors") or
                    overlays_cfg.get("geo_modal_selectors") or
                    []
                ),
                "wait_after_click_ms": (
                    overlays_cfg.get("geo_popup", {}).get("wait_after_click_ms") or
                    500
                ),
            },
            "other": overlays_cfg.get("other_overlays") or {},
        }
    
    def _get_trap_config(self) -> Dict[str, Any]:
        """site_configからTrap設定を取得"""
        nav_cfg = (self.site_config.get("navigation") or {})
        trap_cfg = nav_cfg.get("trap") or {}
        
        # 後方互換性: navigation.trap_url_patterns も読み込む
        legacy_patterns = nav_cfg.get("trap_url_patterns") or []
        
        return {
            "detect_by_url": {
                "patterns": (
                    trap_cfg.get("detect_by_url", {}).get("patterns") or
                    legacy_patterns or
                    []
                ),
                "exact_matches": (
                    trap_cfg.get("detect_by_url", {}).get("exact_matches") or
                    []
                ),
            },
            "detect_by_selector": (
                trap_cfg.get("detect_by_selector") or
                []
            ),
            "recovery_actions": (
                trap_cfg.get("recovery_actions") or
                [
                    {"action": "go_back", "max_attempts": 1},
                    {"action": "goto_target", "target_url_key": "seed_plp_url", "max_attempts": 1},
                ]
            ),
        }
    
    async def _materialize_tiles(
        self,
        *,
        plp_config: Dict[str, Any],
        start_t: float,
        budget_ms: int,
        target_url: Optional[str] = None,
    ) -> int:
        """PLPタイルをマテリアライズ"""
        # 実装は既存の _materialize_plp_tiles() をベースに
        # site_config ベースにリファクタリング
        ...
    
    async def _handle_overlays(self, overlays_handled: List[str]) -> None:
        """Overlay処理（Cookie、Geo、その他）"""
        overlay_config = self._get_overlay_config()
        
        # Cookie処理
        if await self._handle_cookie_banner(overlay_config["cookie"]):
            overlays_handled.append("cookie")
        
        # Geo処理
        if await self._handle_geo_modal(overlay_config["geo"]):
            overlays_handled.append("geo")
        
        # その他のオーバーレイ削除
        await self._remove_generic_overlays(overlay_config["other"])
    
    async def _handle_cookie_banner(self, config: Dict[str, Any]) -> bool:
        """Cookieバナーを処理"""
        selectors = config.get("selectors", [])
        wait_ms = config.get("wait_after_click_ms", 500)
        
        for sel in selectors:
            try:
                loc = self.page.locator(sel).first
                if await loc.count() > 0:
                    await loc.click(timeout=3000)
                    await self.page.wait_for_timeout(wait_ms)
                    self.logger.info(f"[PlpDriver] Cookie banner accepted via: {sel}")
                    return True
            except Exception:
                continue
        return False
    
    async def _handle_geo_modal(self, config: Dict[str, Any]) -> bool:
        """Geo Modalを処理"""
        selectors = config.get("selectors", [])
        wait_ms = config.get("wait_after_click_ms", 500)
        
        for sel in selectors:
            try:
                loc = self.page.locator(sel).first
                if await loc.count() > 0:
                    await loc.click(timeout=3000)
                    await self.page.wait_for_timeout(wait_ms)
                    self.logger.info(f"[PlpDriver] Geo modal dismissed via: {sel}")
                    return True
            except Exception:
                continue
        return False
    
    async def _remove_generic_overlays(self, config: Dict[str, Any]) -> None:
        """その他のオーバーレイを削除"""
        remove_selectors = config.get("remove_selectors", [])
        remove_body_classes = config.get("remove_body_classes", [])
        
        if not remove_selectors and not remove_body_classes:
            return
        
        js_code = f"""
        (() => {{
            const sels = {json.dumps(remove_selectors)};
            document.querySelectorAll(sels.join(',')).forEach(el => el.remove());
            
            const bodyClasses = {json.dumps(remove_body_classes)};
            const body = document.body;
            if (body) {{
                bodyClasses.forEach(cls => body.classList.remove(cls));
                body.style.overflow = '';
            }}
            
            const html = document.documentElement;
            if (html) {{
                html.style.overflow = '';
                bodyClasses.forEach(cls => html.classList.remove(cls));
            }}
        }})();
        """
        
        try:
            await self.page.evaluate(js_code)
        except Exception as e:
            self.logger.debug(f"[PlpDriver] Overlay removal failed: {e}")
    
    def _is_trap_page(self, url: str, trap_config: Dict[str, Any]) -> bool:
        """Trapページかどうかを判定"""
        # NavigationDriver のロジックを再利用
        from app.agents.browser.navigation_driver import NavigationDriver
        
        driver = NavigationDriver(page=None)  # type: ignore
        return driver._looks_like_trap_or_legal(url, self.site_config)
    
    async def _recover_from_trap(
        self,
        *,
        target_url: str,
        trap_config: Dict[str, Any],
    ) -> bool:
        """Trapページから回復する"""
        recovery_actions = trap_config.get("recovery_actions", [])
        
        for action_cfg in recovery_actions:
            action = action_cfg.get("action")
            max_attempts = action_cfg.get("max_attempts", 1)
            
            for attempt in range(max_attempts):
                try:
                    if action == "go_back":
                        await self.page.go_back(wait_until="domcontentloaded")
                        await self.page.wait_for_timeout(1000)
                        if not self._is_trap_page(self.page.url, trap_config):
                            return True
                    
                    elif action == "goto_target":
                        target = (
                            target_url or
                            self.site_config.get(action_cfg.get("target_url_key", "seed_plp_url")) or
                            ""
                        )
                        if target:
                            await self.page.goto(url=target, wait_until="domcontentloaded")
                            await self.page.wait_for_timeout(1000)
                            if not self._is_trap_page(self.page.url, trap_config):
                                return True
                    
                    elif action == "reload":
                        await self.page.reload(wait_until="domcontentloaded")
                        await self.page.wait_for_timeout(1000)
                        if not self._is_trap_page(self.page.url, trap_config):
                            return True
                    
                except Exception as e:
                    self.logger.warning(f"[PlpDriver] Recovery action '{action}' failed: {e}")
                    continue
        
        return False
    
    async def _click_tile_and_navigate(
        self,
        *,
        plp_config: Dict[str, Any],
        timeout_ms: int = 5000,
    ) -> Optional[Page]:
        """タイルをクリックしてPDPに遷移"""
        # 実装は既存の _click_tile_and_navigate_to_pdp() をベースに
        # site_config ベースにリファクタリング
        ...
    
    async def _click_and_wait_for_navigation(
        self,
        click_coro,
        *,
        url_regex: Optional[Pattern[str]],
        timeout_ms: int = 5000,
    ) -> Optional[Page]:
        """クリック後のナビゲーションを待機（新タブ/同タブ/SPA）"""
        # 実装は既存の _click_and_capture_navigation() をベースに
        ...
```

### C.4 BrowserUseAgent 側の差分パッチ

```python
# app/agents/browser_use_agent.py

# === 削除するメソッド ===
# - _ensure_plp_materialized()  # PlpDriverに移行
# - _accept_cookies_if_present()  # PlpDriverに移行
# - _dismiss_geo_modal()  # PlpDriverに移行
# - _kill_overlays()  # PlpDriverに移行
# - _force_plp_recover()  # PlpDriverに移行（NavigationDriver経由）

# === _run_plp_flow() の変更 ===

async def _run_plp_flow(
    self,
    page: Page,
    context: BrowserContext,
    site: str,
    query: str,
    site_config: Dict,
    settings: Dict,
    run_context: RunContext,
    target_url: str,
    *,
    start_t: float,
    budget_ms: int,
    skip_materialize: bool = False,
    nav_outcome: Optional[Any] = None,
    plugin: Optional[Any] = None,
) -> DiscoveryResult:
    """
    Stage 4: PlpDriver を使用した汎用PLPフロー
    """
    
    # Stage 4: PlpDriver を初期化
    from app.agents.browser.plp_driver import PlpDriver
    from app.agents.browser.telemetry import TelemetryClient
    
    telemetry = TelemetryClient(run_context)
    plp_driver = PlpDriver(
        page=page,
        context=context,
        site_config=site_config,
        run_context=run_context,
        telemetry=telemetry,
    )
    
    try:
        # PlpDriver で PLP → PDP ナビゲーション
        plp_result = await plp_driver.navigate_to_pdp(
            target_url=target_url,
            timeout_ms=min(budget_ms, settings.get("overall_plp_budget_ms", 60000)),
        )
        
        # PlpNavigationResult から DiscoveryResult を構築
        # 次のステップ: Extractor で商品情報を抽出
        
        # page を更新（新タブが開かれた場合）
        if plp_result.pdp_opened_in_new_tab:
            page = plp_driver.page
        
        # Extractor で商品情報を抽出
        # ... (既存の抽出ロジック) ...
        
        return DiscoveryResult(
            ok=True,
            site=site,
            query=query,
            # ... (抽出結果) ...
        )
        
    except Exception as e:
        self.logger.error(f"[_run_plp_flow] Failed: {e}", exc_info=True)
        # ... (エラーハンドリング) ...
        return DiscoveryResult(
            ok=False,
            site=site,
            query=query,
            message=str(e),
            # ... (エラー情報) ...
        )
```

### C.5 site_config のサンプル（Moncler 用）

```json
{
  "MONCLER_OFFICIAL": {
    "selectors": {
      "plp": {
        "product_tiles": [
          "article[data-test='product-tile']",
          "li[data-test='product-tile']",
          "div[data-test='product-card']"
        ],
        "product_link": [
          "a[href*='/products/']",
          "a[href*='/p/']",
          "div[data-test='product-card'] a"
        ],
        "container": [
          "[data-test='product-grid']",
          "ul.products-grid",
          ".product-grid"
        ],
        "click_strategy": "link",
        "min_tiles": 12,
        "max_scroll_rounds": 12,
        "scroll_pause_ms": 350
      }
    },
    "navigation": {
      "plp": {
        "url_patterns": {
          "plp": [
            ".*/en-int/(men|women|kids)/.+"
          ],
          "pdp": [
            ".*/en-int/.*/products/[^/]+",
            ".*/en-int/.*/p/[^/]+"
          ]
        }
      },
      "overlays": {
        "cookie_banner": {
          "selectors": [
            "#onetrust-accept-btn-handler",
            "button:has-text('ACCEPT ALL')"
          ]
        },
        "geo_popup": {
          "selectors": [
            "button:has-text('STAY HERE')",
            "button[aria-label*='close' i]"
          ]
        }
      },
      "trap": {
        "detect_by_url": {
          "patterns": [
            ".*/legal/.*",
            ".*/privacy/.*",
            ".*/client-service/.*"
          ]
        },
        "recovery_actions": [
          {
            "action": "goto_target",
            "target_url_key": "seed_plp_url",
            "max_attempts": 3
          }
        ]
      }
    },
    "discovery_settings": {
      "plp": {
        "budget_ms": 60000,
        "scroll_rounds": 12,
        "scroll_pause_ms": 350
      }
    }
  }
}
```

---

## Task D: テスト戦略と追加テスト案

### D.1 既存のテスト

- `tests/test_plp_driver.py`: 基本的なユニットテスト
- `tests/test_browser_use_agent_plp_integration.py`: 統合テスト

### D.2 追加すべきテスト

#### 1. 新タブ遷移テスト

```python
@pytest.mark.asyncio
async def test_plp_driver_new_tab_navigation(mock_page, mock_context, site_config, run_context):
    """新タブでのPDP遷移をテスト"""
    # 新タブが開かれるケースをシミュレート
    new_page = AsyncMock()
    new_page.url = "https://example.com/product/123"
    
    # context.wait_for_event("page") が新しいPageを返すことをモック
    mock_context.wait_for_event.return_value = new_page
    
    driver = PlpDriver(...)
    result = await driver.navigate_to_pdp(...)
    
    assert result.pdp_opened_in_new_tab is True
    assert result.navigation_method == "new_tab"
```

#### 2. SPA遷移テスト（URL変更検知）

```python
@pytest.mark.asyncio
async def test_plp_driver_spa_navigation(mock_page, mock_context, site_config, run_context):
    """SPA遷移（URLのみ変わる）をテスト"""
    # URLが変更されるが、pageオブジェクトは同じ
    original_url = "https://example.com/category"
    new_url = "https://example.com/product/123"
    
    mock_page.url = original_url
    
    # URL変更イベントをシミュレート
    async def url_change():
        mock_page.url = new_url
        await asyncio.sleep(0.1)
    
    driver = PlpDriver(...)
    # URL変更を検知するテスト
    ...
```

#### 3. Trap → Recovery → PDP成功テスト

```python
@pytest.mark.asyncio
async def test_plp_driver_trap_recovery_success(mock_page, mock_context, site_config, run_context):
    """Trap検出 → 回復 → PDP成功のフローをテスト"""
    # 最初はTrapページ
    mock_page.url = "https://example.com/legal"
    site_config["navigation"] = {
        "trap": {
            "detect_by_url": {"patterns": [".*/legal/.*"]},
            "recovery_actions": [
                {"action": "goto_target", "target_url_key": "seed_plp_url", "max_attempts": 1}
            ]
        }
    }
    site_config["seed_plp_url"] = "https://example.com/category"
    
    # 回復後はPLP、その後PDPに遷移
    # ...
```

#### 4. Overlayが2種類以上出るケース

```python
@pytest.mark.asyncio
async def test_plp_driver_multiple_overlays(mock_page, mock_context, site_config, run_context):
    """Cookie + Geo Modal の両方が表示されるケース"""
    # CookieバナーとGeo Modalの両方を表示
    # 両方が処理されることを確認
    ...
```

#### 5. PLP → PDP の URL 正規化テスト

```python
@pytest.mark.asyncio
async def test_plp_driver_url_normalization(mock_page, mock_context, site_config, run_context):
    """PDP URLの正規化をテスト"""
    # フラグメントやクエリパラメータが含まれるURL
    # 正規化後のURLが正しいことを確認
    ...
```

---

## Task E: 最終的に生成してほしい成果物

### E.1 PLP Driver（新バージョン）の完全コード

- `app/agents/browser/plp_driver.py` (拡張版)
- 全メソッドの実装
- ドキュメントコメント

### E.2 BrowserUseAgent への差分パッチ

- `_run_plp_flow()` の変更点
- 削除するメソッドのリスト
- 移行手順

### E.3 site_config（plp部分）の標準テンプレ

- `docs/templates/site_config_plp_template.json`
- Moncler設定のサンプル
- 他サイトへの適用例

### E.4 追加テストファイル一式

- `tests/test_plp_driver_stage4.py` (新規テスト)
- `tests/test_browser_use_agent_plp_integration_stage4.py` (拡張テスト)

### E.5 段階的移行ガイド（後方互換性を維持）

- `docs/migration_guides/STAGE_4_MIGRATION_GUIDE.md`
- 5ステップの移行手順
- 既存設定の移行方法

---

## 次のステップ

1. **Task C の完全実装**: PlpDriver の全メソッドを実装
2. **Task D のテスト作成**: 追加テストを実装
3. **Task E の成果物生成**: 完全コード、パッチ、テンプレ、移行ガイド

続けて、これらの実装を進めますか？

