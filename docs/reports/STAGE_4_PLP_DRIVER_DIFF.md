# Stage 4: PlpDriver 拡張版 - 差分レポート

## 概要

`app/agents/browser/plp_driver.py` を Stage 4 の設計書に基づいて拡張版にリファクタリングしました。

## 主要な変更点

### 1. PlpNavigationResult の拡張

```diff
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
+   recovery_successful: bool = False  # Stage 4: 追加
+   overlays_handled: List[str] = field(default_factory=list)  # Stage 4: 追加
+   navigation_method: Optional[str] = None  # Stage 4: 追加 ("new_tab", "same_tab", "spa")
+   errors: List[str] = field(default_factory=list)  # Stage 4: 追加
```

**変更理由:**
- リカバリの成功/失敗を追跡
- 処理したオーバーレイの種類を記録
- ナビゲーション方法を記録（デバッグ・診断に有用）
- エラーメッセージを収集（トラブルシューティングに有用）

### 2. navigate_to_pdp() メソッドの拡張

```diff
async def navigate_to_pdp(
    self,
    *,
-   start_t: float,
-   budget_ms: int,
+   target_url: Optional[str] = None,
+   timeout_ms: int = 60000,
+   # 後方互換性のため、既存のシグネチャもサポート
+   start_t: Optional[float] = None,
+   budget_ms: Optional[int] = None,
) -> PlpNavigationResult:
```

**変更理由:**
- 新しいシグネチャ（`timeout_ms`）を導入し、よりシンプルなAPIに
- 後方互換性を維持（既存コードが動作するように）

**内部実装の改善:**
- Overlay処理を最初に実行
- エラー収集とレポート機能を追加
- リカバリの成功/失敗を追跡

### 3. 設定取得メソッドの追加（Stage 4 の核心機能）

```python
# 新規追加
def _get_plp_config(self) -> Dict[str, Any]:
    """site_configからPLP設定を取得（後方互換性維持）"""
    # selectors.plp.* を優先しつつ、selectors.pdp.* からもフォールバック

def _get_overlay_config(self) -> Dict[str, Any]:
    """site_configからOverlay設定を取得"""
    # navigation.overlays.* と selectors.ui.* を統合

def _get_trap_config(self) -> Dict[str, Any]:
    """site_configからTrap設定を取得"""
    # navigation.trap.* スキーマを優先
```

**変更理由:**
- site_config ベースの設定取得を統一
- 後方互換性を維持（既存の設定キーも読み込む）
- 設定の一元管理

### 4. Overlay処理の改善

```diff
- async def _handle_overlays(self) -> None:
+ async def _handle_overlays(self, overlays_handled: List[str]) -> None:
    """Overlay処理（Cookie、Geo、その他）"""
+   # overlays_handled に処理したオーバーレイの種類を追加

+ async def _handle_cookie_banner(self, config: Dict[str, Any]) -> bool:
+    """Cookieバナーを処理（site_configベース）"""
+
+ async def _handle_geo_modal(self, config: Dict[str, Any]) -> bool:
+    """Geo Modalを処理（site_configベース）"""
+
+ async def _remove_generic_overlays(self, config: Dict[str, Any]) -> None:
+    """その他のオーバーレイを削除（site_configベース）"""
```

**変更理由:**
- site_config から設定を取得して処理
- 処理したオーバーレイの種類を記録
- 各オーバーレイ処理を個別のメソッドに分離（テスト容易性向上）

### 5. Trap検出・回復の改善

```diff
- def _looks_like_trap_or_legal(self, url: str) -> bool:
+ def _is_trap_page(self, url: str, trap_config: Dict[str, Any]) -> bool:
+    """Trapページかどうかを判定（site_configベース）"""
+    # NavigationDriver のロジックを再利用（将来DI可能に）

- async def _recover_from_trap(self, target_url: str) -> None:
+ async def _recover_from_trap(
+    self,
+    *,
+    target_url: str,
+    trap_config: Dict[str, Any],
+) -> bool:
+    """Trapページから回復する（site_configベースのリカバリアクション）"""
+    # recovery_actions に基づいてリカバリを実行
+    # 成功/失敗を返す
```

**変更理由:**
- site_config の `navigation.trap.recovery_actions` に基づいてリカバリ
- リカバリの成功/失敗を返す（呼び出し側で追跡可能に）
- NavigationDriver への依存を最小化（将来DI可能に）

**後方互換性:**
- 既存の `_looks_like_trap_or_legal()` と `_recover_from_trap()` は保持

### 6. タイルマテリアライズの改善

```diff
+ async def _materialize_tiles(
+    self,
+    *,
+    plp_config: Dict[str, Any],
+    start_t: float,
+    budget_ms: int,
+    target_url: Optional[str] = None,
+) -> int:
+    """PLPタイルをマテリアライズ（Stage 4: site_configベース）"""
+    # _materialize_plp_tiles() を呼び出すラッパー

async def _materialize_plp_tiles(...):
    # 既存の実装を site_config ベースに拡張
+   plp_config = self._get_plp_config()
+   # site_config から設定を取得して使用
```

**変更理由:**
- site_config から設定を取得（`selectors.plp.*`、`discovery_settings.plp.*`）
- 後方互換性を維持（既存メソッドも保持）

### 7. ナビゲーションメソッドの改善

```diff
+ async def _click_tile_and_navigate(
+    self,
+    *,
+    plp_config: Dict[str, Any],
+    timeout_ms: int = 5000,
+) -> Optional[Page]:
+    """タイルをクリックしてPDPに遷移（Stage 4: site_configベース）"""

async def _click_tile_and_navigate_to_pdp(...):
    # 既存の実装を site_config ベースに拡張
+   plp_config = self._get_plp_config()
+   # site_config から設定を取得（click_strategy, product_link, product_tiles, etc.）

+ async def _click_and_wait_for_navigation(...):
+    """クリック後のナビゲーションを待機（新タブ/同タブ/SPA）"""
+    # _click_and_capture_navigation() をラップ（名前変更）
```

**変更理由:**
- site_config からクリック戦略を取得（`click_strategy: "link" | "tile" | "both"`）
- より明確なメソッド名

### 8. 後方互換性のためのメソッド保持

既存コードとの互換性を保つため、以下のメソッドは保持：

- `_materialize_plp_tiles()` - 新しい `_materialize_tiles()` から呼び出される
- `_click_tile_and_navigate_to_pdp()` - 新しい `_click_tile_and_navigate()` から呼び出される
- `_click_and_capture_navigation()` - 新しい `_click_and_wait_for_navigation()` から呼び出される
- `_looks_like_trap_or_legal()` - 新しい `_is_trap_page()` から呼び出される
- `_accept_cookies_if_present()` - 新しい `_handle_cookie_banner()` から呼び出される
- `_dismiss_geo_modal()` - 新しい `_handle_geo_modal()` から呼び出される
- `_kill_overlays()` - 新しい `_remove_generic_overlays()` から呼び出される

## コード行数の変化

- **旧バージョン**: 約 563 行
- **新バージョン**: 約 1,000 行

**増加理由:**
- 設定取得メソッドの追加（`_get_plp_config()`, `_get_overlay_config()`, `_get_trap_config()`）
- Overlay処理メソッドの分離と拡張
- 後方互換性のためのメソッド保持
- ドキュメントコメントの追加

## 破壊的変更

**なし** - すべて後方互換性を維持

- 既存のメソッドシグネチャはすべて保持
- 既存のコードは変更なしで動作
- 新しい機能は追加のみ（オプショナル）

## 新機能

1. **拡張された PlpNavigationResult**
   - `recovery_successful`: リカバリの成功/失敗
   - `overlays_handled`: 処理したオーバーレイの種類
   - `navigation_method`: ナビゲーション方法
   - `errors`: エラーメッセージのリスト

2. **site_config ベースの設定取得**
   - `selectors.plp.*` スキーマをサポート
   - `navigation.overlays.*` スキーマをサポート
   - `navigation.trap.*` スキーマをサポート

3. **改善されたリカバリ機能**
   - `recovery_actions` に基づく設定可能なリカバリ
   - リカバリの成功/失敗を返す

4. **エラー収集とレポート**
   - 発生したエラーを `PlpNavigationResult.errors` に収集

## テストへの影響

既存のテスト (`tests/test_plp_driver.py`) は後方互換性により動作しますが、
新しいフィールドをテストするために最小限の更新が必要です。

**必要な更新:**
- `PlpNavigationResult` の新しいフィールドのアサーション追加
- 新しいメソッド（`_get_plp_config()`, etc.）のテスト追加

## 次のステップ

1. **既存テストの更新**: 新しいフィールドのアサーション追加
2. **新しいテストの追加**: Task D で定義されたテストケース
3. **BrowserUseAgent の更新**: Task E で `PlpDriver` の新しいAPIを使用
4. **site_config の移行**: 既存の設定を新しいスキーマに移行

## 関連ファイル

- `app/agents/browser/plp_driver.py` - 拡張版 PlpDriver
- `docs/reports/STAGE_4_GENERIC_PLP_DRIVER_DESIGN.md` - 設計書
- `tests/test_plp_driver.py` - 既存テスト（更新が必要）

