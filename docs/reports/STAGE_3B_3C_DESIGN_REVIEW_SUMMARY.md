# Stage 3B & 3C 設計レビュー サマリー

## レビュー結果

### ✅ 総合評価: **承認（条件付き）**

設計案は全体的に良好ですが、以下の修正が必要です。

---

## 必須修正項目

### 1. ✅ `compare_and_maybe_update` の扱いを明確化
- **修正済み**: VRT用の関数として設計案から除外
- **理由**: TelemetryService の責務外

### 2. ✅ `_write_artifacts` の確認と削除
- **修正済み**: 実装が存在しないため、設計案から削除

### 3. ✅ `time` モジュールのインポート追加
- **修正済み**: `TelemetryService.record_success` で使用するため追加

### 4. ✅ `record_plp_state` に `site_config` パラメータ追加
- **修正済み**: `selectors` の自動取得を可能にするため追加

### 5. ✅ `BrowserRuntime.page` のエラーハンドリング強化
- **修正済み**: `None` の場合に `ValueError` を投げるように変更

### 6. ✅ `StrategyPlugin.run` のデフォルト実装改善
- **修正済み**: 既存メソッドを呼び出すデフォルト実装を追加

### 7. ✅ 実装計画の詳細化
- **修正済み**: 具体的な変更箇所をリスト化

---

## 修正内容の詳細

### Stage 3B: TelemetryService

#### 修正1: 移行対象メソッドの明確化
- `compare_and_maybe_update` を除外（VRT用）
- `_write_artifacts` を除外（実装が存在しない）

#### 修正2: `record_plp_state` の改善
```python
async def record_plp_state(
    self,
    page: "Page",
    *,
    name: str = "plp_dom_initial",
    selectors: Optional[List[str]] = None,
    site_config: Optional[Dict[str, Any]] = None,  # 追加
) -> None:
    # selectors が None で site_config がある場合、自動取得
    if selectors is None and site_config:
        pdp_cfg = (site_config.get("selectors") or {}).get("pdp", {}) or {}
        selectors = (
            (pdp_cfg.get("pdp_link_selectors") or []) +
            (pdp_cfg.get("plp_container_selectors") or [])
        )
```

#### 修正3: `time` モジュールのインポート
```python
import time  # record_success で使用
```

### Stage 3C: Plugin API

#### 修正1: `BrowserRuntime.page` の強化
```python
@property
def page(self) -> "Page":
    """現在のPageオブジェクトを取得（必須）"""
    if self.session and hasattr(self.session, "page") and self.session.page:
        return self.session.page
    if self._page:
        return self._page
    raise ValueError("Page is not available in BrowserRuntime")
```

#### 修正2: `StrategyPlugin.run` のデフォルト実装
```python
async def run(
    self,
    runtime: "BrowserRuntime",
    ctx: "NavigationContext",
) -> "DiscoveryResult":
    # デフォルト実装（既存メソッドを呼び出す）
    url = self.before_navigate(ctx.entry_url or runtime.page.url, ctx.site_config)
    await runtime.goto(url)
    await self.after_navigate(runtime.page, ctx.site_config)
    # ...
    raise NotImplementedError(...)
```

### 実装計画

#### 修正1: TelemetryService の初期化タイミング
- `BrowserUseAgent.__init__` ではなく、`run` メソッド内で作成
- `run_context` が必要なため

#### 修正2: 具体的な変更箇所のリスト化
- 各 `save_dom` 呼び出し箇所を具体的にリスト化
- 優先順位を明確化

---

## 承認条件

以下の修正が完了したため、**設計案を承認**します：

1. ✅ `compare_and_maybe_update` を設計案から除外
2. ✅ `_write_artifacts` を設計案から削除
3. ✅ `time` モジュールのインポート追加
4. ✅ `record_plp_state` に `site_config` パラメータ追加
5. ✅ `BrowserRuntime.page` のエラーハンドリング強化
6. ✅ `StrategyPlugin.run` のデフォルト実装改善
7. ✅ 実装計画の詳細化

---

## 次のステップ

1. ✅ **設計案の修正完了**: 上記の指摘事項を反映済み
2. ⏳ **実装開始**: Stage 3B Step 1（TelemetryService クラスの骨組み作成）から開始

---

## 追加の推奨事項

### テスト計画の詳細化（オプショナル）

以下のテストケースを追加することを推奨します：

#### Stage 3B のテスト
1. **単体テスト**:
   - `TelemetryService.record_plp_state` の正常系・異常系
   - `TelemetryService.record_success` の正常系・異常系
   - `TelemetryService.record_failure` の正常系・異常系
   - 各内部メソッド（`_save_dom`, `_save_json`, etc.）

2. **統合テスト**:
   - BrowserUseAgent との統合
   - NavigationDriver との統合
   - 失敗シナリオのテスト

#### Stage 3C のテスト
1. **単体テスト**:
   - `BrowserRuntime` の各メソッド
   - `StrategyPlugin` インターフェース

2. **統合テスト**:
   - BrowserUseAgent との統合
   - MonclerPLPStrategy の動作確認
   - 既存プラグインの後方互換性確認

---

## まとめ

設計案は**承認**されました。修正内容を反映した設計案に基づいて、実装を開始できます。

**実装の優先順位**:
1. Stage 3B: TelemetryService（1-2週間）
2. Stage 3C: Plugin API（2-3週間）

**リスク管理**:
- 段階的移行アプローチにより、リスクを最小化
- 後方互換性を維持
- 各ステップで動作確認を実施

