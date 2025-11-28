# Stage 3B & 3C 設計レビュー

## レビュー日時
2025-01-XX

## レビュー対象
- `STAGE_3B_3C_DESIGN_PROPOSAL.md`
- `STAGE_3B_3C_IMPLEMENTATION_PLAN.md`

---

## ✅ 承認事項

### 1. 全体的なアプローチ
- **段階的移行**: リスクを最小化する段階的なアプローチは適切
- **後方互換性**: 既存コードとの後方互換性を維持する方針は正しい
- **責務の分離**: TelemetryService と BrowserRuntime の責務が明確

### 2. Stage 3B: TelemetryService 設計
- **RunPhase Enum**: 実行フェーズを明確に管理する設計は良い
- **FailureContext**: 失敗情報を構造化する設計は適切
- **メソッド設計**: `record_plp_state`, `record_success`, `record_failure` の3つの公開メソッドはシンプルで良い

### 3. Stage 3C: Plugin API 設計
- **BrowserRuntime Facade**: プラグインから見える最小限のインターフェースは適切
- **段階的移行**: 初期は委譲パターンで既存実装を呼び出す方針は安全

---

## ⚠️ 指摘事項と改善提案

### 1. Stage 3B: TelemetryService の改善点

#### 1.1 `compare_and_maybe_update` の扱い

**問題**:
- `compare_and_maybe_update` は `app/utils/visual_regression.py` に定義されている
- VRT（Visual Regression Test）用の関数で、TelemetryService の責務外の可能性がある

**提案**:
- **Option A**: TelemetryService に含めない（VRTは別の責務として扱う）
- **Option B**: TelemetryService に `record_vrt` メソッドを追加し、内部で `compare_and_maybe_update` を呼び出す

**推奨**: **Option A**（VRTは別の責務として扱う）

**修正案**:
```python
# TelemetryService には含めない
# VRT は別途 visual_regression.py で管理
# 必要に応じて TelemetryService から呼び出すことは可能
```

---

#### 1.2 `_write_artifacts` の扱い

**問題**:
- `_write_artifacts` メソッドが `browser_use_agent.py` 内に見つからない
- 設計案に含まれているが、実装が存在しない可能性

**提案**:
- 実装を確認し、存在しない場合は設計案から削除
- 存在する場合は、`record_failure` 内で処理する

**確認結果**: `_write_artifacts` は見つからなかった。設計案から削除を推奨。

---

#### 1.3 `record_plp_state` の引数設計

**問題**:
- `selectors` パラメータが Optional だが、実際の使用箇所では常に指定される可能性が高い

**提案**:
- `selectors` を必須にするか、`site_config` から自動取得するオプションを追加

**修正案**:
```python
async def record_plp_state(
    self,
    page: "Page",
    *,
    name: str = "plp_dom_initial",
    selectors: Optional[List[str]] = None,
    site_config: Optional[Dict[str, Any]] = None,  # 追加
) -> None:
    """
    PLP ロード直後の DOM/スクショ保存
    
    Args:
        page: Playwright Page オブジェクト
        name: 保存ファイル名のベース（拡張子は自動付与）
        selectors: セレクタカウント対象（オプション）
        site_config: サイト設定（selectors が None の場合、ここから取得）
    """
    # selectors が None で site_config がある場合、自動取得
    if selectors is None and site_config:
        pdp_cfg = (site_config.get("selectors") or {}).get("pdp", {}) or {}
        selectors = (
            (pdp_cfg.get("pdp_link_selectors") or []) +
            (pdp_cfg.get("plp_container_selectors") or [])
        )
    
    # ... 既存の処理 ...
```

---

#### 1.4 `record_success` の `time` モジュールのインポート

**問題**:
- `record_success` メソッドで `time.time()` を使用しているが、`time` モジュールのインポートが設計案に含まれていない

**修正案**:
```python
import time  # 追加

async def record_success(...):
    try:
        record = {
            "phase": phase.value,
            "timestamp": time.time(),  # time モジュールが必要
            # ...
        }
```

---

#### 1.5 `RunContext` の `save_content` メソッドの扱い

**確認結果**:
- `RunContext.save_content` は同期的なメソッド（`async` ではない）
- `TelemetryService._maybe_await` で対応する設計は適切

**追加確認事項**:
- `RunContext.save_json` も同期的なメソッドの可能性がある
- `_maybe_await` で対応する設計で問題ない

---

### 2. Stage 3C: Plugin API の改善点

#### 2.1 `BrowserRuntime.page` プロパティの設計

**問題**:
- `BrowserRuntime.page` が `Optional[Page]` を返すが、プラグインが `None` を想定していない可能性がある

**提案**:
- `page` が `None` の場合のエラーハンドリングを明確にする
- または、`page` が必須であることを型で示す

**修正案**:
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

---

#### 2.2 `StrategyPlugin.run` メソッドのデフォルト実装

**問題**:
- 設計案では `StrategyPlugin.run` が `NotImplementedError` を投げるが、既存メソッドのラッパーとして実装可能と記載されている

**提案**:
- デフォルト実装を提供し、既存メソッドを呼び出す形にする

**修正案**:
```python
class StrategyPlugin(Protocol):
    # ... 既存のコード ...
    
    async def run(
        self,
        runtime: "BrowserRuntime",
        ctx: "NavigationContext",
    ) -> "DiscoveryResult":
        """
        新しい統一インターフェース
        
        デフォルト実装では既存の before_navigate, after_navigate などを呼び出す。
        サブクラスでオーバーライド可能。
        """
        # デフォルト実装（既存メソッドを呼び出す）
        url = self.before_navigate(ctx.entry_url or runtime.page.url, ctx.site_config)
        await runtime.goto(url)
        await self.after_navigate(runtime.page, ctx.site_config)
        
        # assert_plp と materialize は NavigationDriver が担当するため、
        # ここでは呼び出さない
        
        # デフォルトでは NotImplementedError を投げる
        # サブクラスで実装するか、既存メソッドを使用する
        raise NotImplementedError(
            "Subclasses should implement run() or use existing methods (before_navigate, after_navigate, etc.)"
        )
```

---

#### 2.3 `BrowserRuntime` の初期化タイミング

**問題**:
- `BrowserUseAgent.run` メソッド内で `BrowserRuntime` を作成する設計だが、`run_context` が `__init__` で利用できない可能性がある

**確認結果**:
- `TelemetryService` は `run_context` を必要とする
- `BrowserUseAgent.run` メソッド内で `TelemetryService` を作成する必要がある

**修正案**:
```python
# BrowserUseAgent.run メソッド内
async def run(self, *, site: str, query: str, site_config: Dict[str, Any], 
              run_context: RunContext, target_url: str, likely_plp: bool) -> DiscoveryResult:
    # ... 既存のコード ...
    
    # TelemetryService の作成（run_context が必要なため、ここで作成）
    from app.agents.browser.telemetry import TelemetryService
    telemetry = TelemetryService(run_context=run_context, logger=self.logger)
    
    # BrowserRuntime の作成
    runtime = BrowserRuntime(
        session=session,
        navigation=navigation_driver,
        extractor=self.extractor,
        telemetry=telemetry,  # ここで注入
        browser_agent=self,
        page=page,
        context=context,
        run_context=run_context,
    )
```

---

#### 2.4 `MonclerPLPStrategy` の移行戦略

**問題**:
- `MonclerPLPStrategy` は `page` を直接使用している箇所が多い
- 新しい `run` メソッドへの移行が複雑

**提案**:
- 段階的移行戦略を明確にする
- 既存メソッド（`before_navigate`, `after_navigate`, etc.）を残しつつ、新しい `run` メソッドを追加

**修正案**:
```python
# MonclerPLPStrategy の移行戦略

# Phase 1: 既存メソッドはそのまま（後方互換性）
# Phase 2: 新しい run メソッドを追加（既存メソッドを内部で呼び出す）
# Phase 3: 段階的に runtime 経由の呼び出しに移行
```

---

### 3. 実装計画の改善点

#### 3.1 Step 4-2 の詳細化

**問題**:
- `save_dom` 呼び出しの置き換えが9箇所と多いが、1箇所ずつ置き換える方針は適切

**追加提案**:
- 各置き換え箇所のリストを明確にする
- 置き換え順序を優先度順に整理

**修正案**:
```markdown
**対象箇所と優先順位**:
1. `_run_plp_flow` 内（優先度: 高）
   - `plp_dom_initial_materialized` (1919行目)
   - `plp_dom_search_fallback` (1933行目)
2. `_run_pdp_flow` 内（優先度: 中）
   - `pdp_dom` (extractor.py 内)
3. `_run_learning_flow` 内（優先度: 低）
   - `learn_plp_dom_for_discovery` (2246行目)
```

---

#### 3.2 `compare_and_maybe_update` の扱い

**問題**:
- 設計案に `compare_and_maybe_update` が含まれているが、VRTは別の責務

**提案**:
- 設計案から削除するか、別途扱うことを明記

**修正案**:
```markdown
### 移行対象メソッド（修正版）

1. `save_dom` (app/utils/observability.py) ✅
2. `write_fail_snapshot` (app/utils/observability.py) ✅
3. `_handle_run_failure` (browser_use_agent.py) ✅
4. `count_selectors` (app/utils/observability.py) ✅
5. `save_json` (app/utils/observability.py) ✅
6. `save_raw_hrefs` (app/utils/observability.py) ✅

**除外**:
- `compare_and_maybe_update` (app/utils/visual_regression.py)
  - VRT（Visual Regression Test）用の関数で、TelemetryService の責務外
  - 必要に応じて TelemetryService から呼び出すことは可能だが、直接移行はしない
```

---

#### 3.3 テスト計画の詳細化

**問題**:
- テスト計画が抽象的

**提案**:
- 具体的なテストケースを追加

**修正案**:
```markdown
### Stage 3B のテスト（詳細）

#### 単体テスト
1. `TelemetryService.record_plp_state`
   - 正常系: DOM保存、セレクタカウント、スクショ取得
   - 異常系: page が None、page が closed
   - エッジケース: selectors が空リスト

2. `TelemetryService.record_success`
   - 正常系: result あり、result なし、metadata あり
   - 異常系: run_context が None

3. `TelemetryService.record_failure`
   - 正常系: page あり、page なし、site_config あり
   - 異常系: failure が None、page が closed

#### 統合テスト
1. BrowserUseAgent との統合
   - `save_dom` 呼び出しの置き換え後の動作確認
   - `_handle_run_failure` の置き換え後の動作確認

2. NavigationDriver との統合
   - `record_plp_state` の呼び出し確認
```

---

## 📋 修正が必要な項目

### 必須修正

1. **`compare_and_maybe_update` の扱いを明確化**
   - 設計案から削除するか、別途扱うことを明記

2. **`_write_artifacts` の確認**
   - 実装が存在しない場合は設計案から削除

3. **`time` モジュールのインポート追加**
   - `TelemetryService.record_success` で使用

4. **`record_plp_state` の `site_config` パラメータ追加**
   - `selectors` の自動取得を可能にする

### 推奨修正

1. **`BrowserRuntime.page` のエラーハンドリング強化**
   - `None` の場合のエラーを明確にする

2. **`StrategyPlugin.run` のデフォルト実装改善**
   - 既存メソッドを呼び出すデフォルト実装を提供

3. **実装計画の詳細化**
   - 各ステップの具体的な変更箇所をリスト化

---

## ✅ 最終評価

### 承認条件

以下の修正を行えば、設計案を承認します：

1. ✅ `compare_and_maybe_update` を設計案から削除（または別途扱うことを明記）
2. ✅ `_write_artifacts` の確認と削除（存在しない場合）
3. ✅ `time` モジュールのインポート追加
4. ✅ `record_plp_state` に `site_config` パラメータを追加
5. ✅ `BrowserRuntime.page` のエラーハンドリング強化
6. ✅ 実装計画の詳細化（具体的な変更箇所のリスト化）

### 総合評価

**評価**: ⭐⭐⭐⭐ (4/5)

**良い点**:
- 段階的移行アプローチが適切
- 責務の分離が明確
- 後方互換性を考慮している

**改善点**:
- 一部の実装詳細が不明確
- テスト計画が抽象的
- VRTの扱いが不明確

**結論**: 上記の修正を行えば、実装を開始して問題ありません。

---

## 次のステップ

1. **設計案の修正**: 上記の指摘事項を反映
2. **修正版のレビュー**: 修正後の設計案を再レビュー
3. **実装開始**: 修正が承認されたら実装を開始

