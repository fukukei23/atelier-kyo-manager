# Stage 3B & 3C 実装計画書

## 概要

Stage 3B（TelemetryService）と Stage 3C（Plugin API）の実装を段階的に進めるための詳細な計画書です。

---

## Stage 3B: TelemetryService 実装計画

### 実装ステップ

#### Step 1: TelemetryService クラスの骨組み作成

**ファイル**: `app/agents/browser/telemetry.py`（新規作成）

**実装内容**:
1. `RunPhase` Enum の定義
2. `FailureContext` データクラスの定義
3. `TelemetryService` クラスの骨組み（メソッドはスタブ）

**パッチサイズ**: 小（約100行）

**リスク**: 低（新規ファイル作成のみ）

---

#### Step 2: 内部メソッドの実装（observability.py から移行）

**実装内容**:
1. `_save_dom` メソッドの実装（`observability.save_dom` から移行）
2. `_save_json` メソッドの実装（`observability.save_json` から移行）
3. `_count_selectors` メソッドの実装（`observability.count_selectors` から移行）
4. `_write_fail_snapshot` メソッドの実装（`observability.write_fail_snapshot` から移行）

**パッチサイズ**: 中（約150行）

**リスク**: 低（既存コードの移動のみ）

**テスト**: 各メソッドの単体テストを作成

---

#### Step 3: 公開メソッドの実装

**実装内容**:
1. `record_plp_state` メソッドの実装
2. `record_success` メソッドの実装
3. `record_failure` メソッドの実装

**パッチサイズ**: 中（約200行）

**リスク**: 中（新しいロジックの追加）

**テスト**: 各メソッドの統合テストを作成

---

#### Step 4: BrowserUseAgent への統合（段階的）

**Step 4-1: TelemetryService インスタンスの追加**

**変更ファイル**: `app/agents/browser_use_agent.py`

**変更内容**:
```python
# __init__ メソッドには追加しない（run_context が run メソッドで利用可能なため）
# run メソッド内で TelemetryService を作成

async def run(self, *, site: str, query: str, site_config: Dict[str, Any], 
              run_context: RunContext, target_url: str, likely_plp: bool) -> DiscoveryResult:
    from app.agents.browser.telemetry import TelemetryService
    
    # TelemetryService の作成（run_context が必要なため、ここで作成）
    self.telemetry = TelemetryService(
        run_context=run_context,
        logger=self.logger,
    )
    
    # ... 既存のコード ...
```

**パッチサイズ**: 小（約10行）

**リスク**: 低（追加のみ）

---

**Step 4-2: save_dom 呼び出しの置き換え（1箇所ずつ）**

**変更ファイル**: `app/agents/browser_use_agent.py`

**変更内容**:
```python
# 変更前
await save_dom(run_context, page, "plp_dom_initial_materialized")

# 変更後
await self.telemetry.record_plp_state(
    page,
    name="plp_dom_initial_materialized",
    selectors=[...],  # 必要に応じて
)
```

**対象箇所と優先順位**:
1. `_run_plp_flow` 内（優先度: 高）
   - `plp_dom_initial_materialized` (1919行目付近)
   - `plp_dom_search_fallback` (1933行目付近)
2. `_run_pdp_flow` 内（優先度: 中）
   - `pdp_dom` (extractor.py 内、120行目付近)
3. `_run_learning_flow` 内（優先度: 低）
   - `learn_plp_dom_for_discovery` (2246行目付近)

**合計**: 約9箇所（1箇所ずつ置き換え、動作確認後次へ）

**パッチサイズ**: 小（1箇所ずつ、約10行/箇所）

**リスク**: 低（1箇所ずつ置き換え、動作確認後次へ）

**テスト**: 各置き換え後に動作確認

---

**Step 4-3: _handle_run_failure の置き換え**

**変更ファイル**: `app/agents/browser_use_agent.py`

**変更内容**:
```python
# 変更前
async def _handle_run_failure(self, e: Exception, site: str, query: str, site_config: Dict,
                              run_context: RunContext, page: Optional[Page]) -> DiscoveryResult:
    # ... 既存の実装 ...
    await write_fail_snapshot(run_context, active_page, final_url_on_fail, e, site_config)
    # ...

# 変更後
async def _handle_run_failure(self, e: Exception, site: str, query: str, site_config: Dict,
                              run_context: RunContext, page: Optional[Page]) -> DiscoveryResult:
    from app.agents.browser.telemetry import RunPhase, FailureContext
    
    failure = FailureContext(
        site_code=site,
        url=final_url_on_fail or (page.url if page else "unknown"),
        phase=RunPhase.PLP_DISCOVERY,  # 適切なフェーズを設定
        exception=e,
        query=query,
        site_config=site_config,
    )
    
    await self.telemetry.record_failure(failure, page=active_page)
    
    # ... 残りの処理（DiscoveryResult生成など） ...
```

**パッチサイズ**: 中（約50行）

**リスク**: 中（失敗処理の変更）

**テスト**: 失敗シナリオのテストを実行

---

#### Step 5: NavigationDriver への統合

**変更ファイル**: `app/agents/browser/navigation_driver.py`

**変更内容**:
```python
# __init__ メソッドに telemetry パラメータを追加（既に存在する場合は更新）
def __init__(
    self,
    page: Page,
    *,
    ensure_plp_materialized: EnsurePlpMaterializedFn,
    trap_checker: Optional[TrapCheckerFn] = None,
    recovery_fn: Optional[RecoveryFn] = None,
    telemetry: Optional[Any] = None,  # TelemetryService 型に更新
    strategy: Any = None,
) -> None:
    # ... 既存のコード ...
    self.telemetry = telemetry

# run_plp_flow メソッド内で使用
async def run_plp_flow(self, ctx: NavigationContext, *, target_url: Optional[str] = None) -> NavigationOutcome:
    # ... 既存のコード ...
    
    # PLP状態の記録
    if self.telemetry:
        await self.telemetry.record_plp_state(
            self.page,
            name="plp_dom_initial",
            selectors=[...],  # site_configから取得
        )
    
    # ...
```

**パッチサイズ**: 小（約20行）

**リスク**: 低（追加のみ）

---

#### Step 6: 既存コードのクリーンアップ（オプショナル）

**変更ファイル**: `app/utils/observability.py`

**変更内容**:
```python
# 非推奨警告を追加
import warnings

async def save_dom(run_context: "RunContext", page: "Page", name: str):
    """
    Saves the current DOM of the page as an HTML file.
    
    .. deprecated:: 3B
        Use :class:`TelemetryService.record_plp_state` instead.
    """
    warnings.warn(
        "save_dom is deprecated. Use TelemetryService.record_plp_state instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... 既存の実装（後方互換性のため残す） ...
```

**パッチサイズ**: 小（各関数に警告を追加）

**リスク**: 低（警告追加のみ）

---

## Stage 3C: Plugin API 実装計画

### 実装ステップ

#### Step 1: BrowserRuntime クラスの骨組み作成

**ファイル**: `app/agents/browser/plugin_api.py`（新規作成）

**実装内容**:
1. `BrowserRuntime` クラスの骨組み
2. 基本的なプロパティ（`page`, `context`）
3. 既存メソッドへの委譲メソッド（スタブ）

**パッチサイズ**: 小（約100行）

**リスク**: 低（新規ファイル作成のみ）

---

#### Step 2: BrowserRuntime の基本メソッド実装

**実装内容**:
1. `save_dom` メソッド（TelemetryService経由、または既存実装に委譲）
2. `take_screenshot` メソッド
3. `goto` メソッド
4. `wait_for_timeout` メソッド
5. `locator` メソッド

**パッチサイズ**: 中（約150行）

**リスク**: 低（既存実装への委譲）

---

#### Step 3: StrategyPlugin インターフェースの更新

**変更ファイル**: `app/agents/plugins/base.py`

**変更内容**:
```python
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.browser.plugin_api import BrowserRuntime
    from app.agents.browser.navigation_driver import NavigationContext
    from app.models.result_models import DiscoveryResult


class StrategyPlugin(Protocol):
    """サイト別のスクレイピング戦略を差し替えるための基底クラス。"""
    site: str = ""
    
    # --- 新しいインターフェース（推奨） ---
    async def run(
        self,
        runtime: "BrowserRuntime",
        ctx: "NavigationContext",
    ) -> "DiscoveryResult":
        """
        新しい統一インターフェース
        
        段階的移行のため、既存メソッドのラッパーとして実装可能。
        デフォルト実装では既存の before_navigate, after_navigate などを呼び出す。
        """
        # デフォルト実装（既存メソッドを呼び出す）
        url = self.before_navigate(ctx.entry_url or runtime.page.url, ctx.site_config)
        await runtime.goto(url)
        await self.after_navigate(runtime.page, ctx.site_config)
        # ... 既存のロジック ...
        raise NotImplementedError("Subclasses should implement run() or override existing methods")
    
    # --- 既存のフック群（後方互換性のため残す） ---
    def before_navigate(self, url: str, ctx) -> str:
        """navigate前にURLを安全に補正するフック。"""
        return url
    
    # ... 既存のメソッド ...
```

**パッチサイズ**: 中（約100行）

**リスク**: 中（インターフェース変更）

---

#### Step 4: BrowserUseAgent の更新

**Step 4-1: BrowserRuntime インスタンスの作成**

**変更ファイル**: `app/agents/browser_use_agent.py`

**変更内容**:
```python
from app.agents.browser.plugin_api import BrowserRuntime

# run メソッド内で BrowserRuntime を作成
async def run(self, *, site: str, query: str, site_config: Dict[str, Any], 
              run_context: RunContext, target_url: str, likely_plp: bool) -> DiscoveryResult:
    # ... 既存のコード ...
    
    # BrowserRuntime の作成
    runtime = BrowserRuntime(
        session=session,  # SessionManager インスタンス
        navigation=navigation_driver,  # NavigationDriver インスタンス
        extractor=self.extractor,  # BrowserExtractionService インスタンス
        telemetry=self.telemetry,  # TelemetryService インスタンス
        browser_agent=self,  # 段階的移行のため
        page=page,
        context=context,
        run_context=run_context,
    )
    
    # プラグイン呼び出し時に runtime を渡す
    if plugin:
        # 既存の before_navigate はそのまま（後方互換性）
        nav_url = plugin.before_navigate(target_url, plugin_ctx) or target_url
        
        # 新しい run メソッドがあれば使用（段階的移行）
        if hasattr(plugin, "run") and callable(getattr(plugin, "run")):
            try:
                nav_ctx = NavigationContext(...)
                result = await plugin.run(runtime, nav_ctx)
                # ... 結果の処理 ...
            except NotImplementedError:
                # run メソッドが実装されていない場合は既存の方法を使用
                pass
```

**パッチサイズ**: 中（約50行）

**リスク**: 中（プラグイン呼び出しの変更）

---

#### Step 5: MonclerPLPStrategy の段階的移行

**変更ファイル**: `app/agents/plugins/moncler_plp_v1.py`

**変更内容**:
```python
# 既存のメソッドはそのまま残す（後方互換性）

# 新しい run メソッドを追加（段階的移行）
async def run(
    self,
    runtime: "BrowserRuntime",
    ctx: "NavigationContext",
) -> "DiscoveryResult":
    """
    新しい統一インターフェース
    
    既存の before_navigate, after_navigate などを内部で呼び出す。
    """
    # before_navigate を呼び出し
    url = self.before_navigate(ctx.entry_url or runtime.page.url, ctx.site_config)
    await runtime.goto(url)
    
    # after_navigate を呼び出し
    await self.after_navigate(runtime.page, ctx.site_config)
    
    # assert_plp を呼び出し
    is_plp = await self.assert_plp(runtime.page, ctx.site_config)
    if not is_plp:
        raise ValueError("Not a PLP page")
    
    # materialize を呼び出し
    materialized = await self.materialize(runtime.page, ctx.site_config)
    if not materialized:
        raise ValueError("PLP materialization failed")
    
    # ... 既存のロジックを runtime 経由で実行 ...
    
    # DiscoveryResult を返す
    return DiscoveryResult(...)
```

**パッチサイズ**: 大（約200行）

**リスク**: 高（プラグインの主要ロジック変更）

**テスト**: プラグインの動作確認を徹底的に実施

---

## 実装の優先順位とタイムライン

### Phase 1: Stage 3B（1-2週間）

1. **Week 1**: Step 1-3（TelemetryService クラスの実装）
2. **Week 2**: Step 4-5（BrowserUseAgent/NavigationDriver への統合）

### Phase 2: Stage 3C（2-3週間）

1. **Week 3**: Step 1-3（BrowserRuntime と StrategyPlugin インターフェース）
2. **Week 4**: Step 4（BrowserUseAgent の更新）
3. **Week 5**: Step 5（MonclerPLPStrategy の移行、テスト）

---

## リスク管理

### Stage 3B のリスク

1. **既存の `observability.py` への依存が広範囲**
   - **対策**: 段階的移行、後方互換性を維持
   - **検証**: 各ステップで動作確認

2. **`_handle_run_failure` の変更による影響**
   - **対策**: 既存の動作を維持しつつ、TelemetryService 経由に変更
   - **検証**: 失敗シナリオのテストを実行

### Stage 3C のリスク

1. **プラグインが既存メソッドに直接依存している**
   - **対策**: 初期は委譲パターンで既存実装を呼び出す
   - **検証**: 既存プラグインの動作確認

2. **`MonclerPLPStrategy` の移行が複雑**
   - **対策**: 段階的移行、既存メソッドを残す
   - **検証**: プラグインの動作確認を徹底的に実施

---

## テスト計画

### Stage 3B のテスト

1. **単体テスト**:
   - `TelemetryService.record_plp_state`
   - `TelemetryService.record_success`
   - `TelemetryService.record_failure`
   - 各内部メソッド（`_save_dom`, `_save_json`, etc.）

2. **統合テスト**:
   - BrowserUseAgent との統合
   - NavigationDriver との統合
   - 失敗シナリオのテスト

### Stage 3C のテスト

1. **単体テスト**:
   - `BrowserRuntime` の各メソッド
   - `StrategyPlugin` インターフェース

2. **統合テスト**:
   - BrowserUseAgent との統合
   - MonclerPLPStrategy の動作確認
   - 既存プラグインの後方互換性確認

---

## 次のアクション

1. **Stage 3B Step 1 の実装開始**: TelemetryService クラスの骨組み作成
2. **設計レビュー**: この計画書のレビューと承認
3. **テスト計画の詳細化**: 各ステップのテストケースを詳細化

