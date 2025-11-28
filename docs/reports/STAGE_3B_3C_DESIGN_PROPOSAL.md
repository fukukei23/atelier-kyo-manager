# Stage 3B & 3C 設計提案書

## 概要

Stage 3A（NavigationDriver）の移行が完了したため、次は Stage 3B（TelemetryService）と Stage 3C（Plugin API）の設計と実装計画を提案します。

---

## Stage 3B: TelemetryService 抽出

### 責務

- **DOM保存・スクリーンショット・失敗時アーティファクト生成**
- **「いつ」「どの段階で」「何が起きたか」を記録**
- **ビジネスロジック（どの PDP を捨てるか等）は持たない**

### 移行対象メソッド

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
- `_write_artifacts` (browser_use_agent.py)
  - 実装が存在しないため、設計案から除外

### クラス設計

#### 1. RunPhase Enum

```python
# app/agents/browser/telemetry.py
from enum import Enum

class RunPhase(str, Enum):
    """実行フェーズを表すEnum"""
    PLP_DISCOVERY = "plp_discovery"      # PLP探索中
    PDP_EXTRACT = "pdp_extract"          # PDP抽出中
    RECOVERY = "recovery"                 # 回復試行中
    LEARNING = "learning"                 # 学習モード
    MATERIALIZE = "materialize"           # PLPマテリアライズ中
```

#### 2. FailureContext データクラス

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class FailureContext:
    """失敗時のコンテキスト情報"""
    site_code: str
    url: str
    phase: RunPhase
    exception: Optional[Exception] = None
    retry_count: int = 0
    query: Optional[str] = None
    site_config: Optional[Dict[str, Any]] = None
    intent_description: Optional[str] = None
```

#### 3. TelemetryService クラス

```python
# app/agents/browser/telemetry.py
from __future__ import annotations

import logging
import time  # record_success で使用
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page
    from app.core.run_context import RunContext
    from app.models.result_models import DiscoveryResult


class RunPhase(str, Enum):
    """実行フェーズを表すEnum"""
    PLP_DISCOVERY = "plp_discovery"
    PDP_EXTRACT = "pdp_extract"
    RECOVERY = "recovery"
    LEARNING = "learning"
    MATERIALIZE = "materialize"


@dataclass
class FailureContext:
    """失敗時のコンテキスト情報"""
    site_code: str
    url: str
    phase: RunPhase
    exception: Optional[Exception] = None
    retry_count: int = 0
    query: Optional[str] = None
    site_config: Optional[Dict[str, Any]] = None
    intent_description: Optional[str] = None


class TelemetryService:
    """
    Stage 3B: 観測機能（Telemetry）を一元管理するサービス
    
    責務:
    - DOM保存・スクリーンショット・失敗時アーティファクト生成
    - 「いつ」「どの段階で」「何が起きたか」を記録
    - ビジネスロジックは持たない
    """
    
    def __init__(
        self,
        run_context: "RunContext",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.run_context = run_context
        self.logger = logger or logging.getLogger(__name__)
    
    async def record_plp_state(
        self,
        page: "Page",
        *,
        name: str = "plp_dom_initial",
        selectors: Optional[List[str]] = None,
        site_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        PLP ロード直後の DOM/スクショ保存
        
        Args:
            page: Playwright Page オブジェクト
            name: 保存ファイル名のベース（拡張子は自動付与）
            selectors: セレクタカウント対象（オプション）
            site_config: サイト設定（selectors が None の場合、ここから自動取得）
        """
        # selectors が None で site_config がある場合、自動取得
        if selectors is None and site_config:
            pdp_cfg = (site_config.get("selectors") or {}).get("pdp", {}) or {}
            selectors = (
                (pdp_cfg.get("pdp_link_selectors") or []) +
                (pdp_cfg.get("plp_container_selectors") or [])
            )
        if not page or page.is_closed():
            return
        
        try:
            # DOM保存
            await self._save_dom(page, name)
            
            # セレクタカウント（指定がある場合）
            if selectors:
                await self._count_selectors(page, selectors, name=f"selector_counts_{name}")
            
            # スクリーンショット（RunContextがサポートしている場合）
            if hasattr(self.run_context, "take_screenshot"):
                try:
                    await self.run_context.take_screenshot(page, f"30_{name}")
                except Exception as e:
                    self.logger.debug(f"[Telemetry] Screenshot failed for {name}: {e}")
        except Exception as e:
            self.logger.warning(f"[Telemetry] Failed to record PLP state '{name}': {e}")
    
    async def record_success(
        self,
        phase: RunPhase,
        *,
        result: Optional["DiscoveryResult"] = None,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        成功時のメタ情報や差分更新など
        
        Args:
            phase: 実行フェーズ
            result: DiscoveryResult（オプション）
            url: 成功時のURL（オプション）
            metadata: 追加メタデータ（オプション）
        """
        try:
            record = {
                "phase": phase.value,
                "timestamp": time.time(),
                "url": url,
            }
            
            if result:
                record["result"] = {
                    "ok": result.ok,
                    "site": result.site,
                    "query": result.query,
                    "message": result.message,
                }
            
            if metadata:
                record.update(metadata)
            
            await self._save_json(record, f"success_{phase.value}")
        except Exception as e:
            self.logger.warning(f"[Telemetry] Failed to record success for {phase}: {e}")
    
    async def record_failure(
        self,
        failure: FailureContext,
        *,
        page: Optional["Page"] = None,
    ) -> None:
        """
        失敗時の DOM / スクショ / ログ一括処理
        
        Args:
            failure: 失敗コンテキスト
            page: Playwright Page オブジェクト（オプション）
        """
        try:
            # DOM保存
            if page and not page.is_closed():
                await self._save_dom(page, "failure_dom")
            
            # スクリーンショット
            screenshot_path: Optional[str] = None
            if page and not page.is_closed() and hasattr(self.run_context, "take_screenshot"):
                try:
                    screenshot_path = await self.run_context.take_screenshot(page, "99_failure")
                except Exception as e:
                    self.logger.debug(f"[Telemetry] Failed to take failure screenshot: {e}")
            
            # セレクタカウント（site_configがある場合）
            visible_counts: Dict[str, int] = {}
            if page and not page.is_closed() and failure.site_config:
                try:
                    pdp_cfg = (failure.site_config.get("selectors") or {}).get("pdp", {}) or {}
                    markers = list(dict.fromkeys(
                        (pdp_cfg.get("title_selectors") or []) +
                        (pdp_cfg.get("price_selectors") or [])
                    ))
                    for s in markers:
                        try:
                            visible_counts[s] = await page.locator(s).count()
                        except Exception:
                            visible_counts[s] = -1
                except Exception as e:
                    self.logger.warning(f"[Telemetry] Could not count visible markers: {e}")
            
            # 失敗レポート生成
            await self._write_fail_snapshot(
                failure=failure,
                screenshot_path=screenshot_path,
                visible_counts=visible_counts,
            )
        except Exception as e:
            self.logger.error(f"[Telemetry] Failed to record failure: {e}", exc_info=True)
    
    # --- 内部メソッド（既存のobservability.pyから移行） ---
    
    async def _save_dom(self, page: "Page", name: str) -> None:
        """DOM保存の内部実装"""
        if not page or page.is_closed():
            return
        try:
            html = await page.content()
            await self._maybe_await(self.run_context.save_content(f"{name}.html", html))
        except Exception as e:
            self.logger.warning(f"[Telemetry] Failed to save DOM for '{name}': {e}")
    
    async def _save_json(self, data: Any, name: str) -> None:
        """JSON保存の内部実装"""
        try:
            await self._maybe_await(self.run_context.save_json(f"{name}.json", data))
        except Exception as e:
            self.logger.warning(f"[Telemetry] Failed to save JSON for '{name}': {e}")
    
    async def _count_selectors(
        self,
        page: "Page",
        selectors: List[str],
        *,
        name: str = "selector_counts",
    ) -> None:
        """セレクタカウントの内部実装"""
        if not page or page.is_closed():
            return
        counts: Dict[str, int] = {}
        for s in selectors or []:
            try:
                counts[s] = await page.locator(s).count()
            except Exception:
                counts[s] = -1
        await self._save_json({"selector_counts": counts}, name)
    
    async def _write_fail_snapshot(
        self,
        failure: FailureContext,
        *,
        screenshot_path: Optional[str] = None,
        visible_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """失敗スナップショットの生成"""
        import time
        import json
        
        note = f"Final URL: {failure.url}\nError: {failure.exception}"
        visible_counts = visible_counts or {}
        
        md_report = f"""# Failure Snapshot

- **Run ID:** `{getattr(self.run_context, 'run_id', 'unknown')}`
- **Timestamp:** `{time.strftime('%Y-%m-%d %H:%M:%S Z', time.gmtime())}`
- **Phase:** `{failure.phase.value}`
- **Site:** `{failure.site_code}`
- **Query:** `{failure.query or 'N/A'}`
- **Retry Count:** `{failure.retry_count}`
- **Note:** `{note}`

## Key Selector Counts at Failure
```json
{json.dumps(visible_counts, indent=2)}
```

## Available Artifacts
- `failure_dom.html` (if page was available)
- `{screenshot_path or '99_failure.png (if captured)'}`
- `plp_dom_initial_materialized.html`
- `plp_dom_search_fallback.html`
- `selector_counts_plp_initial.json`
- `selector_counts_after_search_fallback.json`
- `raw_hrefs_raw_abs.json`
- `network.har`
- `trace.zip`
"""
        await self._maybe_await(self.run_context.save_content("fail_snapshot.md", md_report))
    
    async def _maybe_await(self, x: Any) -> Any:
        """引数が Awaitable であれば await し、そうでなければそのまま返す"""
        import inspect
        return (await x) if inspect.isawaitable(x) else x
```

### 移行手順（Stage 3B）

#### Step 1: TelemetryService クラスの作成
1. `app/agents/browser/telemetry.py` を作成
2. `RunPhase`, `FailureContext`, `TelemetryService` を実装
3. 既存の `observability.py` の関数を内部メソッドとして移行

#### Step 2: BrowserUseAgent への統合
1. `BrowserUseAgent.__init__` に `TelemetryService` を追加
2. `save_dom`, `write_fail_snapshot` の呼び出しを `TelemetryService` に置き換え
3. `_handle_run_failure` を `TelemetryService.record_failure` に置き換え

#### Step 3: NavigationDriver への統合
1. `NavigationDriver` に `TelemetryService` を注入
2. PLP状態の記録を `TelemetryService.record_plp_state` に置き換え

#### Step 4: 既存コードのクリーンアップ
1. `observability.py` の関数を非推奨マーク（後方互換性のため）
2. 段階的に `TelemetryService` に移行

---

## Stage 3C: Plugin API / BrowserRuntime 整理

### 現状の問題

- `StrategyPlugin` が `BrowserUseAgent` の private メソッドに依存している
- 分離した瞬間に全部壊れるリスクがある

### 解決方針

**小さめの Facade をひとつだけ定義**

- `BrowserUseAgent` が Strategy に渡すのは、「全部入り self」ではなく最小限の "サービスセット"
- 初期段階では `BrowserRuntime` のメソッドが内部で既存 `BrowserUseAgent` の実装に委譲する形でも構わない

### クラス設計

#### 1. BrowserRuntime クラス

```python
# app/agents/browser/plugin_api.py
from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page, BrowserContext
    from app.agents.browser.session_manager import SessionManager
    from app.agents.browser.navigation_driver import NavigationDriver
    from app.agents.browser.extractor import BrowserExtractionService
    from app.agents.browser.telemetry import TelemetryService
    from app.core.run_context import RunContext


class BrowserRuntime:
    """
    Stage 3C: プラグインから見える最小限の顔
    
    プラグインが BrowserUseAgent の実装詳細に依存しないようにするための Facade。
    初期段階では既存の BrowserUseAgent のメソッドに委譲する形で実装。
    """
    
    def __init__(
        self,
        *,
        session: Optional["SessionManager"] = None,
        navigation: Optional["NavigationDriver"] = None,
        extractor: Optional["BrowserExtractionService"] = None,
        telemetry: Optional["TelemetryService"] = None,
        # 段階的移行のため、既存の BrowserUseAgent への参照も保持
        browser_agent: Optional[Any] = None,
        page: Optional["Page"] = None,
        context: Optional["BrowserContext"] = None,
        run_context: Optional["RunContext"] = None,
    ) -> None:
        self.session = session
        self.navigation = navigation
        self.extractor = extractor
        self.telemetry = telemetry
        
        # 段階的移行のため（後で削除予定）
        self._browser_agent = browser_agent
        self._page = page
        self._context = context
        self._run_context = run_context
    
    # --- プラグインから使えるメソッド（段階的に NavigationDriver/Extractor/Telemetry に移行） ---
    
    @property
    def page(self) -> "Page":
        """現在のPageオブジェクトを取得（必須）"""
        if self.session and hasattr(self.session, "page") and self.session.page:
            return self.session.page
        if self._page:
            return self._page
        raise ValueError("Page is not available in BrowserRuntime")
    
    @property
    def context(self) -> Optional["BrowserContext"]:
        """現在のBrowserContextを取得"""
        if self.session and hasattr(self.session, "context"):
            return self.session.context
        return self._context
    
    async def save_dom(self, name: str) -> None:
        """DOM保存（TelemetryService経由）"""
        if self.telemetry and self.page:
            await self.telemetry.record_plp_state(self.page, name=name)
        elif self._browser_agent:
            # 段階的移行: 既存の実装に委譲
            from app.utils.observability import save_dom
            await save_dom(self._run_context, self.page, name)
    
    async def take_screenshot(self, name: str) -> Optional[str]:
        """スクリーンショット取得"""
        if self._run_context and self.page and hasattr(self._run_context, "take_screenshot"):
            return await self._run_context.take_screenshot(self.page, name)
        return None
    
    async def goto(self, url: str, **kwargs) -> None:
        """ページ遷移（NavigationDriver経由、または直接）"""
        if self.page:
            await self.page.goto(url, **kwargs)
    
    async def wait_for_timeout(self, timeout: int) -> None:
        """タイムアウト待機"""
        if self.page:
            await self.page.wait_for_timeout(timeout)
    
    def locator(self, selector: str):
        """ロケーター取得"""
        if self.page:
            return self.page.locator(selector)
        raise ValueError("Page is not available")
```

#### 2. StrategyPlugin の更新

```python
# app/agents/plugins/base.py（更新版）
from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.browser.plugin_api import BrowserRuntime
    from app.agents.browser.navigation_driver import NavigationContext
    from app.models.result_models import DiscoveryResult


class StrategyPlugin(Protocol):
    """
    Stage 3C: プラグインの新しいインターフェース
    
    段階的移行のため、既存のメソッドも残すが、新しい `run` メソッドを推奨。
    """
    
    site: str = ""  # 例: "MONCLER_OFFICIAL"
    
    # --- 新しいインターフェース（推奨） ---
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
    
    # --- 既存のフック群（後方互換性のため残す） ---
    def before_navigate(self, url: str, ctx) -> str:
        """navigate前にURLを安全に補正するフック。"""
        return url
    
    async def after_navigate(self, page, ctx) -> None:
        """navigate直後にゲート/クッキー許諾などを処理するフック。"""
        return None
    
    async def assert_plp(self, page, ctx) -> bool:
        """ここがPLPかどうかの判定を強化したい場合。"""
        return True
    
    async def materialize(self, page, ctx) -> bool:
        """PLPのカードを表示させるためのスクロール等。成功ならTrue。"""
        return False
```

### 移行手順（Stage 3C）

#### Step 1: BrowserRuntime クラスの作成
1. `app/agents/browser/plugin_api.py` を作成
2. `BrowserRuntime` クラスを実装（初期は既存メソッドへの委譲）

#### Step 2: StrategyPlugin インターフェースの更新
1. `app/agents/plugins/base.py` を更新
2. 新しい `run` メソッドを追加（既存メソッドは後方互換性のため残す）

#### Step 3: BrowserUseAgent の更新
1. `BrowserRuntime` インスタンスを作成
2. プラグイン呼び出し時に `BrowserRuntime` を渡す

#### Step 4: プラグインの段階的移行
1. `MonclerPLPStrategy` を新しいインターフェースに対応
2. 既存メソッドから `BrowserRuntime` 経由の呼び出しに移行

---

## 実装の優先順位

### Phase 1: Stage 3B（TelemetryService）
1. ✅ TelemetryService クラスの作成
2. ✅ BrowserUseAgent への統合
3. ✅ NavigationDriver への統合
4. ⏳ 既存コードのクリーンアップ

### Phase 2: Stage 3C（Plugin API）
1. ⏳ BrowserRuntime クラスの作成
2. ⏳ StrategyPlugin インターフェースの更新
3. ⏳ BrowserUseAgent の更新
4. ⏳ プラグインの段階的移行

---

## リスクと対策

### Stage 3B のリスク
- **既存の `observability.py` への依存が広範囲**
  - 対策: 段階的移行、後方互換性を維持

### Stage 3C のリスク
- **プラグインが既存メソッドに直接依存している**
  - 対策: 初期は委譲パターンで既存実装を呼び出す

---

## 次のステップ

1. **Stage 3B の実装開始**: TelemetryService クラスの作成
2. **Stage 3C の設計レビュー**: BrowserRuntime の設計を確認
3. **テスト計画**: 各ステージのテストケースを作成

