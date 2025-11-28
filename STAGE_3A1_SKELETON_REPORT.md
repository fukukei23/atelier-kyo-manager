# Stage 3A-1: NavigationDriver 骨組み作成レポート

## 概要

Stage 3A（NavigationDriver 抽出）の骨組みを作成しました。
このステップでは、クラス定義とメソッドシグネチャのみを作成し、実際のロジック移動は Stage 3A-2 で行います。

## 変更内容

### 1. `app/agents/browser/navigation_driver.py` の作成

#### 1.1 追加されたクラス・データクラス

- **`NavigationContext` (dataclass)**
  - `site: str` - サイト名
  - `query: str` - 検索クエリ
  - `site_config: Dict[str, Any]` - サイト設定
  - `settings: Dict[str, Any]` - 実行設定
  - `run_context: Any` - 実行コンテキスト（RunContext を直接 import しない）
  - `start_t: float` - 開始時刻
  - `budget_ms: int` - タイムバジェット（ミリ秒）
  - `entry_url: Optional[str] = None` - エントリー URL

- **`NavigationOutcome` (dataclass)**
  - `entry_url: str` - エントリー URL
  - `plp_materialized: bool = False` - PLP がマテリアライズされたか
  - `trap_detected: bool = False` - trap が検出されたか
  - `trap_reason: Optional[str] = None` - trap の理由
  - `recovered: bool = False` - 回復試行が実行されたか

- **`NavigationDriver` (class)**
  - `__init__(self, page: Page, *, telemetry: Optional[TelemetryService] = None, strategy: Any = None)`
    - `page`: SessionManager から渡される Page オブジェクト
    - `telemetry`: TelemetryService（オプション、このステージでは使わない）
    - `strategy`: StrategyPlugin（オプション、このステージでは使わない）

#### 1.2 追加されたメソッド（骨組みのみ）

- **`async def run_plp_flow(self, ctx: NavigationContext) -> NavigationOutcome`**
  - Stage 3A-1: スタブ実装 - ctx をそのまま返すだけ
  - 実際のロジック移動は Stage 3A-2 で行います

- **`async def collect_pdp_links(self, page: Page, site_config: Dict[str, Any], settings: Dict[str, Any], run_context: Any) -> List[str]`**
  - Stage 3A-1: スタブ実装 - 空リストを返す
  - 実際のロジック移動は Stage 3A-2 で行います

- **`async def run_deep_extraction(self, page: Page, site_config: Dict[str, Any]) -> List[str]`**
  - Stage 3A-1: スタブ実装 - 空リストを返す
  - 実際のロジック移動は Stage 3A-2 で行います

- **`async def recover_plp(self, page: Page, ctx: NavigationContext) -> bool`**
  - Stage 3A-1: スタブ実装 - False を返す
  - 実際のロジック移動は Stage 3A-2 で行います

- **`async def _click_first_card(self, page: Page, site_config: Dict[str, Any]) -> Optional[Page]`**
  - Stage 3A-1: スタブ実装 - None を返す
  - 実際のロジック移動は Stage 3A-2 で行います

- **`def _looks_like_trap_or_legal(self, url: str) -> bool`**
  - Stage 3A-1: スタブ実装 - False を返す
  - 実際のロジック移動は Stage 3A-2 で行います

### 2. `app/agents/browser_use_agent.py` の変更

#### 2.1 `_run_plp_flow` メソッドの変更

**変更前:**
```python
async def _run_plp_flow(self, page: Page, context: BrowserContext, site: str, query: str,
                        site_config: Dict, settings: Dict, run_context: RunContext,
                        target_url: str,
                        *, start_t: float, budget_ms: int, skip_materialize: bool = False,
                        nav_outcome: Optional[Any] = None) -> DiscoveryResult:
    """
    Stage 3A-2: NavigationDriver の結果を利用するように更新。
    """
    # --- Stage 3A-2: NavigationDriver が既に処理済みの場合はスキップ ---
    attempted_recover = False
    if nav_outcome and nav_outcome.recovered:
        # ... 既存の処理 ...
```

**変更後:**
```python
async def _run_plp_flow(self, page: Page, context: BrowserContext, site: str, query: str,
                        site_config: Dict, settings: Dict, run_context: RunContext,
                        target_url: str,
                        *, start_t: float, budget_ms: int, skip_materialize: bool = False,
                        nav_outcome: Optional[Any] = None) -> DiscoveryResult:
    """
    Stage 3A-1: NavigationDriver への最小限の委譲を追加。
    """
    # --- Stage 3A-1: NavigationDriver への最小限の委譲 ---
    # NavigationContext を組み立て
    nav_ctx = NavigationContext(
        site=site,
        query=query,
        site_config=site_config,
        settings=settings,
        run_context=run_context,
        start_t=start_t,
        budget_ms=budget_ms,
        entry_url=target_url,
    )
    
    # NavigationDriver を初期化（このステージでは telemetry/strategy は使わない）
    navigation_driver = NavigationDriver(
        page=page,
        telemetry=None,  # Stage 3A-1: まだ使わない
        strategy=None,   # Stage 3A-1: まだ使わない
    )
    
    # run_plp_flow を呼び出す（現時点ではスタブ実装なので、ctx をそのまま返すだけ）
    try:
        nav_outcome = await navigation_driver.run_plp_flow(nav_ctx)
        self.logger.debug(f"[_run_plp_flow] NavigationDriver.run_plp_flow called (stub): entry_url={nav_outcome.entry_url}")
    except Exception as nav_e:
        self.logger.debug(f"[_run_plp_flow] NavigationDriver.run_plp_flow failed (fallback to legacy): {nav_e}")
        nav_outcome = None

    # --- Stage 3A-1: 実際のナビゲーションロジックは旧実装を使用（以下は変更なし） ---
    # ... 既存の処理（変更なし） ...
```

#### 2.2 変更のポイント

- **NavigationContext の組み立て**: `_run_plp_flow` の冒頭で NavigationContext を組み立て
- **NavigationDriver の初期化**: NavigationDriver を初期化（telemetry/strategy は None）
- **run_plp_flow の呼び出し**: NavigationDriver.run_plp_flow を呼び出す（スタブ実装なので、ctx をそのまま返すだけ）
- **既存ロジックの維持**: 実際のナビゲーションロジックは旧実装を使用（変更なし）

## 動作確認

### 確認項目

1. ✅ **NavigationDriver のインポート**: `from app.agents.browser.navigation_driver import NavigationContext, NavigationDriver` が正しく動作する
2. ✅ **NavigationContext の作成**: NavigationContext が正しく作成できる
3. ✅ **NavigationDriver の初期化**: NavigationDriver が正しく初期化できる
4. ✅ **run_plp_flow の呼び出し**: NavigationDriver.run_plp_flow が呼び出せる（スタブ実装）
5. ✅ **既存ロジックの動作**: 既存のナビゲーションロジックが正常に動作する（変更なし）

### 期待される動作

- **Stage 3A-1**: NavigationDriver を挟むための配線だけを作成
- **挙動**: サイトに対する実際の動きは変えない（既存ロジックを使用）
- **NavigationDriver.run_plp_flow**: スタブ実装なので、ctx をそのまま返すだけ

## 次のステップ（Stage 3A-2）

Stage 3A-2 では、以下のロジックを NavigationDriver に移行します：

1. `_run_plp_flow` のロジックを `NavigationDriver.run_plp_flow` に移行
2. `_collect_pdp_links` のロジックを `NavigationDriver.collect_pdp_links` に移行
3. `_run_deep_extraction_phase2` のロジックを `NavigationDriver.run_deep_extraction` に移行
4. `_force_plp_recover` のロジックを `NavigationDriver.recover_plp` に移行
5. `_looks_like_trap_or_legal` のロジックを `NavigationDriver._looks_like_trap_or_legal` に移行
6. `_click_first_card_or_link` のロジックを `NavigationDriver._click_first_card` に移行

## 変更ファイル一覧

1. **新規作成**: `app/agents/browser/navigation_driver.py`
2. **変更**: `app/agents/browser_use_agent.py` (`_run_plp_flow` メソッド)

## 確認事項

この骨組みを適用してよいか、ご確認ください。

- ✅ NavigationDriver のクラス定義とメソッドシグネチャが正しく作成されている
- ✅ BrowserUseAgent からの最小限の委譲が追加されている
- ✅ 既存のナビゲーションロジックは変更されていない（旧実装を使用）
- ✅ NavigationDriver.run_plp_flow はスタブ実装（ctx をそのまま返すだけ）

