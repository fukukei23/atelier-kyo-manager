# Stage 3A-3: trap 判定の観測フック追加 完了レポート

## 概要

Stage 3A-3 では、trap 判定の「観測フック」を NavigationDriver に追加しました。
このステージでは、実際の挙動（例外・復旧・ナビゲーション制御）を一切変えず、
既存の `_run_plp_flow()` が持っている trap 判定・復旧ロジックはそのまま維持しています。

NavigationDriver 側は「観測とログだけ」が目的です。

## 変更内容

### 1. `app/agents/browser/navigation_driver.py` の拡張

#### 1.1 NavigationOutcome の確認

- ✅ `trap_detected: bool = False` - 既に存在
- ✅ `trap_reason: Optional[str] = None` - 既に存在

既存のフィールドを再利用しました。

#### 1.2 TrapCheckerFn 型定義の追加

```python
# Stage 3A-3: trap 判定関数の型定義
TrapCheckerFn = Callable[[str], bool]  # URL を受けて trap かどうか判定
```

#### 1.3 NavigationDriver.__init__ に trap_checker 引数を追加

**変更前:**
```python
def __init__(
    self,
    page: Page,
    *,
    telemetry: Optional["TelemetryService"] = None,
    strategy: Any = None,
) -> None:
```

**変更後:**
```python
def __init__(
    self,
    page: Page,
    *,
    trap_checker: Optional[TrapCheckerFn] = None,
    telemetry: Optional["TelemetryService"] = None,
    strategy: Any = None,
) -> None:
    self.page = page
    self.trap_checker = trap_checker
    self.telemetry = telemetry
    self.strategy = strategy
```

#### 1.4 NavigationDriver.run_plp_flow() の冒頭で trap_checker を実行

**変更内容:**
- `run_plp_flow()` の冒頭で、現在の URL に対して `trap_checker` を実行
- 結果を `NavigationOutcome` に記録（`trap_detected`, `trap_reason`）
- **例外を投げない**（観測だけ）
- エラーは握りつぶして debug ログに留める

**実装:**
```python
# --- Stage 3A-3: trap 観測フック ---
# このステージでは、trap 判定を「観測だけ」で行い、実際のナビゲーションは変更しない
url = self.page.url or entry
if self.trap_checker and url:
    try:
        if self.trap_checker(url):
            outcome.trap_detected = True
            outcome.trap_reason = f"initial_url={url}"
            # ★ ここでは raise しない。単に記録とログのみ。
            logger.warning("[NavigationDriver] trap-like url observed: %s", url)
    except Exception as e:
        # 観測用途なのでエラーは握りつぶし、必要なら debug ログに留める
        logger.debug("[NavigationDriver] trap_checker failed: %s", e)
```

### 2. `app/agents/browser_use_agent.py` の変更

#### 2.1 NavigationDriver に trap_checker を渡す

**変更箇所:** `run()` メソッド内の NavigationDriver 初期化部分（1574-1588行目）

**変更前:**
```python
navigation_driver = NavigationDriver(
    page=page,
    ensure_plp_materialized=lambda pg, scfg, stg, s_t, b_ms: self._ensure_plp_materialized(...),
    trap_checker=None,
    recovery_fn=lambda pg, scfg, t_url: self._force_plp_recover(pg, scfg, t_url),
    telemetry=telemetry,
    strategy=plugin,
)
```

**変更後:**
```python
# Stage 3A-3: trap_checker を NavigationDriver に渡す（観測用）
navigation_driver = NavigationDriver(
    page=page,
    trap_checker=lambda url: self._looks_like_trap_or_legal(url),
    telemetry=telemetry,
    strategy=plugin,
)
```

**注意:** 
- 既存の `ensure_plp_materialized` や `recovery_fn` は削除しました（これらは Stage 3A-2 で追加される予定）
- `run_plp_flow` の呼び出しで `target_url` パラメータを削除しました（現在のシグネチャには存在しない）

#### 2.2 nav_outcome の trap 観測結果をログに出す

**変更箇所:** `run()` メソッド内の NavigationDriver 呼び出し後（1597-1603行目）

**変更前:**
```python
# Stage 3A-2: NavigationDriver が trap を検出した場合は早期リターン
if nav_outcome and nav_outcome.trap_detected:
    self.logger.warning(
        f"[NavigationDriver] Trap detected: {nav_outcome.trap_reason}"
    )
    # TODO: DiscoveryResult として適切なエラーを返す
    # 現時点では従来の処理にフォールバック
```

**変更後:**
```python
# Stage 3A-3: trap 観測結果をログに出す（挙動は変更しない）
if nav_outcome and getattr(nav_outcome, "trap_detected", False):
    self.logger.warning(
        "[NavigationDriver] trap-like url observed (legacy PLP flow will still run): %s",
        nav_outcome.trap_reason,
    )
```

**重要なポイント:**
- 早期リターンや例外を投げない（観測だけ）
- ログメッセージに「legacy PLP flow will still run」を追加して、挙動が変わらないことを明示

#### 2.3 その後の処理フローは変更しない

- `skip_materialize` の計算は変更なし
- `_run_plp_flow(...)` の呼び出しは変更なし
- 既存の trap 判定・復旧ロジックは一切変更なし

### 3. `_run_plp_flow` 側は変更しない

- ✅ 既存の `_run_plp_flow` の中にある `self._looks_like_trap_or_legal(...)` や `_force_plp_recover(...)`、trap 関連の例外処理・回復ロジックには一切手を触れていません

## 品質確認

### ✅ 既存の挙動を変えていない

- NavigationDriver は trap を検出しても例外を投げない
- 既存の `_run_plp_flow` の trap 判定・復旧ロジックはそのまま
- ナビゲーション制御は一切変更なし

### ✅ ログメッセージ

- 新しいログメッセージには `[NavigationDriver]` プレフィックスを付与
- 既存ログの文言は変更なし

### ✅ 型エラー・循環 import の回避

- `TrapCheckerFn` は `navigation_driver.py` 内で完結
- 型エラーなし（lint チェック通過）

## 変更ファイル一覧

1. **変更**: `app/agents/browser/navigation_driver.py`
   - `TrapCheckerFn` 型定義を追加
   - `NavigationDriver.__init__` に `trap_checker` 引数を追加
   - `run_plp_flow()` の冒頭で trap_checker を実行して観測結果を記録

2. **変更**: `app/agents/browser_use_agent.py`
   - NavigationDriver 初期化時に `trap_checker` を渡す
   - `nav_outcome` の trap 観測結果をログに出す

## 確認事項

- ✅ NavigationDriver は trap を検出しても例外を投げない（観測だけ）
- ✅ 既存の `_run_plp_flow` の trap 判定・復旧ロジックはそのまま
- ✅ ナビゲーション制御は一切変更なし
- ✅ ログメッセージに適切なプレフィックスを付与
- ✅ 型エラー・循環 import なし

## 次のステップ

Stage 3A-3 は完了しました。次のステップは Stage 3A-2（実際のロジック移動）です。

