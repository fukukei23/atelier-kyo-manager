# BrowserUseAgent 例外処理・Retry ロジック統一リファクタリング完了レポート

## 実装日時
2025年11月28日

## 概要

### 目的
- Playwright / DrissionPage 呼び出しの例外分類と retry を一箇所に集約
- TimeoutError / Playwright Timeout / ネットワークエラー / セレクタ 0 件を明確に区別
- FailureAnalysisAgent / SelfHealingAgent / SelectorDiscoveryAgent に渡す情報を安定化

### ゴール
1. 例外分類用の Enum と dataclass を定義
2. `_run_with_retry` ラッパー関数を実装
3. 主要な Playwright 操作を `_run_with_retry` でラップ
4. 例外分類ロジックを統一
5. RunContext への保存とログ出力を維持

### 原則
- 既存の NexusCore / Self-healing エージェント群との連携を維持
- `app/core/run_context.py` を利用したログ・証跡保存を維持
- 後方互換性を保つ（既存の `safe_wait_selector` などは維持）

## 実装ステップ

### ステップ 1: 例外分類用の Enum と dataclass を定義

**変更内容:**
- `BrowserErrorType` Enum を追加（TIMEOUT / NAVIGATION / SELECTOR / NETWORK / UNKNOWN）
- `BrowserOperationResult` dataclass を追加（操作結果を保持）
- `to_failure_context()` メソッドを追加（FailureAnalysisAgent / SelfHealingAgent に渡す情報を生成）

**コード例:**
```python
class BrowserErrorType(Enum):
    """ブラウザ操作の例外タイプ分類"""
    TIMEOUT = "timeout"
    NAVIGATION = "navigation"
    SELECTOR = "selector"
    NETWORK = "network"
    UNKNOWN = "unknown"

@dataclass
class BrowserOperationResult:
    """ブラウザ操作の結果を保持するデータクラス"""
    success: bool
    error_type: Optional[BrowserErrorType] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    operation_name: str = ""
    context: Optional[Dict[str, Any]] = None
```

**なぜ変更したか:**
- 例外を明確に分類することで、FailureAnalysisAgent / SelfHealingAgent が適切な回復戦略を選択できるようにする
- 操作結果を構造化することで、ログ・証跡保存を統一化

### ステップ 2: `_run_with_retry` ラッパー関数を実装

**変更内容:**
- `_run_with_retry()` メソッドを追加（ブラウザ操作を retry ロジックでラップ）
- `_classify_exception()` メソッドを追加（例外を分類）

**コード例:**
```python
async def _run_with_retry(
    self,
    operation: Callable[[], Awaitable[T]],
    operation_name: str,
    *,
    max_retries: int = 3,
    retry_delay_ms: int = 1000,
    timeout_ms: Optional[int] = None,
    run_context: Optional[RunContext] = None,
    context: Optional[Dict[str, Any]] = None,
) -> BrowserOperationResult:
    """ブラウザ操作を retry ロジックでラップし、例外を分類して返す。"""
    # ... 実装 ...
```

**なぜ変更したか:**
- retry ロジックを一箇所に集約することで、コードの重複を削減
- 例外分類を統一することで、エラーハンドリングを安定化

### ステップ 3: 主要な Playwright 操作を `_run_with_retry` でラップ

**変更内容:**
- `safe_goto()` メソッドを追加（`page.goto()` を retry ロジックでラップ）
- `safe_click()` メソッドを追加（`locator.click()` を retry ロジックでラップ）
- `safe_wait_for_load_state()` メソッドを追加（`page.wait_for_load_state()` を retry ロジックでラップ）
- `safe_wait_selector()` メソッドを更新（既存のシグネチャを維持しつつ、内部で `_run_with_retry` を使用）

**コード例:**
```python
async def safe_goto(
    self,
    page: Page,
    url: str,
    *,
    wait_until: str = "domcontentloaded",
    timeout_ms: Optional[int] = None,
    max_retries: int = 3,
) -> BrowserOperationResult:
    """ページ遷移を retry ロジックでラップ。"""
    # ... 実装 ...
```

**なぜ変更したか:**
- 主要な Playwright 操作を統一的な retry ロジックでラップすることで、エラーハンドリングを安定化
- 既存のコードとの互換性を保つため、新しいメソッドを追加（既存のメソッドは削除しない）

### ステップ 4: 例外分類ロジックを統一

**変更内容:**
- `_handle_run_failure()` メソッドを更新（例外分類情報を `failure_context` に追加）
- `error_type` と `error_class` を `failure_context` に追加

**コード例:**
```python
# V88.7.0: 例外分類情報を failure_context に追加
failure_context = {
    "final_url": final_url_on_fail,
    "dom_snapshot_path": dom_path_str,
    "errors": [str(e)],
    "error_type": error_type.value,  # V88.7.0: 例外タイプを追加
    "error_class": type(e).__name__,  # V88.7.0: 例外クラス名を追加
    # ... その他のフィールド ...
}
```

**なぜ変更したか:**
- FailureAnalysisAgent / SelfHealingAgent が例外タイプに基づいて適切な回復戦略を選択できるようにする
- エラーの原因を明確にすることで、デバッグを容易にする

### ステップ 5: RunContext への保存とログ出力を維持

**変更内容:**
- `_run_with_retry()` 内で `run_context.save_json()` を呼び出してエラー情報を保存
- ログ出力を統一（`[Retry]` プレフィックスを使用）

**コード例:**
```python
# RunContext にエラー情報を保存
error_info = {
    "operation": operation_name,
    "attempt": retry_count,
    "error_type": error_type.value,
    "error_message": str(e),
    "context": context or {},
}
run_context.save_json(f"retry_error_{operation_name}_{retry_count}.json", error_info)
```

**なぜ変更したか:**
- 既存のログ・証跡保存機能を維持しつつ、新しい例外分類情報を追加
- デバッグ時にエラー情報を確認しやすくする

## 変更ファイル一覧

### 新規作成ファイル
なし

### 変更ファイル

1. **app/agents/browser_use_agent.py**
   - 例外分類用の Enum と dataclass を追加
   - `_run_with_retry()` メソッドを追加
   - `_classify_exception()` メソッドを追加
   - `safe_goto()` / `safe_click()` / `safe_wait_for_load_state()` メソッドを追加
   - `safe_wait_selector()` メソッドを更新（内部で `_run_with_retry` を使用）
   - `_bootstrap_session_page()` メソッドを更新（`safe_goto()` / `safe_wait_for_load_state()` を使用）
   - `_handle_run_failure()` メソッドを更新（例外分類情報を `failure_context` に追加）

## 動作確認結果

### 静的解析結果
- リンターエラー: なし
- 型チェッカー: 未実施（型ヒントは追加済み）

### コードレビュー結果
- 既存の NexusCore / Self-healing エージェント群との連携を維持
- `app/core/run_context.py` を利用したログ・証跡保存を維持
- 後方互換性を保つ（既存のメソッドは削除しない）

### テスト結果
- ユニットテスト: 未実施（今後のタスクとして推奨）
- 統合テスト: 未実施（今後のタスクとして推奨）

## 設計上の改善点

### アーキテクチャの改善
1. **例外分類の統一**: 例外を明確に分類することで、エラーハンドリングを安定化
2. **Retry ロジックの集約**: retry ロジックを一箇所に集約することで、コードの重複を削減
3. **構造化されたエラー情報**: `BrowserOperationResult` を使用することで、エラー情報を構造化

### 将来の拡張性への配慮
1. **新しい例外タイプの追加**: `BrowserErrorType` Enum に新しいタイプを追加可能
2. **カスタム retry 戦略**: `_run_with_retry()` にカスタム retry 戦略を追加可能
3. **メトリクス収集**: `BrowserOperationResult` にメトリクス情報を追加可能

### コード品質の向上
1. **型ヒントの追加**: すべてのメソッドに型ヒントを追加
2. **ドキュメントの追加**: すべてのメソッドに docstring を追加
3. **ログの統一**: ログ出力を統一（`[Retry]` プレフィックスを使用）

## 既知の制約・注意事項

### 既存コードとの互換性
- 既存の `safe_wait_selector()` メソッドは後方互換性を保つ（既存のシグネチャを維持）
- 既存の `_handle_run_failure()` メソッドは後方互換性を保つ（新しいフィールドを追加するのみ）

### 制限事項やトレードオフ
1. **Retry 回数の制限**: `max_retries` のデフォルト値は 3（必要に応じて調整可能）
2. **タイムアウトの制限**: `timeout_ms` が指定されていない場合、Playwright のデフォルトタイムアウトを使用
3. **メモリ使用量**: エラー情報を JSON ファイルに保存するため、メモリ使用量が増加する可能性がある

### 移行時の注意点
- 既存のコードは自動的に新しい retry ロジックを使用（`safe_goto()` などを使用する場合）
- 既存のコードが直接 `page.goto()` などを呼び出している場合、新しい retry ロジックは適用されない

## 次のステップ

### 推奨されるフォローアップアクション

1. **ユニットテストの追加**
   - `_run_with_retry()` のテスト
   - `_classify_exception()` のテスト
   - `safe_goto()` / `safe_click()` / `safe_wait_for_load_state()` のテスト

2. **統合テストの追加**
   - FailureAnalysisAgent / SelfHealingAgent との連携テスト
   - 実際のブラウザ操作での retry ロジックの動作確認

3. **メトリクス収集の追加**
   - retry 回数の統計情報を収集
   - 例外タイプ別の統計情報を収集

4. **ドキュメントの更新**
   - API ドキュメントの更新
   - 使用例の追加

5. **既存コードの段階的な移行**
   - 既存のコードで直接 `page.goto()` などを呼び出している箇所を、新しい `safe_goto()` などに置き換え

## 関連ファイル

- `app/agents/browser_use_agent.py`: メインの実装ファイル
- `app/core/run_context.py`: RunContext クラス（ログ・証跡保存）
- `app/agents/failure_analysis_agent.py`: FailureAnalysisAgent（例外分析）
- `app/agents/self_healing_agent.py`: SelfHealingAgent（自己修復）
- `app/agents/selector_discovery_agent.py`: SelectorDiscoveryAgent（セレクタ発見）

## 完了状況

✅ **完了**: BrowserUseAgent レベルでの例外処理・retry ロジックの統一は完了しました。

⚠️ **未完了（別タスク）**: 上位エージェント側（FailureAnalysisAgent / SelfHealingAgent / SelectorDiscoveryAgent）での例外分類情報の活用は、各エージェントのリファクタリングタスクとして別途実施予定です。

