# CR-ATELIER-003 Phase D-4 Telemetry 統合と Self-Healing 連携導線 完了レポート

## 実装日時

2025年12月10日

## 概要

CR-ATELIER-003 Phase D-4 は、PLP/PDP フローにおける Telemetry 記録を `BrowserOrchestrator` に集約し、失敗時に Self-Healing / FailureAnalysisAgent が利用できる `failure_context` を標準化することを目的としました。

これにより、将来の Self-Healing 強化の下地が整い、すべての失敗ケースで統一された `failure_context` が生成されるようになりました。

## 実装ステップ

### Step 1: `BrowserOrchestrator.run_plp_to_pdp` に Telemetry 呼び出しを追加

**ファイル**: `app/agents/browser_orchestrator.py`

**変更内容**:
- NavigationDriver.run_plp_flow 実行前に PLP 初期状態を Telemetry に記録
- NavigationDriver.run_plp_flow 実行後に PLP navigation outcome を Telemetry に記録
- pdp_links が 0 の場合、`plp_no_pdp_links` を Telemetry に記録
- pdp_links > 0 の場合、`plp_pdp_links` を Telemetry に記録（最初の10件のみ）
- TrapPageDetected 例外発生時、Telemetry に trap 検出を記録
- NavigationDriver 失敗時、Telemetry に失敗を記録

**実装例**:
```python
# PLP フロー実行前の Telemetry 記録
if telemetry and hasattr(telemetry, 'record_plp_state'):
    await telemetry.record_plp_state(
        page,
        name="plp_dom_initial",
        site_config=site_config,
    )

# PLP フロー実行後の Telemetry 記録
if telemetry and hasattr(telemetry, 'save_json'):
    await telemetry.save_json(
        "plp_navigation_outcome",
        {
            "entry_url": nav_outcome.entry_url,
            "pdp_links_count": len(nav_outcome.pdp_links),
            "trap_detected": nav_outcome.trap_detected,
            "recovered": nav_outcome.recovered,
        },
        tctx,
    )
```

### Step 2: `BrowserOrchestrator.run_pdp` に Telemetry 呼び出しを追加

**ファイル**: `app/agents/browser_orchestrator.py`

**変更内容**:
- prepare_hook 内で PDP DOM を Telemetry に保存
- Visual regression check の結果を Telemetry に記録（可能であれば）

**実装例**:
```python
async def prepare_hook(inner_page: Page) -> None:
    # PDP DOM を Telemetry に保存
    if telemetry_client and hasattr(telemetry_client, 'record_plp_state'):
        await telemetry_client.record_plp_state(
            inner_page,
            name="pdp_dom",
            site_config=site_config,
        )
    
    # VRT 結果を Telemetry に記録
    if telemetry_client and hasattr(telemetry_client, 'save_json') and vrt_result:
        await telemetry_client.save_json(
            "pdp_vrt_result",
            {"url": inner_page.url, "vrt_result": str(vrt_result)},
            tctx,
        )
```

### Step 3: failure_context の標準化

**ファイル**: `app/agents/browser_orchestrator.py`

**変更内容**:
- `_build_failure_context` メソッドを追加し、標準化された `failure_context` を構築
- すべての失敗ケースで `failure_context` を `DiscoveryResult.evidence` に含める

**failure_context の構造**:
```python
{
    "final_url": str,                    # 最終URL
    "error_type": str,                   # エラータイプ（"no_pdp_links" / "pdp_extraction_failed" / "trap_recovery_failed" など）
    "error_class": str,                  # 例外クラス名
    "error_message": str,                # エラーメッセージ
    "site": str,                         # サイトコード
    "query": str,                        # 検索クエリ
    "run_id": str,                       # 実行ID
    "dom_snapshot_path": str,            # DOM スナップショットパス（オプション）
    "screenshot_paths": List[str],       # スクリーンショットパス（オプション、最大6件）
    "site_config_summary": {             # site_config の要約
        "site_code": str,
        "has_plp_selectors": bool,
        "has_pdp_selectors": bool,
    },
}
```

**適用箇所**:
- TrapPageDetected 例外発生時
- NavigationDriver 失敗時
- Trap recovery 失敗時
- PlpDriver 失敗時
- extract_from_pdp_list 失敗時
- extract_single_pdp 失敗時（ValueError / 予期しない例外）
- pdp_links が空で PlpDriver も呼ばれなかった場合

### Step 4: Self-Healing 連携導線の準備

**ファイル**: `app/agents/browser_orchestrator.py`

**変更内容**:
- `_maybe_analyze_failure` メソッドを追加し、Self-Healing 連携のインターフェースを整備
- 現時点ではログ出力のみを行う
- 後続フェーズで FailureAnalysisAgent や Self-Healing Agent を呼べるようにインターフェースを整えておく

**実装例**:
```python
async def _maybe_analyze_failure(
    self,
    failure_context: Dict[str, Any],
    *,
    page: Optional[Page] = None,
) -> None:
    """
    CR-ATELIER-003 Phase D-4: Self-Healing 連携導線（インターフェースのみ）
    
    現時点ではログ出力のみを行う。
    後続フェーズで FailureAnalysisAgent や Self-Healing Agent を呼べるように
    インターフェースを整えておく。
    """
    self.log.info(
        f"[Orchestrator][SelfHealing] Failure detected: "
        f"type={failure_context.get('error_type')}, "
        f"class={failure_context.get('error_class')}, "
        f"url={failure_context.get('final_url')}"
    )
    
    # TODO: Phase D-5 以降で実装予定
    # - FailureAnalysisAgent.analyze(failure_context) を呼び出す
    # - Self-Healing Agent に failure_context を渡す
    # - Selector Discovery Agent に DOM snapshot を渡す
```

### Step 5: テスト追加

**ファイル**: `tests/test_browser_orchestrator_telemetry.py` (新規作成)

**テスト内容**:
- `test_run_plp_to_pdp_records_plp_initial_state`: PLP 初期状態が Telemetry に記録されることを確認
- `test_run_plp_to_pdp_records_no_pdp_links`: pdp_links が 0 の場合、Telemetry に記録されることを確認
- `test_run_pdp_records_pdp_dom`: PDP DOM が Telemetry に記録されることを確認
- `test_failure_context_includes_required_keys`: failure_context に必要なキーが含まれていることを確認
- `test_maybe_analyze_failure_logs_failure`: `_maybe_analyze_failure` がログを出力することを確認
- `test_build_failure_context_includes_site_config_summary`: `_build_failure_context` が site_config_summary を含むことを確認

**テスト方針**:
- TelemetryClient / TelemetryService を AsyncMock に差し替えて、指定メソッドが呼ばれることだけを検証
- failure_context のキーが存在することを確認する軽量テスト

## 変更ファイル一覧

### 新規作成ファイル

- `tests/test_browser_orchestrator_telemetry.py` (約250行)

### 変更ファイル

- `app/agents/browser_orchestrator.py`
  - `run_plp_to_pdp`: Telemetry 記録を追加（PLP 初期状態、navigation outcome、pdp_links、失敗時）
  - `run_pdp`: Telemetry 記録を追加（PDP DOM、VRT 結果）
  - `_build_failure_context`: 標準化された failure_context を構築するメソッドを追加
  - `_maybe_analyze_failure`: Self-Healing 連携導線のインターフェースを追加

## 動作確認結果

### テスト結果

すべての既存テストと新規テストがパスしました。

- `tests/test_browser_use_agent_plp_integration.py`: 6 passed
- `tests/test_plp_driver.py`: 13 passed
- `tests/test_moncler_pdp_url.py`: 17 passed
- `tests/test_browser_orchestrator_telemetry.py`: 6 passed

**合計: 42 passed, 8 warnings**

### 静的解析結果

- リンターエラー: 主に型チェックに関する警告が残っていますが、実行時には問題ありません。
- 実行時エラー: なし
- テスト失敗: なし

## 設計上の改善点

1. **Telemetry 記録の集約**:
   - PLP/PDP フローにおける Telemetry 記録が `BrowserOrchestrator` に集約され、一貫性が向上
   - すべての失敗ケースで Telemetry に記録されるようになり、可観測性が向上

2. **failure_context の標準化**:
   - すべての失敗ケースで統一された `failure_context` が生成されるようになり、Self-Healing 連携が容易に
   - `error_type`、`error_class`、`error_message` などの標準フィールドにより、エラー分析が容易に

3. **Self-Healing 連携導線の準備**:
   - `_maybe_analyze_failure` メソッドにより、将来の Self-Healing 強化の下地が整備
   - インターフェースが明確に定義され、後続フェーズでの実装が容易に

## 達成状況 (Phase D-4 完了条件)

Phase D-4 の完了条件に対する達成状況は以下の通りです。

### 1. Telemetry 記録の集約
- ✅ **PLP/PDP フローにおける Telemetry 記録を BrowserOrchestrator に集約**: 達成。`run_plp_to_pdp` と `run_pdp` の両方で Telemetry 記録を実装しました。

### 2. failure_context の標準化
- ✅ **失敗時に Self-Healing / FailureAnalysisAgent が利用できる failure_context を標準化**: 達成。`_build_failure_context` メソッドにより、すべての失敗ケースで統一された `failure_context` が生成されます。

### 3. Self-Healing 連携導線の準備
- ✅ **将来の Self-Healing 強化の下地を作る**: 達成。`_maybe_analyze_failure` メソッドにより、インターフェースが整備されました。

### 4. 既存テストの維持
- ✅ **既存の 36 テストを一切壊さずに**: 達成。すべての既存テストがパスし、新規テストも追加されました（合計42テスト）。

## 既知の制約・注意事項

1. **Self-Healing 連携の実装**: 現時点では `_maybe_analyze_failure` はログ出力のみを行います。Phase D-5 以降で FailureAnalysisAgent や Self-Healing Agent の呼び出しを実装予定です。

2. **Telemetry 記録のエラーハンドリング**: Telemetry 記録が失敗した場合、警告ログを出力して続行します。これにより、Telemetry の失敗がメインフローに影響を与えないようにしています。

3. **failure_context のフィールド**: `dom_snapshot_path` や `screenshot_paths` は、RunContext が対応している場合のみ含まれます。これにより、異なる RunContext 実装でも動作します。

## 次のステップ

Phase D-4 の完了をもって、Telemetry 統合と Self-Healing 連携導線の準備は完了しました。次のステップとして、以下のタスクが推奨されます。

### Phase D-5: Self-Healing 連携の実装

1. **FailureAnalysisAgent との連携**
   - `_maybe_analyze_failure` 内で `FailureAnalysisAgent.analyze(failure_context)` を呼び出す
   - 分析結果をログに記録し、必要に応じて Telemetry に保存

2. **Self-Healing Agent との連携**
   - Self-Healing Agent に `failure_context` を渡す
   - リカバリ試行の結果を Telemetry に記録

3. **Selector Discovery Agent との連携**
   - Selector Discovery Agent に DOM snapshot を渡す
   - 新しいセレクタ候補を提案し、Telemetry に記録

### Phase D-6: 実ブラウザ E2E 検証

1. **大規模な E2E テストの実施**
   - 実際の Moncler サイトでの E2E テスト
   - Telemetry 記録と failure_context の生成を検証

2. **パフォーマンス最適化**
   - Telemetry 記録のオーバーヘッドを測定
   - 必要に応じて非同期処理やバッチ処理を導入

## 関連ファイル

- `app/agents/browser_orchestrator.py` - Telemetry 統合と failure_context 標準化
- `app/agents/browser/telemetry.py` - TelemetryClient / TelemetryService
- `tests/test_browser_orchestrator_telemetry.py` - Telemetry 統合のテスト
- `tests/test_browser_use_agent_plp_integration.py` - 統合テスト
- `tests/test_plp_driver.py` - PLP ドライバーテスト
- `tests/test_moncler_pdp_url.py` - Moncler PDP URL テスト
- `docs/spec/CR-ATELIER-003_PHASE_D1_PDP_ANALYSIS.md` - Phase D-1 分析レポート
- `docs/completion_reports/CR_ATELIER_003_PHASE_D3_PDP_DELEGATOR_COMPLETION_REPORT.md` - Phase D-3 完了レポート

## まとめ

CR-ATELIER-003 Phase D-4 は、PLP/PDP フローにおける Telemetry 記録を `BrowserOrchestrator` に集約し、失敗時に Self-Healing / FailureAnalysisAgent が利用できる `failure_context` を標準化することを成功裏に完了しました。

これにより、将来の Self-Healing 強化の下地が整い、すべての失敗ケースで統一された `failure_context` が生成されるようになりました。すべての既存テストがパスし、新規テストも追加されました（合計42テスト）。

Phase D-5 では、Self-Healing 連携の実装を進める予定です。

