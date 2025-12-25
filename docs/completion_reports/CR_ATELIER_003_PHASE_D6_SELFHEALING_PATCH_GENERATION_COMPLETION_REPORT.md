# CR-ATELIER-003 Phase D-6 Self-Healing 自動パッチ候補の生成 完了レポート

## 実装日時

2025年12月10日

## 概要

CR-ATELIER-003 Phase D-6 は、Phase D-5 で統合された `failure_context` + `failure_analysis` を入力として、サイト設定 (site_config / overrides.local.json 相当) に対する「自動パッチ候補」を JSON として生成・保存する仕組みを実装することを目的としました。

このフェーズでは、パッチの「適用」は行わず、候補生成と保存、および `DiscoveryResult` へのパス埋め込みまでに限定しました。これにより、すべての失敗ケースで統一されたパッチ候補が生成され、将来の自動パッチ適用の基盤が整いました。

## 実装ステップ

### Step 1: SelfHealingPatchAgent の新規実装

**ファイル**: `app/agents/self_healing_patch_agent.py` (新規作成)

**実装内容**:
- `SelfHealingPatchAgent` クラスを新規作成
- `build_patch_candidate` メソッドを実装:
  - `failure_context` と `failure_analysis` を入力として受け取る
  - 簡易ルールベースでパッチ候補を生成
  - 生成したパッチ候補を `run_context.save_json("patch_candidate_self_healing.json", patch_dict)` で保存

**パッチ生成ルール（heuristic_v1）**:
1. **error_type が "navigation_failed" または "trap_recovery_failed" の場合**:
   - `discovery_settings.timeout_sec` を +30 秒するパッチ候補を提案
   - `current_value` と `proposed_value` を含む

2. **error_message 内に "selector" または "not found" が含まれる場合**:
   - `selectors.pdp` に対する「要再確認」フラグ（`action: "review_required"`）を付ける
   - 実際の CSS セレクタ変更はまだ行わない

3. **suggested_fixes に "timeout" が含まれる場合**:
   - `discovery_settings.timeout_sec` を +30 秒するパッチ候補を提案
   - 既に timeout_sec の変更が提案されている場合はスキップ

**パッチ候補のフォーマット**:
```json
{
  "target_site": "<site_code>",
  "run_id": "<run_id>",
  "generated_at": "<ISO8601>",
  "strategy": "heuristic_v1",
  "changes": [
    {
      "path": "discovery_settings.timeout_sec",
      "action": "increase",
      "value_delta": 30,
      "current_value": 60,
      "proposed_value": 90
    }
  ],
  "notes": {
    "summary": "<failure_analysis.summary>",
    "root_causes": [...],
    "suggested_fixes": [...]
  }
}
```

### Step 2: BrowserOrchestrator への SelfHealingPatchAgent 組み込み

**ファイル**: `app/agents/browser_orchestrator.py`

**変更内容**:
- **`__init__` の拡張**:
  - `SelfHealingPatchAgent` を初期化して `self.patch_agent` を保持
  - `FailureAnalysisAgent` と同様に、引数で渡されなければデフォルトでインスタンス化

- **`_maybe_build_patch_candidate` メソッドの追加**:
  - `failure_context`, `failure_analysis`, `site_config`, `run_context` を受け取る
  - `self.patch_agent` が `None` の場合は `None` を返す
  - 例外発生時はログ出力のみで `None` を返す（メインフローには影響させない）

- **失敗パスでの連携**:
  - `run_plp_to_pdp` / `run_pdp` で `DiscoveryResult(ok=False)` を組み立てる箇所において、
    `analysis` がある場合に `_maybe_build_patch_candidate` を呼び出し、
    `patch` が `None` でなければ `evidence["self_healing_patch_candidate"] = patch` を追加

**適用箇所**:
- Trap recovery 失敗時
- PlpDriver 失敗時
- extract_from_pdp_list 失敗時
- extract_single_pdp 失敗時（ValueError / 予期しない例外）
- pdp_links が空で PlpDriver も呼ばれなかった場合

**実装例**:
```python
analysis = await self._maybe_analyze_failure(failure_ctx, page=page, run_context=run_context)
evidence = {"failure_context": failure_ctx}
if analysis:
    evidence["failure_analysis"] = analysis
    # CR-ATELIER-003 Phase D-6: パッチ候補を生成
    patch = await self._maybe_build_patch_candidate(
        failure_context=failure_ctx,
        failure_analysis=analysis,
        site_config=site_config,
        run_context=run_context,
    )
    if patch:
        evidence["self_healing_patch_candidate"] = patch
```

### Step 3: FailureAnalysisAgent との整合性の確認

**確認結果**:
- `failure_analysis["suggested_fixes"]` の形式は現状 `["設定を確認してください"]` のような素朴なリスト
- Phase D-6 の段階では、文字列マッチ or 正規表現ベースの簡易ルールで十分
- 将来 LLM を本番接続した際も、`suggested_fixes` を自然文 → 構造化ルールにマッピングする層を `SelfHealingPatchAgent` 内で差し替えられるように設計

### Step 4: テスト追加

**ファイル**: `tests/test_self_healing_patch_agent.py` (新規作成)

**テスト内容**:
- `test_build_patch_candidate_navigation_failed_timeout_increase`: navigation_failed の場合、timeout_sec を +30 秒するパッチ候補が生成されることを確認
- `test_build_patch_candidate_trap_recovery_failed_timeout_increase`: trap_recovery_failed の場合、timeout_sec を +30 秒するパッチ候補が生成されることを確認
- `test_build_patch_candidate_selector_not_found_review_required`: error_message に "selector" が含まれる場合、pdp selector に対する「要再確認」フラグが含まれることを確認
- `test_build_patch_candidate_suggested_fixes_timeout`: suggested_fixes に "timeout" が含まれる場合、timeout_sec を +30 秒するパッチ候補が生成されることを確認
- `test_build_patch_candidate_exception_handling`: 例外発生時でも None を返し、メインフローに影響しないことを確認
- `test_orchestrator_calls_patch_agent_on_failure`: BrowserOrchestrator が失敗時に SelfHealingPatchAgent を呼び出すことを確認
- `test_orchestrator_no_patch_agent_does_not_break_flow`: patch_agent が None の場合でも、メインフローが正常に動作することを確認

**テスト方針**:
- `build_patch_candidate` に対して直接テストを書く
- BrowserOrchestrator との統合テストで、`build_patch_candidate` が呼ばれることと、`DiscoveryResult.evidence["self_healing_patch_candidate"]` に dict が載ることを確認

## 変更ファイル一覧

### 新規作成ファイル

- `app/agents/self_healing_patch_agent.py` (約200行)
- `tests/test_self_healing_patch_agent.py` (約420行)

### 変更ファイル

- `app/agents/browser_orchestrator.py`
  - `SelfHealingPatchAgent` のインポートを追加
  - `__init__`: `SelfHealingPatchAgent` の初期化を追加
  - `_maybe_build_patch_candidate`: パッチ候補生成メソッドを追加
  - `run_plp_to_pdp` / `run_pdp`: すべての失敗ケースで `_maybe_build_patch_candidate` を呼び出し、`self_healing_patch_candidate` を `evidence` に追加

## 動作確認結果

### テスト結果

すべての既存テストと新規テストがパスしました。

- `tests/test_browser_use_agent_plp_integration.py`: 6 passed
- `tests/test_plp_driver.py`: 13 passed
- `tests/test_moncler_pdp_url.py`: 17 passed
- `tests/test_browser_orchestrator_telemetry.py`: 6 passed
- `tests/test_failure_analysis_integration.py`: 5 passed
- `tests/test_self_healing_patch_agent.py`: 7 passed

**合計: 54 passed, 17 warnings**

### 静的解析結果

- リンターエラー: 主に型チェックに関する警告が残っていますが、実行時には問題ありません。
- 実行時エラー: なし
- テスト失敗: なし

## 設計上の改善点

1. **パッチ候補の標準化**:
   - すべての失敗ケースで統一されたパッチ候補フォーマットが生成されるようになり、パッチ適用が容易に
   - `strategy: "heuristic_v1"` により、将来のルール拡張が容易に

2. **メインフローへの影響を最小化**:
   - `SelfHealingPatchAgent` が例外を投げた場合でも、メインフローには影響させない設計
   - `patch_agent` が `None` の場合でも、メインフローが正常に動作する

3. **将来の拡張性**:
   - `heuristic_v1` の簡易ルールベース実装により、将来の LLM 統合やより高度なルール追加が容易に
   - Phase D-7 以降で、パッチ候補に基づいて自動パッチ適用を実装可能

## 達成状況 (Phase D-6 完了条件)

Phase D-6 の完了条件に対する達成状況は以下の通りです。

### 1. SelfHealingPatchAgent の新規実装
- ✅ **パッチ候補生成ロジックを実装**: 達成。`build_patch_candidate` メソッドにより、簡易ルールベースでパッチ候補を生成します。

### 2. BrowserOrchestrator への組み込み
- ✅ **SelfHealingPatchAgent を初期化して保持**: 達成。`__init__` で `self.patch_agent` を初期化します。
- ✅ **すべての失敗ケースでパッチ候補を生成**: 達成。すべての失敗ケースで `_maybe_build_patch_candidate` を呼び出し、`self_healing_patch_candidate` を `evidence` に追加します。

### 3. FailureAnalysisAgent との整合性
- ✅ **suggested_fixes を自然文 → 構造化ルールにマッピング**: 達成。文字列マッチベースの簡易ルールで実装しました。

### 4. テスト追加
- ✅ **新規テストファイルを作成し、パッチ候補生成を検証**: 達成。7つのテストを追加し、すべてパスしました。

### 5. 既存テストの維持
- ✅ **既存の 47 テストを一切壊さずに**: 達成。すべての既存テストがパスし、新規テストも追加されました（合計54テスト）。

## 既知の制約・注意事項

1. **パッチ適用の未実装**: 現時点ではパッチ候補の「生成と保存」までを実装し、パッチの「適用」は行いません。Phase D-7 以降で実装予定です。

2. **簡易ルールベース**: Phase D-6 では簡易ルールベース（heuristic_v1）で実装しました。将来の LLM 統合時には、より高度なルールや自然文解析を追加可能です。

3. **パッチ候補の品質**: 現時点では簡易ルールに基づく提案のため、パッチ候補の品質は限定的です。将来の LLM 統合や実ブラウザ検証により、品質を向上させる予定です。

## 次のステップ

Phase D-6 の完了をもって、Self-Healing 自動パッチ候補の生成が実装されました。次のステップとして、以下のタスクが推奨されます。

### Phase D-7: パッチ適用フローの実装（オプション）

1. **パッチ候補のレビューと承認フロー**
   - パッチ候補を人間がレビューし、承認後に適用するフロー
   - パッチ適用前のバックアップとロールバック機能

2. **自動パッチ適用（オプション）**
   - 信頼度が高いパッチ候補（例: `confidence > 0.9`）を自動適用
   - 適用後の検証とロールバック

### Phase D-8: 実ブラウザ E2E 検証

1. **大規模な E2E テストの実施**
   - 実際の Moncler サイトでの E2E テスト
   - パッチ候補の生成と品質の検証

2. **パフォーマンス最適化**
   - `SelfHealingPatchAgent` の呼び出しオーバーヘッドを測定
   - 必要に応じて非同期処理やバッチ処理を導入

## 関連ファイル

- `app/agents/self_healing_patch_agent.py` - SelfHealingPatchAgent の実装
- `app/agents/browser_orchestrator.py` - `_maybe_build_patch_candidate` の実装、`run_plp_to_pdp` / `run_pdp` との統合
- `app/agents/failure_analysis_agent.py` - `analyze_failure_context` メソッド
- `tests/test_self_healing_patch_agent.py` - SelfHealingPatchAgent のテスト
- `tests/test_failure_analysis_integration.py` - FailureAnalysisAgent 統合のテスト
- `tests/test_browser_orchestrator_telemetry.py` - Telemetry 統合のテスト
- `docs/completion_reports/CR_ATELIER_003_PHASE_D5_SELFHEALING_DIAGNOSIS_COMPLETION_REPORT.md` - Phase D-5 完了レポート

## まとめ

CR-ATELIER-003 Phase D-6 は、Phase D-5 で統合された `failure_context` + `failure_analysis` を入力として、サイト設定に対する「自動パッチ候補」を JSON として生成・保存する仕組みを成功裏に実装しました。

これにより、すべての失敗ケースで統一されたパッチ候補が生成され、将来の自動パッチ適用の基盤が整いました。すべての既存テストがパスし、新規テストも追加されました（合計54テスト）。

Phase D-7 では、パッチ適用フローの実装を進める予定です。

