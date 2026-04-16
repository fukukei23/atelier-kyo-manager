# CR-ATELIER-003 Phase D-7 完了レポート: Self-Healing パッチ適用フローの実装（オフライン適用ツール）

## 実装日時

2025年12月10日

## 概要

Phase D-6 で生成される `self_healing_patch_candidate`（`patch_candidate_self_healing.json`）を、人間レビュー前提の「オフライン適用ツール」を通じて `overrides.local.json` に安全に適用できる状態にした。

このフェーズでは「パッチ候補の適用ツール」と「単体テスト」までを実装し、Orchestrator 実行中の自動適用は行わない（将来フェーズに委譲）。

## 実装ステップ

### Step 1: 事前調査

**実施内容**:
- `patch_applier` という名前の専用ユーティリティは存在しないことを確認
- 既存の `moncler_patch_builder.py` を確認し、パッチ候補生成ロジックを理解
- `overrides.local.json` の構造を確認

**結果**:
- JSON ファイルを直接操作するシンプルな実装を採用
- RFC6902 スタイルの JSON Patch を参考にした diff 形式を採用

### Step 2: SelfHealingPatchAdapter の実装

**ファイル**: `app/agents/self_healing_patch_adapter.py`

**実装内容**:
- `SelfHealingPatchAdapter` クラスを実装
- `to_diff()` メソッドで `patch_candidate` の `changes` を diff 形式に変換
- 対応する action:
  - `increase`: 数値の増加（例: `timeout_sec` +30秒）
  - `review_required`: selector 要再確認フラグ（`__meta_flags/need_review` を追加）
  - `set`: 値を直接設定

**変換ルール**:
- dot 区切りのパス（例: `"discovery_settings.timeout_sec"`）を JSON Pointer 形式（例: `"/discovery_settings/timeout_sec"`）に変換
- `current_value` が指定されていない場合、`site_config` から現在の値を取得
- `value_delta` を加算して新しい値を計算

### Step 3: SelfHealingPatchApplier の実装

**ファイル**: `app/agents/self_healing_patch_applier.py`

**実装内容**:
- `SelfHealingPatchApplier` クラスを実装
- `apply_patch_candidate()` メソッドでパッチ候補を `overrides.local.json` に適用
- 処理フロー:
  1. パッチ候補を JSON から読み込み
  2. `overrides.local.json` を読み込み
  3. 対象サイトの `site_config` を取得
  4. `SelfHealingPatchAdapter.to_diff()` で diff を生成
  5. バックアップファイルを作成（`.bak-YYYYMMDD-HHMMSS` 形式）
  6. diff を適用
  7. 更新された `overrides.local.json` を保存
  8. 適用結果を返す

**エラーハンドリング**:
- ファイルが存在しない場合、適切なエラーメッセージを返す
- diff 適用に失敗した場合、バックアップから復元
- 無効な JSON の場合、エラーメッセージを返す

### Step 4: CLI スクリプトの実装

**ファイル**: `scripts/apply_self_healing_patch.py`

**実装内容**:
- コマンドライン引数:
  - `--run-id`: Run ID（必須）
  - `--site`: サイトコード（必須）
  - `--overrides`: `overrides.local.json` のパス（オプション、デフォルト: `app/config/sites/overrides.local.json`）
  - `--candidate-path`: パッチ候補ファイルのパス（オプション、デフォルト: `instance/runs/<run_id>/patch_candidate_self_healing.json`）
  - `--dry-run`: ドライランモード（実際には適用しない）
- 適用前の確認メッセージを表示
- 適用結果をログに出力

**使用例**:
```bash
python scripts/apply_self_healing_patch.py \
    --run-id "RUN_20251210_XXXXXX" \
    --site "MONCLER_OFFICIAL" \
    --overrides "app/config/sites/overrides.local.json"
```

### Step 5: テストの実装

**ファイル**:
- `tests/test_self_healing_patch_adapter.py`（9テスト）
- `tests/test_self_healing_patch_applier.py`（7テスト）

**テスト内容**:

**SelfHealingPatchAdapter**:
- `timeout_sec` increase の変換
- `current_value` が指定されていない場合の処理
- selector review required フラグの追加
- `set` action の処理
- 空の changes の処理
- 複数の changes の処理
- JSON Pointer 変換の確認
- ネストされた値の取得
- パス存在チェック

**SelfHealingPatchApplier**:
- 正常系: パッチ候補が正常に適用される
- changes が空の場合、適用されない
- パッチ候補ファイルが存在しない場合、エラーを返す
- `overrides.local.json` が存在しない場合、エラーを返す
- 対象サイトが存在しない場合、エラーを返す
- 無効な JSON の場合、エラーを返す
- 複数の changes が含まれる場合、すべて適用される

## 変更ファイル一覧

### 新規作成ファイル

1. `app/agents/self_healing_patch_adapter.py`
   - Self-Healing パッチ候補を diff 形式に変換するアダプタ

2. `app/agents/self_healing_patch_applier.py`
   - パッチ候補を `overrides.local.json` に適用するアプライヤ

3. `scripts/apply_self_healing_patch.py`
   - オフライン実行用の CLI スクリプト

4. `tests/test_self_healing_patch_adapter.py`
   - `SelfHealingPatchAdapter` のテスト（9テスト）

5. `tests/test_self_healing_patch_applier.py`
   - `SelfHealingPatchApplier` のテスト（7テスト）

### 変更ファイル

なし（新規実装のみ）

## 動作確認結果

### テスト結果

```bash
python -m pytest tests/test_self_healing_patch_adapter.py tests/test_self_healing_patch_applier.py tests/test_self_healing_patch_agent.py tests/test_failure_analysis_integration.py tests/test_browser_orchestrator_telemetry.py tests/test_browser_use_agent_plp_integration.py tests/test_plp_driver.py tests/test_moncler_pdp_url.py -q
```

**結果**: 70 passed, 17 warnings

### 静的解析結果

- リンターエラー: なし
- 型チェッカー: 警告のみ（実行時には問題なし）

### CLI スクリプトの動作確認

```bash
python scripts/apply_self_healing_patch.py --help
```

**結果**: 正常にヘルプメッセージが表示される

## 設計上の改善点

### 1. 拡張性

- `SelfHealingPatchAdapter` は `strategy` フィールドに対応しており、将来的に異なる戦略（例: `heuristic_v2`, `llm_based`）を追加可能
- `action` の種類を追加する場合、`to_diff()` メソッドに新しい分岐を追加するだけで対応可能

### 2. 安全性

- バックアップファイルを自動生成し、適用失敗時に復元可能
- `--dry-run` オプションで実際の適用前に確認可能
- 適用前に対象ファイル・サイトコード・変更内容を表示

### 3. エラーハンドリング

- ファイルが存在しない場合、適切なエラーメッセージを返す
- 無効な JSON の場合、エラーメッセージを返す
- diff 適用に失敗した場合、バックアップから復元

## 既知の制約・注意事項

### 1. 自動適用は未実装

- Phase D-7 では、Orchestrator 実行中の自動適用は実装していない
- すべてのパッチ適用は手動で `scripts/apply_self_healing_patch.py` を実行する必要がある

### 2. Moncler 専用の簡易ルール

- 現在の実装は Moncler 向けの簡易ルール（`heuristic_v1`）のみ対応
- 他のサイトやより複雑なルールには対応していない

### 3. CSS セレクタの自動生成は未実装

- selector 要再確認フラグは追加するが、実際の CSS セレクタ文字列の自動生成は未実装
- これは Phase D-8 以降で検討する

### 4. メタデータフィールドの扱い

- `__meta_flags/need_review` は `overrides.local.json` に追加されるが、既存のコードがこのフィールドを参照するかは未確認
- 将来的には、このフィールドを参照するロジックを追加する必要がある

## 次のステップ

### Phase D-8（将来フェーズ）

1. **自動適用フローの実装**
   - `BrowserOrchestrator` から `SelfHealingPatchApplier` を呼び出す
   - 適用前に人間の承認を求める仕組み（オプション）

2. **LLM による CSS セレクタ生成**
   - `FailureAnalysisAgent` や `SelectorDiscoveryAgent` の結果から、実際の CSS セレクタ文字列を生成
   - 生成されたセレクタを `overrides.local.json` に適用

3. **汎用パッチルールの実装**
   - Moncler 以外のサイトにも対応できる汎用的なパッチルール
   - サイトごとのカスタマイズ可能なルール設定

4. **メタデータフィールドの活用**
   - `__meta_flags/need_review` を参照するロジックの追加
   - セレクタの優先順位や信頼度の管理

### その他の改善点

1. **パッチ適用の履歴管理**
   - 適用されたパッチの履歴を保存
   - ロールバック機能の強化

2. **パッチ適用前の検証**
   - `pytest` を自動実行して、パッチ適用後の動作を確認
   - 問題がある場合は適用を中止

3. **パッチ候補の品質評価**
   - パッチ候補の信頼度やリスクを評価
   - 低品質なパッチ候補は自動的に却下

## まとめ

Phase D-7 では、Self-Healing パッチ候補を `overrides.local.json` に安全に適用するオフライン適用ツールを実装した。すべてのテストがパスし、CLI スクリプトも正常に動作することを確認した。

次のフェーズ（Phase D-8）では、自動適用フローや LLM による CSS セレクタ生成などの高度な機能を実装する予定。

