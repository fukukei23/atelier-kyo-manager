# CR-ATELIER-002 Step 6 完了レポート

## 実装日時

2025年12月8日

## 概要

CR-ATELIER-002 Step 6「Self-Healing & Selector Discovery 実装フェーズ」を完了しました。

本ステップでは、Step 5で設計した以下を実コード化しました：

1. **Telemetry: `moncler_plp_pdp_outcome` の実装**
2. **Self-Healing Agent: Moncler 用の自己修復解析**
3. **Selector Discovery Agent: DOM から新セレクタを提案**
4. **NavigationDriver / Extractor からの連携フロー実装**
5. **pytest による検証コードの追加**

### 目的

- Moncler PLP→PDP 抽出の最終結果を構造化された Telemetry として保存
- 抽出失敗時に Self-Healing Agent と Selector Discovery Agent を自動的に起動
- 失敗分析とセレクタ提案の自動化

### ゴール

Step 6 では、以下の実装を完了することを目指しました：

- Telemetry に `moncler_plp_pdp_outcome` を記録できるようになった
- Self-Healing Agent が Moncler 専用の失敗ハンドリングを実行できるようになった
- Selector Discovery Agent が Moncler 専用のセレクタ提案を実行できるようになった
- NavigationDriver から Self-Healing / Selector Discovery を自動的にトリガーできるようになった
- すべてのテストがパスする状態になった

## 実装ステップ

### Step 6-1: Telemetry ヘルパー (telemetry.py) の追加

**変更ファイル**: `app/agents/browser/telemetry.py`

**実施内容**:
- `TelemetryService` に `record_moncler_plp_pdp_outcome()` メソッドを追加
- 設計書に準拠した outcome データ構造を保存
- 既存の `_save_json()` メソッドを活用

**追加したメソッド**:
```python
async def record_moncler_plp_pdp_outcome(
    self,
    outcome: Dict[str, Any],
) -> None:
    """
    CR-ATELIER-002 Step 6: Moncler PLP→PDP 抽出結果を Telemetry に記録
    
    Args:
        outcome: 抽出結果の辞書。以下のフィールドを含む:
            - plp_materialized: bool
            - tiles_detected: int
            - pdp_links_raw: int
            - pdp_links_accepted: int
            - selector_layers_used: list[str]
            - layer_stats: dict
            - locale_corrections: int
            - trap_detected: bool
            - current_url: str
            - run_id: str
            - timestamp: str
    """
```

**実装のポイント**:
- `timestamp` がなければ自動生成（ISO8601形式）
- キー名は設計書に準拠（`moncler_plp_pdp_outcome`）
- 既存の `_save_json()` メソッドを活用して実装

### Step 6-2: Extractor 側の outcome 生成 (extractor.py)

**変更ファイル**: `app/agents/browser/extractor.py`

**実施内容**:
- `extract_moncler_pdp_links()` を拡張して outcome 情報を生成
- 使用されたレイヤ（primary / secondary / tertiary）を判定
- outcome 情報を ctx に格納（後方互換性を保つ）

**追加したロジック**:
```python
# 使用されたレイヤを判定
layers_used: List[str] = []
if layer_stats.get("primary_raw", 0) > 0 or layer_stats.get("primary_accepted", 0) > 0:
    layers_used.append("primary")
if layer_stats.get("secondary_raw", 0) > 0 or layer_stats.get("secondary_accepted", 0) > 0:
    layers_used.append("secondary")
if layer_stats.get("tertiary_raw", 0) > 0 or layer_stats.get("tertiary_accepted", 0) > 0:
    layers_used.append("tertiary")

# outcome 情報を構築
outcome_info = {
    "links": urls,
    "raw_count": len(raw_hrefs),
    "accepted_count": len(urls),
    "layer_stats": layer_stats,
    "layers_used": layers_used,
    "rejection_stats": rejection_stats,
    "current_url": target_url,
}

# ctx に格納（後方互換性のため）
if isinstance(ctx, dict):
    ctx["moncler_outcome"] = outcome_info
elif hasattr(ctx, "__dict__"):
    try:
        setattr(ctx, "moncler_outcome", outcome_info)
    except Exception:
        pass
```

**実装のポイント**:
- 既存の戻り値（`List[str]`）を維持し、outcome 情報は ctx に格納
- NavigationDriver 側で outcome 情報を取得できるように設計

### Step 6-3: NavigationDriver から Telemetry 保存＋トリガ判定 (navigation_driver.py)

**変更ファイル**: `app/agents/browser/navigation_driver.py`

**実施内容**:

1. **NavigationOutcome の拡張**:
   - `locale_corrections: int = 0` フィールドを追加
   - `moncler_outcome: Optional[Dict[str, Any]] = None` フィールドを追加

2. **Locale補正の回数カウント**:
   - `run_plp_flow()` の最初で `locale_correction_count = 0` を初期化
   - `_ensure_expected_locale()` が呼ばれるたびに、URL が変更された場合のみカウント
   - Recovery 後、Header search fallback 後など、すべての Locale Guard 呼び出しでカウント

3. **Telemetry への outcome 保存**:
   - `collect_pdp_links()` の後、Moncler 専用の outcome 情報を取得
   - `tiles_detected` を取得（materialized または tiles_detected が True の場合）
   - outcome dict を構築して Telemetry に保存

4. **Self-Healing / Selector Discovery のトリガー判定**:
   - トリガ条件をチェック：
     - `pdp_links_raw == 0`
     - `pdp_links_accepted == 0`
     - `layers_used` に "secondary" や "tertiary" が含まれる
     - `trap_detected == True`
     - `locale_corrections >= 3`
   - トリガ条件を満たす場合、`_trigger_moncler_self_healing()` を呼び出す

5. **`_trigger_moncler_self_healing()` メソッドの実装**:
   - Self-Healing Agent と Selector Discovery Agent を呼び出す
   - failure_payload と discovery_payload を構築
   - 各エージェントの Moncler 専用メソッドを呼び出す

**実装のポイント**:
- Locale補正の回数は、URL が実際に変更された場合のみカウント
- Telemetry 保存と Self-Healing トリガーは、Moncler サイトの場合のみ実行
- エラーハンドリングを適切に行い、失敗しても続行できるように設計

### Step 6-4: Self-Healing Agent 実装（Moncler 専用ハンドラ追加）

**変更ファイル**: `app/agents/self_healing_agent.py`

**実施内容**:
- `SelfHealingAgent` に `handle_moncler_failure()` メソッドを追加
- 失敗理由に基づいて分析と提案を生成
- 実際の site_config 書き換えは行わず、提案のみを返す

**追加したメソッド**:
```python
async def handle_moncler_failure(self, failure_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    CR-ATELIER-002 Step 6-4: Moncler 専用の失敗ハンドリング
    
    Args:
        failure_payload: 失敗情報を含む辞書
            - site: str
            - url: str
            - failure_reason: str
            - dom_snapshot_path: Optional[str]
            - layer_stats: Dict[str, Any]
            - rejection_stats: Dict[str, Any]
            - selectors_current: Dict[str, Any]
            - run_id: Optional[str]
            - timestamp: Optional[str]
    
    Returns:
        Dict[str, Any]: 分析結果
            - analysis: str
            - root_cause: str
            - suggested_actions: List[str]
            - confidence: float
    """
```

**実装のポイント**:
- 失敗理由（`raw_zero`, `rejected_all`, `secondary_or_tertiary_used`, `trap_detected`, `locale_corrections_exceeded`）に基づいて分析
- 提案はログに保存（将来の site_config パッチ作成に使用）
- 実際の site_config 書き換えは Step 7 以降の責務

### Step 6-5: Selector Discovery Agent 実装

**変更ファイル**: `app/agents/selector_discovery_agent.py`

**実施内容**:
- `SelectorDiscoveryAgent` に `propose_moncler_selectors()` メソッドを追加
- ルールベースでセレクタ候補を生成（将来は LLM を使用可能）
- 最低3件の候補を返す

**追加したメソッド**:
```python
async def propose_moncler_selectors(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    CR-ATELIER-002 Step 6-5: Moncler 専用のセレクタ提案
    
    Args:
        payload: セレクタ発見のための情報
            - dom_snapshot_path: Optional[str]
            - selectors_current: Dict[str, Any]
            - layer_stats: Dict[str, Any]
            - rejection_stats: Dict[str, Any]
            - run_id: Optional[str]
    
    Returns:
        Dict[str, Any]: セレクタ提案結果
            - candidate_selectors: List[str]
            - confidence_scores: List[float]
            - recommended_layer: str ("primary" | "secondary" | "tertiary")
    """
```

**実装のポイント**:
- Primary / Secondary / Tertiary の各レイヤで失敗した場合、適切な候補を生成
- 既存のセレクタと重複を排除
- 信頼度の高い順にソート
- 最低3件の候補を返す（不足する場合は既存のセレクタから追加）

### Step 6-6: pytest 追加・修正

**新規作成ファイル**:
- `tests/test_moncler_self_healing.py`: Self-Healing Agent のテスト
- `tests/test_moncler_selector_discovery.py`: Selector Discovery Agent のテスト

**実施内容**:

1. **`test_moncler_self_healing.py`**:
   - `test_handle_moncler_failure_raw_zero`: raw_zero の場合のハンドリング
   - `test_handle_moncler_failure_rejected_all`: rejected_all の場合のハンドリング
   - `test_handle_moncler_failure_secondary_used`: secondary_or_tertiary_used の場合のハンドリング
   - `test_handle_moncler_failure_trap_detected`: trap_detected の場合のハンドリング
   - `test_handle_moncler_failure_locale_corrections_exceeded`: locale_corrections_exceeded の場合のハンドリング
   - `test_handle_moncler_failure_unknown`: 不明な失敗理由の場合のハンドリング
   - `test_failure_payload_format`: failure_payload のフォーマット検証

2. **`test_moncler_selector_discovery.py`**:
   - `test_propose_moncler_selectors_minimum_three`: 最低3件のセレクタを返すこと
   - `test_propose_moncler_selectors_recommended_layer`: recommended_layer が正しい値であること
   - `test_propose_moncler_selectors_primary_success`: Primary layer が成功した場合のテスト
   - `test_propose_moncler_selectors_secondary_success`: Secondary layer が成功した場合のテスト
   - `test_propose_moncler_selectors_tertiary_success`: Tertiary layer が成功した場合のテスト
   - `test_propose_moncler_selectors_confidence_scores_sorted`: confidence_scores が降順でソートされていること
   - `test_propose_moncler_selectors_no_duplicates`: 既存のセレクタと重複しないこと

**テスト結果**:
```
============================== 31 passed in 1.78s ==============================
```

すべてのテストがパスしました。

## 変更ファイル一覧

### 新規作成ファイル

1. **`tests/test_moncler_self_healing.py`**:
   - Self-Healing Agent の Moncler 専用ハンドリングのテスト（7件）

2. **`tests/test_moncler_selector_discovery.py`**:
   - Selector Discovery Agent の Moncler 専用セレクタ提案のテスト（7件）

### 変更ファイル

1. **`app/agents/browser/telemetry.py`**:
   - `TelemetryService.record_moncler_plp_pdp_outcome()` メソッドを追加

2. **`app/agents/browser/extractor.py`**:
   - `extract_moncler_pdp_links()` を拡張して outcome 情報を生成
   - outcome 情報を ctx に格納

3. **`app/agents/browser/navigation_driver.py`**:
   - `NavigationOutcome` に `locale_corrections` と `moncler_outcome` フィールドを追加
   - `run_plp_flow()` で Locale補正の回数をカウント
   - Telemetry に `moncler_plp_pdp_outcome` を保存
   - Self-Healing / Selector Discovery のトリガー判定と実行
   - `_trigger_moncler_self_healing()` メソッドを追加

4. **`app/agents/self_healing_agent.py`**:
   - `SelfHealingAgent.handle_moncler_failure()` メソッドを追加

5. **`app/agents/selector_discovery_agent.py`**:
   - `SelectorDiscoveryAgent.propose_moncler_selectors()` メソッドを追加

## 動作確認結果

### 静的解析結果

- **リンター**: エラーなし
- **型チェッカー**: エラーなし

### テスト結果

**pytest実行結果**:
```
============================== 31 passed in 1.78s ==============================
```

**テスト内容**:
- `test_moncler_pdp_url.py`: 17件のテスト（既存）
- `test_moncler_self_healing.py`: 7件のテスト（新規）
- `test_moncler_selector_discovery.py`: 7件のテスト（新規）

**すべてのテストがパスしました。**

### 実ブラウザ検証（未実施）

実ブラウザ検証は Step 7 以降で実施予定です。

## 設計上の改善点

### アーキテクチャの改善

1. **Telemetry の構造化**:
   - `moncler_plp_pdp_outcome` として構造化されたデータを保存
   - レイヤごとの統計情報を記録
   - Locale補正の回数を記録

2. **Self-Healing の自動化**:
   - 抽出失敗時に自動的に Self-Healing Agent を起動
   - 失敗理由に基づいた分析と提案を生成
   - 将来の site_config パッチ作成に活用可能

3. **Selector Discovery の自動化**:
   - 抽出失敗時に自動的に Selector Discovery Agent を起動
   - セレクタ候補を自動生成
   - 信頼度の高い順にソート

### 将来の拡張性への配慮

1. **LLM 統合の準備**:
   - Selector Discovery Agent は現時点ではルールベースだが、将来は LLM を使用可能な設計
   - DOM スナップショットのパスを payload に含めることで、LLM による分析が可能

2. **他サイトへの展開**:
   - Moncler 専用の実装だが、他のサイトにも適用可能な設計パターンを確立
   - エージェント間の連携フローを明確化

3. **site_config 自動パッチ**:
   - Self-Healing Agent と Selector Discovery Agent の提案を、将来の site_config 自動パッチ作成に活用可能
   - Step 7 以降で実装予定

### コード品質の向上

1. **エラーハンドリング**:
   - Telemetry 保存や Self-Healing トリガーが失敗しても続行できるように設計
   - 適切なログ出力を実装

2. **テストの充実**:
   - Self-Healing Agent と Selector Discovery Agent のテストを追加
   - すべての失敗理由とレイヤパターンをテスト

3. **ログの充実**:
   - `[Telemetry][Moncler]`, `[SelfHealing][Moncler]`, `[SelectorDiscovery][Moncler]` などのプレフィックスを付与
   - grep で検索しやすいログ設計

## 既知の制約・注意事項

### 既存コードとの互換性

- 既存の `extract_moncler_pdp_links()` の戻り値（`List[str]`）を維持
- outcome 情報は ctx に格納することで、既存コードとの互換性を保持

### 制限事項やトレードオフ

1. **site_config 自動パッチ**:
   - Self-Healing Agent と Selector Discovery Agent の提案は生成されるが、実際の site_config 書き換えは Step 7 以降の責務
   - 現時点では提案のみをログに保存

2. **LLM 統合**:
   - Selector Discovery Agent は現時点ではルールベース
   - 将来は LLM を使用してより高度なセレクタ提案が可能

3. **実ブラウザ検証**:
   - 実ブラウザ検証は Step 7 以降で実施予定
   - 現時点では pytest によるユニットテストのみ実施

### 移行時の注意点

- `NavigationOutcome` に新しいフィールドを追加したため、既存コードで `NavigationOutcome` を使用している場合は確認が必要
- `locale_correction_count` のカウントロジックは、URL が実際に変更された場合のみカウントするように実装

## 次のステップ

### 推奨されるフォローアップアクション

1. **Step 7: site_config 自動パッチ適用**:
   - Self-Healing Agent と Selector Discovery Agent の提案を site_config に自動適用
   - 提案の検証と適用ロジックの実装

2. **実ブラウザ検証**:
   - 実ブラウザで Moncler PLP→PDP 抽出を実行
   - Telemetry に `moncler_plp_pdp_outcome` が保存されることを確認
   - Self-Healing / Selector Discovery が適切にトリガーされることを確認

3. **LLM 統合**:
   - Selector Discovery Agent に LLM を統合して、より高度なセレクタ提案を実現
   - DOM スナップショットを LLM に渡して分析

4. **他サイトへの展開**:
   - Moncler 以外のサイトにも Self-Healing / Selector Discovery を適用
   - 汎用的な設計パターンを確立

5. **完了レポートの作成**:
   - CR-ATELIER-002全体の完了レポートを作成
   - Step 1〜6の統合的な評価を実施

### 関連ファイル

- `docs/spec/CR-ATELIER-002_STEP5_SPEC.md`: Step 5 の設計書
- `docs/spec/CR-ATELIER-002_STEP5_SELF_HEALING_INTERFACE.md`: Self-Healing / Selector Discovery 連携設計
- `app/agents/browser/telemetry.py`: TelemetryService
- `app/agents/browser/extractor.py`: Moncler 専用 PDP 抽出ロジック
- `app/agents/browser/navigation_driver.py`: NavigationDriver
- `app/agents/self_healing_agent.py`: Self-Healing Agent
- `app/agents/selector_discovery_agent.py`: Selector Discovery Agent
- `tests/test_moncler_self_healing.py`: Self-Healing Agent のテスト
- `tests/test_moncler_selector_discovery.py`: Selector Discovery Agent のテスト

---

**作成者**: AI Assistant  
**レビュー**: 未実施  
**ステータス**: 実装完了（実ブラウザ検証は Step 7 以降）

