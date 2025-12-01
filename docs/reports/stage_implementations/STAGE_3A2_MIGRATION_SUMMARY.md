# Stage 3A-2 移行サマリー

## 完了した作業

### 1. NavigationDriver への機能追加

#### 1.1 `_looks_like_trap_or_legal` メソッドの移動
- **場所**: `app/agents/browser/navigation_driver.py`
- **実装**: 静的メソッド `NavigationDriver.looks_like_trap_or_legal(url: str) -> bool`
- **内容**: BrowserUseAgent から移動した trap 判定ロジック
  - URL正規化（/en-jp/en-int/ → /en-int/）
  - ハッシュ除去
  - ロケール判定（/en-jp, コーポレートサイト、ロケーションゲート、リーガルキーワード）

#### 1.2 `_force_plp_recover` コールバックの追加
- **型定義**: `RecoveryFn = Callable[[Page, Dict[str, Any], Optional[str]], Awaitable[None]]`
- **実装**: NavigationDriver の `__init__` に `recovery_fn: Optional[RecoveryFn] = None` を追加
- **用途**: BrowserUseAgent の `_force_plp_recover` をコールバックとして受け取る

#### 1.3 `NavigationOutcome` の拡張
- **追加フィールド**: `recovered: bool = False`
- **用途**: 回復試行が実行されたかどうかを記録

#### 1.4 `run_plp_flow` の拡張
- **追加パラメータ**: `target_url: Optional[str] = None`
- **実装内容**:
  1. **初期 trap 判定**: `looks_like_trap_or_legal` で判定
  2. **回復試行**: `recovery_fn` が提供されている場合、trap 検出時に自動回復を試行
  3. **materialize**: 既存の `ensure_plp_materialized` を呼び出し
  4. **materialize 後の trap 再チェック**: materialize 後に再度 trap 判定を行い、必要に応じて回復試行

### 2. BrowserUseAgent の更新

#### 2.1 NavigationDriver インスタンス化の更新
- **場所**: `app/agents/browser_use_agent.py` (1797行目付近)
- **変更**: `recovery_fn` パラメータを追加
  ```python
  recovery_fn=lambda pg, scfg, t_url: self._force_plp_recover(pg, scfg, t_url)
  ```

#### 2.2 `run_plp_flow` 呼び出しの更新
- **変更**: `target_url` パラメータを追加
  ```python
  nav_outcome = await navigation_driver.run_plp_flow(nav_ctx, target_url=nav_url)
  ```

#### 2.3 `_run_plp_flow` メソッドの更新
- **追加パラメータ**: `nav_outcome: Optional[Any] = None`
- **実装内容**:
  - NavigationDriver の結果を利用して、初期 trap 判定と回復試行をスキップ
  - `nav_outcome.recovered` が `True` の場合、従来の trap 判定処理をスキップ
  - `nav_outcome.trap_detected` が `True` の場合、早期エラーを返す

## 移行の安全性

### ✅ 後方互換性
- NavigationDriver が使われていない場合（`nav_outcome=None`）、従来の処理が実行される
- 既存の `_looks_like_trap_or_legal` メソッドは BrowserUseAgent に残っている（段階的移行のため）

### ✅ エラーハンドリング
- NavigationDriver の処理が失敗した場合、従来の処理にフォールバック
- trap 検出時は適切なエラーメッセージを返す

### ✅ テスト容易性
- NavigationDriver の機能は独立してテスト可能
- コールバック関数により、BrowserUseAgent の実装詳細から分離

## 完了した追加作業（Stage 3A-2 続き）

### materialize 後の trap 再チェックの移行 ✅
- **場所**: `app/agents/browser_use_agent.py` (1904-1916行目)
- **実装内容**:
  - NavigationDriver の結果を利用して、materialize 後の trap 再チェックをスキップ
  - `nav_outcome.plp_materialized` が `True` の場合、NavigationDriver で処理済みとみなす
  - NavigationDriver が使われていない場合のみ、従来の処理を実行

### 次のステップ（Stage 3A-2 続き）

### 残りの移行対象
1. **`_pause_for_operator` の呼び出し**: NavigationDriver に移動するか検討（オプショナル）
2. **完全な移行**: BrowserUseAgent の `_looks_like_trap_or_legal` を NavigationDriver の静的メソッド呼び出しに置き換え
3. **materialize 失敗時の処理**: NavigationDriver で materialize が失敗した場合のエラーハンドリングを強化

### 推奨される次の作業
1. NavigationDriver の `run_plp_flow` に `pause_callback` を追加（オプショナル）
2. BrowserUseAgent の `_looks_like_trap_or_legal` を NavigationDriver の静的メソッド呼び出しに置き換え
3. NavigationDriver で materialize が失敗した場合のエラーハンドリングを強化

## ファイル変更サマリー

### 変更ファイル
1. `app/agents/browser/navigation_driver.py`
   - `looks_like_trap_or_legal` 静的メソッドを追加
   - `RecoveryFn` 型を追加
   - `NavigationOutcome.recovered` フィールドを追加
   - `run_plp_flow` に初期 trap 判定と回復試行ロジックを追加

2. `app/agents/browser_use_agent.py`
   - NavigationDriver インスタンス化時に `recovery_fn` を追加
   - `run_plp_flow` 呼び出し時に `target_url` を追加
   - `_run_plp_flow` に `nav_outcome` パラメータを追加
   - NavigationDriver の結果を利用するように更新

### テスト推奨事項
1. NavigationDriver の `looks_like_trap_or_legal` のテスト
2. NavigationDriver の `run_plp_flow` のテスト（trap 検出、回復試行、materialize）
3. BrowserUseAgent の統合テスト（NavigationDriver との連携）

## 注意事項

- `_looks_like_trap_or_legal` は BrowserUseAgent にも残っている（段階的移行のため）
- materialize 後の trap 再チェックは NavigationDriver に移行済み（NavigationDriver が使われていない場合のみ従来処理を実行）
- materialize 失敗時の処理は `_run_plp_flow` で行っている（段階的移行のため）

## 移行の進捗状況

### ✅ 完了
1. 初期 trap 判定と回復試行 → NavigationDriver に移行済み
2. materialize 後の trap 再チェック → NavigationDriver に移行済み
3. NavigationDriver の結果を利用するロジック → 実装済み

### ⏳ 残り
1. `_pause_for_operator` の呼び出し（オプショナル）
2. BrowserUseAgent の `_looks_like_trap_or_legal` を NavigationDriver の静的メソッド呼び出しに置き換え
3. materialize 失敗時の処理を NavigationDriver に移行（オプショナル）

