# ドキュメント

このディレクトリには、プロジェクトの各種ドキュメントが格納されています。

## 公式ドキュメント

Atelier Kyo Manager の正式な設計書は `docs/official/` 配下にあります。

- [system_design.md](official/system_design.md)
- [db_schema.md](official/db_schema.md)
- [n8n_integration.md](official/n8n_integration.md)
- [saas_design.md](official/saas_design.md)
- [architecture_overview.mmd](official/architecture_overview.mmd)

## ディレクトリ構成

### `official/refactoring/`
公式のリファクタリング完了レポートを格納するディレクトリです。重要なリファクタリング作業の完了レポートが保存されます。

例:
- `BROWSER_USE_AGENT_EXCEPTION_RETRY_REFACTOR.md`

### `completion_reports/`
完了レポートを格納するディレクトリです。各作業完了時に自動的に作成されるレポートが保存されます。

命名規則: `<作業識別子>_COMPLETION_REPORT.md`

例:
- `STAGE_3A2_COMPLETION_REPORT.md`
- `STAGE_3B_COMPLETION_REPORT.md`
- `AUTO_FIX_LOOP_COMPLETION_REPORT.md`

### `chat_records/`
チャット記録を格納するディレクトリです。各セッションで実施した作業の記録が保存されます。

命名規則: `CHAT_RECORD_YYYYMMDD_<概要>.md`

例:
- `CHAT_RECORD_20251128_AUTO_FIX_AND_TESTS.md`

### `reports/`
各種レポートをカテゴリ別に分類して格納するディレクトリです。

#### `reports/stage_implementations/`
STAGE実装に関するレポート（52個）
- STAGE実装計画、設計、テスト結果など

#### `reports/test_related/`
テスト関連のレポート（29個）
- テスト実行結果、テストガイド、テスト修正など

#### `reports/bugfixes/`
バグ修正・トラブルシューティング関連のレポート（17個）
- バグ修正手順、原因分析、修正結果など

#### `reports/analysis/`
分析レポート（4個）
- プロジェクト分析、ルール分析など

#### `reports/other/`
その他のレポート（38個）
- 上記カテゴリに該当しない各種レポート

## レポートの作成

### 完了レポートの自動作成

プロジェクトルール（`.cursorrules`）に従い、以下の作業が完了した際は自動的に完了レポートが作成されます：

- リファクタリング作業
- 機能追加
- アーキテクチャ変更
- 重要なバグ修正
- 移行作業

### 完了レポートの形式

完了レポートには以下のセクションが含まれます：

1. **実装日時**: 作業完了日
2. **概要**: 目的、ゴール、原則
3. **実装ステップ**: 各ステップの変更内容
4. **変更ファイル一覧**: 新規作成ファイル、変更ファイル
5. **動作確認結果**: 静的解析結果、コードレビュー結果、テスト結果
6. **設計上の改善点**: アーキテクチャの改善、将来の拡張性への配慮
7. **既知の制約・注意事項**: 既存コードとの互換性、制限事項
8. **次のステップ**: 推奨されるフォローアップアクション

## 移動スクリプト

ルートディレクトリに `move_all_reports.py` が用意されています。このスクリプトを実行すると、プロジェクトルートに散在している完了レポートとチャット記録が自動的に適切なディレクトリに移動されます。

```bash
python3 move_all_reports.py
```

## 今後のレポート保存先

プロジェクトルール（`.cursorrules`）に従い、今後作成される完了レポートは自動的に `docs/completion_reports/` ディレクトリに保存されます。

チャット記録は `docs/chat_records/` ディレクトリに保存することを推奨します。

## リファクタリング完了レポート

公式のリファクタリング完了レポートは `docs/official/refactoring/` 配下に格納されています。

- [BrowserUseAgent 例外処理・Retry ロジック統一](official/refactoring/BROWSER_USE_AGENT_EXCEPTION_RETRY_REFACTOR.md)

## TODO / フォローアップタスク

### BrowserUseAgent リファクタリング関連

以下のタスクは、BrowserUseAgent 例外処理・retry ロジック統一リファクタリングのフォローアップとして実施予定です。

1. **BrowserUseAgent のモジュール分割**
   - UI操作ヘルパー（`_accept_cookies_if_present`, `_dismiss_geo_modal` 等）を別モジュールに分離
   - ナビゲーション関連（`_bootstrap_session_page`, `_force_plp_recover` 等）を別モジュールに分離
   - PDP抽出関連を別モジュールに分離

2. **timeout の site_config.discovery_settings.timeout_sec への完全委譲**
   - ハードコードされた timeout 値を `site_config.discovery_settings.timeout_sec` に統一
   - すべての Playwright 操作で設定値を使用するように変更

3. **Playwright 生呼び出しの safe_* への全置換**
   - 既存のコードで直接 `page.goto()` などを呼び出している箇所を、新しい `safe_goto()` などに置き換え
   - `page.wait_for_selector()` → `safe_wait_selector()` への置き換え
   - `locator.click()` → `safe_click()` への置き換え

4. **stealth.py の共通化と browser_use_moncler_patch.py との統合**
   - サイト固有のパッチロジックを共通化
   - `browser_use_moncler_patch.py` の機能を `stealth.py` に統合または共通モジュール化

### Phase 1 ロードマップ関連

**注意**: BrowserUseAgent レベルでの例外処理・retry ロジックの統一は完了しましたが、上位エージェント側（FailureAnalysisAgent / SelfHealingAgent / SelectorDiscoveryAgent）での例外分類情報の活用は、各エージェントのリファクタリングタスクとして別途実施予定です。

詳細は [BrowserUseAgent 例外処理・Retry ロジック統一リファクタリング完了レポート](official/refactoring/BROWSER_USE_AGENT_EXCEPTION_RETRY_REFACTOR.md) を参照してください。

