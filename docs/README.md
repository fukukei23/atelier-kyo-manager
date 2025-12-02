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

