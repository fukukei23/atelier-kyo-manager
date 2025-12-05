# CR-ATELIER-SPEC-STANDARDIZATION 完了レポート

## 実装日時
2025-12-04

## 概要

atelier-kyo-manager プロジェクトに対して、NexusCore プロジェクトで実施した「Spec 管理ルールの標準化」と同等の作業を実施しました。

### 目的
- 仕様書（Spec）を `docs/spec/` 配下に集約して管理する
- Spec 用テンプレートを 1 つ用意する
- README と .cursorrules に「Spec 保存ルール」を明文化する
- 以後、新しい CR や機能実装の前に「Spec を書く／保存する」運用を徹底する

### 原則
- 既存の Spec Kit Mode 設定（`.spec/` ディレクトリ）との互換性を維持
- 既存の完了レポートやドキュメントへの影響を最小限に抑制
- ファイル命名規則を `CR-ATELIER-XXX_...` 形式に統一

## 実施内容サマリー

1. **ディレクトリ構造の整備**
   - `docs/spec/` ディレクトリを作成
   - `docs/spec/SPEC_TEMPLATE.md` を作成

2. **既存 Spec の移動**
   - `docs/moncler_plp_autorun_spec.md` → `docs/spec/CR-ATELIER-MONCLER-PLP-AUTORUN.md`
   - `.spec/CR-AKM-001_MONCLER_PLP_TELEMETRY_AND_PLP_STATE_RECORDING.md` → `docs/spec/CR-ATELIER-001_MONCLER_PLP_TELEMETRY_AND_PLP_STATE_RECORDING.md`

3. **ルールの明文化**
   - `.cursorrules` に「Spec Storage Rules」と「Spec Auto-Generate Rules」を追記
   - `README.md` に「Specification (Spec) 管理ルール」セクションを追加

## 追加 / 変更 / 移動されたファイル一覧

### 新規作成ファイル
- `docs/spec/SPEC_TEMPLATE.md` - Spec テンプレート
- `docs/spec/CR-ATELIER-SPEC-STANDARDIZATION_COMPLETION_REPORT.md` - 本完了レポート

### 移動されたファイル
- `docs/moncler_plp_autorun_spec.md` → `docs/spec/CR-ATELIER-MONCLER-PLP-AUTORUN.md`
  - メタデータ（Project, CreatedAt, SourcePath, MovedAt）を追加
- `.spec/CR-AKM-001_MONCLER_PLP_TELEMETRY_AND_PLP_STATE_RECORDING.md` → `docs/spec/CR-ATELIER-001_MONCLER_PLP_TELEMETRY_AND_PLP_STATE_RECORDING.md`
  - メタデータを追加し、CR番号を `CR-ATELIER-001` に統一

### 変更されたファイル
- `.cursorrules`
  - 「Spec Storage Rules (atelier-kyo-manager)」セクションを追加
  - 「Spec Auto-Generate Rules (atelier-kyo-manager)」セクションを追加
- `README.md`
  - 「Specification (Spec) 管理ルール」セクションを追加（セクション7として挿入）
  - 既存の「注意事項（規約）」をセクション8に変更

## 既存 Spec の有無と対応内容

### 発見された既存 Spec ファイル

1. **`docs/moncler_plp_autorun_spec.md`**
   - **内容**: Moncler PLP 自動回復フロー仕様書
   - **作成日**: 2025-11-15
   - **対応**: `docs/spec/CR-ATELIER-MONCLER-PLP-AUTORUN.md` に移動し、メタデータを追加

2. **`.spec/CR-AKM-001_MONCLER_PLP_TELEMETRY_AND_PLP_STATE_RECORDING.md`**
   - **内容**: Moncler PLP Telemetry & PLP State Recording の仕様書
   - **作成日**: 2025-12-04
   - **対応**: `docs/spec/CR-ATELIER-001_MONCLER_PLP_TELEMETRY_AND_PLP_STATE_RECORDING.md` に移動し、CR番号を `CR-ATELIER-001` に統一

### 移動対象外のファイル

- `.spec/templates/spec_template.md` - これは Spec Kit Mode 用のテンプレートのため、そのまま残す
- `docs/moncler/PLP_EXTRACTION_FIX_TASK_TEMPLATE.md` - タスクテンプレートであり、Spec ではないため移動対象外

## .cursorrules に追加したルールの概要

### Spec Storage Rules (atelier-kyo-manager)

- 仕様書（Spec）、CR プロンプト、設計メモで長期的に参照するものは、必ず `docs/spec/` 以下に保存する
- ファイル名は `CR-ATELIER-<ID>_... .md` 形式を推奨
- 新しい CR や大きな実装タスクを開始する前に、仕様書が存在しない場合は `docs/spec/SPEC_TEMPLATE.md` をコピーして Spec を作成する

### Spec Auto-Generate Rules (atelier-kyo-manager)

- ユーザーからの自然文の「やりたいこと」が長文で与えられた場合、いきなりコードを変更せず、まず `docs/spec/` に CR-ATELIER-XXX_... .md として Spec を起こすことを優先する
- Spec 作成後、その Spec を前提として CR プロンプト（Cursor 用指示書）を生成し、実装フェーズに進む
- Spec が無いまま大きな機能追加・リファクタリングを行うことは禁止

## README に追記した内容の概要

### Specification (Spec) 管理ルール（セクション7）

以下の4項目を追加：

1. このプロジェクトの仕様書（Spec）は `docs/spec/` に集約します
2. 新しい CR や大きめの機能追加を行う場合は、必ず事前に Spec を作成します
3. 新規 Spec は `docs/spec/SPEC_TEMPLATE.md` をコピーして作成します
4. ファイル名は `CR-ATELIER-XXX_... .md` 形式を推奨します

## 今後の運用上の注意点

### Spec を書くタイミング

1. **新しい CR を開始する前**
   - 機能追加、リファクタリング、バグ修正など、複数ファイルに影響する変更を行う前に Spec を作成

2. **大きな実装タスクの開始前**
   - 影響範囲が大きい変更や、アーキテクチャレベルの変更を行う前に Spec を作成

3. **設計メモを残したい場合**
   - 長期的に参照する価値がある設計メモは、Spec として `docs/spec/` に保存

### Spec の作成手順

1. `docs/spec/SPEC_TEMPLATE.md` をコピー
2. ファイル名を `CR-ATELIER-XXX_[タイトル].md` 形式に変更
3. テンプレートの各セクションを埋める
4. Status を `Draft` → `In-Progress` → `Done` と更新

### 既存の Spec Kit Mode との関係

- `.spec/` ディレクトリは Spec Kit Mode 用として残す
- `docs/spec/` は正式な Spec の保存場所として使用
- 将来的には `.spec/` から `docs/spec/` への移行を検討

## 設計上の改善点

### アーキテクチャの改善
- **Spec の一元管理**: すべての Spec を `docs/spec/` に集約することで、仕様書の検索性が向上
- **命名規則の統一**: `CR-ATELIER-XXX` 形式に統一することで、プロジェクト内の Spec を識別しやすくなる

### 将来の拡張性への配慮
- Spec テンプレートは必要に応じて拡張可能な構造になっている
- メタデータ（Project, CreatedAt, SourcePath, MovedAt）を追加することで、Spec の履歴管理が容易になる

### コード品質の向上
- Spec を事前に作成することで、実装前に設計を明確化できる
- Spec が存在することで、コードレビュー時の参照資料として活用できる

## 既知の制約・注意事項

### 既存コードとの互換性
- ✅ 既存の Spec Kit Mode 設定（`.spec/` ディレクトリ）との互換性を維持
- ✅ 既存の完了レポートやドキュメントへの影響はない

### 制限事項やトレードオフ
- `.spec/` と `docs/spec/` の2つのディレクトリが存在するが、将来的な統合を検討する必要がある
- 既存の Spec Kit Mode のルール（`.spec/CR-AKM-...` 形式）と新しいルール（`docs/spec/CR-ATELIER-...` 形式）が併存している

### 移行時の注意点
- 既存の Spec ファイルは移動時にメタデータを追加したが、元のファイルは削除していない（手動で削除する必要がある）
- `.spec/` ディレクトリ内のファイルは、必要に応じて `docs/spec/` に移動することを推奨

## 次のステップ

### 即座に実施すべきこと
1. **既存 Spec ファイルの整理**
   - `.spec/` ディレクトリ内のファイルを確認し、必要に応じて `docs/spec/` に移動
   - 元のファイル（`docs/moncler_plp_autorun_spec.md` など）を削除するか、`docs/spec/` へのリンクに置き換える

2. **Spec テンプレートの活用**
   - 新しい CR や機能追加を行う際は、必ず `docs/spec/SPEC_TEMPLATE.md` をコピーして Spec を作成

### 今後のタスク
- `.spec/` と `docs/spec/` の統合を検討
- Spec のレビュープロセスの確立
- Spec のバージョン管理方法の検討

## 関連ファイル

- `docs/spec/SPEC_TEMPLATE.md` - Spec テンプレート
- `docs/spec/CR-ATELIER-MONCLER-PLP-AUTORUN.md` - Moncler PLP 自動回復フロー仕様書
- `docs/spec/CR-ATELIER-001_MONCLER_PLP_TELEMETRY_AND_PLP_STATE_RECORDING.md` - Moncler PLP Telemetry & PLP State Recording 仕様書
- `.cursorrules` - Spec 保存ルールと自動生成ルール
- `README.md` - Spec 管理ルール

