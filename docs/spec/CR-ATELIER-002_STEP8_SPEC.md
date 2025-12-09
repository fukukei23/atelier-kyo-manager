# CR-ATELIER-002 Step 8

Moncler 実ブラウザ検証 & 手動パッチ適用ガイド整備

**Version:** 1.0

**Status:** Implementation

**Owner:** Atelier Kyo / NexusCore Line

**Related Steps:** Step 3 / Step 4 / Step 5 / Step 6 / Step 7

**Related Docs:**

- `docs/completion_reports/CR_ATELIER_002_STEP3_COMPLETION_REPORT.md`
- `docs/completion_reports/CR_ATELIER_002_STEP4_COMPLETION_REPORT.md`
- `docs/completion_reports/CR_ATELIER_002_STEP5_COMPLETION_REPORT.md`
- `docs/completion_reports/CR_ATELIER_002_STEP6_COMPLETION_REPORT.md`
- `docs/completion_reports/CR_ATELIER_002_STEP7_COMPLETION_REPORT.md`
- `docs/official/moncler_patch_apply_guide.md`（本 Step で作成）

---

## 1. Overview & Context

### 1.1 これまでの実装（Step 1〜7）

CR-ATELIER-002 Step 1〜7 までで、Moncler 公式サイト向けの PLP→PDP 抽出ロジックについて以下を実装済み：

- **Step 3**: Moncler 向け PLP→PDP 抽出ロジックの具体化・修正
  - `site_config`（`overrides.local.json`）と PLP/PDP 抽出コードの整合性改善
  - URL バリデーション（Moncler 本体の `/products/` のみを正として扱い、外部ドメインを除外）

- **Step 4**: 実ブラウザ検証と最終修正
  - 実 DOM サンプリング & 構造メモ作成
  - URL バリデーションとロケール制御の実 DOM ベース調整
  - Telemetry / ログの実データに合わせた具体化

- **Step 5**: Moncler PLP→PDP 抽出ロバスト化 & Self-Healing 連携設計
  - セレクタ戦略のレイヤリング設計（Primary, Secondary, Tertiary）
  - Redirect / Locale 挙動の扱い整理
  - Self-Healing / Selector Discovery 連携設計

- **Step 6**: Self-Healing & Selector Discovery 実装フェーズ
  - Telemetry: `moncler_plp_pdp_outcome` の実装
  - Self-Healing Agent: Moncler 用の自己修復解析
  - Selector Discovery Agent: DOM から新セレクタを提案
  - NavigationDriver / Extractor からの連携フロー実装

- **Step 7**: Moncler site_config パッチ適用フロー — 提案結果の安全な反映
  - Self-Healing / Selector Discovery の結果から `analysis.json` と `patch_candidate.json` を自動生成
  - 現行 `overrides.local.json` との差分を計算し、before/after を明示
  - Markdown レポートを生成して人間がレビューしやすい形式を提供

### 1.2 Step 8 の目的

Step 8 の目的は、以下を満たすことです：

1. **実ブラウザ検証シナリオの整理**
   - Moncler run 実行手順の明確化
   - `self_healing/moncler` 配下ファイルの生成確認方法の整理

2. **手動パッチ適用ガイドの作成**
   - `docs/official/moncler_patch_apply_guide.md` の新規作成
   - Moncler run 失敗時に Self-Healing が生成したパッチ候補を、人間が確認して `overrides.local.json` に手動で反映するための手順書

3. **ドキュメント整備**
   - README との整合性確認（必要なら README にガイドへのリンクを1行追加）

---

## 2. Scope (In-Scope / Out-of-Scope)

### 2.1 In-Scope（Step 8 でやること）

1. **Moncler 実行手順の整理**
   - run コマンドの明確化（WSL 前提で記載）
   - 確認ポイントの整理（`self_healing/moncler` 配下ファイルの確認方法）

2. **`self_healing/moncler` 配下ファイルの読み方の整理**
   - `<RUN_ID>_analysis.json`: run のメタ情報、self-healing / selector-discovery の分析結果
   - `<RUN_ID>_patch_candidate.json`: `overrides.local.json` へのパッチ候補
   - `docs/reports/MONCLER_PATCH_<RUN_ID>.md`: 人間が読む用の要約

3. **手動パッチ適用ガイドの作成**
   - `docs/official/moncler_patch_apply_guide.md` の新規作成
   - バックアップ取得 → パッチ確認 → 適用 → テスト → ロールバックまでの一連の流れを明文化

4. **README 更新（必要に応じて）**
   - `docs/official/moncler_patch_apply_guide.md` へのリンクを1行追加

### 2.2 Out-of-Scope（Step 8 ではやらないこと）

- **`overrides.local.json` への自動パッチ適用**
  - 手動承認フローを維持（安全性を優先）

- **GitHub Actions / CI の導入**
  - パッチ適用の自動化は将来の Step の対象

- **Moncler 以外のサイトへの一般化**
  - Step 8 時点では **MONCLER_OFFICIAL 専用** とする

---

## 3. Implementation Plan (Step 8)

### Step 8-1: 実ブラウザ検証シナリオ（想定コマンド）を整理

**実施内容**:
- Moncler run 実行コマンドの明確化
- `self_healing/moncler` 配下ファイルの確認方法の整理
- ガイド内に「想定コマンド」として明示

**想定コマンド例**:
```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

# Moncler ラン実行（例）
python -m app.scripts.run_site moncler --query "down jacket" --headful

# 最新の self_healing ファイル確認（例）
ls -1 instance/self_healing/moncler
```

### Step 8-2: `patch_candidate.json` / `analysis.json` / `MONCLER_PATCH_*.md` の読み方を整理

**実施内容**:
- 各ファイルの構造と意味を説明
- パッチ候補の読み方（before/after の見方）を明文化
- リスク評価の見方を説明

**ファイル構造**:
- `<RUN_ID>_analysis.json`: Self-Healing 分析結果と Selector Discovery の提案結果
- `<RUN_ID>_patch_candidate.json`: `overrides.local.json` へのパッチ候補（before/after 形式）
- `docs/reports/MONCLER_PATCH_<RUN_ID>.md`: 人間がレビューしやすい Markdown レポート

### Step 8-3: 手動パッチ適用ガイド `docs/official/moncler_patch_apply_guide.md` を新規作成

**実施内容**:
- 目的・前提・典型的な実行コマンド例を記載
- ファイルの読み方・意味を説明
- 手動パッチ適用手順（バックアップ → 確認 → 適用 → テスト → ロールバック）を明文化
- 注意事項を記載

**必須セクション**:
1. 目的
2. 前提
3. 典型的な実行コマンド例（WSL 前提で記載）
4. ファイルの読み方・意味
5. 手動パッチ適用手順（メイン）
6. ロールバック方法
7. 注意事項

### Step 8-4: README との整合性確認（必要なら README にガイドへのリンクを1行追加）

**実施内容**:
- README.md の「ドキュメント」や「Moncler ライン」の説明セクションに、`docs/official/moncler_patch_apply_guide.md` へのリンクを1行追加するだけに留める
- UI・API の説明やアーキテクチャ概要は変更しない

---

## 4. Testing Strategy

### 4.1 pytest 実行

**コマンド**:
```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python -m pytest
```

**確認事項**:
- 既存の 40 テスト（Step7 時点）が全てパスすること
- 今回の Step 8 では、基本的には新規テスト追加は必須ではない（ドキュメント中心のため）

### 4.2 Moncler ランの確認（実行コマンド例だけガイドに書く）

**想定コマンド**:
```bash
python -m app.scripts.run_site moncler --query "down jacket" --headful
ls -1 instance/self_healing/moncler
```

**確認事項**:
- `instance/self_healing/moncler/` 配下に `<RUN_ID>_analysis.json` と `<RUN_ID>_patch_candidate.json` が生成されること
- `docs/reports/MONCLER_PATCH_<RUN_ID>.md` が生成されること（オプション）

**注意**:
- Cursor から実行できる環境であれば、実際に実行して log/patch ファイル生成を確認
- 実行が難しい場合は、ガイド内で「想定コマンド」として明示するだけでよい

---

## 5. Acceptance Criteria (Step 8 完了条件)

Step 8 が完了したと見なす条件は以下：

1. **`docs/spec/CR-ATELIER-002_STEP8_SPEC.md` が作成され、**
   - Step 8 の目的・スコープ・実装計画・テスト戦略が明文化されている

2. **`docs/official/moncler_patch_apply_guide.md` が作成され、以下を満たす：**
   - Moncler run → self_healing → patch_candidate までの流れが時系列で説明されている
   - `overrides.local.json` への手動パッチ適用手順が、バックアップ〜適用〜ロールバックまで一通り記載されている
   - すべて日本語で書かれている

3. **既存テスト (`python -m pytest`) がグリーンである**

4. **`.cursorrules` / README のポリシーを破る変更（UI層編集、自動パッチ適用など）が行われていない**

---

## 6. 今後の拡張（Step 9 以降候補）

- Moncler 以外のサイトへの一般化（セレクタパッチの共通フォーマット化）
- Streamlit / Flask ダッシュボードでのパッチレビュー UI
- Approval Agent による半自動承認フロー
- Telemetry ベースの「サイトごとのヘルススコア」可視化

