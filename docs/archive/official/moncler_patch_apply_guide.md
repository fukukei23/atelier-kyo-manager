# Moncler パッチ適用ガイド

## 目的

本ガイドは、**Moncler run 失敗時に Self-Healing が生成したパッチ候補を、人間が確認して `overrides.local.json` に手動で反映するための手順書**です。

Self-Healing Agent と Selector Discovery Agent が分析・提案したセレクタや trap URL パターンの変更案を、安全に `app/config/sites/overrides.local.json` に適用する手順を説明します。

---

## 前提

以下の条件を満たしていることを前提とします：

1. **Moncler run が実行済み**
   - `python -m app.scripts.run_site moncler --query "down jacket" --headful` のようなコマンドで run 済み

2. **Self-Healing がパッチ候補を生成済み**
   - `instance/self_healing/moncler/` 配下に `<RUN_ID>_analysis.json` と `<RUN_ID>_patch_candidate.json` が生成されている

3. **Markdown レポートが存在する場合（オプション）**
   - `docs/reports/MONCLER_PATCH_<RUN_ID>.md` が存在する場合は、それも参照する

---

## 典型的な実行コマンド例（WSL 前提）

### Moncler run の実行

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

# Moncler ラン実行（例）
python -m app.scripts.run_site moncler --query "down jacket" --headful
```

### 最新の self_healing ファイル確認

```bash
# 最新の self_healing ファイル確認（例）
ls -1 instance/self_healing/moncler

# 出力例:
# 20251209_123456_789_analysis.json
# 20251209_123456_789_patch_candidate.json
```

### 最新の run ディレクトリ確認

```bash
# 最新の run ディレクトリを確認
LATEST=$(ls -td instance/runs/2025* | head -1)
echo "LATEST = $LATEST"
ls -1 "$LATEST/self_healing/moncler/"
```

---

## ファイルの読み方・意味

### `<RUN_ID>_analysis.json`

**内容**: run のメタ情報、self-healing / selector-discovery の分析結果

**主要フィールド**:
- `run_id`: 実行ID
- `site`: サイトコード（"MONCLER_OFFICIAL"）
- `timestamp`: タイムスタンプ
- `current_url`: 現在のURL
- `moncler_outcome`: Telemetry に保存した outcome 情報
  - `plp_materialized`: PLP がマテリアライズされたか
  - `tiles_detected`: 検出されたタイル数
  - `pdp_links_raw`: 生の PDP リンク数
  - `pdp_links_accepted`: 承認された PDP リンク数
  - `layer_stats`: セレクタレイヤーの統計
  - `rejection_stats`: 拒否理由の統計
  - `locale_corrections`: ロケール補正の回数
  - `trap_detected`: Trap ページが検出されたか
- `self_healing`: Self-Healing Agent の分析結果
  - `analysis`: 分析内容
  - `root_cause`: 根本原因
  - `suggested_actions`: 推奨アクション
  - `confidence`: 信頼度
- `selector_discovery`: Selector Discovery Agent の提案結果
  - `candidate_selectors`: 候補セレクタのリスト
  - `recommended_layer`: 推奨レイヤー（"primary", "secondary", "tertiary"）
  - `confidence_scores`: 信頼度スコア

**確認ポイント**:
- `root_cause` を確認して、失敗の根本原因を理解する
- `suggested_actions` を確認して、推奨されるアクションを把握する
- `confidence` が低い（0.5 以下）場合は、提案の信頼性が低い可能性がある

### `<RUN_ID>_patch_candidate.json`

**内容**: `overrides.local.json` へのパッチ候補

**主要フィールド**:
- `target_file`: 対象ファイル（`app/config/sites/overrides.local.json`）
- `site`: サイトコード（"MONCLER_OFFICIAL"）
- `run_id`: 実行ID
- `timestamp`: タイムスタンプ
- `summary`: パッチの概要
- `changes`: 変更内容
  - `selectors.plp.pdp_link_selectors`: PDP リンクセレクタの変更（before/after）
  - `selectors.plp.tile_selectors`: タイルセレクタの変更（before/after）
  - `navigation.trap_url_patterns`: Trap URL パターンの追加（append）
- `risk_assessment`: リスク評価
  - `overall`: 全体リスク（"LOW", "MEDIUM", "HIGH"）
  - `notes`: 注意事項

**確認ポイント**:
- `changes` の `before` と `after` を比較して、変更内容を理解する
- `risk_assessment.overall` が "HIGH" の場合は、特に慎重にレビューする
- `risk_assessment.notes` を確認して、注意事項を把握する

**変更形式の説明**:
- `before` / `after`: 既存の値を新しい値に置き換える
- `append`: 既存のリストに新しい値を追加する
- `remove`: 既存のリストから値を削除する（Step 7 では原則使用しない）

### `docs/reports/MONCLER_PATCH_<RUN_ID>.md`

**内容**: 人間が読む用の要約（root_cause / risk_assessment / patch_diff 概要）

**主要セクション**:
- Run 情報（run_id, 日時, URL）
- 失敗の概要（Self-Healing 分析要約）
- 提案パッチ一覧（before / after）
- 推奨度（OK / 要レビュー / 危険）
- 推奨アクション
- 適用方法

**確認ポイント**:
- Markdown レポートがある場合は、まずこちらを読んで全体像を把握する
- その後、`patch_candidate.json` の詳細を確認する

---

## 手動パッチ適用手順（メイン）

### 1. バックアップ取得

**重要**: パッチ適用前に必ずバックアップを取得してください。

```bash
cd /home/yn441611/atelier-kyo-manager

# バックアップ取得（日付を付与）
cp app/config/sites/overrides.local.json app/config/sites/overrides.local.json.bak_$(date +%Y%m%d)
```

または、Git で管理している場合は：

```bash
# Git でバックアップ（推奨）
git add app/config/sites/overrides.local.json
git commit -m "Backup before applying Moncler patch <RUN_ID>"
```

### 2. パッチ候補の確認

**`<RUN_ID>_patch_candidate.json` をエディタで開き、以下を確認**：

1. **変更内容の確認**
   - `changes` セクションを確認
   - `selectors.plp.pdp_link_selectors` の `before` と `after` を比較
   - `selectors.plp.tile_selectors` の `before` と `after` を比較
   - `navigation.trap_url_patterns` の `append` を確認

2. **リスク評価の確認**
   - `risk_assessment.overall` が "HIGH" の場合は、特に慎重にレビュー
   - `risk_assessment.notes` を確認

3. **推奨アクションの確認**
   - `analysis.json` の `self_healing.suggested_actions` を確認
   - または `MONCLER_PATCH_<RUN_ID>.md` の「推奨アクション」セクションを確認

**確認例**:
```bash
# JSON ファイルを開く（例）
cat instance/self_healing/moncler/20251209_123456_789_patch_candidate.json | jq .

# または、エディタで開く
code instance/self_healing/moncler/20251209_123456_789_patch_candidate.json
```

### 3. `overrides.local.json` の編集

**`app/config/sites/overrides.local.json` の `MONCLER_OFFICIAL` ブロックをエディタで開き、パッチ候補の「after」側を人間の判断で反映**：

#### 3-1. PDP リンクセレクタの更新

**変更例**:
```json
{
  "MONCLER_OFFICIAL": {
    "selectors": {
      "plp": {
        "pdp_link_selectors": [
          // before: ["a[href*='/products/']"]
          // after: 以下に置き換え
          "article[data-component*='ProductCard'] a[href*='/products/']",
          "div[data-testid='product-card'] a[href*='/products/']"
        ]
      }
    }
  }
}
```

#### 3-2. タイルセレクタの更新

**変更例**:
```json
{
  "MONCLER_OFFICIAL": {
    "selectors": {
      "plp": {
        "tile_selectors": [
          // before: ["div:has(a[href$='.html'])"]
          // after: 以下に置き換え
          "article[data-component*='ProductCard']",
          "div[data-testid='product-card']"
        ]
      }
    }
  }
}
```

#### 3-3. Trap URL パターンの追加

**変更例**:
```json
{
  "MONCLER_OFFICIAL": {
    "navigation": {
      "trap_url_patterns": [
        // 既存のパターン...
        "/search",
        "/404",
        "/client-service",
        // append: 以下を追加
        ".*/en-[a-z]{2}/en-int/.*",
        ".*/en-[a-z]{2}/en-[a-z]{2}/.*"
      ]
    }
  }
}
```

**注意事項**:
- JSON の構文エラーに注意（カンマの有無、引用符の閉じ忘れなど）
- 既存の設定を壊さないように注意
- 変更は `MONCLER_OFFICIAL` ブロック内に限定する

### 4. 変更内容の確認（Git diff）

**変更後に `git diff` で変更内容を確認**：

```bash
git diff app/config/sites/overrides.local.json
```

**確認ポイント**:
- 変更内容が期待通りか
- 不要な変更が含まれていないか
- JSON の構文エラーがないか

### 5. テスト実行

**`python -m pytest` を実行し、テストがグリーンであることを確認**：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

# すべてのテストを実行
python -m pytest

# または、Moncler 関連のテストのみ実行
python -m pytest tests/test_moncler_pdp_url.py tests/test_moncler_self_healing.py tests/test_moncler_selector_discovery.py tests/test_moncler_patch_builder.py -q -v
```

**確認ポイント**:
- すべてのテストがパスすること
- 特に Moncler 関連のテストがパスすること

### 6. 実ブラウザ検証（推奨）

**必要に応じて Moncler run を再実行し、`collected_pdp_links >= 1` になっているかを確認**：

```bash
# Moncler run を再実行
python -m app.scripts.run_site moncler --query "down jacket" --headful

# 最新の run ディレクトリを確認
LATEST=$(ls -td instance/runs/2025* | head -1)
echo "LATEST = $LATEST"

# result.json を確認
cat "$LATEST/result.json" | jq '.nav_outcome.collected_pdp_links'

# または、system.log を確認
tail -50 "$LATEST/system.log" | grep -i "collected.*pdp"
```

**確認ポイント**:
- `collected_pdp_links >= 1` であること
- Trap ページが検出されていないこと
- ロケール補正が過度に発生していないこと

### 7. Git コミット（推奨）

**変更を Git にコミット**：

```bash
git add app/config/sites/overrides.local.json
git commit -m "Apply Moncler patch from run <RUN_ID>

- Updated pdp_link_selectors: <変更内容の要約>
- Updated tile_selectors: <変更内容の要約>
- Added trap_url_patterns: <追加内容の要約>

Related: CR-ATELIER-002 Step 8"
```

---

## ロールバック方法

パッチ適用後に問題が発生した場合、以下の手順でロールバックできます。

### 方法1: バックアップファイルから復元

```bash
cd /home/yn441611/atelier-kyo-manager

# バックアップファイルから復元
cp app/config/sites/overrides.local.json.bak_YYYYMMDD app/config/sites/overrides.local.json

# または、mv で上書き
mv app/config/sites/overrides.local.json.bak_YYYYMMDD app/config/sites/overrides.local.json
```

### 方法2: Git でロールバック

```bash
# 直前のコミットを取り消す
git reset --hard HEAD~1

# または、特定のコミットに戻る
git log --oneline app/config/sites/overrides.local.json
git checkout <commit-hash> -- app/config/sites/overrides.local.json
```

### 方法3: 手動で変更を元に戻す

`patch_candidate.json` の `before` の値を `overrides.local.json` に手動で反映する。

---

## 注意事項

### Self-Healing の提案はあくまで候補

- Self-Healing Agent と Selector Discovery Agent の提案は、**あくまで候補**です
- 人間が内容を確認してから適用すること
- 提案の信頼度（`confidence`）が低い場合は、特に慎重にレビューする

### BUYMA 規約・サイト規約の遵守

- BUYMA 規約やサイトの規約を侵さない範囲で利用すること
- 自動ログイン・自動出品は行わない（規約違反となる可能性あり）
- 国内 EC サイト（楽天市場・Amazon.co.jp 等）を仕入先にしない

### 他サイトへの流用

- 本ガイドは **MONCLER_OFFICIAL 専用** です
- 他サイトへの流用は別 CR（ATELIER-003 など）で行う想定です

### リスク評価が "HIGH" の場合

- `risk_assessment.overall` が "HIGH" の場合は、特に慎重にレビューする
- 実ブラウザ検証を必ず実施する
- 必要に応じて、段階的に適用する（一部の変更のみ適用して検証）

### JSON の構文エラーに注意

- `overrides.local.json` を編集する際は、JSON の構文エラーに注意する
- カンマの有無、引用符の閉じ忘れなどに注意
- エディタの JSON バリデーション機能を活用する

---

## トラブルシューティング

### パッチ適用後にテストが失敗する

1. JSON の構文エラーを確認
2. `git diff` で変更内容を確認
3. バックアップから復元して、段階的に適用する

### パッチ適用後に Moncler run が失敗する

1. `system.log` を確認して、エラーメッセージを確認
2. `failure_dom.html` を確認して、DOM 構造を確認
3. 必要に応じて、ロールバックして別のアプローチを検討

### パッチ候補が生成されない

1. Self-Healing のトリガー条件を確認
2. `moncler_outcome` が Telemetry に保存されているか確認
3. `navigation_driver.py` の `_trigger_moncler_self_healing()` が呼び出されているか確認

---

## 関連ドキュメント

- `docs/spec/CR-ATELIER-002_STEP8_SPEC.md`: Step 8 の仕様書
- `docs/completion_reports/CR_ATELIER_002_STEP7_COMPLETION_REPORT.md`: Step 7 の完了レポート
- `app/agents/moncler_patch_builder.py`: パッチ生成ユーティリティの実装
- `app/config/sites/overrides.local.json`: 現行の site_config

---

**最終更新日**: 2025年12月9日  
**バージョン**: 1.0

