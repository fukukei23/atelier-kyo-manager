# PLP 抽出修復タスクテンプレート

## タスク概要

Moncler Phase1.5 の dry-run (2025-12-03) の結果、PLP → PDP 抽出ロジックが壊れていることが判明した。

**重要**: Stealth / retry / exception classification は正常であり、問題はセレクタまたは materialization ロジックに限定される。

## 現状確認

### ✅ 正常に動作している機能

- **Stealth モジュール**: SessionManager 経由で Stealth が適用されている
- **例外分類・Retry ロジック**: Timeout エラーが適切にキャッチされ、retry が動作している
- **Bot 検知回避**: 403/429 エラーは発生していない
- **Proxy 設定**: Proxy が正しく設定され、使用されている

### ⚠️ 問題が発生している機能

- **PLP → PDP 抽出**: Tile counts は検出されているが、実際の PDP リンク抽出に失敗（0 links）
- **Materialization**: PLP materialization がタイムアウトしている
- **セレクタマッチング**: Phase 1a/1b（quick extract）でリンクが 0 になっている

## 修正すべき内容（必須）

### 1. PLP DOM snapshot の解析

**目的**: 正しい PDP リンクの位置を特定する

**実施内容**:
- `instance/runs/**` のスクリーンショットを確認
- `instance/runs/**` のログファイルを確認
- `instance/runs/**` の HTML スナップショット（`failure_dom.html` など）を読み取り
- 実際の DOM 構造を分析し、PDP リンクのセレクタパターンを特定

**確認ポイント**:
- PDP リンクの実際の HTML 構造
- セレクタがマッチしない理由
- Shadow DOM や nested elements の有無
- 動的ロードによる DOM 変更の有無

### 2. site_config の selectors.block の修正

**目的**: PLP セレクタの定義を現行 DOM に合わせて更新する

**実施内容**:
- `app/config/sites/overrides.local.json` の `MONCLER_OFFICIAL.selectors.pdp.pdp_link_selectors` を更新
- DOM snapshot の解析結果に基づいて、正しいセレクタを追加
- 既存のセレクタが機能しない理由を特定し、修正

**確認ポイント**:
- セレクタの優先順位
- セレクタのマッチングロジック
- 複数セレクタのフォールバック戦略

### 3. navigation_driver.py の PLP materialization 修正

**目的**: Phase 1a/1b（quick extract）でリンクが 0 にならないよう修正

**実施内容**:
- `app/agents/browser/navigation_driver.py` の `run_plp_flow()` メソッドを確認
- Phase 1a/1b（quick extract）のロジックを修正
- Phase 2（deep extract）で shadow DOM / nested elements に対応
- セレクタマッチングのロジックを改善

**確認ポイント**:
- `_extract_pdp_links()` メソッドの動作
- セレクタの評価順序
- タイムアウト設定
- リトライロジック

### 4. Moncler patch との整合確認

**目的**: モーダルや地理ブロック UI がリンク抽出を妨げていないか確認

**実施内容**:
- `app/agents/browser_use_moncler_patch.py` の動作を確認
- モーダル処理が DOM 構造に影響を与えていないか確認
- 地理ブロック UI がリンク抽出を妨げていないか確認

**確認ポイント**:
- モーダル処理のタイミング
- DOM 構造への影響
- セレクタ評価のタイミング

## 制約

### 変更禁止領域

- **Stealth 共通化ロジック**: `scraping/stealth.py` は触らない
- **Retry ロジック**: `app/agents/browser_use_agent.py` の `_run_with_retry()` などは触らない
- **例外分類ロジック**: `BrowserErrorType` や `_classify_exception()` は触らない
- **UI 層**: `app/templates/**`, `app/forms/**`, `app/static/**` は触らない

### 変更可能領域

- **site_config**: `app/config/sites/overrides.local.json` の `MONCLER_OFFICIAL.selectors` セクション
- **navigation_driver.py**: `app/agents/browser/navigation_driver.py` の PLP 抽出ロジック
- **Moncler patch**: `app/agents/browser_use_moncler_patch.py` の UI 操作ロジック（必要に応じて）

### 修正方針

- **最小差分**: site_config → navigation_driver → extractors の最小差分で修復する
- **段階的修正**: 1つずつ修正し、各修正後に dry-run で確認
- **ログ確認**: 各修正後にログを確認し、問題が解決したか検証

## 出力必須：変更レポート

作業完了後は、必ず以下の形式で「変更レポート」を出力してください：

### 1. Reasoning（なぜこの変更を行ったか）

- PLP 抽出が失敗している原因
- 修正のアプローチ
- 修正の優先順位

### 2. Diff Summary（修正されたファイルと主要差分の要点）

- 変更ファイル一覧
- 主要な変更内容
- 変更の影響範囲

### 3. Next Action（次に行うべきこと）

- 修正後の dry-run 実行
- 追加の検証項目
- 次のフェーズへの移行

## 実行手順

### Step 1: DOM snapshot の解析

```bash
# 最新の実行ログを確認
ls -lt instance/runs/ | head -5

# スクリーンショットを確認
ls instance/runs/<run_id>/screenshots/

# HTML スナップショットを確認（あれば）
cat instance/runs/<run_id>/failure_dom.html
```

### Step 2: site_config の修正

```bash
# site_config を編集
vim app/config/sites/overrides.local.json

# 修正内容を確認
git diff app/config/sites/overrides.local.json
```

### Step 3: navigation_driver.py の修正

```bash
# navigation_driver.py を編集
vim app/agents/browser/navigation_driver.py

# 修正内容を確認
git diff app/agents/browser/navigation_driver.py
```

### Step 4: dry-run で確認

```bash
# dry-run で設定を確認
python -m app.scripts.run_site moncler --dry-run

# 実際の実行で動作確認
python -m app.scripts.run_site moncler --query "down jacket" --headful
```

## 関連ファイル

- `app/config/sites/overrides.local.json`: サイト設定（セレクタ定義）
- `app/agents/browser/navigation_driver.py`: PLP 抽出ロジック
- `app/agents/browser_use_moncler_patch.py`: Moncler サイト固有パッチ
- `instance/runs/<run_id>/`: 実行ログ・スクリーンショット・DOM スナップショット
- `docs/moncler/PHASE1_5_DRY_RUN_REPORT.md`: Dry-run レポート

## 参考情報

- **Dry-run レポート**: `docs/moncler/PHASE1_5_DRY_RUN_REPORT.md`
- **Stealth 共通化レポート**: `docs/completion_reports/STEALTH_COMMONALIZATION_COMPLETION_REPORT.md`
- **例外分類・Retry リファクタレポート**: `docs/official/refactoring/BROWSER_USE_AGENT_EXCEPTION_RETRY_REFACTOR.md`

