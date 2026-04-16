# 完全自動修正ループ - 使い方ガイド

## 概要

`auto_fix_and_retry.py` は、テスト実行 → ログ解析 → 問題検出 → 自動修正 → 再実行を自動的にループするシステムです。

## 機能

### 1. 自動テスト実行
- `tools/run_browser_use.py` を自動実行
- ログを `browser_test_YYYYMMDD_HHMMSS.log` に保存

### 2. ログ解析
- 最新のログファイルを自動検索
- 以下の問題を自動検出：
  - PDP リンクが0件
  - PLP materialization 失敗
  - セレクタが見つからない
  - タイムアウト
  - MonclerPLPStrategy でタイルが見つかっているか

### 3. 修正案生成
- 検出された問題に対して、具体的な修正案を自動生成
- 自動修正可能なものと手動確認が必要なものを分類

### 4. 自動修正適用
**現在実装済みの自動修正:**
- `pdp_link_selectors` に `'a[href*="/products/"]'` が含まれていない場合、自動的に追加

**手動確認が必要な修正:**
- セレクタの更新（コード変更が必要）
- Materialization 条件の調整（ロジック変更が必要）

### 5. 再実行ループ
- 修正適用後、自動的に再実行
- 最大3回までリトライ
- 成功した場合はループを終了

## 使い方

### 基本的な使い方

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python auto_fix_and_retry.py
```

### 実行例

```
================================================================================
完全自動修正と再実行ループ開始
================================================================================

[試行 1/3]

[実行] テストを開始します...
[ログ] browser_test_20251128_013000.log

[解析] ログファイルを解析中...

--- 検出された問題 ---
  ✗ PDP リンクが0件
  ✗ PLP materialization が失敗
  ✗ タイムアウトが発生
  ℹ MonclerPLPStrategy で 6 個のタイルが見つかりました

--- 修正案 ---
自動修正可能:
  1. [selector_mismatch] MonclerPLPStrategy ではタイルが見つかっているが、collect_pdp_links で見つからない
     アクション: check_and_add_selector
     ファイル: app/config/sites/overrides.local.json

[適用] 自動修正を適用中...
[修正] app/config/sites/overrides.local.json に 'a[href*="/products/"]' を追加しました
  ✓ MonclerPLPStrategy ではタイルが見つかっているが、collect_pdp_links で見つからない を修正しました

[再実行] 修正後に再実行します...

[試行 2/3]
...
```

## 実装されている自動修正

### 1. pdp_link_selectors の自動追加

**条件:**
- PDP リンクが0件
- MonclerPLPStrategy でタイルが見つかっている
- `pdp_link_selectors` に `'a[href*="/products/"]'` が含まれていない

**動作:**
- `app/config/sites/overrides.local.json` の `MONCLER_OFFICIAL.selectors.plp.pdp_link_selectors` に自動追加
- 先頭に追加（優先度を高くするため）

## 制限事項

### 自動修正できないもの

1. **コード変更が必要な修正**
   - セレクタの更新（コード内のセレクタ文字列の変更）
   - Materialization 条件の調整（ロジック変更）

2. **判断が必要な修正**
   - 複数の修正案がある場合
   - 既存の動作に影響する可能性がある修正

### 安全性の考慮

- 自動修正は、設定ファイル（JSON）の更新のみ
- コード変更は手動確認が必要
- 最大リトライ回数は3回（無限ループを防止）

## トラブルシューティング

### エラー: ファイルが見つかりません

- `app/config/sites/overrides.local.json` が存在するか確認
- プロジェクトルートで実行しているか確認

### エラー: JSON 読み込み/書き込みエラー

- JSON ファイルの構文エラーを確認
- ファイルの権限を確認

### 自動修正が適用されない

- ログを確認して、問題が正しく検出されているか確認
- 修正案が「自動修正可能」に分類されているか確認

## 次のステップ

1. **自動修正の拡張**
   - より多くの問題パターンに対応
   - コード変更の自動適用（慎重に）

2. **ログ解析の改善**
   - より詳細な問題分析
   - 修正案の精度向上

3. **レポート生成**
   - 各試行の結果をレポートに保存
   - 修正履歴の記録

