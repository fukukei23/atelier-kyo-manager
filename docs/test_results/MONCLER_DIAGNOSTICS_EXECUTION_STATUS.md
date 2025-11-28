# MONCLER Drission 診断スクリプト実行状況

## 実行日時
2025-11-28

## 実装状況

### ✅ 実装完了

1. **MonclerDrissionHandler に診断フックを追加**
   - ✅ `debug` フラグによる診断モードの有効化
   - ✅ `_ensure_diag_dir` メソッドの実装
   - ✅ `_save_diag_snapshot` メソッドの実装
   - ✅ `run()` メソッド内で診断フックを呼び出す

2. **診断スクリプトの作成**
   - ✅ `scripts/run_moncler_drission_diagnostics.py` の作成
   - ✅ コマンドライン引数の解析
   - ✅ ロガーの設定
   - ✅ site_config の読み込み
   - ✅ RunContext の初期化

3. **ドキュメント**
   - ✅ `scripts/README.md` の作成
   - ✅ 完了レポートの作成

## 実行方法

### Windows 環境での実行

```bash
# 基本的な実行
python scripts/run_moncler_drission_diagnostics.py \
  --query "down jacket" \
  --target_url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \
  --headless

# 複数回実行
python scripts/run_moncler_drission_diagnostics.py \
  --query "jacket" \
  --runs 3 \
  --out_base "artifacts/moncler_test"
```

### 出力先

各実行ごとに `artifacts/moncler_drission/YYYYMMDD_HHMMSS/` ディレクトリが作成され、以下が保存されます：

- `success_plp.html` / `.png` / `.json` (成功時)
- `error_plp.html` / `.png` / `.json` (失敗時)
- `error_no_items.html` / `.png` / `.json` (商品が見つからない場合)
- `run.log` (ログファイル)

## 注意事項

### 実行環境

1. **Windows 環境推奨**
   - DrissionPage は Windows 環境で動作します
   - WSL環境では実際のブラウザ操作はできません

2. **DrissionPage のインストール**
   - `pip install DrissionPage` が必要です
   - Chrome または Chromium がインストールされている必要があります

### 確認事項

- ✅ インポート: すべてのモジュールが正しくインポート可能
- ✅ サイト設定: MONCLER_OFFICIAL の設定が読み込める
- ✅ RunContext: 正常に初期化可能
- ✅ MonclerDrissionHandler: 診断モードで初期化可能

## 次のステップ

1. **Windows 環境での実機テスト**
   - 実際に診断スクリプトを実行
   - 出力される診断情報の確認

2. **診断情報の活用**
   - 保存された HTML、PNG、JSON を分析
   - セレクタの問題点を特定

## 関連ファイル

- **実装ファイル**: `app/specialized/moncler_handler.py`
- **診断スクリプト**: `scripts/run_moncler_drission_diagnostics.py`
- **ドキュメント**: `scripts/README.md`
- **完了レポート**: `docs/completion_reports/MONCLER_DRISSION_DIAGNOSTICS_COMPLETION_REPORT.md`

