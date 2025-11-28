# MONCLER Drission 診断ランナー＋ログ出力 - 完了レポート

## 実装日時
2025-11-28

## 概要

MONCLER 専用の `MonclerDrissionHandler` を単体で自動実行し、成功・失敗それぞれについてスクリーンショット、HTML、JSON、ログを保存する「診断モード」を追加しました。

### 目的
- Drissionルートがどこで詰まっているかを自動診断
- どのセレクタが効いていないかを後から確認可能にする
- 人手でブラウザを触らなくても問題を特定できるようにする

## 実装ステップ

### Step 1: MonclerDrissionHandler に診断フックを追加

**変更ファイル**: `app/specialized/moncler_handler.py`

#### 1-1. debug フラグの追加

`__init__` メソッドに `debug: bool = False` パラメータを追加し、診断モードを有効化できるようにしました。

```python
def __init__(
    self,
    *,
    runtime_kwargs: dict | None = None,
    user_data_path: str | None = None,
    debug: bool = False,
) -> None:
    self.debug = debug
    self._diag_dir: Optional[Path] = None  # 診断用ディレクトリ
```

#### 1-2. 診断用メソッドの追加

以下のメソッドを追加しました：

- **`_ensure_diag_dir`**: 診断用ディレクトリを確保
- **`_save_diag_snapshot`**: HTML、PNG、JSON をまとめて保存

```python
def _save_diag_snapshot(
    self,
    out_dir: Path,
    *,
    page: "ChromiumPage",
    name_prefix: str,
    payload: dict | None = None,
) -> None:
    """診断用に HTML, PNG, JSON をまとめて保存する"""
    # HTML保存
    # スクリーンショット保存
    # JSON payload保存
```

#### 1-3. run() メソッド内で診断フックを呼び出す

`run()` メソッド内で、以下のタイミングで診断スナップショットを保存するようにしました：

- **成功時**: `success_plp.html` / `.png` / `.json`
- **商品が見つからない場合**: `error_no_items.html` / `.png` / `.json`
- **エラー時**: `error_plp.html` / `.png` / `.json`

### Step 2: 診断スクリプトの作成

**新規ファイル**: `scripts/run_moncler_drission_diagnostics.py`

#### 2-1. 機能

- コマンドライン引数の解析
- ロガーの設定（ファイル + コンソール）
- site_config の読み込み
- RunContext の初期化
- MonclerDrissionHandler の実行（診断モード有効）
- 複数回実行対応（`--runs` オプション）

#### 2-2. コマンドライン引数

- `--query`: 検索クエリ（デフォルト: "down jacket"）
- `--target_url`: 直接指定する PLP URL（オプション）
- `--headless`: ヘッドレスモードで実行
- `--runs`: 実行回数（デフォルト: 1）
- `--out_base`: 出力ベースディレクトリ（デフォルト: "artifacts/moncler_drission"）

### Step 3: README の追加

**新規ファイル**: `scripts/README.md`

診断スクリプトの使用方法、引数、出力先について説明を追加しました。

## 変更ファイル一覧

### 変更ファイル
- `app/specialized/moncler_handler.py`
  - `__init__` に `debug` フラグを追加
  - `_ensure_diag_dir` メソッドを追加
  - `_save_diag_snapshot` メソッドを追加
  - `run()` メソッド内で診断フックを呼び出す

### 新規作成ファイル
- `scripts/run_moncler_drission_diagnostics.py`: 診断スクリプト
- `scripts/README.md`: スクリプトの使用方法

## 動作確認結果

### ✅ 実装完了項目

1. **診断フックの追加**
   - ✅ `debug` フラグによる診断モードの有効化
   - ✅ 診断用ディレクトリの自動生成
   - ✅ HTML、PNG、JSON の保存機能

2. **診断スクリプト**
   - ✅ コマンドライン引数の解析
   - ✅ ロガーの設定（ファイル + コンソール）
   - ✅ site_config の読み込み
   - ✅ RunContext の初期化
   - ✅ 複数回実行対応

3. **ドキュメント**
   - ✅ README の作成
   - ✅ 使用方法の説明

### コード品質

- ✅ リンターエラーなし
- ✅ 既存の BrowserUseAgent / Playwright ルートには変更なし
- ✅ エラーハンドリングが適切に実装されている

## 設計上の改善点

### アーキテクチャの改善

1. **診断モードの分離**
   - 診断機能は `debug` フラグで制御され、通常の実行には影響しない
   - 診断用ディレクトリは自動生成され、タイムスタンプで管理される

2. **柔軟な出力先設定**
   - `diagnostics_dir` と `run_id` を `runtime_kwargs` で指定可能
   - デフォルトは `artifacts/moncler_drission` を使用

3. **包括的な診断情報**
   - 成功時・失敗時それぞれについて HTML、PNG、JSON を保存
   - ログファイルも自動生成される

### 将来の拡張性への配慮

1. **他のサイトへの拡張**
   - 診断機能は汎用的に実装されており、他のサイトにも適用可能

2. **診断情報の拡張**
   - `payload` パラメータで任意の JSON データを保存可能
   - 将来的に追加の診断情報を保存しやすい構造

## 既知の制約・注意事項

### 実行環境

1. **DrissionPage のインストール**
   - DrissionPage がインストールされている必要があります
   - Windows 環境での実行を推奨します

2. **Chrome/Chromium のインストール**
   - Chrome または Chromium がインストールされている必要があります

### 制限事項

- 診断モードは `MonclerDrissionHandler` 単体での実行を前提としています
- `BrowserUseAgent` 経由での実行時は診断モードは無効です（`debug=False` がデフォルト）

## 使用方法

### 基本的な実行

```bash
python scripts/run_moncler_drission_diagnostics.py \
  --query "down jacket" \
  --target_url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \
  --headless
```

### 複数回実行

```bash
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

## 次のステップ

### 推奨されるフォローアップアクション

1. **実機テスト**
   - Windows 環境で実際に診断スクリプトを実行
   - 出力される診断情報の確認

2. **診断情報の活用**
   - 保存された HTML、PNG、JSON を分析
   - セレクタの問題点を特定

3. **診断機能の拡張**
   - 追加の診断情報の保存（例: ネットワークリクエスト、コンソールログ）
   - 診断結果の自動分析機能

## 関連ファイル

- **実装ファイル**: `app/specialized/moncler_handler.py`
- **診断スクリプト**: `scripts/run_moncler_drission_diagnostics.py`
- **ドキュメント**: `scripts/README.md`
- **統合ファイル**: `app/agents/browser_use_agent.py`（変更なし）

## 結論

MONCLER Drission 診断ランナー＋ログ出力機構の実装が完了しました。これにより、人手でブラウザを触らなくても、Drissionルートがどこで詰まっているか、どのセレクタが効いていないかを後から確認できるようになりました。

