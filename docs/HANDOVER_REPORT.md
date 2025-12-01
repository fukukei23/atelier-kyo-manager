# AIエディタ移管用 申し送り資料

## 作成日時
2025年11月28日

## プロジェクト概要

**プロジェクト名**: atelier-kyo-manager  
**目的**: BUYMA Growth Hub / Multi-Agent Automation System  
**場所**: `/home/yn441611/atelier-kyo-manager` (WSL Ubuntu環境)

このプロジェクトは、BUYMA無在庫転売向けの「多サイト対応スクレイピング + 価格リサーチ用 自律エージェント群」です。

## 最近完了した作業（Task C, D, E）

### Task C: Stage 4 – 汎用 PLP Driver の切り出し

**目的**: PLP → PDP のナビゲーションロジックを BrowserUseAgent から切り出し、サイト非依存の「汎用 PLP Driver」モジュールにまとめる。

**実装内容**:
- `app/agents/browser/plp_driver.py` を新規作成
- `PlpDriver` クラスと `PlpNavigationResult` データクラスを定義
- PLP タイルのマテリアライズ、Trap検出、Overlay処理を実装
- BrowserUseAgent から PLP 関連ロジックを移動

**主要ファイル**:
- `app/agents/browser/plp_driver.py` - PlpDriver クラス
- `app/agents/browser_use_agent.py` - PlpDriver を使用するように修正

### Task D: Stage 5 – Extractor の site_config 駆動化

**目的**: PDP 抽出ロジックを「サイトごとにハードコード」するのではなく、`site_config` JSON で定義されたセレクタ・正規化ルールに基づいて抽出する汎用 Extractor に置き換える。

**実装内容**:
- `app/agents/browser/product_extractor.py` を新規作成
- `ProductExtractor` クラスと `ProductInfo` データクラスを定義
- 価格を `float` に変換する処理を追加
- `metadata` フィールドを追加して抽出結果の詳細を記録
- HTML保存ファイル名を `pdp_dom.html` → `pdp_raw.html` に変更
- BrowserExtractionService を ProductExtractor を使用するように修正

**主要ファイル**:
- `app/agents/browser/product_extractor.py` - ProductExtractor クラス
- `app/agents/browser/extractor.py` - BrowserExtractionService クラス（ProductExtractor を使用）

### Task E: テストとリグレッション防止

**目的**: 大規模リファクタリング後の破綻を防ぐため、PlpDriver と ProductExtractor のユニットテストを追加。

**実装内容**:
- `tests/test_plp_driver.py` - PlpDriver のユニットテスト（7つのテスト関数）
- `tests/test_product_extractor.py` - ProductExtractor のユニットテスト（8つのテスト関数）
- `tests/test_browser_use_agent_plp_integration.py` - BrowserUseAgent と PlpDriver の統合テスト（2つのテスト関数）

**テスト内容**:
- Happy path: PLP → PDP success
- Trap / Legal page detection
- Overlay handling
- Full PDP extraction
- Partial selectors / missing elements
- Price normalization (float変換)

## 環境情報

### 実行環境

- **OS**: WSL Ubuntu (Windows上で動作)
- **プロジェクトパス**: `/home/yn441611/atelier-kyo-manager`
- **仮想環境**: `venv/` または `myenv/` (WSL環境では `venv/bin/activate`)
- **Python**: Python 3.8+

### 重要なディレクトリ構造

```
atelier-kyo-manager/
├── app/
│   ├── agents/
│   │   ├── browser/
│   │   │   ├── plp_driver.py          # Task C: 新規作成
│   │   │   ├── product_extractor.py   # Task D: 新規作成
│   │   │   ├── navigation_driver.py   # 既存（汎用化済み）
│   │   │   ├── extractor.py           # Task D: 修正
│   │   │   └── session_manager.py     # 既存
│   │   └── browser_use_agent.py       # Task C: 修正
│   ├── config/
│   │   └── sites/
│   │       ├── base.json              # サイト設定のベース
│   │       └── overrides.local.json   # ローカルオーバーライド
│   └── core/
│       └── run_context.py             # RunContext クラス
├── tests/
│   ├── test_plp_driver.py             # Task E: 新規作成
│   ├── test_product_extractor.py      # Task E: 新規作成
│   └── test_browser_use_agent_plp_integration.py  # Task E: 新規作成
└── docs/
    └── completion_reports/            # 完了レポート
```

### 依存関係

**主要なパッケージ**:
- `playwright` - ブラウザ自動化
- `pytest`, `pytest-asyncio` - テストフレームワーク
- `Flask` - Webフレームワーク
- `selenium`, `selenium-stealth` - スクレイピング
- `beautifulsoup4` - HTML解析

**インストール方法**:
```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
pip install -r requirements.txt
```

## 重要な設計原則

### 1. サイト非依存化

- **原則**: サイト固有のロジックは `site_config` JSON に定義し、コード側は汎用的なロジックのみを持つ
- **実装**: `PlpDriver` と `ProductExtractor` は `site_config` を引数に取り、設定に基づいて動作する
- **例**: Moncler固有のURL補正（`/en-int/`）は `site_config["locale"]["normalize_rules"]` に定義

### 2. 責務の分離

- **PlpDriver**: PLP → PDP ナビゲーション
- **ProductExtractor**: PDP からの商品情報抽出
- **SessionManager**: ブラウザセッション管理
- **RunContext**: アーティファクト保存とログ管理
- **BrowserUseAgent**: これらを統合する薄いハブ

### 3. 型安全性

- `ProductInfo.price`: `Optional[float]` (Task Dで `Optional[str]` から変更)
- `ProductInfo.list_price`: `Optional[float]` (Task Dで `Optional[str]` から変更)
- `ProductInfo.metadata`: `Dict[str, Any]` (Task Dで追加)

### 4. テスト容易性

- モックを使用したユニットテスト
- 実際のブラウザは不要（Playwrightのモックを使用）
- 非同期テスト（`pytest-asyncio` を使用）

## 重要なファイルとその役割

### 新規作成ファイル

1. **`app/agents/browser/plp_driver.py`**
   - `PlpDriver` クラス: PLP → PDP ナビゲーション
   - `PlpNavigationResult` データクラス: ナビゲーション結果

2. **`app/agents/browser/product_extractor.py`**
   - `ProductExtractor` クラス: PDP からの商品情報抽出
   - `ProductInfo` データクラス: 抽出された商品情報
   - `PriceRules` データクラス: 価格正規化ルール

3. **`tests/test_plp_driver.py`**
   - PlpDriver のユニットテスト（7つのテスト関数）

4. **`tests/test_product_extractor.py`**
   - ProductExtractor のユニットテスト（8つのテスト関数）

5. **`tests/test_browser_use_agent_plp_integration.py`**
   - BrowserUseAgent と PlpDriver の統合テスト（2つのテスト関数）

### 変更されたファイル

1. **`app/agents/browser_use_agent.py`**
   - PlpDriver を使用するように修正
   - PLP 関連ロジックを削減

2. **`app/agents/browser/extractor.py`**
   - ProductExtractor を使用するように修正
   - ProductInfo を Dict に変換する際、`raw_html_path` と `metadata` を追加

3. **`requirements.txt`**
   - `pytest` と `pytest-asyncio` を追加

## ユーザールール（.cursorrules より）

### 基本方針

1. **言語**: 常に日本語で会話すること
2. **分析**: 変更を提案する前に、必ず既存のコードの文脈を分析すること
3. **尊重**: 明示的な許可なく、既存の機能を削除しないこと
4. **形式**: 見出しや箇条書きを使い、読みやすいMarkdown形式で出力すること

### カスタムコマンドとトリガー

ユーザーの入力に「現状のコードを分析・スキャン・理解したい」という意図（例：「分析開始」「解析して」「分析頼む」「みてみて」「現状どう？」「/analyze」など）が含まれていた場合、即座に以下のプロセスを実行：

1. **アクション**: プロジェクト全体のコンテキストを読み込み、依存関係グラフ（全体図）を構築する
2. **分析ターゲット**: リポジトリをスキャンし、以下をまとめる：
   - 主要コンポーネント: 主要な機能ブロックは何か
   - ボトルネック: パフォーマンスやロジックの妨げになっている箇所はどこか
   - 最大リスクファイル: 壊れやすい、または重要なファイルはどれか
   - 推奨されるモジュール化戦略: MASアーキテクチャに向けて、どうコードを分割すべきか
   - 結合度の高いファイル: 複雑に絡み合いすぎているファイルはどれか
3. **出力**: 「BUYMA Growth Hub」のアーキテクチャ視点に基づき、詳細なレポートを日本語で出力すること

### 完了レポート作成ルール

以下の作業が完了した際は、自動的に完了レポートを作成する：

- **リファクタリング作業**: 複数ファイル・モジュールにまたがる段階的なリファクタリング
- **機能追加**: 複数コンポーネントに影響する大きな機能追加
- **アーキテクチャ変更**: 新しいクラス・モジュール・アーキテクチャパターンの導入
- **重要なバグ修正**: 複数ファイルに影響する修正や、影響範囲の大きい修正
- **移行作業**: コード移行、依存関係更新、フレームワークアップグレード

**レポート形式**: `docs/completion_reports/<作業識別子>_COMPLETION_REPORT.md`

**必須セクション**:
1. 実装日時
2. 概要
3. 実装ステップ
4. 変更ファイル一覧
5. 動作確認結果
6. 設計上の改善点
7. 既知の制約・注意事項
8. 次のステップ

## プロジェクトルール（workspace rules より）

### Atelier-Kyo Project Firewall

**触って良いディレクトリ**:
- `/home/yn441611/atelier-kyo-manager` 配下のみ

**触ってはいけない領域**:
- `/home/yn441611/NexusCore`
- `/home/yn441611/*` (atelier-kyo-manager以外)
- `/etc`, `/usr`, `/var`, `/root` など OS ディレクトリ
- WSL 全体の設定ファイル
- Windows 側の `C:\Users\…` 配下

**禁止操作**:
- `rm -rf`（パス内容不問）
- `sudo` を含むコマンド
- `systemctl` / `service` 操作
- `git reset --hard` / `git clean -fdx`
- `apt remove` / `apt install`

### Atelier-Kyo Safe Shell & File Operations

**絶対に実行しないコマンド**:
- `rm -rf /` などルートディレクトリ削除系
- `rm -rf .*` / `rm -rf .git` / `rm -rf venv` / `rm -rf myenv`
- `git reset --hard` / `git clean -fdx`
- OS や WSL の設定を変えるコマンド

**ファイル操作の原則**:
- 破壊的操作（削除・上書き）は **常に diff ベース** で行い、1 ファイル単位の変更に留める
- 不要ファイルを削除する必要がある場合は、まず `archive/` などに退避する案を優先して提案する

### Atelier-Kyo Safe Test Execution Rules

**実行環境**:
- すべての pytest / shell 実行は **WSL Ubuntu** 上を前提とする
- プロジェクトルートは `/home/yn441611/atelier-kyo-manager` とみなす
- venv は次の優先順で有効化する：
  ```bash
  source venv/bin/activate 2>/dev/null || source myenv/Scripts/activate 2>/dev/null || true
  ```

**pytest の実行方法（標準）**:
```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate 2>/dev/null || source myenv/Scripts/activate 2>/dev/null || true
python -m pytest tests/
```

**禁止事項**:
- プロジェクトルート（`.`）に pytest を実行してはならない
- `git reset --hard` / `git clean -fdx` を実行してはならない
- `rm -rf` を含む破壊的操作を行わない

### Atelier-Kyo Test Quality Guidelines

**テストの基本原則**:
- 1テスト = 1責務（1 assertion グループ）
- 内部実装詳細ではなく **公開APIの振る舞い** を検証する
- 実際の LLM 呼び出しは禁止（すべてモック or スタブを使う）
- 外部 API / 時刻 / 乱数などに依存しないこと

**禁止・注意事項**:
- `time.sleep` や長時間I/Oを使わない
- 実際のファイル削除を行うテストは禁止（`tmp_path` / `tmp_path_factory` を使用）
- ランダム性を扱う場合は seed を固定
- 一度に多数の副作用を伴うテストを書かない

## 重要な注意事項

### 1. WSL環境での作業

- **コマンド実行**: WSL Ubuntu環境で実行すること
- **パス**: すべてのパスは `/home/yn441611/atelier-kyo-manager` を基準とする
- **仮想環境**: `venv/bin/activate` を使用（Windows環境の場合は `myenv/Scripts/activate`）

### 2. テスト実行

- **前提**: `pytest` と `pytest-asyncio` が必要
- **実行方法**: 
  ```bash
  source venv/bin/activate
  pip install pytest pytest-asyncio
  python -m pytest tests/test_plp_driver.py tests/test_product_extractor.py tests/test_browser_use_agent_plp_integration.py -v
  ```
- **注意**: テストはモックを使用しているため、実際のブラウザは不要

### 3. site_config の構造

**重要なキー**:
- `selectors.pdp.*`: PDP抽出用セレクタ
- `selectors.ui.*`: UI要素（Cookieバナーなど）のセレクタ
- `navigation.overlays.*`: オーバーレイ処理用セレクタ
- `navigation.trap_url_patterns`: Trapページ検出用パターン
- `discovery_settings.*`: 発見設定（タイムアウト、スクロール回数など）
- `price_rules.*`: 価格正規化ルール
- `locale.*`: ロケール設定

**設定例** (`app/config/sites/overrides.local.json`):
```json
{
  "MONCLER_OFFICIAL": {
    "selectors": {
      "pdp": {
        "title": ["h1.product-title"],
        "price": [".product-price"],
        "pdp_link_selectors": ["a.product-link"]
      }
    },
    "price_rules": {
      "strip_chars": ["¥", ",", " "],
      "decimal_separator": ".",
      "thousands_separator": ","
    }
  }
}
```

### 4. コード変更時の注意

- **既存の動作を壊さない**: MONCLER_OFFICIAL の既存動作は維持すること
- **フォールバック処理**: 既存のフォールバックロジックは維持（段階的な移行のため）
- **型の一貫性**: `price` と `list_price` は `float` 型に統一（Task Dで変更）
- **HTML保存**: PDP抽出時は `pdp_raw.html` として保存（Task Dで変更）

### 5. Git管理

- **Git LFS**: `app/agents/browser_use_agent.py` は以前 Git LFS で追跡されていたが、通常のGit追跡に戻した
- **コミット**: 小さな変更ごとにコミットすることを推奨
- **破壊的操作**: `git reset --hard` や `git clean -fdx` は禁止

## 次のステップ（推奨）

### 1. テストの実行と修正

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
pip install pytest pytest-asyncio
python -m pytest tests/test_plp_driver.py tests/test_product_extractor.py tests/test_browser_use_agent_plp_integration.py -v
```

エラーがあれば修正する。

### 2. E2Eテストの追加

- Moncler統合テストを追加
- 実際のブラウザでの動作確認

### 3. site_config スキーマの標準化

- PlpDriver と ProductExtractor が使用する `site_config` キーの標準化
- ドキュメント化

### 4. フォールバックロジックの削減

- すべての抽出を ProductExtractor に統一
- 既存のフォールバックロジックを段階的に削除

### 5. 新しい抽出フィールドの追加

- レビュー、在庫状況、配送情報などの抽出
- `ProductInfo` データクラスに新しいフィールドを追加

## 関連ドキュメント

### 完了レポート

- `docs/completion_reports/TASK_C_D_E_ENHANCED_COMPLETION_REPORT.md` - Task C, D, E の詳細レポート
- `docs/completion_reports/TASK_D_PRODUCT_EXTRACTOR_SITE_CONFIG_COMPLETION_REPORT.md` - Task D の詳細レポート
- `docs/completion_reports/TASK_E_TESTING_COMPLETION_REPORT.md` - Task E の詳細レポート

### テスト実行ガイド

- `tests/EXECUTE_TESTS.md` - 詳細な実行手順とトラブルシューティング
- `tests/RUN_TESTS.md` - テスト実行方法のまとめ
- `tests/README_TESTING.md` - テスト実行ガイド

### その他

- `.cursorrules` - プロジェクトルールとユーザールール
- `requirements.txt` - 依存関係一覧

## 連絡先・サポート

- **プロジェクトパス**: `/home/yn441611/atelier-kyo-manager`
- **環境**: WSL Ubuntu
- **主要言語**: Python 3.8+
- **フレームワーク**: Playwright, Flask, pytest

---

**重要**: このプロジェクトは継続的に開発中です。変更を行う際は、必ず既存の動作を壊さないように注意し、テストを実行して確認してください。

