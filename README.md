# Atelier Kyo Manager / BUYMA Growth Hub

> **フレームワークについて**: このプロジェクトは [AI Augmented System Design Framework (AASDF)](https://github.com/fukukei23/ai-augmented-system-design-framework) をベースに開発されています。  
> フレームワークの詳細・使い方・セキュリティ方針については、上記リポジトリを参照してください。

**Atelier Kyo Manager** は、BUYMA 無在庫転売向けの

- リサーチ
- 在庫監視
- AI 説明文生成
- 利益計算
- ブランド公式サイト向けスクレイピング自律エージェント

を統合した **開発用コードベース兼 SaaS プロトタイプ** です。

- 規約遵守の **手動出品** を前提とした支援ツール
- 仕入先は海外ショップのみ（国内 EC 仕入れは使用しない方針）
- NexusCore 系の自律型 AI エージェント群と連携し、  
  Moncler などブランド公式サイトの PLP/PDP 抽出・自己修復を行う

---

## 1. コンセプト（SaaS / Multi-Agent 視点）

本リポジトリは、将来的な **「BUYMA Growth Hub」SaaS** を見据えた開発基盤です。

- BUYMA ショップ運営に必要な **データ収集・分析・意思決定** を一元化
- NexusCore ベースの **マルチエージェント / Self-Healing スクレイピング** を採用
- **観測性（Observability）・差分管理・完了レポート** を重視した運用設計
- 後にマルチテナント化・API 提供を見据えた **疎結合モジュール構成**

---

## 2. 主な機能

### 2.1 商品管理

- CSV / Google Sheets からの商品情報インポート
- 仕入 URL / 仕入価格 / 販売価格 / 利益率の一元管理
- 商品ステータス（出品待ち / 出品済み / 在庫切れ など）の管理

### 2.2 利益計算

- 代表的な計算要素
  - 仕入値
  - 国際送料
  - BUYMA 手数料（例: 7.7%）
  - 為替レート
  - 関税 / 消費税
- 利益率 10%以上を目標とした自動計算
- 「販売価格 → 想定利益 / 利益率」のシミュレーション

### 2.3 在庫チェック / 価格監視

- Celery 定期タスクで仕入先 URL を巡回
- 在庫切れ / 価格上昇 / ページ構造変更を検知
- Slack / Streamlit ダッシュボード / n8n 経由での通知連携

### 2.4 AI コンテンツ生成

- OpenAI / Claude 等 LLM による
  - 商品説明文生成
  - タイトル生成
  - キャッチコピー生成
- ブランド世界観・希少性・迅速発送・安心感などをプロンプトで制御

### 2.5 買付パートナー管理

- 外注・買付パートナー情報の管理
- SLA（発送リードタイム）遅延の検知・可視化
- 将来的なスコアリング・アラート連携を想定

### 2.6 ブラウザ自動操作 / Self-Healing エージェント（NexusCore 連携）

- Playwright / DrissionPage によるブランド公式サイトのスクレイピング
- **Moncler 向け PLP → PDP 抽出ロジック**
  - トラップページ検出
  - ロケール制御（/en-int, shipToCountry など）
  - セレクタ自己学習（learned_selectors.json）
- NavigationDriver / SessionManager / Telemetry による観測性強化
- Self-Healing Agent による
  - セレクタ壊れ検知
  - 修正案生成
  - Telemetry ベースの再学習サイクル
- **パッチ適用ガイド**: Moncler run 失敗時のパッチ候補を手動で適用する手順は [`docs/official/moncler_patch_apply_guide.md`](docs/official/moncler_patch_apply_guide.md) を参照

### 2.7 Self-Healing / Selector-Healing ダッシュボード

- Self-Healing と Selector Auto-Healing の効果・挙動を可視化
- メトリクス（`docs/reports/self_healing_metrics.jsonl`）を読み込み、以下を表示：
  - Overall KPIs（総実行回数、成功率、平均試行回数など）
  - サイト別メトリクス
  - 日次推移チャート
  - Selector-Healing フォーカスビュー
- **起動方法**:
  ```bash
  streamlit run dashboard_self_healing.py
  ```

---

## 3. アーキテクチャ概要

### 3.1 技術スタック

- Backend: Flask または FastAPI
- Task Queue: Celery + Redis
- Scraping: Playwright / DrissionPage
- Database: PostgreSQL（本番想定） / SQLite（開発）
- Dashboard: Streamlit（開発用ビュー）
- Orchestration: NexusCore / 自律エージェント群
- Integration: n8n（Slack / Google Sheets / Gmail 連携）

### 3.2 論理コンポーネント

- **Core Layer** (`app/core/**`)
  - 価格計算 / 在庫チェック / 仕入サイト管理など BUYMA ドメインロジック

- **Agent Layer** (`app/agents/**`)
  - browser_use_agent, selector_discovery_agent, self_healing_agent, profitability_agent など

- **Browser Layer** (`app/agents/browser/**`)
  - NavigationDriver, PLP Driver, Extractor, Telemetry, SessionManager 等

- **Extractor Layer** (`app/extractors/**`)
  - Moncler 等ブランドごとの PLP/PDP 抽出ロジック

- **Config Layer** (`app/config/**`)
  - スクレイピング対象サイト / LLM コスト / 統合設定

- **Docs / Spec / Reports** (`docs/**`)
  - 設計 / Spec / 完了レポート / チャット記録 / テスト結果

---

## 4. ディレクトリ構成（抜粋）

```text
atelier-kyo-manager/
├─ app/
│  ├─ core/                 # BUYMA ドメインロジック
│  ├─ agents/               # NexusCore 連携エージェント群
│  │   ├─ browser/          # NavigationDriver, Extractor, Telemetry など
│  │   ├─ browser_use_agent.py
│  │   ├─ browser_use_moncler_patch.py
│  │   ├─ selector_discovery_agent.py
│  │   ├─ self_healing_agent.py
│  │   └─ ...
│  ├─ extractors/           # ブランド別抽出ロジック
│  ├─ templates/            # Flask UI（編集禁止領域）
│  ├─ static/               # CSS / JS / 画像（編集禁止領域）
│  ├─ routes.py             # Flask ルーティング（薄いレイヤ）
│  ├─ models.py             # SQLAlchemy モデル
│  ├─ extensions.py         # Flask 拡張初期化
│  └─ ...
│
├─ app/config/
│  ├─ sites/
│  │   ├─ base.json
│  │   ├─ overrides.local.json   # Moncler 等サイト別設定
│  │   └─ ...
│  └─ ...
│
├─ docs/
│  ├─ official/             # 公式設計ドキュメント
│  │   ├─ system_design.md
│  │   ├─ db_schema.md
│  │   ├─ n8n_integration.md
│  │   ├─ saas_design.md
│  │   └─ architecture_overview.mmd
│  ├─ spec/                 # CR-ATELIER 系 Spec
│  ├─ completion_reports/   # 完了レポート
│  ├─ chat_records/         # チャット記録（.gitignore 済）
│  ├─ reports/
│  └─ test_results/
│
├─ instance/                # 実行時データ（runs, logs, sessions, screenshots）※自動生成
├─ exports/                 # エクスポート成果物（ZIP, CSV など）※自動生成
├─ scripts/                 # ユーティリティスクリプト
├─ tools/                   # 開発補助ツール
├─ tests/                   # pytest
├─ .cursorrules             # Cursor 用ルール（編集ポリシー）
└─ README.md
```

## 5. セットアップ

### 5.1 仮想環境（venv）

```bash
# プロジェクトディレクトリに移動
cd /home/yn441611/atelier-kyo-manager

# 仮想環境を作成（初回のみ）
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate

# 依存パッケージをインストール
pip install -r requirements.txt

# 仮想環境を無効化（必要に応じて）
deactivate
```

#### Ubuntu WSL での実行例

```bash
# WSL の bash で実行
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

# Moncler 用ブラウザラン
python -m app.scripts.run_site moncler --query "down jacket"
```

#### よくある問題

**問題1: `source: command not found`**

- PowerShell ではなく bash で実行する
- WSL ターミナルで直接実行するか、`wsl bash -c "..."` を利用

**問題2: 仮想環境が見つからない**

```bash
ls -la venv/          # ディレクトリが存在するか確認
python3 -m venv venv  # 存在しない場合は作成
```

**問題3: パッケージがインストールされない**

```bash
pip install --upgrade pip

# 仮想環境の python になっているか
which python
# → /home/yn441611/atelier-kyo-manager/venv/bin/python であれば OK
```

#### 便利コマンド

```bash
pip list                      # インストール済みパッケージ一覧
pip freeze > requirements.txt # requirements の更新
python --version              # Python バージョン確認
```

### 5.2 .env 例

```ini
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost:5432/atelier_kyo
REDIS_URL=redis://localhost:6379/0
```

## 6. 実行方法（代表例）

### 6.1 API サーバ

```bash
# Flask の場合
flask run

# FastAPI の場合
uvicorn app.main:app --reload
```

### 6.2 Celery ワーカー

```bash
celery -A app.celery_app worker -l info
celery -A app.celery_app beat -l info
```

### 6.3 Streamlit ダッシュボード

```bash
# 既存の Streamlit ダッシュボード
streamlit run app/streamlit_app.py

# Self-Healing / Selector-Healing ダッシュボード（CR-ATELIER-003 Phase D-11）
streamlit run dashboard_self_healing.py
```

### 6.4 Moncler スクレイピング（調査用）

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

python -m app.scripts.run_site moncler --query "down jacket" --headful
```

実行結果は `instance/runs/YYYYMMDD_*/` 以下に保存されます（`system.log`, `failure_dom.html`, `screenshots/` など）。

## 7. 編集ポリシー（人間 & AI 共通）

詳細なルールは `.cursorrules` を参照。README では要点のみ記載します。

### 7.1 積極的に編集してよい領域 ✅

- **`app/core/**`**：BUYMA ドメインロジック
- **`app/agents/**`**：NexusCore 連携エージェント
- **`app/extractors/**`**：ブランド別抽出ロジック
- **`docs/**`**：設計 / Spec / レポート
- **`tests/**`**：pytest テスト
- **`README.md`**：本ドキュメント

### 7.2 原則編集禁止領域 ❌

- **`app/templates/**`**：Flask UI テンプレート
- **`app/forms/**`**：WTForms 定義
- **`app/static/**`**：CSS / JS / 画像
- **`app/routes.py`**：ルーティング（極力薄く保持）
- **`instance/**`**：実行時データ（自動生成）
- **`export/**`, `exports/**`**：成果物
- **`**/logs/**`, `**/screenshots/**`**：ログ・スクリーンショット

### 7.3 慎重に編集すべき領域 ⚠️

- **`app/models.py`**：DB スキーマ（既存カラムの削除・型変更は避ける）
- **`app/extensions.py`**：Flask 拡張の初期化
- **`app/config/**`**：サイト設定 / API 設定
- **`scripts/**`, `tools/**`**：運用・開発ツール

## 8. Spec 駆動開発ルール（Spec Kit Mode）

このリポジトリでは、**仕様駆動開発（Specification-Driven Development, SDD）** を採用します。

### 8.1 フロー

**Specify（仕様策定）**

- 新しい CR / 大きめの機能追加を行う前に、`docs/spec/` に Spec を作成
- ファイル名: `CR-ATELIER-XXX_<タイトル>.md`
- 必須セクション:
  - Overview & Context
  - Scope（In-Scope / Out-of-Scope）
  - Implementation Plan
  - Testing Strategy

**Plan（計画）**

- 実装ステップを Step1, Step2... と具体化
- どのテストをどのコマンドで実行するか明記

**Implement（実装）**

- 「Spec OK」「この Spec で実装」と明示された後にコード変更を行う
- コード提案時は「Plan のどのステップか」を意識する

**Report（報告）**

- 実装完了後、完了レポートを作成（次節）

## 9. 完了レポート作成ルール

### 9.1 作成タイミング

以下の作業完了時には必ず完了レポートを作成する：

- 複数モジュールにまたがるリファクタリング
- 複数コンポーネントに影響する機能追加
- 重要バグ修正
- アーキテクチャ変更
- 依存関係更新 / 移行作業

### 9.2 保存場所・命名

- **パス**: `docs/completion_reports/`
- **例**:
  - `docs/completion_reports/CR_ATELIER_002_STEP3_COMPLETION_REPORT.md`
  - `docs/completion_reports/STAGE_3A3_COMPLETION_REPORT.md`

### 9.3 必須セクション

1. **実装日時**
2. **概要**（目的・ゴール・前提）
3. **実装ステップ**（何を・なぜ）
4. **変更ファイル一覧**（新規 / 変更）
5. **動作確認結果**（lint / tests / 手動確認）
6. **設計上の改善点**（アーキテクチャ・拡張性・品質）
7. **既知の制約・注意事項**
8. **次のステップ**（フォローアップ案）

## 10. チャット記録の自動保存ルール

### 10.1 作成トリガー

- ユーザーが「Chat記録して」「チャットログ残して」等を明示したとき
- 重要な作業セッション完了時（完了レポート作成時など）
- セッション終了時（可能な範囲で）

### 10.2 保存場所・命名

- **パス**: `docs/chat_records/`
- **ファイル名形式**: `CHAT_RECORD_YYYYMMDD_<概要>.md`
- **例**:
  - `docs/chat_records/CHAT_RECORD_20251128_MONCLER_DRISSION_DIAGNOSTICS.md`

チャット記録は `.gitignore` 済みで GitHub にはコミットされません。

## 11. BUYMA 規約に関する注意事項

- BUYMA への **自動ログイン・自動出品** は行わない（規約違反となる可能性あり）
- 国内 EC サイト（楽天市場・Amazon.co.jp 等）を仕入先にしない
- 発送期限（18日）・キャンセル率など、BUYMA 側の運営ルールを遵守することを前提とする

このリポジトリはあくまで **「リサーチ・在庫監視・情報整理・支援」のためのツール** であり、
最終的な出品・取引操作はユーザー自身が責任をもって手動で行う前提です。

## 12. ライセンス・免責（ドラフト）

現時点では個人利用・検証目的を想定したコードベースです。

実運用・商用利用時は、BUYMA 利用規約・各仕入先サイトの利用規約・個人情報保護法等を遵守してください。

本リポジトリの利用によって発生したいかなる損害についても、作者は責任を負いません。
