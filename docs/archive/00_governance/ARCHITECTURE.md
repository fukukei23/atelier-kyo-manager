ARCHITECTURE.md

プロジェクト

Name: atelier-kyo-manager

Domain: BUYMA 無在庫転売支援ツール

バージョン情報

Version: 1.0.3

LastUpdated: 2026-01-10

Commit:d27937ec0902fa88508f61ab4c313f8f8d5066c

1. 本書の位置づけ（重要）

本ドキュメントは、本プロジェクトにおける
**システム構成・責務境界・依存関係の正本（Single Source of Truth）**である。

実装・修正・設計判断は必ず本書を基準に行う

実装と乖離した場合は、本書を更新する

課題・評価・改善案は本書に記載しない
（それらは review / analysis 文書に分離する）

2. 全体構成

実行形態

ローカルPC上での単一ユーザー利用

Flask Web アプリとして起動

SQLite を正規DBとする

起動点

runserver.py
Flask アプリのエントリポイント。create_app() を呼び出し、アプリを起動する。

3. ディレクトリ構造（正）

.
├─ runserver.py
├─ app/
│  ├─ __init__.py
│  ├─ routes.py
│  ├─ models/
│  │  └─ __init__.py   ← モデル定義の正本（集約・公開）
│  ├─ utils/
│  ├─ agents/
│  ├─ config/
│  └─ templates/
├─ tests/
├─ data/
│  └─ exports/
├─ docs/
│  └─ 00_governance/
│     ├─ MASTER_PROTOCOL_TEMPLATE.md
│     ├─ PROJECT_PROFILE_BUYMA.md
│     ├─ ARCHITECTURE.md
│     └─ README.md


補足:
app/models.py は互換目的で残存する可能性があるが、正本は app/models/__init__.py（および配下の定義ファイル）とする。将来的に models.py は削除予定。

4. 責務境界（拘束ルール）

Web / Routing 層

app/routes.py

HTTP リクエスト受付

業務ロジックを持たない

utils / agents へ処理を委譲する

ドメインロジック層

app/utils/

価格計算、CSV処理、データ加工など

Flask / Web に依存しない設計とする

必要に応じて Model や Agent を呼び出す

CSV導線 / Sourcing Evaluation

app/utils/sourcing_csv_adapter.py（adapter）

CSVセルの正規化を担当する

数値パース（カンマ区切り対応）を行う

空欄 / N/A / "-" 等を "unknown" に正規化する

partial / complete の判定は行わない

downstream のロジックを一切知らない

app/utils/sourcing_input_schema.py（schema）

必須項目チェックを行う

型・範囲チェックを行う

unknown を含む場合の partial / complete 判定を行う

business rule の入口とする

app/utils/sourcing_csv_batch_runner.py（batch runner）

I/O（CSV入力・JSONL出力）を担当する

Fail-Fast / Fail-Soft を制御する

invalid 行を downstream に 絶対に流さない

streaming 前提で 1 行ずつ処理する

Agent 層

app/agents/

LLM やブラウザ操作など外部依存処理

失敗時は例外を送出（Fail-Fast）

データ層

app/models/__init__.py

ORM（SQLAlchemy）経由のみ

生SQLは原則禁止

5. データフロー（概要）

UI / Route
  ↓
Domain Logic (utils) ──┬──→ Agent (LLM / Browser)
  │                    │
  └────────────────────┴──→ Model (ORM)
                              ↓
                            SQLite


※ Logic層は、Agent（外部処理）とModel（DB処理）の両方をコーディネートする。

6. データベース方針

SQLite を正規DBとする

DB接続情報は環境変数経由

PostgreSQL への将来移行を阻害しない設計とする

マルチテナントは現時点では考慮しない

7. テスト構成

pytest を使用

tests/ 配下に集約

仕様変更時はテスト更新を必須とする

テストが Pass しない変更はマージ不可

8. ログ・エラー方針

例外の握りつぶしは禁止

障害時は即停止（Fail-Fast）

ログはトラブルシュート目的に限定

9. 変更ルール

以下に該当する場合、本書の更新を必須とする。

ファイル構成の変更

責務の移動

永続データ構造の変更

起動方法の変更

正本宣言
本 ARCHITECTURE.md は docs/00_governance/ 配下における唯一のアーキテクチャ正本である。
