# CLAUDE.md — atelier-kyo-manager

> AI開発用プロジェクトガイドライン

## ⚠️ 必須事前読込

**実装作業を開始する前に `docs/経営者判断.md` を必ず一読すること。**
経営者のビジネス判断・手数料設定・関税率・ブランド戦略等が記載されており、
これに基づいて実装を行う。本ファイルと矛盾する場合は経営者判断ファイルを優先する。

## プロジェクト概要

BUYMA×Buyandshipを利用した転売管理システム。Flask製Webアプリ。
- 商品管理・価格計算・スクレイピング・画像処理・自動出品パイプライン
- 注文管理（18日ルール）・キャッシュフロー予測
- AIチャットボット・自動発注ステートマシン

## アーキテクチャ

```
Flask MVC + Service Layer
  routes/    → HTTPリクエスト処理（Blueprint）
  services/  → ビジネスロジック
  models/    → SQLAlchemy ORM（SQLite）
  utils/     → 汎用ユーティリティ（LLM制御・為替・価格計算）
```

- エントリ: `app/__init__.py` → `create_app()` ファクトリパターン
- DB: SQLite + Flask-Migrate
- 認証: Flask-Login（`/auth/login`, `/auth/logout`）
- CSRF: Flask-WTF

## 起動コマンド

```bash
# Flask 開発サーバー起動
FLASK_APP=app ./venv/bin/python -m flask run   # http://localhost:5000

# または Makefile.local 経由（推奨）
make run   # FLASK_APP=app で flask run を実行
```

## テスト実行

```bash
# 基本実行
./venv/bin/python -m pytest tests/ -x -q

# カバレッジ付き
./venv/bin/python -m pytest tests/ --cov=app --cov-report=term-missing -q

# 特定テスト
./venv/bin/python -m pytest tests/test_xxx.py -v
```

## コーディング規約

- Python 3.10+（`from __future__ import annotations` 推奨）
- 型ヒント必須
- 1関数1責務
- docstringは最小限（"why"のみ記載）
- テストは `tests/` に配置、命名規則: `test_<module>_<scenario>.py`

## 主要モジュール

### routes/（Blueprint — 14モジュール）
| ファイル | URLプレフィックス | 役割 |
|---|---|---|
| products.py | `/`, `/manage`, `/products` | F01商品CRUD + F02BUYMA拡張 + CSV入出力 + パイプライン |
| orders.py | `/orders`, `/cashflow` | F05注文管理（18日ルール） + F08キャッシュフロー予測 |
| partners.py | `/partners`, `/customers` | F06パートナー + F13リピーター管理 |
| analytics.py | `/analytics`, `/dashboard` | F09ブランド分析 + FR-018ダッシュボード |
| listing_templates.py | `/listing-templates` | F03出品テンプレート管理 |
| faq_templates.py | `/faq-templates` | FR-010基盤 FAQテンプレートCRUD |
| prohibited_sources.py | `/api/prohibited-sources` | F04禁制品買付先チェックAPI |
| listing_progress.py | `/listing-progress` | F07出品進捗トラッカー |
| shipment_notifications.py | `/shipment-notifications` | FR-012発送通知管理 |
| stock_checks.py | `/stock-check`, `/api/stock-check` | F10在庫＆価格チェック（クイック更新API含む） |
| popularity.py | `/popularity` | F11人気度トラッキング |
| region_recommendations.py | `/region-recommendations` | F12買付先地域レコメンド |
| auto_orders.py | `/auto-orders` | FR-009 AI自動発注ステートマシン |
| chatbot.py | `/chatbot` | FR-010顧客対応AI ChatBot |
| misc.py | `/auto-research`, `/api/warehouses` | 互換リダイレクト + リサーチ + API倉庫 |
| warehouse_webhook.py | `/api/warehouse/events` | Webhook受信 |

### services/（ビジネスロジック）
| サービス | 役割 |
|---|---|
| AutoOrderService | 自動発注ステートマシン |
| ChatBotService | FAQ照合・AI回答生成 |
| ImageService | 画像DL・背景除去 |
| NotificationService | Slack通知 |
| PipelineService | 出品パイプライン統合 |
| PriceScraper | 価格・在庫スクレイピング |
| template_service | 出品テキスト生成・CSV出力 |
| warehouse_event_service | 倉庫Webhookイベント処理 |

### models/（SQLAlchemy）
Product, Order, User, Partner, CustomerInquiry, FaqTemplate,
ListingProgress, ListingTemplate, PopularityTracker, ProhibitedSource,
RegionRecommendation, RepeatCustomer, ShipmentNotification, StockCheck

## 環境変数

`.env.template` を参照。主要変数:
- `FLASK_APP` — アプリエントリ
- `BUYANDSHIP_*` — Buyandship認証
- `OPENAI_API_KEY`, `GEMINI_API_KEY` — LLM API
- `LLM_ORDER_*` — LLMルーティング設定
- `CRAWLER_*` — スクレイピング設定

## 注意事項

- `instance/` にSQLite DBが配置される（gitignore済み）
- `data/` に実行時データが生成される
- Webhook受信は `warehouse_webhook.py` で処理
- LLM呼び出しは `utils/ai_llm_controller.py` で一元管理

## 作業記録ルール

- プロジェクト固有の変更履歴は `docs/変更履歴.md` に記録すること
- リファクタリングバックログの進捗は `docs/リファクタリングバックログ.md` に✅完了マークを付けること
- SSOT側（`obsidian-ssot/01_DECISIONS/atelier-kyo-manager/`）には判断理由と技術的詳細を記録し、プロジェクト側（`docs/変更履歴.md`）には変更概要を記録する
