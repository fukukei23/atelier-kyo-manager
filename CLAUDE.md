# CLAUDE.md — atelier-kyo-manager

> AI開発用プロジェクトガイドライン

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

### routes/（Blueprint）
| ファイル | URLプレフィックス | 役割 |
|---|---|---|
| analytics.py | `/` | ダッシュボード・テンプレート・FAQ・禁止ソース・在庫チェック |
| orders.py | `/orders`, `/cashflow` | 注文CRUD・キャッシュフロー |
| partners.py | `/partners`, `/customers` | パートナー・リピーター管理 |
| products.py | `/`, `/manage`, `/products` | 商品CRUD・パイプライン・CSV入出力 |
| misc.py | `/auto-research`, `/api/warehouses` | リサーチ・倉庫API |
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
