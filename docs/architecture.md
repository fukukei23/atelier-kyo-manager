# atelier-kyo-manager 設計書

**日付**: 2026-04-16
**ステータス**: ドラフト
**目的**: 本システムの「どう作るか」の技術構成を定義する

---

## 1. システムアーキテクチャ

```
┌─────────────────────────────────────────────────┐
│                Frontend (Flask Templates)         │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│               app.py (ルーティング・統合)          │
├─────────────────────────────────────────────────┤
│  Data Collection Layer                           │
│  ├─ brand_site_collector.py (ブランド公式スクレイパー) │
│  ├─ brightdata_scraper.py   (BUYMAスクレイピング)    │
│  └─ matcher.py              (SKU/型番マッチング)     │
├─────────────────────────────────────────────────┤
│  Business Logic Layer                            │
│  ├─ cost_calculator.py   (実ベース利益計算)         │
│  ├─ pricing_calculator.py (価格最適化・既存)        │
│  ├─ shipping_tracker.py  (18日ルール管理)           │
│  ├─ warehouse_manager.py (転送倉庫管理)             │
│  └─ order_manager.py     (注文・発注管理/Phase2)    │
├─────────────────────────────────────────────────┤
│  AI & External Layer                             │
│  ├─ OpenAI API  (画像処理・説明文生成)              │
│  └─ Slack Webhook (アラート通知)                   │
├─────────────────────────────────────────────────┤
│  Data Store                                      │
│  ├─ SQLite (instance/)                           │
│  └─ CSV (export/)                                │
└─────────────────────────────────────────────────┘
         │                              │
    pw-stealth-enhanced           Bright Data API
    (PyPI公開済み・ stealth基盤)     (BUYMA側スクレイピング)
```

---

## 2. ディレクトリ構造

```
atelier-kyo-manager/
├── app.py                      # Flaskメイン
├── config.py                   # 環境変数・定数管理（New）
├── models/                     # データモデル定義（New）
│   ├── __init__.py
│   ├── product.py              # 商品マスタ
│   ├── order.py                # 注文管理
│   └── shipping.py             # 発送・転送管理
├── services/                   # ビジネスロジック層（New）
│   ├── cost_calculator.py      # 実ベース利益計算（FR-005）
│   ├── shipping_tracker.py     # 18日ルール管理（FR-006）
│   ├── warehouse_manager.py    # 転送倉庫管理
│   └── order_manager.py        # 注文・発注管理（Phase 2）
├── scrapers/                   # スクレイピング層
│   ├── brand_site_collector.py # ブランド公式サイト用（New）
│   ├── brightdata_scraper.py   # BUYMAスクレイピング（既存）
│   └── matcher.py              # SKU/型番マッチング（New）
├── price_optimization_system/  # 価格最適化（既存）
├── pricing_calculator.py       # 利益計算（既存）
├── instance/                   # SQLite DB
├── export/                     # CSVエクスポート
├── venv/
└── .env
```

---

## 3. データモデル

### Product（仕入れ商品マスタ）
| カラム | 型 | 説明 |
|--------|-----|------|
| id | INTEGER PK | 自動採番 |
| brand_name | TEXT | ブランド名 |
| product_name | TEXT | 商品名 |
| sku | TEXT | 型番（マッチングキー） |
| source_url | TEXT | 公式サイトURL |
| base_price | REAL | 現地価格 |
| currency | TEXT | 通貨（USD/EUR） |
| image_urls | TEXT | 画像URL（カンマ区切り） |

### Listing（BUYMA出品情報）
| カラム | 型 | 説明 |
|--------|-----|------|
| id | INTEGER PK | 自動採番 |
| product_id | INTEGER FK | Product参照 |
| buyma_url | TEXT | BUYMA出品URL |
| selling_price | REAL | 販売価格(JPY) |
| description | TEXT | AI生成説明文 |

### Order（注文・発注管理）
| カラム | 型 | 説明 |
|--------|-----|------|
| id | INTEGER PK | 自動採番 |
| listing_id | INTEGER FK | Listing参照 |
| order_date | DATETIME | 注文日 |
| deadline_date | DATETIME | 発送期限 |
| status | TEXT | PENDING/ORDERED/WAREHOUSE/SHIPPED/DELIVERED/ALERT |
| tracking_number | TEXT | 追跡番号 |
| purchase_price | REAL | 実仕入れ額 |
| shipping_fee | REAL | 転送送料 |
| customs_duty | REAL | 関税 |
| platform_fee | REAL | BUYMA手数料 |

### ShippingWarehouse（転送倉庫マスタ）
| カラム | 型 | 説明 |
|--------|-----|------|
| id | INTEGER PK | 自動採番 |
| company_name | TEXT | 倉庫サービス名 |
| base_fee_table | TEXT | 送料テーブル(JSON) |
| estimated_days | INTEGER | 基本転送日数 |

---

## 4. 外部サービス連携

| サービス | 用途 | 連携モジュール | 状態 |
|---------|------|---------------|------|
| Bright Data | BUYMA側スクレイピング | brightdata_scraper.py | 動作確認済み |
| pw-stealth-enhanced | ブランド公式サイト巡回 | brand_site_collector.py | PyPI公開済み |
| OpenAI API | 画像処理・説明文生成 | AIレイヤー | 実装済み |
| 転送倉庫API | 送料見積・ステータス取得 | warehouse_manager.py | 未調査 |
| Slack Webhook | アラート通知 | shipping_tracker.py | 未実装 |

---

## 5. 設定管理

```python
# config.py（新設）
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///instance/app.db'
    BRIGHT_DATA_PROXY = os.getenv('BRIGHT_DATA_PROXY_URL')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    SHIPPING_RULE_DAYS = 18
    PLATFORM_FEE_RATE = 0.077  # BUYMA手数料 7.7%
```

---

## 6. SaaS化に向けた考慮事項

| 項目 | 現在(MVP) | SaaS化時(Phase 3) |
|------|----------|------------------|
| DB | SQLite | PostgreSQL/MySQL |
| 認証 | なし | Flask-Login + OAuth |
| テナント分離 | なし | 全テーブルにuser_id |
| タスク処理 | 同期 | Celery + Redis |
| フロントエンド | Flask Templates | API化→React等 |
| デプロイ | ローカル | Docker + クラウド |
