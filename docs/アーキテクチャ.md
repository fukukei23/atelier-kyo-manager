# atelier-kyo-manager アーキテクチャ

> **最終更新**: 2026-04-20
> **状態**: Phase 1 完了・Phase 2 計画中

---

## 1. システム概要

atelier-kyo-managerは、BUYMAでのファッションレセラー業務を包括的に支援するWebアプリケーション。商品リサーチから利益計算、画像処理、出品テキスト生成、注文管理まで、一連のパイプラインを自動化・効率化する。

### 技術スタック

| レイヤー | 技術 | 状態 |
|---|---|---|
| バックエンド | Python 3.11+ / Flask | 実装済み |
| DB / ORM | SQLAlchemy + Alembic (SQLite) | 実装済み |
| 認証 | Flask-Login | 実装済み |
| フロントエンド | Flask Templates (Jinja2) + Tailwind CSS | 実装済み |
| AI / LLM | Gemini / DeepSeek / OpenAI (ルーティング) | 実装済み |
| 画像処理 | rembg (背景除去) + Selenium (画像収集) | 実装済み |
| スクレイピング | Bright Data + Selenium | 実装済み |
| テスト | pytest (231テスト / 36ファイル) | 実装済み |

---

## 2. アプリケーション起動構造

### Flask Factory Pattern

```python
# app/__init__.py
def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(get_config(config_name))

    # 拡張初期化
    db.init_app(app)
    login_manager.init_app(app)

    # Blueprint登録
    register_blueprints(app)

    return app
```

- `create_app()` によるFactory Patternを採用
- テスト・開発・本番で設定を切り替え可能
- 全ての拡張は `app/extensions.py` で一元管理

### 拡張モジュール

```python
# app/extensions.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()
```

---

## 3. ディレクトリ構造

```
atelier-kyo-manager/
├── app/
│   ├── __init__.py              # Flask Factory (create_app)
│   ├── auth.py                  # Flask-Login認証ロジック
│   ├── decorators.py            # login_required等デコレータ
│   ├── extensions.py            # db, login_manager拡張
│   ├── forms.py                 # WTFormsフォーム定義
│   │
│   ├── config/                  # 設定モジュール
│   │   ├── config.py            # Config クラス (Dev/Test/Prod)
│   │   ├── config_loader.py     # 設定読み込みユーティリティ
│   │   ├── generate_secrets.py  # 秘密鍵生成スクリプト
│   │   ├── loader.py            # YAML設定ファイル読み込み
│   │   ├── paths.py             # パス管理
│   │   ├── protocols.py         # インターフェース定義
│   │   ├── sites/               # サイト別設定JSON (Moncler, Gucci等)
│   │   └── integrations/        # 外部サービス連携設定
│   │
│   ├── models/                  # データモデル (11テーブル)
│   │   ├── product.py           # 商品マスタ (pipeline_status含む)
│   │   ├── order.py             # 注文管理 (18日ルール、利益計算)
│   │   ├── partner.py           # 仕入れ先パートナー
│   │   ├── user.py              # ユーザー認証 (Flask-Login)
│   │   ├── listing_template.py  # 出品テンプレート
│   │   ├── listing_progress.py  # 出品進捗トラッキング
│   │   ├── stock_check.py       # 価格・在庫チェック
│   │   ├── popularity_tracker.py # 人気度スコア
│   │   ├── region_recommendation.py # 仕入れ地域推奨
│   │   ├── repeat_customer.py   # リピート顧客管理
│   │   ├── prohibited_source.py # 禁止仕入れ先
│   │   └── result_models.py     # 結果・集計モデル
│   │
│   ├── routes/                  # ルート (Blueprint) 6モジュール
│   │   ├── products.py          # 商品CRUD + パイプライン + 出品候補
│   │   ├── orders.py            # 注文管理 (18日ルール、利益計算)
│   │   ├── partners.py          # パートナー管理
│   │   ├── analytics.py         # 売上分析
│   │   ├── warehouse_webhook.py # 転送倉庫Webhook (FR-012基盤)
│   │   └── misc.py              # その他ユーティリティルート
│   │
│   ├── services/                # ビジネスロジック 5サービス
│   │   ├── image_service.py     # 画像DL + 背景除去 (FR-002)
│   │   ├── pipeline_service.py  # パイプライン統合管理 (FR-002/003)
│   │   ├── template_service.py  # 出品テキスト生成 (FR-003)
│   │   ├── price_scraper.py     # 価格スクレイピング (Bright Data)
│   │   └── warehouse_event_service.py # 倉庫イベント処理
│   │
│   ├── core/pricing/            # 価格計算エンジン
│   │   ├── calculator.py        # 実ベース利益計算 (FR-005)
│   │   ├── rules.py             # 計算ルール定義
│   │   └── schemas.py           # データスキーマ
│   │
│   ├── agents/                  # AIエージェント群
│   │   ├── browser/             # ブラウザ操作モジュール
│   │   ├── plugins/             # サイト別プラグイン
│   │   │   ├── moncler_plp_v1.py
│   │   │   ├── gucci_plp_v1.py
│   │   │   ├── prada_plp_v1.py
│   │   │   ├── farfetch_plp_v1.py
│   │   │   └── ssense_plp_v1.py
│   │   ├── ai_vision_agent.py
│   │   ├── self_healing_agent.py
│   │   ├── price_intelligence_agent.py
│   │   └── (他多数)
│   │
│   ├── utils/                   # ユーティリティ
│   │   ├── ai_llm_controller.py # LLMルーティング (Gemini/DeepSeek/OpenAI)
│   │   ├── ai_image_crawler.py  # Selenium画像収集
│   │   ├── ai_background_remover.py # rembg背景除去
│   │   ├── ai_generate_descriptions.py # AI説明文生成
│   │   ├── csv_handler.py       # CSV出力
│   │   ├── fx_utils.py          # 為替レート取得
│   │   └── (他多数)
│   │
│   └── templates/               # 26 HTMLテンプレート
│       ├── auth/login.html
│       ├── pipeline_result.html
│       ├── listing_candidates.html
│       ├── orders.html
│       └── (他計26ファイル)
│
├── tests/                       # 36テストファイル / 231テスト
├── migrations/                  # Alembicマイグレーション
├── docs/                        # プロジェクトドキュメント (5ファイル)
├── seed_data.py                 # テストデータ投入（冪等性保証）
└── instance/                    # SQLite DB
```

---

## 4. データモデル（11テーブル）

### ER図（概要）

```
User ──────────────────────────── 認証・認可

Product ──── Order ──── Partner   商品・注文・仕入れ先の核
  │            │
  ├─ ListingTemplate             出品テンプレート
  ├─ ListingProgress             出品進捗
  ├─ StockCheck                  価格・在庫チェック
  ├─ PopularityTracker           人気度スコア

RegionRecommendation             仕入れ地域推奨
RepeatCustomer                   リピート顧客管理
ProhibitedSource                 禁止仕入れ先
ResultModels                     結果・集計データ
```

### Product（商品マスタ）

| フィールド | 型 | 説明 |
|---|---|---|
| id | Integer (PK) | 商品ID |
| name | String | 商品名 |
| brand | String | ブランド名 |
| purchase_price | Float | 仕入れ原価 |
| selling_price | Float | 販売価格 |
| pipeline_status | String | pending/running/success/partial/failed |
| stock_status | Boolean | 在庫有無 |
| listing_status | String | listed/draft/sold/archived |
| source_type | String | overseas/domestic |
| source_region | String | 仕入れ地域 (FR/IT/US等) |
| item_category | String | bag/jacket/shoes/tops等 |

### Order（注文管理）

| フィールド | 型 | 説明 |
|---|---|---|
| id | Integer (PK) | 注文ID |
| product_id | Integer (FK) | 商品ID |
| partner_id | Integer (FK) | パートナーID |
| order_date | DateTime | 注文日 |
| status | String | pending/shipped/completed/cancelled |
| selling_price | Float | 販売価格 |
| purchase_cost | Float | 仕入れ原価 |
| customs_duty | Float | 関税 |
| profit | Float | 利益（自動計算） |

### その他モデル

| モデル | 役割 |
|---|---|
| Partner | 仕入れ先パートナー管理 |
| User | Flask-Login認証ユーザー |
| ListingTemplate | Jinja2風出品テンプレート |
| ListingProgress | 日次出品数トラッキング |
| StockCheck | 価格・在庫変動チェック |
| PopularityTracker | 閲覧数・お気に入り・問合せ数 |
| RegionRecommendation | 仕入れ地域別利益率・信頼性 |
| RepeatCustomer | リピート顧客セグメント管理 |
| ProhibitedSource | BUYMA禁止仕入れ先 |

---

## 5. レイヤー構造

```
[Browser]
    │
    ▼
[Flask Routes (Blueprint)]  ← app/routes/*.py
    │
    ├── 認証チェック (app/auth.py, app/decorators.py)
    ├── フォームバリデーション (app/forms.py)
    │
    ▼
[Services]  ← app/services/*.py
    │
    ├── ImageService     → 画像DL + 背景除去
    ├── PipelineService  → パイプライン統合管理
    ├── TemplateService  → 出品テキスト生成
    ├── PriceScraper     → 価格スクレイピング
    └── WarehouseEventService → 倉庫イベント処理
    │
    ▼
[Core]  ← app/core/pricing/
    │
    └── PricingCalculator → 実ベース利益計算
    │
    ▼
[Agents]  ← app/agents/
    │
    ├── Browser Plugins  → 5サイト別スクレイパー
    ├── Self-Healing     → エラー自動修復
    └── Price Intelligence → 価格分析
    │
    ▼
[Utils]  ← app/utils/
    │
    ├── ai_llm_controller.py → LLM APIルーティング
    ├── ai_image_crawler.py  → Selenium画像収集
    └── ai_background_remover.py → rembg背景除去
    │
    ▼
[Models (SQLAlchemy)]  ← app/models/*.py (11テーブル)
    │
    ▼
[SQLite / Alembic Migrations]
```

---

## 6. 外部サービス連携

| サービス | 用途 | 実装場所 | 状態 |
|---|---|---|---|
| LLM API (Gemini/DeepSeek/OpenAI) | 説明文生成・画像分析 | `app/utils/ai_llm_controller.py` | 実装済み |
| rembg | 背景除去 | `app/utils/ai_background_remover.py` | 実装済み |
| Selenium | 画像収集・ブラウザ自動化 | `app/utils/ai_image_crawler.py` | 実装済み |
| Bright Data | BUYMA価格スクレイピング | `app/services/price_scraper.py` | 実装済み |
| Buyandship | 転送倉庫Webhook | `app/routes/warehouse_webhook.py` | 基盤のみ |
| Slack | 発注・発送通知 | 未実装 | Issue #18 |

### ブラウザ自動化プラグイン

| サイト | プラグイン | 状態 |
|---|---|---|
| Moncler | `plugins/moncler_plp_v1.py` | 実装済み |
| Gucci | `plugins/gucci_plp_v1.py` | 実装済み |
| Prada | `plugins/prada_plp_v1.py` | 実装済み |
| Farfetch | `plugins/farfetch_plp_v1.py` | 実装済み |
| SSENSE | `plugins/ssense_plp_v1.py` | 実装済み |

---

## 7. 認証・認可

Flask-Login + User modelによる認証基盤を実装済み。

- `app/auth.py`: ログイン/ログアウトロジック
- `app/models/user.py`: UserMixin継承、パスワードハッシュ化
- `app/decorators.py`: login_requiredデコレータ

---

## 8. 設定管理

`app/config/` モジュールで一元管理:

- `config.py`: Config/Development/Testing/Productionクラス
- `sites/`: サイト別設定JSON (セレクタ、URL等)
- `integrations/`: 外部サービス連携設定

---

## 9. SaaS化への考慮

| 項目 | 現状 | Phase 2計画 |
|---|---|---|
| 認証 | Flask-Login実装済み | マルチテナント対応 |
| DB | SQLite (Alembic対応済み) | PostgreSQL移行 |
| テナント分離 | なし | tenant_idによる論理分離 |
| バックグラウンド処理 | 同期のみ | Celery + Redis (Issue #15) |
| コンテナ化 | なし | Docker (Issue #21) |
