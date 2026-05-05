# atelier-kyo-manager

A personal resale management system for BUYMA x Buyandship. Automates product listing pipelines, order state machines, and customer support with an AI chatbot and multi-provider LLM routing.

BUYMA x Buyandshipを利用した転売管理システム（個人用）。出品パイプライン、注文ステートマシン、顧客対応AIチャットボット、LLMルーティングを統合したFlaskアプリケーション。

---

## アーキテクチャ

Flask App Factoryパターン（`create_app`）によるアプリケーション初期化。

```
Flask App Factory (create_app)
  ├── Blueprint (6モジュール)    ← HTTPルーティング
  │     analytics / orders / partners / products / misc / warehouse_webhook
  ├── Service層 (8モジュール)    ← ビジネスロジック
  │     auto_order / chatbot / image / notification / pipeline / price_scraper / template / warehouse_event
  ├── Models (16種)              ← SQLAlchemyデータモデル
  ├── Utils (20+モジュール)      ← ai_llm_controller / fx_utils / pricing_calculator 等
  └── Templates / Static         ← Jinja2 + CSS/JS
```

**データフロー**: HTTPリクエスト → Blueprint → Service層 → Model/外部API → Response

---

## 主要ビジネスロジック

| サービス | 役割 |
|---|---|
| `auto_order_service.py` | 自動発注ステートマシン。`pending → sourcing → cart_added → checkout → payment_done → shipped → completed` の状態遷移を管理 |
| `pipeline_service.py` | 出品パイプライン統合。画像収集 → 背景除去（rembg） → AI説明文生成 → 出品テキスト生成を一括実行 |
| `chatbot_service.py` | 顧客対応AIチャットボット。FAQテンプレートマッチ → AI回答生成 → エスカレーション判定の3段階分類 |
| `warehouse_event_service.py` | 倉庫Webhook受信。転送宅配便の荷物受領イベント・写真処理 |
| `order.py` | 18日ルールダッシュボード。決済方法別延長期限マッピング（クレカ45日/銀行振込90日等）、BUYMA手数料計算 |
| `price_scraper.py` | 仕入先価格スクレイピング（Playwrightヘッドレスブラウザ） |
| `notification_service.py` | Slack通知（注文・出荷・エラー等のイベント通知） |
| `image_service.py` | 画像処理（背景除去rembg、リサイズ、OpenCV加工） |

### LLMルーティング

`ai_llm_controller.py` が複数プロバイダーを統一的に管理:

- **対応プロバイダー**: OpenAI / Gemini / Local Llama (llama.cpp) / transformers
- **ディスクキャッシュ**: 500MB（diskcache）— 同一プロンプトの再呼び出しを高速化
- **OpenTelemetry計装**付き — トレーシングでAPI呼び出しを監視
- プロバイダー間のフォールバックとレートリミットをサポート

---

## 技術スタック

| カテゴリ | 技術 |
|---|---|
| 言語 | Python 3.x |
| フレームワーク | Flask + SQLAlchemy + Flask-Login + Flask-WTF |
| データベース | SQLite (Flask-Migrate) |
| スクレイピング | Playwright / Selenium |
| 画像処理 | Pillow / OpenCV / rembg |
| LLM | OpenAI API / Gemini API / Local LLM |
| テスト | pytest (588 passed, カバレッジ29%) |

---

## セットアップ（WSL2前提）

```bash
git clone https://github.com/fukukei23/atelier-kyo-manager.git
cd atelier-kyo-manager
make venv              # venv作成
make install           # 依存インストール
cp .env.template .env  # .envを編集
flask db upgrade       # DBマイグレーション
flask run              # 開発サーバー起動
```

`Makefile.local` でプロジェクト固有のターゲットを拡張可能。

---

## 主要コマンド

| コマンド | 説明 |
|---|---|
| `make venv` | venv作成 |
| `make install` | 依存インストール |
| `make install-dev` | 開発用依存も含めてインストール |
| `make test` | pytest実行 |
| `make lint` | ruff / flake8実行 |
| `make format` | black実行（導入時） |
| `make clean-pyc` | `__pycache__`削除 |
| `make info` | Python / pip / プロジェクト情報表示 |
| `flask run` | 開発サーバー起動 |

---

## ディレクトリ構造

```
app/
  __init__.py              # Flask app factory (create_app)
  auth.py                  # 認証Blueprint
  config/                  # 設定モジュール
  models/                  # SQLAlchemyモデル (16種)
    product.py             #   商品（価格計算・BUYMA連携）
    order.py               #   注文（18日ルール・利益計算）
    user.py                #   ユーザー認証
    partner.py             #   パートナー管理
    customer_inquiry.py    #   顧客問い合わせ（AIチャットボット）
    faq_template.py        #   FAQテンプレート
    listing_progress.py    #   出品進捗
    listing_template.py    #   出品テンプレート
    popularity_tracker.py  #   人気追踪
    prohibited_source.py   #   禁止ソース
    region_recommendation.py # 地域レコメンド
    repeat_customer.py     #   リピーター顧客
    shipment_notification.py # 出荷通知
    stock_check.py         #   在庫・価格チェック
  routes/                  # Blueprint ルート (6モジュール)
    analytics.py           #   分析ダッシュボード・API
    orders.py              #   注文管理・キャッシュフロー
    partners.py            #   パートナー・リピーター顧客
    products.py            #   商品CRUD・パイプライン・CSV
    misc.py                #   リダイレクト・倉庫API
    warehouse_webhook.py   #   倉庫Webhook受信
  services/                # ビジネスロジック (8モジュール)
    auto_order_service.py  #   自動発注ステートマシン
    chatbot_service.py     #   AIチャットボット
    image_service.py       #   画像処理
    notification_service.py #  Slack通知
    pipeline_service.py    #   出品パイプライン
    price_scraper.py       #   価格スクレイピング
    template_service.py    #   テンプレート生成
    warehouse_event_service.py # 倉庫イベント処理
  utils/                   # ユーティリティ (20+モジュール)
    ai_llm_controller.py   #   LLMルーティング
    fx_utils.py            #   為替計算
    pricing_calculator.py  #   価格計算
  templates/               # Jinja2テンプレート
  static/                  # 静的ファイル
tests/                     # pytestテスト
docs/                      # ドキュメント
config/                    # 設定ファイル
migrations/                # DBマイグレーション
```

---

## 環境変数（主要）

| 変数 | 説明 |
|---|---|
| `FLASK_APP` | Flaskアプリエントリポイント |
| `BUYANDSHIP_EMAIL` | Buyandshipログイン用 |
| `OPENAI_API_KEY` | OpenAI APIキー |
| `GEMINI_API_KEY` | Gemini APIキー |
| `LLM_ORDER_DEFAULT` | デフォルトLLMプロバイダー |
| `CRAWLER_HEADLESS` | ヘッドレスブラウザ設定 |

---

## ライセンス

Private Repository
