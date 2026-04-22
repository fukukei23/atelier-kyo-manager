# atelier-kyo-manager

BUYMA×Buyandshipを利用した転売管理システム（個人用）

## 技術スタック

- **バックエンド**: Python 3.x / Flask + SQLAlchemy + Flask-Login + Flask-WTF
- **データベース**: SQLite (Flask-Migrate)
- **スクレイピング**: Playwright / Selenium
- **画像処理**: Pillow / OpenCV / rembg
- **テスト**: pytest (588 passed, カバレッジ29%)

## セットアップ（WSL2前提）

```bash
git clone https://github.com/fukukei23/atelier-kyo-manager.git
cd atelier-kyo-manager
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.template .env   # .envを編集
flask db upgrade
flask run
```

## 主要コマンド

| コマンド | 説明 |
|---|---|
| `make venv` | venv作成 |
| `make install` | 依存インストール |
| `make test` | pytest実行 |
| `make lint` | ruff/flake8実行 |
| `make clean-pyc` | `__pycache__`削除 |
| `flask run` | 開発サーバー起動 |

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

## 環境変数（主要）

| 変数 | 説明 |
|---|---|
| `FLASK_APP` | Flaskアプリエントリポイント |
| `BUYANDSHIP_EMAIL` | Buyandshipログイン用 |
| `OPENAI_API_KEY` | OpenAI APIキー |
| `GEMINI_API_KEY` | Gemini APIキー |
| `LLM_ORDER_DEFAULT` | デフォルトLLMプロバイダー |
| `CRAWLER_HEADLESS` | ヘッドレスブラウザ設定 |

## ライセンス

Private Repository
