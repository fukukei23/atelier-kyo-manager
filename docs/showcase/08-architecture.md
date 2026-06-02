---
title: アーキテクチャ
parent: 機能ショーケース
nav_order: 8
---

# アーキテクチャ

## アプリケーション構成

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

## データフロー

```
HTTPリクエスト → Blueprint(ルーティング) → Service層(ビジネスロジック) → Model/外部API → Response
                                      │
                                      ├── SQLAlchemy ORM (DB操作)
                                      ├── LLM API (AI機能)
                                      ├── Playwright (スクレイピング)
                                      └── Slack API (通知)
```

## 主要ビジネスロジック

| サービス | 役割 |
|---|---|
| `auto_order_service.py` | 自動発注ステートマシン |
| `pipeline_service.py` | 出品パイプライン統合 |
| `chatbot_service.py` | AIチャットボット |
| `warehouse_event_service.py` | 倉庫Webhook受信 |
| `price_scraper.py` | 価格スクレイピング |
| `notification_service.py` | Slack通知 |
| `image_service.py` | 画像処理（rembg・OpenCV） |
| `template_service.py` | テンプレート生成 |

## テスト品質

- **2,070テストケース**（pytest）
- モジュール構成: routes 14 + services 8 + models 16 + utils 20+
- 実行コマンド: `make test`
- カバレッジ: `pytest --cov=app --cov-report=term-missing`

## モバイル対応

![モバイル表示]({{ site.baseurl }}/screenshots/07-mobile.png)

スマートフォンからも全機能にアクセス可能。外出先でも注文状況確認・価格チェックができます。

- **レスポンシブデザイン**: Bootstrap CSS Gridで画面サイズに自動調整
- **タッチ操作**: ボタンサイズ・間隔がモバイル最適化
- **軽量化**: 画像遅延読み込み・CSS/JS圧縮で高速表示
