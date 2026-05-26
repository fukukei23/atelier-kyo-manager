# リファクタリング調査トラッカー

> 自動調査ループ用。各エリアの調査状況を記録。
> 既にRound 1-4で対応済みのBUYMAスクレイパー系は除外。

## 調査対象一覧

### Routes（14モジュール）
- [x] R01: products.py — F01商品CRUD + F02BUYMA拡張 + CSV + パイプライン
- [x] R02: orders.py — F05注文管理（18日ルール）+ F08キャッシュフロー
- [x] R03: partners.py — F06パートナー + F13リピーター管理
- [x] R04: analytics.py — F09ブランド分析 + ダッシュボード
- [x] R05: listing_templates.py — F03出品テンプレート管理
- [x] R06: faq_templates.py — FAQテンプレートCRUD
- [x] R07: prohibited_sources.py — F04禁制品買付先チェックAPI
- [x] R08: listing_progress.py — F07出品進捗トラッカー
- [x] R09: shipment_notifications.py — 発送通知管理
- [x] R10: stock_checks.py — F10在庫＆価格チェック
- [x] R11: popularity.py — F11人気度トラッキング
- [x] R12: region_recommendations.py — F12買付先地域レコメンド
- [x] R13: auto_orders.py — AI自動発注ステートマシン
- [x] R14: chatbot.py — AIチャットボット
- [x] R15: misc.py — 互換リダイレクト + リサーチ + API
- [x] R16: warehouse_webhook.py — Webhook受信
- [x] R17: brand_prices.py — Round 1-4で対応済み

### Services（11モジュール）
- [x] S01: auto_order_service.py — R13で対応済み
- [x] S02: chatbot_service.py — R14で対応済み
- [x] S03: image_service.py — 画像DL・背景除去
- [x] S04: notification_service.py — Slack通知
- [x] S05: pipeline_service.py — 出品パイプライン統合
- [x] S06: price_scraper.py — 価格・在庫スクレイピング
- [x] S07: product_csv_service.py — CSV入出力
- [x] S08: template_service.py — 出品テキスト生成
- [x] S09: warehouse_event_service.py — R16で対応済み
- [x] S10: brand_price_scraper.py — Round 1-4で対応済み
- [x] S11: buyma_price_scraper.py — Round 1-4で対応済み

### Models（18モジュール）
- [x] M01: 全モデル共通 — リレーション・インデックス・クエリパターン

### Utils（25+モジュール）
- [x] U01: ai_llm_controller.py — LLM制御・ルーティング
- [x] U02: ai_llm_controller_responses.py — レスポンス処理
- [x] U03: ai_image_crawler.py — Playwright画像収集
- [x] U04: ai_background_remover.py — 背景除去
- [x] U05: ai_generate_descriptions.py — AI説明文生成
- [x] U06: fx_utils.py — 為替レート取得
- [x] U07: sourcing_pipeline.py — 調達パイプライン
- [x] U08: sourcing_csv_adapter.py — CSV アダプタ
- [x] U09: sourcing_csv_batch_runner.py — バッチ実行
- [x] U10: browser_utils.py — ブラウザユーティリティ
- [x] U11: buyma_catalog_manager.py — BUYMAカタログ管理
- [x] U12: その他utils群 — scout系/observability/diagnostics等

### Agents（15モジュール）
- [x] A01: browser_orchestrator.py — ブラウザオーケストレーション
- [x] A02: price_intelligence_agent.py — 価格インテリジェンス
- [x] A03: profitability_agent.py — 収益性分析
- [x] A04: supplier_scout_agent.py — サプライヤー探索
- [x] A05: その他agents — ai_vision/selector_discovery等

### Config（7モジュール）
- [x] C01: config.py + cost_table.py — 設定・コストテーブル

### Templates
- [x] T01: 全テンプレート共通 — セキュリティ(XSS/CSRF)・アクセシビリティ・一貫性

### Tests
- [x] X01: テスト全体 — カバレッジ・テスト品質・モックの一貫性

### アーキテクチャ全体
- [x] ARCH: 全体 — 循環依存・レイヤー違反・関心の分離
