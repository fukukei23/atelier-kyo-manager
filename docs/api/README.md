# atelier-kyo-manager API Reference

## 認証

| エンドポイント | メソッド | 説明 | 認証 |
|---|---|---|---|
| `/auth/login` | GET/POST | ログイン | 不要 |
| `/auth/logout` | GET | ログアウト | 不要 |

セッションベース認証（Flask-Login）。他のエンドポイントはログインが必要。

---

## 商品管理（products.py）

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/` | GET | ホームページ |
| `/manage` | GET/POST | 商品登録・更新・一覧 |
| `/products` | GET | 登録済み商品一覧 |
| `/products/<id>/edit` | GET/POST | 商品編集 |
| `/products/<id>/delete` | POST | 商品削除 |
| `/import_csv` | POST | CSV一括インポート |
| `/export_csv` | GET | CSVエクスポート |
| `/listing-candidates` | GET | 出品候補一覧 |
| `/listing-candidates/export` | GET | 出品候補CSV出力 |
| `/products/<id>/generate-listing` | GET | 出力テキスト生成プレビュー |
| `/generate-buyma-csv` | GET | BUYMA用CSV生成 |
| `/products/<id>/run-pipeline` | POST | パイプライン実行（画像→背景除去→説明→出品） |
| `/run-pipeline-batch` | POST | パイプライン一括実行 |
| `/products/<id>/pipeline-result` | GET | パイプライン実行結果 |
| `/products/<id>/upload-images` | POST | 手動画像アップロード |

---

## 注文管理（orders.py）

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/orders` | GET | 注文一覧（18日ルール対応） |
| `/orders/new` | GET/POST | 注文作成 |
| `/orders/<id>/edit` | GET/POST | 注文編集 |
| `/orders/<id>/delete` | POST | 注文削除 |
| `/orders/<id>/extend` | POST | 注文期限延長リクエスト |
| `/api/orders/dashboard` | GET | 18日ルールダッシュボードAPI |
| `/cashflow` | GET | キャッシュフロー予測ダッシュボード |

---

## パートナー管理（partners.py）

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/partners` | GET | パートナー一覧 |
| `/partners/new` | GET/POST | パートナー作成 |
| `/partners/<id>/edit` | GET/POST | パートナー編集 |
| `/partners/<id>/delete` | POST | パートナー削除 |
| `/customers` | GET | リピーター顧客一覧 |
| `/customers/new` | GET/POST | 顧客作成 |
| `/customers/<id>/edit` | GET/POST | 顧客編集 |
| `/customers/<id>/delete` | POST | 顧客削除 |

---

## 分析・ダッシュボード（analytics.py）

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/dashboard` | GET | KPI・パイプライン・在庫・出品状況ダッシュボード |
| `/brand-analytics` | GET | ブランドTier利益率ダッシュボード |
| `/templates` | GET | 出品テンプレート一覧 |
| `/templates/new` | GET/POST | テンプレート作成 |
| `/templates/<id>/edit` | GET/POST | テンプレート編集 |
| `/templates/<id>/delete` | POST | テンプレート削除 |
| `/faq-templates` | GET | FAQテンプレート一覧 |
| `/faq-templates/new` | GET/POST | FAQテンプレート作成 |
| `/faq-templates/<id>/delete` | POST | FAQテンプレート削除 |
| `/listing-progress` | GET | 出品進捗一覧 |
| `/listing-progress/new` | GET/POST | 進捗レコード作成 |
| `/listing-progress/<id>/delete` | POST | 進捗レコード削除 |
| `/stock-check` | GET | 在庫・価格チェック一覧 |
| `/stock-check/new` | GET/POST | 在庫チェック作成 |
| `/stock-check/<id>/delete` | POST | 在庫チェック削除 |
| `/shipment-notifications` | GET | 出荷通知一覧 |
| `/shipment-notifications/new` | GET/POST | 出荷通知作成 |
| `/shipment-notifications/<id>/notify` | POST | 出荷通知済みに更新 |
| `/shipment-notifications/<id>/delete` | POST | 出荷通知削除 |
| `/popularity` | GET | 人気追踪一覧 |
| `/popularity/new` | GET/POST | 人気レコード作成 |
| `/popularity/<id>/delete` | POST | 人気レコード削除 |
| `/regions` | GET | 地域レコメンド一覧 |
| `/regions/new` | GET/POST | 地域レコメンド作成 |
| `/regions/<id>/delete` | POST | 地域レコメンド削除 |
| `/auto-orders` | GET | 自動発注ステータス一覧 |
| `/auto-orders/<id>/start` | POST | 自動発注開始 |
| `/auto-orders/<id>/step` | POST | 自動発注ステップ実行 |
| `/auto-orders/<id>/error` | POST | 自動発注エラー報告 |
| `/chatbot` | GET | チャットボットダッシュボード |
| `/chatbot/<id>` | GET | 問い合わせ詳細 |
| `/chatbot/create` | GET/POST | 問い合わせ作成 |
| `/chatbot/<id>/approve` | POST | チャットボット回答承認 |
| `/chatbot/<id>/reject` | POST | チャットボット回答拒否 |

---

## API エンドポイント

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/api/faq-match` | POST | 問い合わせテキスト→FAQ照合 |
| `/api/check-source` | GET | URL禁止ソースチェック |
| `/api/prohibited-sources` | GET | 禁止ソース一覧 |
| `/api/prohibited-sources` | POST | 禁止ソース追加 |
| `/api/prohibited-sources/<id>` | DELETE | 禁止ソース削除 |
| `/api/stock-check/<id>/fetch` | POST | 在庫価格自動取得 |
| `/api/stock-check/quick-update` | POST | インライン価格更新 |
| `/api/stock-check/quick-add` | POST | クイック在庫チェック追加 |
| `/api/stock-check/fetch-all` | POST | 全在庫価格一括取得 |
| `/api/products-list` | GET | ドロップダウン用商品一覧 |
| `/api/warehouses` | GET | 国別倉庫一覧（Buyandship） |
| `/api/warehouse/events` | POST | 倉庫Webhookイベント受信 |

---

## その他（misc.py）

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/auto-research` | GET/POST | 自動リサーチページ |
| `/image-crawler` | GET | 個別リサーチページ |
| `/dashboard` | GET | `/cashflow` へリダイレクト |
| `/listing-templates` | GET | `/templates` へリダイレクト |
| `/region-recommendations` | GET | `/regions` へリダイレクト |
| `/repeat-customers` | GET | `/customers` へリダイレクト |

---

## 共通レスポンス形式

- HTML: Jinja2テンプレートレンダリング
- JSON API: `{"success": true/false, "data": ..., "error": ...}`
- リダイレクト: `redirect(url_for(...))`
- CSRF: 全POSTリクエストにCSRFトークン必須（Flask-WTF）
