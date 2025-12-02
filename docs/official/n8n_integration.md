# Atelier Kyo Manager × n8n 連携仕様書  

版: 2025-12-01



---



## 1. 目的



Atelier Kyo Manager のバックエンドと n8n を連携し、以下の運用自動化を行う。



- 在庫アラート → Slack 通知

- 価格見直しレポート → Slack / Google Sheets

- 買付パートナー SLA 違反検知 → リマインド通知

- 顧客フォローアップ候補の抽出



---



## 2. 連携方式の概要



- Atelier Kyo Manager → n8n

  - HTTP POST によるイベント通知（Webhook）

- n8n → Atelier Kyo Manager

  - `/api/hooks/...` 系エンドポイントへの HTTP POST



---



## 3. バックエンド → n8n（イベント通知）



### 3.1 共通 JSON フォーマット



```json

{

  "event_type": "stock_alert",

  "payload": {

    "product_id": 123,

    "status": "OUT_OF_STOCK"

  },

  "timestamp": "2025-12-01T10:00:00Z"

}

```



### 3.2 ヘッダ



`X-Internal-Auth: <secret-token>` を必ず付与



### 3.3 主な event_type



- **stock_alert**

  在庫変化（in stock → out of stock 等）



- **research_highlight**

  高利益率候補を検出したとき



- **partner_sla**

  買付パートナーの応答遅延



- **customer_followup**

  フォローアップ候補の顧客



---



## 4. n8n → バックエンド（hooks）



### 4.1 エンドポイント例



- `POST /api/hooks/pricing/report-request`

- `POST /api/hooks/partners/sla-scan`

- `POST /api/hooks/customers/followup-candidates`



### 4.2 認証



`X-API-Key: <INTERNAL_API_KEY>` を必須とする



キーは `.env` および n8n の Credentials で管理



---



## 5. 代表ワークフロー



### WF-1: 在庫アラート



1. Celery の在庫チェックタスクが在庫変化を検出

2. バックエンドが `stock_alert` イベントを n8n Webhook に送信

3. n8n が Slack メッセージを生成し、`#atelier-kyo-stock-alerts` 等に通知

4. 任意で Google Sheets にもログ追加



### WF-2: 価格見直しレポート



1. n8n の Cron が毎朝 `/api/hooks/pricing/report-request` を叩く

2. バックエンドが「利益率が閾値以下の listing」を集計して返却

3. n8n が Slack / Sheets にレポート送信



---



## 6. セキュリティ



- BUYMA アカウント情報は n8n には置かない

- Webhook の URL は外部に漏らさない

- Internal トークン / API キーは必ず環境変数で管理



---



## 7. 実装メモ（Python 側）



`config.py` に以下を定義する想定



- `N8N_WEBHOOK_STOCK_ALERT_URL`

- `N8N_WEBHOOK_INTERNAL_TOKEN`



`services/notifications.py` 等に n8n 通知用ヘルパを実装



例: `send_stock_alert_to_n8n(product_id, status, ...)`
