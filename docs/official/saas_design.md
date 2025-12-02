# Atelier Kyo Manager - SaaS Expansion Design  

版: 2025-12-01



---



## 1. 目的



Atelier Kyo Manager を単一ユーザー向けツールから、  

複数ユーザー・複数テナントで利用できる SaaS へ展開するための設計方針をまとめる。



---



## 2. マルチテナント戦略



- 初期段階:

  - 単一 DB に `tenant_id` カラムを追加する方式

  - 各テーブル（products, listings, orders 等）に `tenant_id` を持たせる

- 将来:

  - 大口テナント向けに DB 分割（DB per tenant）も検討可能



---



## 3. 認証・認可



- 認証:  

  - JWT ベースのセッション

  - メール + パスワード or 外部 IdP 連携（将来）

- 認可:

  - ロール: `owner` / `operator`

  - テナント境界でのデータ分離（`tenant_id` でスコープ制御）



---



## 4. 課金（Billing）



- Stripe Billing を想定

  - Free / Pro / Enterprise などのプラン

  - ワークフロー:

    1. バックエンドで Stripe Checkout Session 作成

    2. Stripe Webhook → n8n → バックエンドのテナント状態更新

- 制限:

  - 商品数・在庫チェック頻度・AI 実行回数などをプランに応じて制限



---



## 5. モジュール分離



- Core: 商品・在庫・利益計算

- AI: 説明文生成・価格提案・リサーチ要約

- Partner: 買付パートナー管理・SLA ログ

- Orchestrator: n8n 連携（通知・レポート）

- Admin: テナント・ユーザー管理、課金状態の管理



---



## 6. 監査ログとセキュリティ



- `compliance_logs` に SaaS 特有の操作も追加

  - テナント作成・削除

  - ロール変更

  - プラン変更

- API キー発行:

  - テナント毎に API キーを複数発行可能

  - ローテーションを前提とした設計

