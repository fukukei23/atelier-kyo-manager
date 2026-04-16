# Atelier Kyo Manager

DB スキーマ設計  

版: 2025-12-01



---



## 1. 概要



- RDB: PostgreSQL（開発時は SQLite でも可）

- ORM: SQLAlchemy（を想定）



買付・出品・在庫・注文・パートナー・AI・コンプライアンスを管理するため、以下のテーブルを想定します。



---



## 2. products（商品）



| カラム名        | 型           | 説明                    |

|----------------|--------------|-------------------------|

| id             | BIGSERIAL PK | 商品ID                  |

| brand          | TEXT         | ブランド                |

| name           | TEXT         | 商品名                  |

| buy_url        | TEXT         | 仕入URL                 |

| base_cost      | NUMERIC      | 仕入価格                |

| currency       | TEXT         | 通貨コード（JPY 等）    |

| source_site    | TEXT         | 仕入サイト              |

| stock_quantity | INTEGER      | 在庫数の目安            |

| category       | TEXT         | カテゴリ                |

| created_at     | TIMESTAMPTZ  | 作成日時                |

| updated_at     | TIMESTAMPTZ  | 更新日時                |



---



## 3. listings（出品）



| カラム名             | 型           | 説明                         |

|---------------------|--------------|------------------------------|

| id                  | BIGSERIAL PK | 出品ID                       |

| product_id          | BIGINT FK    | products.id                  |

| platform            | TEXT         | BUYMA 等                     |

| listing_price       | NUMERIC      | 販売価格                     |

| min_profit_rate     | NUMERIC      | 最低利益率（%）              |

| expected_profit_rate| NUMERIC      | 想定利益率（%）              |

| buyma_fee_rate      | NUMERIC      | 手数料率（例: 0.077）        |

| shipping_rate       | NUMERIC      | 送料率（例: 0.07）           |

| duty_estimate       | NUMERIC      | 関税見積り額                 |

| listing_csv_path    | TEXT         | 出品 CSV のファイルパス      |

| compliance_checked  | BOOLEAN      | 禁止リスト等のチェック済み   |

| created_at          | TIMESTAMPTZ  | 作成日時                     |

| updated_at          | TIMESTAMPTZ  | 更新日時                     |



---



## 4. stock_snapshots（在庫スナップショット）



在庫・価格の変化を時系列で追跡するための履歴テーブル。



| カラム名          | 型           | 説明                      |

|------------------|--------------|---------------------------|

| id               | BIGSERIAL PK |                           |

| product_id       | BIGINT FK    | products.id               |

| check_time       | TIMESTAMPTZ  | チェックした日時          |

| is_in_stock      | BOOLEAN      | 在庫があるかどうか        |

| price_at_check   | NUMERIC      | チェック時の価格          |

| raw_html_hash    | TEXT         | HTML のハッシュ値         |

| strategy_version | TEXT         | スクレイピング戦略バージョン |



---



## 5. customers（顧客） & orders（注文）



### customers



| カラム名      | 型           | 説明              |

|--------------|--------------|-------------------|

| id           | BIGSERIAL PK | 顧客ID            |

| buyer_handle | TEXT         | BUYMA ユーザー名  |

| email        | TEXT         | メールアドレス    |

| created_at   | TIMESTAMPTZ  | 作成日時          |



### orders



| カラム名           | 型           | 説明                              |

|-------------------|--------------|-----------------------------------|

| id                | BIGSERIAL PK | 注文ID                            |

| listing_id        | BIGINT FK    | listings.id                       |

| customer_id       | BIGINT FK    | customers.id                      |

| status            | TEXT         | ordered / paid / shipped / ...   |

| order_date        | TIMESTAMPTZ  | 受注日                            |

| planned_ship_date | TIMESTAMPTZ  | 予定出荷日（18日ルール管理）     |

| actual_ship_date  | TIMESTAMPTZ  | 実出荷日                          |

| tracking_number   | TEXT         | 追跡番号                          |

| total_price       | NUMERIC      | 売上金額                          |

| estimated_profit  | NUMERIC      | 利益見積り                        |

| cancellation_reason | TEXT       | キャンセル理由                    |



---



## 6. partners / partner_activity_logs（買付パートナー）



### partners



| カラム名           | 型           | 説明                          |

|-------------------|--------------|-------------------------------|

| id                | BIGSERIAL PK | パートナーID                 |

| name              | TEXT         | パートナー名                 |

| platform          | TEXT         | CloudWorks 等                |

| response_sla_hours| INTEGER      | 目標応答時間（時間）         |

| rating            | NUMERIC      | 内部評価（任意）             |

| status            | TEXT         | active / paused / archived   |

| created_at        | TIMESTAMPTZ  | 作成日時                     |



---



## 7. ai_logs / compliance_logs



### ai_logs（AI 実行ログ）



| カラム名       | 型           | 説明            |

|---------------|--------------|-----------------|

| id            | BIGSERIAL PK |                 |

| job_type      | TEXT         | 実行種別        |

| input_summary | TEXT         | 入力の要約      |

| output_summary| TEXT         | 出力の要約      |

| llm_model     | TEXT         | モデル名        |

| cost_estimate | NUMERIC      | コスト見積り    |

| created_at    | TIMESTAMPTZ  | 実行日時        |



### compliance_logs（コンプライアンスログ）



| カラム名   | 型           | 説明                                  |

|-----------|--------------|---------------------------------------|

| id        | BIGSERIAL PK |                                      |

| action_type | TEXT       | terms_accept / listing_check 等      |

| result    | TEXT         | pass / warning / block               |

| details   | TEXT         | 補足情報                             |

| created_at| TIMESTAMPTZ  | 記録日時                             |



---



## 8. インデックス（例）



- products: `(brand, status)`

- orders: `(status, planned_ship_date)`

- stock_snapshots: `(product_id, check_time)`

- compliance_logs: `(action_type, created_at)`
