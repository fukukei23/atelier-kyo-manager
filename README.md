# Atelier Kyo Manager



BUYMA 無在庫転売向けの半自動支援ツールです。  

手動出品（規約遵守）を前提に、リサーチ・在庫監視・AI説明文生成・利益計算を一元管理します。



---



## 1. 主な機能



### 商品管理

- CSV / Google Sheets インポート

- 仕入URL・価格・利益率の管理



### 利益計算

- 仕入値 + 送料 7% + BUYMA 手数料 7.7% + 関税

- 利益率 10%以上を維持するための自動計算



### 在庫チェック

- Celery 定期タスクで仕入先URLを巡回

- 在庫切れ / 価格上昇を検知

- Slack / ダッシュボード通知（n8n 連携）



### AI コンテンツ生成

- OpenAI / Claude による説明文・タイトル生成

- ブランド世界観・希少性・迅速発送を盛り込む



### 買付パートナー管理

- 外注・買付パートナーの管理

- SLA 遅延アラート



---



## 2. アーキテクチャ概要



- Backend: Flask or FastAPI

- Task Queue: Celery + Redis

- Scraping: Playwright / DrissionPage

- Database: PostgreSQL（本番） / SQLite（開発）

- Dashboard: Streamlit

- Optional: n8n（Slack / Sheets / Gmail）



詳細は `docs/official/system_design.md` を参照してください。



---



## 3. ディレクトリ構成（抜粋）



```text

atelier-kyo-manager/

├─ app/                # Flask/FastAPI & Celery

├─ docs/

│   ├─ official/       # 公式設計ドキュメント

│   │   ├─ system_design.md

│   │   ├─ db_schema.md

│   │   ├─ n8n_integration.md

│   │   ├─ saas_design.md

│   │   └─ architecture_overview.mmd

│   ├─ chat_records/

│   ├─ codex_notes/

│   ├─ completion_reports/

│   ├─ reports/

│   ├─ test_results/

│   └─ ...

└─ ...

```



## 4. セットアップ



```bash

python -m venv .venv

source .venv/bin/activate    # Windows の場合: .venv\Scripts\Activate.ps1

pip install -r requirements.txt

```



`.env` 例：



```ini

OPENAI_API_KEY=sk-...

DATABASE_URL=postgresql://user:pass@localhost:5432/atelier_kyo

REDIS_URL=redis://localhost:6379/0

```



## 5. 実行



### API（例）

```bash

flask run

# または

uvicorn app.main:app --reload

```



### Celery

```bash

celery -A app.celery_app worker -l info

celery -A app.celery_app beat -l info

```



### Streamlit

```bash

streamlit run app/streamlit_app.py

```



## 6. ドキュメント



- システム設計: `docs/official/system_design.md`

- DB 設計: `docs/official/db_schema.md`

- n8n 連携: `docs/official/n8n_integration.md`

- SaaS 設計: `docs/official/saas_design.md`



## 7. 注意事項（規約）



- BUYMA への自動ログイン・自動出品は禁止

- 国内ECサイトを仕入先にしない

- キャンセル率・発送期限（18日）を守ることを前提とする
