---
name: 包括的リポジトリレビュー — 2026-08-12
date: 2026-08-12
reviewer: Claude (claude/atelier-kyo-manager-review-xyjbb2)
type: issue-backlog
status: open
---

# atelier-kyo-manager — 包括的リポジトリレビュー

> 実測環境: Python 3.11 / venv 新規構築 / `pytest tests/ --cov=app`
> 結果: **2,149 passed, 6 skipped, coverage 46%**（ruff check / format ともに green）
>
> にもかかわらず、**利益計算の主要な本番経路6つが例外で全滅している**。
> 以下は実際に実行して再現を確認した指摘のみを記載する（推測ゼロ）。

## 実測サマリ

| 指標 | 実測値 |
|---|---|
| app 本体 | 37,854 行 / 330 ファイル |
| tests | 24,277 行 / 107 ファイル |
| テスト結果 | 2,149 passed, 6 skipped, 0 failed |
| カバレッジ | 46%（18,392 stmts / 9,990 miss） |
| ruff check / format | ✅ All checks passed |
| コミット済み秘密情報 | ✅ なし |
| `eval` / `exec` / raw SQL / `shell=True` | ✅ なし |

### サブシステム別 LOC

| ディレクトリ | LOC | 主要カバレッジ帯 |
|---|---:|---|
| `app/agents/` | **19,817**（app の 52%） | 10〜30% |
| `app/utils/` | 6,362 | 0〜100%（二極化） |
| `app/services/` | 4,214 | 16〜100% |
| `app/routes/` | 2,839 | 31〜100% |
| `app/models/` | 1,250 | 93〜100% |
| `app/core/` | 460 | **100%** |
| `app/web/` | 538 | **0%（完全な死にコード）** |

---

## 優先度: 🔴 P0（機能停止級）

### ISSUE-101: `PriceIntegrityError` が正常系を全滅させている

- **症状**: `calculate_pricing()` は冒頭で `inp.validate_sources()` を呼び、`PriceSource.UNKNOWN` を検出すると例外を投げる。
  ところが呼び出し元の多くが `PricingInput` を **price source 未設定（デフォルト = UNKNOWN）** で構築しているため、
  **正常な入力を渡しても必ず `PriceIntegrityError` になる**。

- **再現（実行済み）**:

  ```
  # Order.calc_profit()
  >>> o = Order(order_number="B-1", selling_price=50000, purchase_cost=30000,
  ...           source_type="overseas", order_date=now)
  >>> o.calc_profit()
  PriceIntegrityError: 仕入価格のデータソースが信頼できません: unknown。
                       販売価格のデータソースが信頼できません: unknown。

  # sourcing パイプライン（完全な入力JSON）
  >>> run_pipeline("in.json", "p.json", "t.json")
  PriceIntegrityError: （同上）
  ```

- **影響を受ける本番経路（6件）**:

  | 呼び出し元 | 到達経路 | 影響 |
  |---|---|---|
  | `app/models/order.py:83` `Order.calc_profit()` | `POST /orders/new`, `POST /orders/<id>/edit` | **F05 注文の新規登録・編集が常に失敗** |
  | `app/utils/sourcing_profitability.py:129` | `sourcing_pipeline.run_pipeline()` | sourcing CSV パイプライン全滅 |
  | `app/services/price_comparison_service.py:310` | 価格比較 | 例外 |
  | `app/services/brand_price_service.py:230` | ブランド価格調査 | 例外 |
  | `app/services/price_monitor_service.py:371` | `_update_profitability()` | 例外 |
  | `app/agents/profitability_agent.py:116` | 収益性エージェント | 例外 |

  正しく `PriceSource` を設定しているのは `app/services/ssense_buyma_pipeline.py:249-250` のみ。
  `app/models/product.py:174-175` は DB カラムから読むため、カラムが埋まっていれば動作する。

- **さらに悪い点**: `routes/orders.py` の `@handle_db_error()` が例外を握りつぶし、
  `flash("操作に失敗しました: …")` + redirect で終わる。
  つまり**エラーとして表面化せず、注文が保存されないだけ**になる。

- **推奨対応**:
  1. `Order` に `purchase_price_source` / `selling_price_source` カラムを追加（`Product` と同様）。
     移行までの暫定は `calculate_pricing(inp, skip_source_validation=True)` を明示的に指定。
  2. `sourcing_input_schema` に price source を必須項目として追加し、
     `calculate_profitability()` で `PricingInput` に引き渡す。
  3. `price_comparison_service` / `brand_price_service` / `price_monitor_service` は
     スクレイパ由来の値なので `PriceSource.BROWSER_VERIFIED` を明示。
  4. **恒久対策**: `PricingInput.purchase_price_source` のデフォルト値 `UNKNOWN` を廃し、
     必須引数にする。そうすればこの種の欠落がコンパイル時（型チェック時）に露見する。

- **想定工数**: 1〜2日
- **完了条件**: 上記6経路すべてについて、**モックなしの** 正常系テストが green

---

### ISSUE-102: テストが ISSUE-101 をモックで隠蔽している

- **症状**: 2,149 件 green は、バグを回避するようテストが書き換えられた結果を含む。

  `tests/test_sourcing_profitability_coverage.py:164-166`:
  ```python
  # calculate_profitability 内部で PricingInput(price_source=UNKNOWN) を
  # 構築して PriceIntegrityError を raise するため、テストではモック化。
  with patch("app.utils.sourcing_profitability.calculate_pricing", return_value=fake_result):
  ```
  → `app/utils/sourcing_profitability.py` は **coverage 100%** だが、本番経路は一度も動いていない。

  `tests/order/test_order_model.py:150-171` / `tests/test_models_extra.py:62-80`:
  ```python
  def test_calc_profit_basic():
      """... Order.calc_profit の動作検証と分離）。"""
      # Order.calc_profit を呼ばず、calculate_pricing を直接叩いている
  ```
  → 関数名は `test_calc_profit_*` だが `Order.calc_profit` を一度も呼んでいない。

- **評価**: 2026-05-15 レビューの ISSUE-003（over-mocking の懸念）が、**具体的な実害として顕在化**している。
  カバレッジ 46% という数値以上に、「green であること」の信頼性が損なわれている。

- **推奨対応**:
  - ISSUE-101 の修正後、上記テストからモックを外して実経路を検証する形に戻す
  - 「本番経路が例外を投げるのでモックする」という判断が出た時点で、テストではなく**実装の bug チケット**を切る運用に変える
  - `--cov` に加えて、主要ユースケースの smoke test（`POST /orders/new` が 302 + DB 1件）を追加

- **想定工数**: 半日（ISSUE-101 と同時）

---

## 優先度: 🟠 P1（数字が合わない / セキュリティ）

### ISSUE-103: 実効手数料 14.2% がコア計算に一切適用されていない

- **症状**: `docs/経営者判断.md §3` の `EFFECTIVE_FEE_RATE = 0.142`（実効総手数料 14.2%）が、
  `calculate_pricing()` から**参照されていない**。実際に課金されるのは成約手数料 7.7%（国内）/ 5.5%（海外）のみ。

- **誤った記述が2箇所ある**:
  - `app/core/pricing/rules.py:57-58`: `buyma_effective_fee_rate` … 「→ calculator.py で使用」← **使用していない**
  - `app/agents/profitability_agent.py:101`: 「実取引の計算（14.2%実効レート）は `Product._calculate_pricing()` 経由で行う」
    ← `Product._calculate_pricing()` も `calculate_pricing()` を呼ぶだけなので 7.7%/5.5%

- **14.2% を使っているのは `app/utils/sourcing_profitability.py:79`（partial 経路）のみ**。
  結果として同一商品でも入力の欠落有無で原価が変わり、
  **`profit_upper_bound`（上限のはず）が complete 経路の `profit` を下回る**という逆転が起きる。

  ```
  売価 ¥50,000 のとき
    complete 経路の手数料 = 50,000 × 0.077 = ¥3,850
    partial  経路の手数料 = 50,000 × 0.142 = ¥7,100   ← 差 ¥3,250
  ```

- **経営インパクト**: `docs/経営者判断.md §9` は「利益率目標 20%、現状 11〜14%」を未解決として挙げているが、
  もし実効 14.2% が正なら **現在の利益計算は手数料を 6.5〜8.7pt 過小計上しており、利益が過大表示されている**。
  §9 の「決済手数料 3.6% or 5.5% 未確認」「実効総手数料 14.2% との関係を整理が必要」は
  **コードの実挙動を確定させないと利益率の議論自体が成立しない**。

- **推奨対応**:
  1. BUYMA 公式で決済手数料を確定させる（§9 の宿題）
  2. 確定値に基づき `calculate_pricing()` に一本化。`buyma_effective_fee_rate` を使うのか
     `commission + payment` の分解にするのかを決め、**使わない方の定数を削除**する
  3. `rules.py` / `profitability_agent.py` の誤った docstring を修正
  4. `sourcing_profitability` の partial 経路を complete 経路と同じ料率に揃える

- **想定工数**: 半日（公式確認の待ち時間を除く）

---

### ISSUE-104: ログインに Open Redirect（CWE-601）

- **対象**: `app/auth.py:42-43`
  ```python
  next_page = request.args.get("next")
  return redirect(next_page or url_for("main.index"))
  ```
- **症状**: `next` の検証がないため `/auth/login?next=https://evil.example/` でログイン後に外部へ誘導できる。
  フィッシングの踏み台になる。
- **推奨対応**: 相対 URL かつ同一ホストのみ許可する。
  ```python
  from urllib.parse import urlparse, urljoin

  def _is_safe_url(target: str) -> bool:
      ref = urlparse(request.host_url)
      test = urlparse(urljoin(request.host_url, target))
      return test.scheme in ("http", "https") and ref.netloc == test.netloc

  next_page = request.args.get("next")
  return redirect(next_page if next_page and _is_safe_url(next_page) else url_for("main.index"))
  ```
- **想定工数**: 30分（テスト込み）

---

### ISSUE-105: セッション Cookie / セキュリティヘッダが未設定

- **症状**: `SESSION_COOKIE_SECURE` / `SESSION_COOKIE_HTTPONLY` / `SESSION_COOKIE_SAMESITE` /
  `REMEMBER_COOKIE_*` のいずれも設定されていない（grep で 0 件）。
  `app/auth.py:39` は `login_user(user, remember=True)` を無条件で指定しているため、
  永続 Cookie が HTTP 平文で流れうる。セキュリティヘッダ（CSP / X-Frame-Options）も未設定。

- **併せて**: `app/config/config.py:75`
  ```python
  SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
  ```
  デフォルト値のガードは `STAGE == "prod"` のときのみ。`AK_STAGE` の既定値は `"test"` なので、
  **`AK_STAGE` を設定し忘れた本番デプロイは既知の SECRET_KEY で起動する**（= セッション偽造可能）。

- **推奨対応**:
  - `AppConfig.get_flask_config()` に Cookie フラグを追加（`SECURE` は `STAGE != "test"` で True）
  - `SECRET_KEY` のガードを「デフォルト値なら常に警告、`STAGE != test` なら raise」に変更
  - `@app.after_request` で最低限のセキュリティヘッダを付与
- **想定工数**: 2〜3時間

---

### ISSUE-106: Webhook が fail-open、かつ Blueprint が未登録

- **対象**: `app/routes/warehouse_webhook.py`

- **症状 (a) — 未登録**: `warehouse_webhook.router` は `create_app()` でも `app/routes/__init__.py` でも
  `register_blueprint` されていない。`CLAUDE.md` は `/api/warehouse/events` を提供機能として記載しているが、
  **実際には到達不能**。テスト側もこれを認識しており手動登録している:
  ```python
  # tests/test_warehouse_webhook.py:29
  # warehouse_webhook Blueprint を手動登録（create_app に未登録のため）
  ```

- **症状 (b) — fail-open**: `warehouse_webhook.py:27`
  ```python
  if secret and not _verify_signature(body, signature, secret):
      abort(401)
  ```
  `secret` が空（環境変数の設定漏れ、`forward2me.json` 欠落）だと**署名検証を丸ごとスキップ**し、
  誰でも倉庫イベントを注入できる。設定ミスが「認証なしで公開」に化ける典型パターン。

- **症状 (c)**: 登録した場合、Flask-WTF の `CSRFProtect` が全 POST を保護するため、
  外部からの Webhook は CSRF エラーで弾かれる。`@csrf.exempt` が必要。

- **推奨対応**:
  1. `create_app()` に `app.register_blueprint(warehouse_webhook.router)` を追加
  2. `@csrf.exempt` を付与
  3. fail-open を fail-closed に変更（`secret` 未設定なら 503 を返してログに警告）
  4. 機能として不要なら**ファイルごと削除**し、`CLAUDE.md` の記載も消す

- **想定工数**: 2時間

---

## 優先度: 🟡 P2（構造・保守性）

### ISSUE-107: マイグレーションにベースラインが無い

- **症状**: `migrations/versions/` は5件のみで、`down_revision = None` の起点は
  `fa15d94c15de_add_fr005_pricing_fields.py` ——「テーブル追加」ではなく「カラム追加」。
  空DBに `flask db upgrade` しても**テーブルが作られない**。
  実際にスキーマを作っているのは `app/__init__.py:57` の `db.create_all()`。

- **リスク**: モデル 20 個に対しマイグレーション 5 件。`create_all()` は既存テーブルを変更しないため、
  **開発機（create_all で最新）と本番（migration で古い）でスキーマが恒久的に乖離**する。

- **推奨対応**:
  1. `flask db revision --autogenerate` でベースライン（全テーブル作成）を生成し、既存の起点をその後ろに繋ぐ
  2. `create_app()` から `db.create_all()` を削除。テスト用は conftest 側で明示的に呼ぶ
  3. CI に「`flask db upgrade` した空DB と、モデル定義の autogenerate 差分が空であること」のチェックを追加

- **想定工数**: 半日

---

### ISSUE-108: `app/web/` が完全な死にコード（二重 app factory）

- **症状**: `app/web/__init__.py`（538行のうち67 stmts）に**2つ目の `create_app()`** が存在する。
  ヘッダのコメントは `app/__init__.py`、日付は 2025-08-23。
  `app/web/dashboard.py` (215 stmts) ともども **coverage 0%、参照ゼロ**（grep で確認）。
  この factory は `login_manager` を初期化しないため、仮に使われても全 `@login_required` が壊れる。

- **推奨対応**: `app/web/` を削除。ダッシュボード機能が必要なら `app/routes/analytics.py` 側に統合。
- **想定工数**: 1時間

---

### ISSUE-109: 利益閾値 ¥10,000 が3箇所にハードコード

- **対象**:
  - `app/services/price_comparison_service.py:111` — `p >= max(10_000, c * 0.05)`
  - `app/services/price_comparison_service.py:318` — `result.profit > max(10_000, ...)`
  - `app/services/brand_price_service.py:239` — `result.profit > max(10_000, ...)`

- **症状**: `docs/経営者判断.md §2`「最低絶対利益額 ¥10,000/商品」という**経営判断**が
  `app/config/constants.py`（SSOT を謳うファイル）ではなくサービス層に散っている。
  さらに §2 は「一律パーセンテージではなく価格帯に応じた段階設定にする」と決めているが、
  実装は `max(10_000, cost*0.05)` の一律ルールのまま。

- **推奨対応**: `constants.py` に `MIN_ABSOLUTE_PROFIT_JPY = 10_000` と判定関数 `is_profitable()` を置き、3箇所を置換。
  段階設定への移行はその関数1箇所の変更で済む形にする。
- **想定工数**: 2時間

---

### ISSUE-110: `Order.calc_profit()` がコスト項目を3つしか見ていない

- **対象**: `app/models/order.py:83-93`
  ```python
  inp = PricingInput(
      purchase_price=nz(self.purchase_cost),
      selling_price=nz(self.selling_price),
      customs_duty=nz(self.customs_duty),
  )
  ```
- **症状**: `shipping_cost` / `warehouse_shipping_cost` / `procurement_fee` / `transaction_fee` /
  `original_currency` / `exchange_rate` / `item_category` をすべて無視。
  `Product._calculate_pricing()`（12項目すべて渡す）と**同一商品で結果が食い違う**。
  さらに `item_category` が空なので関税は常にデフォルト 10%（バッグ 11% / 革 12% が効かない）。
- **推奨対応**: `Order` に不足カラムを追加するか、`product_id` 経由で `Product` の値を引き継ぐ。
  少なくとも `item_category` は渡す。
- **想定工数**: 半日（ISSUE-101 と同時が効率的）

---

### ISSUE-111: 為替レート 0 のサイレントフォールバック

- **対象**: `app/core/pricing/calculator.py:28-31`
  ```python
  if inp.original_currency != "JPY" and inp.exchange_rate > 0:
      purchase_price_jpy = inp.purchase_price * inp.exchange_rate
  else:
      purchase_price_jpy = inp.purchase_price   # ← EUR 500 が ¥500 になる
  ```
- **症状**: `exchange_rate` が 0 / 未設定のとき、外貨建て価格を**そのまま円として扱う**。
  €500 が ¥500 になり、利益が桁違いに過大表示される。例外も警告も出ない。
- **評価**: `CLAUDE.md`「価格調査の鉄則（推測・架空データ禁止）」の精神に真っ向から反する。
  `PriceSource` で入口を厳格に守っているのに、換算で静かに壊れる。
- **推奨対応**: `original_currency != "JPY"` かつ `exchange_rate <= 0` は `PriceIntegrityError` を投げる。
- **想定工数**: 1時間

---

### ISSUE-112: 例外の握りつぶしが多い

- **実測**: `except Exception` 623箇所 / `except …: pass` 77箇所（app 配下）。
- **具体例**: `app/utils/decorators.py:28-31` の `handle_db_error` は
  `except Exception` で**すべて**を捕まえ、`flash(f"操作に失敗しました: {e}")` で
  例外文字列をそのままユーザーに表示する。ISSUE-101 の `PriceIntegrityError` が
  UI 上「操作に失敗しました」の一言に化けていたのはこれが原因。
  SQLAlchemy の例外なら SQL 断片が画面に出る可能性もある（軽微な情報漏洩）。
- **推奨対応**:
  - `handle_db_error` は `SQLAlchemyError` に限定し、それ以外は再送出して 500 + ログに落とす
  - ユーザー向けメッセージは定型文にし、詳細は `logger.exception` へ
- **想定工数**: 半日

---

### ISSUE-114: `.gitignore` の行が壊れていて `.overrides_backups/` が無視されていない

- **対象**: `.gitignore:189`
  ```
  .overrides_backups/# ONNX model files (generated by ai_model_builder.py)
  ```
- **症状**: パターンと次のセクションのコメントが**1行に結合**している。
  `.gitignore` は行末コメントを解釈しないため、この行は
  `.overrides_backups/# ONNX model files (generated by ai_model_builder.py)` という
  リテラルなパス指定として扱われ、**`.overrides_backups/` は無視されない**。
- **再現**: `pytest tests/` を実行しただけで、以下が未追跡ファイルとして残る。
  ```
  ?? .overrides_backups/overrides.json.corrupted.20260812_133436_605389.bak
  ?? .overrides_backups/overrides.json.pre-update.20260812_133436_589420.bak
  ... （テスト1回で6件）
  ```
  テストのたびに `git status` が汚れ、誤コミットの温床になる。
- **推奨対応**: 2行に分割する。
  ```
  .overrides_backups/

  # ONNX model files (generated by ai_model_builder.py)
  ```
- **想定工数**: 5分

---

### ISSUE-113: `app/agents/` が本体の 52% を占めるがカバレッジ 10〜30%

- **実測**: `app/agents/` 19,817 行 = app 全体（37,854行）の 52%。
  一方、BUYMA 業務の中核（`routes` + `models` + `core` + 主要 `services`）は合計 ~4,500 行。
  `agents/browser/*` の多くが coverage 0〜30%（`plp_flow.py` 7%、`nav_plp_materializer.py` 10%、
  `nav_locale_guard.py` 12%）。
- **評価**: セルフヒーリング型ブラウザ自動化として単体では意欲的だが、
  `docs/経営者判断.md §5` が示す実戦ルート（YOOX / SSENSE / Italist / Cettire = curl_cffi 中心）と
  接続されておらず、**投資対効果が最も低い領域にコード量の半分が集中している**。
  Moncler 専用コード（`moncler_*` 7ファイル）は `§7 ブランド戦略` の Tier 1/2 にも登場しない。
- **推奨対応**（判断が必要なので提案のみ）:
  - `agents/browser/` を「実戦で使っているもの」「実験」に仕分けし、後者は別リポジトリか `experiments/` へ隔離
  - Moncler 専用実装は汎用化するか削除するかを決める
  - カバレッジ目標を app 全体ではなく「中核4ディレクトリ 80%」に設定し直す（現状 `core` は既に 100%）

---

## ✅ 良かった点

- **`app/core/pricing/` の設計が良い**。`PriceSource` による出所トラッキングは、
  `docs/経営者判断.md` に記録された過去の失敗（キーワード推測で ¥13,000〜25,000 過大評価）への
  対策として的確。coverage 100%、ロジックも読みやすい。
  ISSUE-101 は**設計の欠陥ではなく配線漏れ**であり、直せば設計思想がそのまま活きる。
- **セキュリティの基礎が固い**: 秘密情報のコミットなし（`.env.template` のみ）、
  `eval`/`exec`/生SQL/`shell=True` ゼロ、パスワードは `werkzeug.generate_password_hash`、
  CSRF はグローバル有効かつテンプレートにも `csrf_token()` あり。
- **`@login_required` の網羅性が高い**: 登録ルート 110 件に対し、未ログインで到達できる GET は 8 件のみ
  （実測: test_client で全 GET ルートを叩いて確認）。内訳は `/`（トップ）、`/auth/login`、
  および保護済みページへ転送するだけの旧URL互換リダイレクト6件で、いずれも実害なし。
- **CI が実際に green**: `ruff check` / `ruff format --check` ともに通過、346ファイル整形済み。
  `ruff.toml` の ignore が3つ（E501/E402/F821、実債務 70件）まで絞れており、
  「Phase 1: CI通過優先 → Phase 2: ignore を1つずつ外す」という方針も明記されていて健全。
- **モデル層のカバレッジが 93〜100%** と高い。
- **ドキュメント体系（Layer A/B/C）が実用的**。経営判断とコードの対応表があるため、
  本レビューでも ISSUE-103 / ISSUE-109 のような「判断と実装の乖離」を短時間で特定できた。

---

## 推奨する着手順序

| 順 | ISSUE | 理由 | 工数 |
|---|---|---|---|
| 1 | **ISSUE-101** | 注文登録・sourcing パイプラインが動いていない。他すべての前提 | 1〜2日 |
| 2 | **ISSUE-102** | 101 を直したらモックを外す。同時にやらないと再発する | 半日 |
| 3 | **ISSUE-111** | 為替の静かな破壊。101 と同じ pricing 層なのでまとめて | 1時間 |
| 4 | **ISSUE-104 / 105** | 外部公開するなら必須。小さい | 3時間 |
| 5 | **ISSUE-103** | 利益率の議論の前提。BUYMA 公式確認待ちがあるので早めに着手 | 半日＋確認 |
| 6 | **ISSUE-106 / 108** | 死にコードの整理。判断（残す/消す）だけで大半が片づく | 3時間 |
| 7 | **ISSUE-107** | デプロイ前に必須。今すぐでなくてよい | 半日 |
| 8 | **ISSUE-114** | 5分で終わる。いつでも | 5分 |
| 8 | ISSUE-109 / 110 / 112 | 保守性。順次 | 各半日 |
| 9 | ISSUE-113 | 経営判断が必要。単独で議論する | — |

---

## 補足: 本レビューの前提

- `70_PROMPTS/coding/repo-comprehensive-review.md` はリモート実行環境に存在しなかったため、
  一般的な包括レビュー観点（機能正当性 / セキュリティ / アーキテクチャ / テスト品質 / ドキュメント整合性）で実施した。
- `docs/経営者判断.md` および `docs/要件定義.md`（Layer A）を読んだうえで、
  記載済みの経営判断は所与として扱い、**実装との乖離**のみを指摘している。
- すべての P0 / P1 指摘は実際にコードを実行して再現を確認した。推測による指摘は含まない。
