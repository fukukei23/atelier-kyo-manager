# 実装 Plan — ISSUE-101/102 PricingInput source 配線と信頼性ゲート修復

> spec: `docs/superpowers/specs/2026-08-14-issue101-102-pricing-source-fix-design.md`（approved）
> 改訂案 B'（MiniMax+Gemini レビュー統合）・セッション 2ba9
> 工数見込み: 3〜5日・Tier1・feature branch 運用

## 依存グラフ（垂直スライスの方針）

```
[Phase1: 検出の整備（6経路は skip で動作維持）]
  T0 branch ─┬─ T1 PricingInput必須化 + 6経路一時skip
             ├─ T2 handle_db_error 改修（例外伝播+ロールバック）
             ├─ T3 AST リファレンステスト新設
             └─ T4 mypy strict CI 追加
  CP1（Phase1完了・回帰ゼロ・6経路 skip で動作）
        │
        ▼
[Phase2: source 明示・skip 撤去]
  T5 Order マイグレーション（source select α + 参照URL β カラム）
  T6 Order.calc_profit source 適用（MANUAL_INPUT/ESTIMATED）+ フォームUI
  T7 ②⑤⑥ BROWSER_VERIFIED/API_VERIFIED 明示
  T8 ③④ BuymaPrice.method enum + ESTIMATED 経路
  T9 profit_status（CONFIRMED/ESTIMATED）+ 本番発注ブロック
  T10 6経路の skip_source_validation 撤去
  T11 smoke 6本 + ISSUE-102 モック撤去
  CP2（Phase2完了・完了条件全達成）
```

## Phase1 — 検出の整備（skip で動作維持）

### T0: feature branch 作成
- `git checkout -b fix/issue-101-102-pricing-source`
- **検証**: `git branch --show-current`

### T1: PricingInput デフォルト廃止 + 6経路一時 skip（垂直パス: 必須化しても動き続ける）
1. `app/core/pricing/schemas.py:62-63` — `purchase_price_source`/`selling_price_source` のデフォルト `UNKNOWN` を削除（必須引数化）。ただし後方互換のため一時的に `None` 許容 + `validate_sources` で `None` を UNKNOWN 扱い（T10 で完全化）
2. `calculator.py` の `skip_source_validation` は既存（確認済み）
3. 6経路（①order.py:86 ②sourcing:129 ③price_comparison:310 ④brand_price:230 ⑤monitor:371 ⑥agent:116）の `calculate_pricing(inp)` 呼び出しに一時的に `skip_source_validation=True` 追加（Phase2 で撤去）
4. **テスト（TDD・先）**: 既存2149件が green のまま（skip で動作維持）を確認する回帰テスト
- **検証**: `./venv/bin/python -m pytest tests/ -x -q`（2149件 green 維持）

### T2: handle_db_error 改修（ISSUE-102 の根・並行）
1. `routes/orders.py` の `@handle_db_error()` 現状確認（例外を except→flash+redirect で握りつぶし）
2. `PriceIntegrityError`/`TypeError` 等は**握りつぶさず上位伝播**（500 or エラー画面）+ トランザクション ロールバック
3. 既存の「妥当な DB エラー（IntegrityError 等）」ハンドリングは維持（回帰なし）
- **テスト（TDD・先）**: `PriceIntegrityError` 発生時に 500 が返る・DB ロールバックされることを検証
- **検証**: `./venv/bin/python -m pytest tests/order/ tests/test_routes_errors.py -v`（新規＋回帰）

### T3: AST リファレンステスト新設
1. `tests/test_pricing_input_construction.py` 新規 — AST で `PricingInput(...)` 構築箇所を全列挙
2. `purchase_price_source`/`selling_price_source` 未設定 かつ `skip_source_validation=True` 也でない構築 = CI 失敗
3. Phase1 では 6経路は skip 許容リスト（Phase2 で解消）
- **検証**: `./venv/bin/python -m pytest tests/test_pricing_input_construction.py -v`（6経路が許容リスト・他に未設定なし）

### T4: mypy strict CI 追加
1. `mypy.ini` or `pyrightconfig.json` を strict に（`disallow_untyped_defs` 等・段階的で可）
2. `.github/workflows/ci.yml` に mypy job 追加
- **検証**: `./venv/bin/mypy app/core/pricing/ --strict`（pricing 層から strict 適用）

### CP1: Phase1 完了チェックポイント
- [x] 6経路が skip で従来通り動作（2149件 green）
- [x] handle_db_error 改修（例外伝播）
- [x] AST テスト・mypy strict 導入
- [ ] **commit**: `feat(pricing): Phase1 必須化+検出+handle_db_error（skipで動作維持）`
- **検証**: `./venv/bin/python -m pytest tests/ -q && ./venv/bin/mypy app/core/pricing/ --strict`

## Phase2 — source 明示・skip 撤去

### T5: Order マイグレーション（α select + β URL カラム）
1. `app/models/order.py` — `purchase_price_source`/`selling_price_source`（String・PriceSource値）+ `purchase_price_ref_url`/`selling_price_ref_url`（String nullable）カラム追加
2. Flask-Migrate でマイグレーション生成
- **検証**: `./venv/bin/python -m flask db migrate -m "add price source columns"` → `flask db upgrade` → `sqlite3 instance/*.db ".schema order"` でカラム確認

### T6: Order.calc_profit source 適用 + フォーム UI（垂直パス: Order 経路が正しく計算）
1. `Order.calc_profit()` — フォーム入力の source select（実価格/推定）を `PricingInput` に渡す。実価格=MANUAL_INPUT / 推定=ESTIMATED
2. `routes/orders.py` POST `/orders/new`・`/orders/<id>/edit` — フォームに source select α ＋ 参照URL β 追加
3. templates — select・URL 入力フィールド追加
4. ①の `skip_source_validation=True` 撤去
- **テスト（TDD・先）**: `Order.calc_profit` が MANUAL_INPUT で計算成功・ESTIMATED で profit_status=ESTIMATED になること
- **検証**: `./venv/bin/python -m pytest tests/order/test_order_model.py -v`（実経路・モックなし）

### T7: ②⑤⑥ BROWSER_VERIFIED/API_VERIFIED 明示
1. ②`sourcing_profitability.py:129`・⑤`price_monitor_service.py:371`・⑥`profitability_agent.py:116` — データ生成元で source 判定（スクレイパ=BROWSER_VERIFIED / API=API_VERIFIED）
2. 各経路の `skip_source_validation=True` 撤去
- **テスト（TDD・先）**: 各経路が source 明示で計算成功
- **検証**: `./venv/bin/python -m pytest tests/test_sourcing_profitability*.py tests/test_price_monitor*.py -v`

### T8: ③④ BuymaPrice.method enum + ESTIMATED 経路
1. `BuymaPrice`（該当データ構造）に `method: Literal["BROWSER","MARKUP","MANUAL","API"]` 追加
2. ③`price_comparison_service.py:310`・④`brand_price_service.py:230` — `buyma_price` が `MARKUP`(cheapest*markup_rate) なら `ESTIMATED`、`BROWSER` なら `BROWSER_VERIFIED`
3. 各経路の `skip_source_validation=True` 撤去
- **テスト（TDD・先）**: MARKUP で ESTIMATED 計算・BROWSER で CONFIRMED 計算
- **検証**: `./venv/bin/python -m pytest tests/test_price_comparison*.py tests/test_brand_price*.py -v`

### T9: profit_status + 本番発注ブロック（完全ブロック）
1. `Order.profit_status` カラム（CONFIRMED/ESTIMATED・T5 マイグレーションに含めても可）
2. ESTIMATED の Order は `status` が pending から shipped/completed に遷移不可（ドラフト保持）
3. 実価格 source 更新→再計算で CONFIRMED に遷移→発注可能
4. ダッシュボードで「推定・要実価格確定」警告表示
- **テスト（TDD・先）**: ESTIMATED Order の status 遷移ブロック・CONFIRMED 再計算で遷移可能
- **検証**: `./venv/bin/python -m pytest tests/order/test_estimated_block.py -v`（新規）

### T10: 6経路の skip 完全撤去 + AST 許容リスト清算
1. 全 `skip_source_validation=True` 撤去確認
2. AST リファレンステストの許容リストを空に（source 未設定=0）
- **検証**: `./venv/bin/python -m pytest tests/test_pricing_input_construction.py -v`（許容リスト0・全 source 明示）

### T11: smoke 6本 + ISSUE-102 モック撤去
1. smoke 6本新規（6経路の DB保存/エラー正伝播・モックなし）
2. `tests/test_sourcing_profitability_coverage.py:164-166`・`test_order_model.py:150-171`・`test_models_extra.py:62-80` のモックを外し実経路検証に書換
- **検証**: `./venv/bin/python -m pytest tests/test_smoke_pricing_paths.py tests/test_sourcing_profitability_coverage.py tests/order/test_order_model.py -v`

### CP2: Phase2 完了チェックポイント（= 完了条件）
- [x] smoke 6経路6本 green
- [x] AST grep（source 未設定=0）
- [x] 既存2149件回帰ゼロ（ISSUE-102 モック撤去分は置換・最終2192 passed/6 skipped）
- [x] mypy strict green
- [x] ruff clean
- [x] ESTIMATED 発注ブロック機能
- [ ] **commit**: `feat(pricing): Phase2 source明示+ESTIMATED発注ブロック（ISSUE-101/102完全修復）`
- **最終検証**: `./venv/bin/python -m pytest tests/ -q && ./venv/bin/mypy app/ --strict && ./venv/bin/ruff check .`

## マージ
- feature branch で CP1+CP2 完了後、main に一括マージ（PR・Tier1慎重）
- PR #99（既存レビュー・CONFLICTING）との整合は別タスク（バックログL25）

## リスク対応
- **見落とし経路の TypeError**: T3 AST + T4 mypy で機械検出
- **モック隠蔽再発**: T2 handle_db_error + T11 モック撤去
- **スキーマ変更失敗**: T5 マイグレーションを慎重に（DBバックアップ）
- **UI スコープ膨張**: T6 フォーム変更は最小（select＋URL のみ）
