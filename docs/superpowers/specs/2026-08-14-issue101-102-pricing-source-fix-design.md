---
date: 2026-08-14
status: approved
tags: [atelier-kyo-manager, ISSUE-101, ISSUE-102, pricing, Tier1, bug-fix]
関連:
  - docs/REVIEW_ISSUES_2026-08-12.md（ISSUE-101〜114 正典・PR #99ブランチ）
  - obsidian-ssot/00_SYSTEM/マルチLLMレビュー/2026-08-14_atelier-ISSUE101-102-修正方針-B案レビュー.revised_proposal.md（改訂案 B'）
  - obsidian-ssot/00_SYSTEM/マルチLLMレビュー/2026-08-14_atelier-ISSUE101-102-修正方針-B案レビュー.review_log.md
  - docs/経営者判断.md §3,§9（手数料・価格調査の鉄則）
---

# ISSUE-101/102 修正設計 — PricingInput price source 配線と信頼性ゲートの本質修復

## 1. 背景・問題

### 1.1 ISSUE-101（機能停止級）
`PricingInput`（`app/core/pricing/schemas.py:62-63`）の `purchase_price_source`/`selling_price_source` のデフォルトが `PriceSource.UNKNOWN`。`calculate_pricing()`（`calculator.py:7-23`）は冒頭 `validate_sources()` で `UNKNOWN`/`KEYWORD_GUESS` を `PriceIntegrityError` で弾く。ところが**本番6経路すべて** source 未設定で `PricingInput` を構築するため、正常な入力でも必ず例外 → 計算全滅。

`routes/orders.py` の `@handle_db_error()` が例外を握りつぶし `flash`+`redirect` で終わるため**注文が DB に保存されず表面化しない**。

### 1.2 ISSUE-102（テスト隠蔽）
CI 2149件 green は本バグを回避するようテストが書かれた結果。`tests/test_sourcing_profitability_coverage.py:164-166` は「`PriceIntegrityError` を raise するためモック化」と明記。`tests/order/test_order_model.py:150-171`・`tests/test_models_extra.py:62-80` は関数名 `test_calc_profit_*` なのに `Order.calc_profit` を一度も呼んでいない。`app/utils/sourcing_profitability.py` は coverage 100% だが本番経路は一度も動いていない。

### 1.3 対象6経路（Explore確認済・全て source 未設定）
| # | 箇所 | データ性質 |
|---|---|---|
| 1 | `app/models/order.py:86` `Order.calc_profit()` | ユーザーフォーム入力値 |
| 2 | `app/utils/sourcing_profitability.py:129` | パイプライン入力データ |
| 3 | `app/services/price_comparison_service.py:310` | buyma_price 推定(cheapest*markup_rate)の場合あり |
| 4 | `app/services/brand_price_service.py:230` | 同上・推定の場合あり |
| 5 | `app/services/price_monitor_service.py:371` | 監視データ |
| 6 | `app/agents/profitability_agent.py:116` | supplier/market データ |
| 参考✅ | `app/services/ssense_buyma_pipeline.py:249-250` | 唯一 `BROWSER_VERIFIED` 明示 |

### 1.4 経営者判断「価格調査の鉄則」（`docs/経営者判断.md`・CLAUDE.md）
推測データ（KEYWORD_GUESS/UNKNOWN）で利益計算させない。Fendi ¥13,000〜25,000 過大評価事故（2026-05）の再発防止が導入の動機。**本 spec はこの鉄則を維持・強化する**。

## 2. 目標・非目標

### 2.1 目標
- 6経路の `PriceIntegrityError` 全滅を修復し、各経路が**正しい source で**計算できるようにする
- 鉄則を**強化**する（推定データが「信頼済」として紛れ込む抜け道を塞ぐ）
- ISSUE-102（モック隠蔽）の根 `@handle_db_error` 握りつぶしを改修
- 「source 未設定」を機械検出（mypy strict + AST）し、将来の再発を構造防止

### 2.2 非目標（本 spec の対象外）
- ISSUE-103（実効手数料14.2%未適用）— 経営者による BUYMA 公式手数料確定待ち（別タスク）
- ISSUE-104/105（Open Redirect・Cookie）— セキュリティ群（別タスク）
- ISSUE-111（為替0フォールバック）— pricing 層で本 spec と関連するが、別 PR で対応（本 spec の `ESTIMATED` 仕組みと統合可能）
- 既存レビュー PR #99（`claude/atelier-kyo-manager-review-xyjbb2`）のマージ判断 — 別タスク（バックログL25・CONFLICTING 状態）

## 3. 設計（改訂案 B'・7改訂点）

### 3.1 [改訂1] Order.calc_profit — MANUAL_INPUT 固定 → ユーザー選択/参照URL
**現状**: `Order.calc_profit()`（order.py:86-90）は source 系引数を渡さず（デフォルト UNKNOWN）→ 必ず例外。
**改訂**: `MANUAL_INPUT` 固定でなく、ユーザーが入力価格の信頼性を明示する仕組みを導入。2案（実装時に決定）:
- **案α**: Order フォームに「仕入価格の確度」「販売価格の確度」select（実価格確認済/推定）を追加 → Order に source 情報カラム追加（マイグレーション）
- **案β**: Order フォームに「参照URL」（仕入先・BUYMA）を optional 保持 → URL 有無で source 自動判定（URL有=BROWSER_VERIFIED／無=ESTIMATED 警告）

**理由（MiniMax#2/#7 + Gemini#1）**: `MANUAL_INPUT` 固定は「ユーザー入力＝信頼」の楽観。ユーザーが BUYMA 推定値をコピペ手入力すれば推定値が信頼済で通り Fendi 事故と同じ抜け道。鉄則の「システムで強制」に合致しない。
**判断要点**: α（カラム追加・より明示的）か β（URL・UX 軽い）か。スキーマ変更は両案で必要（元案「不要」は撤回）。

### 3.2 [改訂2] 推定経路(③④) — 完全拒否 → D案混合（profit_status=ESTIMATED）
**現状案（撤回）**: 推定 source は `PriceIntegrityError` で完全拒否。
**改訂**: 推定 source でも計算は実施するが、結果に `estimate_mark=True` を付与。DB 保存時は `profit_status` を `CONFIRMED` / `ESTIMATED` で区別:
- `ESTIMATED` の注文は**本番発注確定を弾く**（ドラフト状態で保持）
- 経営者通知（Slack/EMAIL）「推定価格で計算された注文あり」
- ダッシュボードで「推定・要実価格確定」警告表示
- 30日以内に `MANUAL_INPUT`/`BROWSER_VERIFIED` で再計算されない注文は警告昇格

**理由（MiniMax#1/#5 + Gemini#2/#5）**: 完全拒否は③④の2経路を停止（ISSUE-101 と同パターンの障害再発）。鉄則は「排除」でなく「最終判断（本番発注）に使わせない」。`ESTIMATED` 分離で「参考値として計算は見せる」UX と「推定で発注はさせない」安全性を両立。

### 3.3 [改訂3] 検出 — 必須引数化 + mypy strict + AST リファレンステスト
- `PricingInput` の `purchase_price_source`/`selling_price_source` のデフォルト `UNKNOWN` を**廃止**（必須引数化・位置引数 or 型チェックで欠落を即検出）
- **mypy strict** を CI に追加（`ci.yml`）→ 型レベルで source 未設定を検出
- **AST リファレンステスト新設**: `tests/test_pricing_input_construction.py` で AST で `PricingInput(...)` 構築箇所を全列挙し、`purchase_price_source`/`selling_price_source` が未設定の箇所 = CI 失敗とする（6経路に戻らない構造的保証）

**理由（MiniMax#3/#6 + Gemini#3）**: Python は動的型付け→必須引数化だけでは実行時まで欠落が露見しない。モックがデフォルトを設定し続ければ ISSUE-102（モック隠蔽）が再発。mypy + AST で静的・機械的に検出。

### 3.4 [改訂4] 進め方 — C案的段階性（Phase1/Phase2）
- **Phase1（検出の整備）**: 改訂3（必須化+mypy+AST）+ 改訂5（handle_db_error）。この段階では6経路は一時的に `skip_source_validation=True` で**従来通り動かし続ける**（必須化したが source 未設定の経路は skip で例外回避）。AST テストは `skip_source_validation=True` 使用箇所を「Phase2 で解消すべき一時除外」として許容リスト管理。
- **Phase2（経路別 source 意味付け）**: 6経路を1つずつ source 明示（①改訂1・②⑤⑥ BROWSER/API_VERIFIED・③④改訂2のESTIMATED）し、`skip_source_validation=True` を順次撤去。

**理由（MiniMax#4 + Gemini#4）**: Phase1 単独でもリリース可能（skip で動作維持・検出だけ整備）。Phase2 で意味付け。Cは「見かけ倒し」でなく「失敗を即座に表面化」。B一括は判断誤りを内包しやすい（両LLM指摘）。
**判断要点**: Phase1/Phase2 を別 PR（別リリース）にするか、1 feature branch 内の commit 分割にするか。

### 3.5 [改訂5] @handle_db_error 改修（並行・ISSUE-102 の根）
`@handle_db_error()`（`routes/orders.py`）が例外を握りつぶす構造を改修:
- `PriceIntegrityError`/`TypeError` 等は**握りつぶさず上位に伝播**（500 エラーまたは適切なエラー画面）
- 伝播時はトランザクション **ロールバック必須**
- 既存の「妥当な DB エラー（IntegrityError 等）」のハンドリングは維持（回帰なし）

**理由（MiniMax#10）**: これを直さないと B' が成功しても次の Issue で「例外握りつぶし→DB未保存→CI緑」が再発（ISSUE-102 と同根）。

### 3.6 [改訂6] ③④推定判定基準の明文化
`BuymaPrice`（または該当データ構造）に `method: Literal["BROWSER","MARKUP","MANUAL"]` フィールド追加:
- `BROWSER` = BUYMA 公式/Selenium で取得（`BROWSER_VERIFIED`・取得時刻+取得URL 必須）
- `MARKUP` = `cheapest * markup_rate` 推定（`ESTIMATED`）
- `MANUAL` = 人間入力（`MANUAL_INPUT`）
- `API` = BUYMA API 経由（`API_VERIFIED`・`BROWSER_VERIFIED` と分離）

**理由（MiniMax#8/#9）**: 「推定の場合あり」という曖昧判定だと、実装者が「これは推定でない」と水増しし鉄則が骨抜きになる。method enum で機械的に判定。

### 3.7 [改訂7] 完了条件 — smoke 6経路6本 + AST grep
- **smoke test 6本**: 各経路（①〜⑥）の `PricingInput` 経由計算が DB 保存（または ESTIMATED 保存）まで至る、またはエラーが正伝播することを検証（モック不使用）
- **AST grep テスト**: `PricingInput(...)` 構築箇所のうち `source` 未設定（かつ `skip_source_validation=True` でない）= 0 を CI 失敗条件
- 既存2149件テストの回帰ゼロ（ただし ISSUE-102 のモックテストは実経路検証に書き換え）

## 4. テスト戦略

| 種別 | 内容 |
|---|---|
| **AST リファレンステスト**（新規）| `PricingInput` 構築箇所全列挙・source 未設定=CI失敗 |
| **smoke 6本**（新規）| 6経路の DB 保存/エラー正伝播（モックなし）|
| **ESTIMATED テスト**（新規）| 推定 source で計算→`profit_status=ESTIMATED` 保存・本番発注ブロック |
| **モック撤去**（ISSUE-102）| `test_sourcing_profitability_coverage.py`・`test_order_model.py` のモックを外し実経路検証 |
| **回帰** | 既存2149件 green 維持（モック撤去分は置き換え）|
| **mypy strict** | CI 追加・source 未設定を型検出 |

## 5. リスク・判断要点（ユーザー確認事項）

### 5.1 判断要点（2026-08-14 確定・spec approved）
1. **改訂1**: **α（source select）主軸 ＋ β（参照URL optional 参考記録）** — ユーザーが意識して「実価格/推定」を選ぶ（Fendi事故の無意識コピペ防止）＋ URL で客観証拠。マイグレーション要（purchase_price_source/selling_price_source or 確度 select カラム＋参照URL カラム）
2. **Phase1/Phase2**: **feature branch 1本・論理的に commit 分割（main 一括マージ・Tier1慎重）** — Phase1(skipで動作維持・検出整備)→Phase2(source明示・skip撤去)は依存強く別リリースすると中途半端状態が本番に乗るため一括マージ
3. **ESTIMATED ブロック**: **完全ブロック（ドラフト保持・実価格で source 更新→再計算まで発注不可）** — 承認ゲートは形骸化リスク（ISSUE-102モック隠蔽と同根）で採用しない。緊急時は例外的に notes に理由＋手動 DB 操作（ゲートでなく例外運用）
4. **経営者判断**: **合致（ESTIMATED は「実価格ベース移行」を促す仕組み）** — 経営者判断.md §9「マークアップ率調整か実価格ベース移行が必要」をシステムで支える。手数料14.2%未確定は ISSUE-103 別軸（source信頼性軸と独立・本spec完了条件には影響しない）

### 5.2 リスク
- **最悪ケース**: PricingInput 必須化で見落とし経路（6以外）が実行時 TypeError → mypy strict + AST で軽減（spec 改訂3）
- **過大評価事故再発**: MANUAL_INPUT 抜け道 → 改訂1（選択化）で防止
- **モック隠蔽再発**: handle_db_error 改修（改訂5）+ AST（改訂3）で防止
- **スコープ膨張**: UI（改訂1）・ESTIMATED（改訂2）・handle_db_error（改訂5）と広範 → Phase 分割（改訂4）で管理

### 5.3 工数見込み
3〜5日（Phase1: 1.5日 / Phase2: 2日 / handle_db_error: 0.5日 / UI: 0.5〜1日）

## 6. 関連
- ISSUE 正典: `docs/REVIEW_ISSUES_2026-08-12.md`（PR #99・ISSUE-101〜114）
- 改訂案経緯: `obsidian-ssot/00_SYSTEM/マルチLLMレビュー/2026-08-14_atelier-ISSUE101-102-修正方針-B案レビュー.review_log.md`（MiniMax 11件 + Gemini 5件）
- sentaku 推奨B→B' 淘汰経緯: セッション 2ba9（本セッション）
- 鉄則導入経緯: `obsidian-ssot/01_DECISIONS/atelier-kyo-manager/2026-06-11_価格データ信頼性チェック実装.md`
- 部分対応: `obsidian-ssot/01_DECISIONS/atelier-kyo-manager/2026-07-12_残39件中5件PriceSource明示修正と残34件の分類.md`
