# CR-ATELIER-002 Step 5: Moncler PLP→PDP 抽出ロバスト化 & Self-Healing 連携

- **Status:** Design Phase
- **Author:** [AI Assistant]
- **Date:** 2025-12-08
- **Related CR:** CR-ATELIER-002_MONCLER_PLP_PDP_EXTRACTION_FIX.md

## 1. Overview & Context

### 1.1 背景

CR-ATELIER-002 Step 1〜4 で以下を実施済み：

- Moncler PLP→PDP 抽出ロジックを /products/ パターン前提に整理
- site_config と moncler_plp_v1.py のセレクタ定義を同期
- URL バリデーション・ロケール制御（/en-int, shipToCountry=GB）を明示
- 二重ロケールパターン /en-lt/en-int/... を Trap/Reject 対象として扱う
- Telemetry に PDP 抽出の raw href / rejection_stats を記録
- pytest による URL バリデーション／抽出ロジックのユニットテストを追加

しかし、実ブラウザ検証では以下の問題が残っている：

- LocaleGuard が補正した URL から、サーバ側のリダイレクトにより再び `/en-lt/en-int/search` のような「二重ロケール + search ページ」に飛ばされるケースがある
- Moncler 専用 PDP 抽出ロジックが呼ばれているが、`raw=0` で要素が見つからないケースがある
  → DOM 構造の変化 / 選択セレクタ不足 / タイミング問題のいずれか

### 1.2 Step 5 の目的

Step 5 では、以下を狙う：

1. **実 DOM ベースの PLP→PDP 抽出ロジックのロバスト化**
   - `failure_dom.html` / 将来的な `success_dom.html` 等から、現実の DOM 構造をサンプリングし、セレクタ候補を再設計・多層化する。

2. **Self-Healing / Selector Discovery との連携**
   - Telemetry に蓄積した情報を活用し、`selector_discovery_agent` / `self_healing_agent` に Moncler PLP/PDP の改善タスクを渡せる足場を整える。

3. **「フィールド検証 → 失敗時の自動学習ループ」の一部を確立**
   - 単発のハードコーディングではなく、将来他サイトにも展開可能な「PLP→PDP 抽出の失敗分析 → セレクタ再学習」フローのテンプレート化。

## 2. Scope

### 2.1 In-Scope（Step 5 でやること）

1. **実 DOM スナップショットに基づく PLP/PDP 構造の再分析**
   - `instance/runs/*/failure_dom.html` 等を対象に、
     - 「listing なのか search なのか」
     - PDP リンクがどの階層の要素にぶら下がっているか
     - `/products/` 以外の product パターン（例: `/p/`, `/product/` など）が存在するか
   - これをコード内コメントまたは `docs/spec/` の補助資料として整理。

2. **セレクタ戦略の多層化**
   - `moncler_plp_v1.py` / `extractor.py` で
     - **Primary**: `site_config.selectors.plp.pdp_link_selectors`（/products/ 前提）
     - **Secondary**: DOM 構造ベースの「カード内 a[href]」/ data-component / data-testid など
     - **Tertiary**: /products/ 以外の Moncler 固有パターン（見つかった場合）
   - これらを「優先度付き」のロジックとして実装する設計を固める。

3. **リダイレクト挙動に対する防御強化（実挙動ベース）**
   - `navigation_driver._ensure_expected_locale()` の責務を整理し、
     - Locale 修正（/en-int 化）
     - 「Search へのリダイレクト」を検出して Telemetry に記録
     - 必要であれば Search ページを「PLP 相当」とみなす fallback ポリシーの検討
   - Step 5 ではまだ「実装方針」と「ロジック案」まで（実装は Step 6 以降でも可）。

4. **Self-Healing 連携の設計**
   - Telemetry の `moncler_pdp_links_debug` / `moncler_plp_state` 等から、
     - 「セレクタが raw=0 だったケース」
     - 「Trap 判定で終了したケース」
     - 「Locale 補正が繰り返されたケース」
   - これらを `selector_discovery_agent` / `self_healing_agent` に渡すための「タスクペイロード設計」と「API / 関数インタフェース案」をまとめる。

5. **Acceptance Criteria (Step 5 時点) の明文化**
   - 「Step 5 時点でどこまでを成功とみなすか」を明文化する。
   - 例:
     - Moncler 1 カテゴリに対して、安定して 1 件以上の PDP URL を取得できる run が再現性高く得られている状況（ただし、完全自動復旧ループの実装は Step 6 以降）。

### 2.2 Out-of-Scope（Step 5 ではやらないこと）

- DrissionPage への完全移行 or 併用設計（別 CR / Step で扱う）
- NexusCore 側 Orchestrator からのフル自動起動フロー統合
- Moncler 以外ブランドへの展開（汎用化は Step 6+ のテーマ）
- BUYMA 側との API / UI 統合（本 CR の範囲外）

## 3. Implementation Plan（実装計画）

Step 5 自体は **設計フェーズがメイン**なので、ここでは「コードに落とし込む前提での、具体的な設計タスク」を列挙する。

### Step 5-1: 実 DOM サンプリング & 構造メモ作成

**対象ファイル**:
- `instance/runs/*/failure_dom.html`（複数のrunから選定）
- `docs/spec/CR-ATELIER-002_MONCLER_DOM_NOTES.md`（新規作成）

**実施内容**:
- `instance/runs/*/failure_dom.html` のうち、Moncler / search / PLP っぽいものを数件選ぶ。
- それぞれについて、
  - `<main>` / `<section>` / `div[data-component*='Product']` / `ul/li` などを起点に、
  - 「商品カード」「商品名」「価格」「PDP へのリンク」を人間の目で確認。
- 見つかったパターンを、`docs/spec/CR-ATELIER-002_MONCLER_DOM_NOTES.md` のような追加資料に記録する（セレクタ例、パス例を含める）。

### Step 5-2: セレクタ戦略のレイヤリング設計

**対象ファイル**:
- `app/agents/plugins/moncler_plp_v1.py`
- `app/agents/browser/extractor.py`
- 必要に応じて `app/agents/browser/plp_driver.py`

**設計内容**:
- `moncler_plp_v1.py` に、以下のようなセレクタレイヤをコメント＆定数で整理：
  - `PRIMARY_PDP_LINK_SELECTORS`：site_config 準拠 (/products/)
  - `SECONDARY_PDP_LINK_SELECTORS`：DOM から判明した `article[data-component*='ProductCard'] a[href]` 等
  - `GENERIC_FALLBACK_SELECTORS`：汎用 `div:has(a[href*='/products/'])` など
- `extractor.extract_moncler_pdp_links()` 側で、
  - Primary → Secondary → Fallback の順で
  - 「どのレイヤで何件ヒットしたか」を Telemetry に記録する設計案を明記。
  - 「どのレイヤで抽出されたリンクを優先採用するか」のポリシーをコメントで定義。

### Step 5-3: Redirect / Locale 挙動の扱い整理

**対象ファイル**:
- `app/agents/browser/navigation_driver.py`

**設計内容**:
- `_ensure_expected_locale()` の責務を再整理（コメントレベル）：
  - **Pre-condition**: Moncler の PLP/検索 URL
  - **Post-condition**:
    - `page.url` が `/en-int/...` で始まり
    - 「明らかな Trap（検索トップ / ロケールゲート / 404）」でないこと
- 「Search ページを PLP として扱うか」のポリシーを決める：
  - 例: `/en-int/search` であっても、DOM 上に product tile が並んでいるなら PLP 同等として扱う。
- どの URL パターンを「Trap」ではなく「許容 PLP」とみなすかを一覧化。
- 上記ポリシーをもとに、
  - `TrapDetector` / `NavigationDriver` / `LocaleGuard` の役割分担をコメントと仕様レベルで整理する。

### Step 5-4: Self-Healing / Selector Discovery 連携設計

**対象ファイル**:
- `app/agents/self_healing_agent.py`
- `app/agents/selector_discovery_agent.py`
- `app/agents/browser/telemetry.py`
- 可能なら `app/agents/failure_analysis_agent.py`

**設計内容**:
- Telemetry 側で Moncler PLP/PDP 関連のメトリクスをまとめるキーを定義：
  - `moncler_plp_pdp_outcome` のような logical key で、
    - `plp_materialized` / `tiles_detected` / `pdp_links_raw` / `pdp_links_accepted`
    - `selector_layers_used`（primary / secondary / fallback）
    - `locale_corrections` 回数
  - を JSON で記録する設計（実装は Step 6 以降でもよい）。
- `self_healing_agent` / `selector_discovery_agent` に対して、
  - 「どの Telemetry キーを見て、どういうタスクを起こすか」の I/F をテキストで定義。
  - 例:
    - **条件**: `pdp_links_raw == 0` が一定回数連続
    - **アクション**: `selector_discovery_agent` に Moncler PLP の DOM スナップショット + 現行セレクタを渡し、新セレクタ候補を生成させる。

### Step 5-5: Acceptance Criteria（Step 5 時点）

設計上のゴールとして、Step 5 では以下を満たす状態を狙う：

**仕様レベル**:
- PLP/PDP 抽出ロジックの「レイヤ構造」がコメント + 定数で明確になっている。
- Locale / Trap / Search の扱いポリシーが仕様として書かれている。
- Self-Healing 連携のための Telemetry キーとエージェントへのインタフェース案が定義されている。

**実行レベル（任意だが望ましい）**:
- 手動で `python -m app.scripts.run_site moncler --query "down jacket" --headful` を複数回実行し、
  - 最低 1 パターン以上で `collected_pdp_links >= 1` となる run が存在する。
- `tests/test_moncler_pdp_url.py` を含む pytest が引き続きパスする。

## 4. Testing Strategy（Step 5 時点）

Step 5 は主に「設計＋軽微実験」が中心だが、必要に応じて以下を併用する。

### 4.1 ユニットテスト

**既存**: `tests/test_moncler_pdp_url.py`
- URL バリデーション / rejection reason のテストは引き続き利用。

**追加候補**:
- DOM フィクスチャを用意し、`extract_moncler_pdp_links()` に対して
  - Primary layer での抽出
  - Secondary / Fallback layer での抽出
  - `raw=0` / `accepted=0` 時の Telemetry 出力
- ただし、これらの具体的実装は Step 6 に回してもよい。

### 4.2 実ブラウザ検証（手動）

**コマンド例**:
```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python -m app.scripts.run_site moncler --query "down jacket" --headful
```

**確認ポイント**:
- `instance/runs/YYYYMMDD_*/result.json` の `ok` / `nav_outcome.collected_pdp_links`
- `instance/runs/YYYYMMDD_*/system.log` における
  - LocaleGuard ログ
  - PLP→PDP ログ
  - Telemetry 保存ログ

## 5. Risks & Open Questions

1. **Moncler 側のロケール・検索挙動が頻繁に変わる可能性**
   - → 継続的に DOM/URL パターンの変化を監視する必要あり。

2. **/products/ 以外の product URL パターンが存在する場合の扱い**
   - → Step 5 の DOM 分析で実際に出てきた場合のみ対応範囲に含める。

3. **Search ページを PLP 相当とみなすかどうか**
   - → 誤検知（ノイズの多い検索結果）とのトレードオフがあり、ビジネス的要件も影響する。

## 6. 実装状況（Step 5 完了時点）

### 6.1 Step 5-1: 実 DOM サンプリング & 構造メモ作成 ✅

**完了内容**:
- `docs/spec/CR-ATELIER-002_MONCLER_DOM_NOTES.md` を作成
- PLPページ、Searchページ、TrapページのDOM構造仮説を整理
- セレクタ候補とURLパターンを分析

### 6.2 Step 5-2: セレクタ戦略のレイヤリング設計 ✅

**完了内容**:
- `app/agents/plugins/moncler_plp_v1.py`:
  - `MONCLER_PLP_PDP_LINK_SELECTORS_PRIMARY` を定義
  - `MONCLER_PLP_PDP_LINK_SELECTORS_SECONDARY` を定義
  - `MONCLER_PLP_PDP_LINK_SELECTORS_TERTIARY` を定義
- `app/agents/browser/extractor.py`:
  - Primary → Secondary → Tertiary の順で抽出を試みる実装を追加
  - レイヤごとの統計情報（`layer_stats`）を記録
  - Telemetry に `layer_stats` を保存

### 6.3 Step 5-3: Redirect / Locale 挙動の扱い整理 ✅

**完了内容**:
- `docs/spec/CR-ATELIER-002_STEP5_LOCALE_TRAP_POLICY.md` を作成
- LocaleGuard、TrapDetector、URLバリデーションの役割分担を明確化
- Searchページの扱いポリシーを定義
- `app/agents/browser/navigation_driver.py`:
  - `_ensure_expected_locale()` の責務をコメントで明確化
  - Pre-condition / Post-condition を明記

### 6.4 Step 5-4: Self-Healing / Selector Discovery 連携設計 ✅

**完了内容**:
- `docs/spec/CR-ATELIER-002_STEP5_SELF_HEALING_INTERFACE.md` を作成
- Telemetry キー `moncler_plp_pdp_outcome` のデータ構造を定義
- Self-Healing Agent へのタスクペイロード設計を定義
- Selector Discovery Agent へのタスクペイロード設計を定義
- エージェント間の連携フローを設計

### 6.5 Step 5-5: Acceptance Criteria（Step 5 時点）の明文化 ✅

**完了内容**:
- Step 5 時点での成功基準を明文化
- 仕様レベルと実行レベルの両方を定義
- 実装状況を各ステップごとに記録

## 7. 次のステップ（Step 6 以降）

1. **実装**:
   - Telemetry への `moncler_plp_pdp_outcome` 記録を実装
   - Self-Healing Agent の Moncler 専用ロジックを実装
   - Selector Discovery Agent の Moncler 専用ロジックを実装

2. **テスト**:
   - レイヤリング実装のテスト
   - 失敗検出のテスト
   - セレクタ提案のテスト

3. **実ブラウザ検証**:
   - 複数回実行して、`collected_pdp_links >= 1` となる run が存在することを確認

