# CR-ATELIER-002: MONCLER PLP→PDP Extraction Fix

- **Status:** Draft
- **Author:** [AI Assistant]
- **Date:** 2025-12-05
- **Related CR:** CR-ATELIER-001_HOTFIX_TELEMETRY_RECORD_PLP_STATE_IMPLEMENTATION.md

## 1. Overview & Context

### 目的 (Why)
Moncler PLP 実行時に、想定している PLP ではなく「トラップページ」に落ちており、その状態で PLP→PDP 抽出ロジックが走るため、PDP リンクが 0 件になっている問題を解決する。

CR-ATELIER-001 で Telemetry 記録は修正済みであり、本 CR は「PLP→PDP 抽出ロジックとロケール制御」を対象とする。

### 背景 (Background)
- CR-ATELIER-001 により、PLP 初期状態の観測データ（DOM スナップショット、セレクタカウント）が取得可能になった
- Moncler PLP 実行時に、以下のような「トラップページ」に落ちている：
  - `It's not here` と表示される 404 ページ
  - 「Select your location」モーダルのみが表示されているロケーションゲート
  - `/en-int/...` で始まるはずの URL が、途中から `/en-lt/en-int/...` などに変化し、検索結果ページ (`.../search`) にいる状態
- その状態で PLP→PDP 抽出ロジックが走るため、PDP リンクが 0 件になっている
- 実行開始時は `https://www.moncler.com/en-int/...&shipToCountry=GB` の形で始まるが、途中で `/en-lt/en-int/...` に変化したり、フッターの国表記が `LITHUANIA | ENGLISH`, `GERMANY | ENGLISH` に変わるなど、想定と異なるロケールに遷移してしまっている

### 現状の問題
1. **トラップページの検出不足**
   - 404 ページ（`It's not here`）
   - ロケーションゲート（`Select your location` + product リストなし）
   - 想定外ロケール＋検索ページ（`/en-lt/en-int/search` 等）
   これらが「PLP ではない状態」として検知できていない

2. **ロケールの一貫性が保たれていない**
   - URL パスの先頭が `/en-int/` でなくなる
   - クエリパラメータ `shipToCountry` が `GB` でなくなる
   - フッターの国表記が `GB` 以外になる

3. **PLP→PDP リンク抽出ロジックの不備**
   - 現状の `plp_to_pdp` セレクタ群では、Moncler の現行 DOM 構造に対して PDP リンクを 1 件も抽出できていない
   - `failure_dom.html` の DOM 構造と乖離している可能性が高い

### 参照 (References)
- `docs/spec/CR-ATELIER-001_HOTFIX_TELEMETRY_RECORD_PLP_STATE_IMPLEMENTATION.md` - Telemetry 実装完了レポート
- `app/agents/browser/navigation_driver.py` - PLP→PDP 抽出ロジック、trap 検出ロジック
- `app/config/sites/overrides.local.json` - Moncler サイト設定
- `app/agents/plugins/moncler_plp_v1.py` - Moncler PLP 戦略プラグイン
- `docs/moncler/PLP_EXTRACTION_FIX_TASK_TEMPLATE.md` - 以前のタスクテンプレート

## 2. Scope

### ✅ In-Scope (やること)

1. **Moncler 向けの PLP→PDP 抽出ロジックの修正**
   - `NavigationDriver.collect_pdp_links` の Moncler 向けロジックを改善
   - Moncler の実際の DOM 構造に合わせたセレクタの更新
   - URL バリデーションルールの見直し

2. **404 ページ / location gate / search-only ページなど「PLP ではない状態」の検出とハンドリング**
   - `NavigationDriver` に `_detect_trap_page(ctx)` のようなヘルパーを追加
   - 404 (`It's not here`)、location gate (`Select your location` + product リストなし)、想定外ロケール＋検索ページ (`/en-lt/en-int/search` 等) を判定
   - trap 検出時の挙動（Telemetry 保存＋再ナビゲート or 専用例外）を定義

3. **実行中の URL が `/en-int/...` + `shipToCountry=GB` を継続的に保つためのロケール制御**
   - URL パスとクエリから、現在のロケール（国 / 言語）を判定するユーティリティを実装
   - `/en-int/` かつ `shipToCountry=GB` でない場合は、location gate や footer を経由して GB / EN に戻す、もしくはホーム＋指定 PLP URL へ再ナビゲートするロジックを追加

### ❌ Out-of-Scope (やらないこと)

- 他ブランド向けの PLP→PDP ロジックの一般化
- 検索クエリ戦略や Self-Healing エージェントの大幅な仕様変更
- TelemetryClient / TelemetryService 自体の仕様変更（これは CR-ATELIER-001 で扱う）
- Pricing / 在庫ロジックの変更
- PLP materialization ロジックの再設計（Phase 1 で完了済み）

## 3. Problems to Solve & Accept Criteria

### 3-1. Trap ページ検出（404 / search / location gate）

#### Problem
Moncler 実行中に、以下のような「PLP ではない状態」に落ちるが、現在はそれを検知できていない：

- `h1` に `It's not here` を含む 404 ページ
- 本文に `Select your location` がある location gate（`Select your location` 一覧のみで、product リストが存在しない）
- `/en-lt/en-int/search` など、想定外のロケール＋検索ページ URL にいる状態

#### Accept Criteria
上記のようなパターンを検出した場合、`NavigationDriver` は「PLP ではない」と判断し、以下のいずれかを行うこと：

- **(a) Telemetry に現在の DOM を保存した上で、正しい PLP URL（例: `/en-int/women/outerwear/all-down-jackets/?shipToCountry=GB`）へ再ナビゲートする**
- **(b) 「PLP ではなく trap ページである」ことを示す専用例外を投げ、上位の Self-Healing ロジックで扱えるようにする**

trap 判定は、メッセージ文言（`It's not here`, `Select your location` など）と URL パターンの両方を利用する。

### 3-2. 国・ロケールの一貫性確保

#### Problem
実行開始時は `https://www.moncler.com/en-int/...&shipToCountry=GB` の形で始まるが、

- 途中で `/en-lt/en-int/...` に変化したり、
- フッターの国表記が `LITHUANIA | ENGLISH`, `GERMANY | ENGLISH` に変わるなど、
- 想定と異なるロケールに遷移してしまい、そのまま PLP→PDP 抽出が走っている

#### Accept Criteria
run の開始から終了まで、以下を満たすこと：

- URL パスの先頭は常に `/en-int/` であること
- クエリパラメータ `shipToCountry` は常に `GB` に維持されていること

location gate で別の国が選択されてしまった場合や、フッターの国が `GB` 以外になった場合は、自動的に GB / EN に戻す処理が動作すること。

ロケール修正後は、ホームまたは正しい PLP URL に戻り、そこで改めて PLP materialization / PDP 抽出を行うこと。

### 3-3. PLP→PDP リンク抽出ロジックの修正

#### Problem
現状の `plp_to_pdp` セレクタ群では、Moncler の現行 DOM 構造に対して PDP リンクを 1 件も抽出できていない。

`failure_dom.html` の DOM 構造と乖離している可能性が高い。

#### Accept Criteria
`failure_dom.html`（および最新の Moncler 実 DOM）を元に、以下を満たすようにすること：

- 実際の product card の構造（例: `article[data-component="ProductCard"] a[href*="/products/"]` のような形）を Spec に明記する
- Moncler の「down jacket」クエリで実行したとき、少なくとも 1 件以上の PDP URL が抽出されること
- 抽出された URL が 404 ページではなく、`/products/` を含む正規の PDP であること
- 抽出された PDP URL から、後続の PDP 情報抽出パイプラインが実行可能であること

## 4. Implementation Plan

### Step 1: Trap ページ検出ヘルパーの追加

1. **`NavigationDriver` に `_detect_trap_page(ctx)` ヘルパーを追加**
   - 404 (`It's not here`) の検出
     - `h1` 要素に `It's not here` が含まれるかチェック
     - URL パターンに `/404` や `not-found` が含まれるかチェック
   - location gate (`Select your location` + product リストなし) の検出
     - 本文に `Select your location` が含まれるかチェック
     - product リスト（`[data-component="ProductCard"]` など）が存在しないかチェック
   - 想定外ロケール＋検索ページ (`/en-lt/en-int/search` 等) の検出
     - URL パスに `/en-lt/` や `/en-de/` など、`/en-int/` 以外のロケールが含まれるかチェック
     - URL パスに `/search` が含まれるかチェック

2. **trap 検出時の挙動を定義**
   - Telemetry 保存: `record_plp_state` を呼び出して現在の DOM を保存
   - 再ナビゲート: 正しい PLP URL（例: `/en-int/women/outerwear/all-down-jackets/?shipToCountry=GB`）へ再ナビゲート
   - 専用例外: `TrapPageDetected` のような専用例外を投げ、上位の Self-Healing ロジックで扱えるようにする

3. **`NavigationDriver.run_plp_flow` に trap 検出ロジックを組み込む**
   - PLP materialization 前後で trap 検出を実行
   - trap が検出された場合、上記の挙動を実行

### Step 2: ロケール一貫性チェックと自動修正

1. **ロケール判定ユーティリティの実装**
   - URL パスからロケール（`/en-int/`, `/en-lt/` など）を抽出する関数
   - クエリパラメータから `shipToCountry` を取得する関数
   - 現在のロケールが期待値（`/en-int/` + `shipToCountry=GB`）と一致するかチェックする関数

2. **ロケール修正ロジックの実装**
   - location gate を経由して GB / EN に戻す処理
     - `Select your location` モーダルを開く
     - `GB` または `United Kingdom` を選択
     - モーダルを閉じる
   - footer を経由して GB / EN に戻す処理
     - footer の国選択を開く
     - `GB` または `United Kingdom` を選択
   - ホーム＋指定 PLP URL へ再ナビゲートする処理
     - ホーム（`https://www.moncler.com/en-int`）に移動
     - 指定 PLP URL（例: `/en-int/women/outerwear/all-down-jackets/?shipToCountry=GB`）へ再ナビゲート

3. **`NavigationDriver` の PLP フローの入口付近に組み込む**
   - `run_plp_flow` の開始時にロケールチェックを実行
   - ロケールが期待値と一致しない場合、ロケール修正ロジックを実行
   - ロケール修正後、ホームまたは正しい PLP URL に戻り、改めて PLP materialization / PDP 抽出を行う

### Step 3: Moncler 向け PLP→PDP セレクタの更新

1. **DOM 構造の確認**
   - `failure_dom.html` と実ブラウザでの DOM を参照
   - product card コンテナの構造を特定
   - PDP へのリンク要素の構造を特定

2. **セレクタの更新**
   - `app/config/sites/overrides.local.json` の Moncler 設定を更新
   - `selectors.plp.pdp_link_selectors` に Moncler の実際の DOM 構造に合わせたセレクタを追加
     - 例: `article[data-component="ProductCard"] a[href*="/products/"]`
     - 例: `div[data-testid="product-tile"] a[href*="/en-int/"][href*="/products/"]`
   - `selectors.pdp.pdp_link_selectors` も必要に応じて更新

3. **URL バリデーションロジックの見直し**
   - `NavigationDriver.collect_pdp_links` 内の `looks_like_product_url` 関数を確認
   - Moncler の URL パターン（`/products/` を含む）に対応するように修正
   - OneTrust などのノイズリンクを除外するロジックを追加

4. **`MonclerPLPStrategy` または `plp_to_pdp` 関連のコードを更新**
   - 上記セレクタに合わせてコードを更新
   - URL バリデーションロジックも `/products/` を前提に見直す

### Step 4: テスト追加

1. **Moncler 専用の E2E / integration テストを追加**
   - 正常系: 想定 PLP にいるときに、1 件以上の PDP URL が抽出されること
   - trap 系: 404 / location gate / 想定外ロケールに落ちたときに、trap として扱われること（再ナビ or 専用例外）

2. **最小限の pytest を 1〜2 個追加**
   - `_detect_trap_page` のテスト
   - ロケール判定ユーティリティのテスト
   - `collect_pdp_links` の Moncler 向けテスト

## 5. Testing Strategy

### テスト方針
- E2E に近い「実行テスト」を中心にする
- Moncler 向け `run_site` を実行し、実際に PDP リンクが抽出されることを確認
- trap ページ検出とロケール制御の動作を確認

### 手動テストコマンド

```bash
# 1. 仮想環境の有効化
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

# 2. Moncler run の実行
python -m app.scripts.run_site moncler --query "down jacket" --headful

# 3. ログの確認
LATEST=$(ls -td instance/runs/2025* | head -1)
echo "Latest run: $LATEST"

# 4. ロケール関連のログ確認
grep -E "locale|shipToCountry|en-int|en-lt|LITHUANIA|GERMANY" "$LATEST/system.log" | tail -20

# 5. Trap 検出関連のログ確認
grep -E "trap|It's not here|Select your location|404" "$LATEST/system.log" | tail -20

# 6. PLP→PDP 抽出関連のログ確認
grep -E "PLP→PDP|collect_pdp_links|PDP links found" "$LATEST/system.log" | tail -20
```

### 検証観点

#### 正常系
- ✅ Moncler PLP から PDP リンクが 1 件以上抽出される
- ✅ 抽出された URL が Moncler の PDP URL パターン（`/products/` を含む）に一致する
- ✅ 抽出された URL が OneTrust などのノイズリンクでない
- ✅ 実行中に `/en-lt/en-int/...` や `LITHUANIA | ENGLISH` などに変化していないこと
- ✅ URL パスの先頭が常に `/en-int/` であること
- ✅ クエリパラメータ `shipToCountry` が常に `GB` に維持されていること

#### Trap ページ検出
- ✅ 404 ページ（`It's not here`）にいるときは、PLP→PDP 抽出に進まず、Spec 通りの挙動（再ナビゲート or 専用例外）になること
- ✅ location gate（`Select your location` + product リストなし）にいるときは、PLP→PDP 抽出に進まず、Spec 通りの挙動になること
- ✅ 想定外ロケール＋検索ページ（`/en-lt/en-int/search` 等）にいるときは、PLP→PDP 抽出に進まず、Spec 通りの挙動になること

#### ロケール制御
- ✅ location gate で別の国が選択されてしまった場合、自動的に GB / EN に戻す処理が動作すること
- ✅ フッターの国が `GB` 以外になった場合、自動的に GB / EN に戻す処理が動作すること
- ✅ ロケール修正後は、ホームまたは正しい PLP URL に戻り、そこで改めて PLP materialization / PDP 抽出が行われること

#### 既存機能への影響
- ✅ 他サイト（Moncler 以外）の PLP→PDP 抽出ロジックに影響がないこと

### 自動テスト

1. **pytest / integration test の追加**
   - `_detect_trap_page` のテスト
     - 404 ページの検出テスト
     - location gate の検出テスト
     - 想定外ロケール＋検索ページの検出テスト
   - ロケール判定ユーティリティのテスト
     - URL パスからロケールを抽出するテスト
     - クエリパラメータから `shipToCountry` を取得するテスト
     - ロケールが期待値と一致するかチェックするテスト
   - `collect_pdp_links` の Moncler 向けテスト
     - 正常系: 想定 PLP にいるときに、1 件以上の PDP URL が抽出されること
     - trap 系: 404 / location gate / 想定外ロケールに落ちたときに、trap として扱われること

2. **CI での実行**
   - 上記の pytest / integration test が CI でグリーンになること

## 6. Risks / Notes

### リスク要因

1. **DOM 構造の変更リスク**
   - Moncler のサイト構造が変更された場合、セレクタが機能しなくなる可能性
   - 対策: 複数のセレクタパターンを用意し、フォールバックロジックを実装

2. **他ロケール（en-int, en-jp 等）での差異**
   - ロケールによって DOM 構造や URL パターンが異なる可能性
   - 対策: 各ロケールに対応したセレクタと URL パターンを用意

3. **OneTrust などのノイズリンクの混入**
   - Cookie 同意やプライバシーポリシーのリンクが PDP リンクとして抽出される可能性
   - 対策: URL バリデーションルールを強化し、ノイズリンクを除外

4. **ロケール制御の複雑さ**
   - location gate や footer の構造が変更された場合、ロケール修正ロジックが機能しなくなる可能性
   - 対策: 複数のロケール修正方法を用意し、フォールバックロジックを実装

### 既知の制約

- Moncler のサイト構造が変更された場合、セレクタの再調整が必要
- ロケールによって URL パターンが異なるため、各ロケールに対応した設定が必要
- trap ページ検出は、メッセージ文言と URL パターンの両方を利用するため、Moncler 固有の実装になる可能性がある

### 移行時の注意点

- `app/config/sites/overrides.local.json` の Moncler 設定を更新する際は、既存の設定をバックアップする
- 変更後は、Moncler 以外のサイトに影響がないことを確認する
- trap ページ検出ロジックは、既存の `_looks_like_trap_or_legal` メソッドと統合する必要がある

## 7. 次のステップ

1. **実装の開始**
   - Step 1 から順に実装を進める
   - 各ステップで動作確認を実施

2. **動作確認の実施**
   - Moncler run を実行し、PDP リンク抽出を確認
   - trap ページ検出とロケール制御の動作を確認
   - 問題が発生した場合は、ログと DOM スナップショットを確認して切り分け

3. **完了レポートの作成**
   - 実装完了後、`docs/spec/CR-ATELIER-002_MONCLER_PLP_PDP_EXTRACTION_FIX_COMPLETION_REPORT.md` を作成
