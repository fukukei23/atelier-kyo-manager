# CR-ATELIER-002 Step 7

Moncler site_config パッチ適用フロー — 提案結果の安全な反映

**Version:** 1.0

**Status:** Draft (Step 7 Spec)

**Owner:** Atelier Kyo / NexusCore Line

**Related Steps:** Step 3 / Step 4 / Step 5 / Step 6

**Related Docs:**

- `docs/spec/CR-ATELIER-002_STEP5_SPEC.md`
- `docs/spec/CR-ATELIER-002_STEP5_MONCLER_ROBUST_EXTRACTION_DESIGN.md`
- `docs/spec/CR-ATELIER-002_STEP5_SELF_HEALING_INTERFACE.md`
- `docs/completion_reports/CR_ATELIER_002_STEP4_COMPLETION_REPORT.md`
- `docs/completion_reports/CR_ATELIER_002_STEP5_COMPLETION_REPORT.md`
- `docs/completion_reports/CR_ATELIER_002_STEP6_COMPLETION_REPORT.md`

---

## 1. Overview & Context

### 1.1 背景

CR-ATELIER-002 Step 3〜6 までにおいて、Moncler 公式サイト向けの PLP→PDP 抽出ロジックについて以下を実装済み：

- `site_config`（`overrides.local.json`）と PLP/PDP 抽出コードの整合性改善
- Locale / Trap / TrapDetector / LocaleGuard の連携強化
- Telemetry による `moncler_plp_pdp_outcome` の構造化記録
- Self-Healing Agent / Selector Discovery Agent による
  - 失敗要因分析
  - セレクタ候補の自動提案
- Moncler 専用のテスト群（pytest）によるカバレッジ確保

現時点（Step 6）では、**「提案」までは自動化されているが、`site_config` への反映（パッチ適用）は人手ベースで ad-hoc** になっている。

### 1.2 Step 7 の目的

Step 7 の目的は、以下を満たす **「安全な site_config パッチ適用フロー」** を設計・実装すること。

1. Self-Healing / Selector Discovery の結果を
   - 一定のフォーマットの **パッチ候補 JSON** として出力する
   - 必要に応じて Markdown レポートとして人間がレビューしやすい形に整形する

2. `app/config/sites/overrides.local.json` への変更は
   - **直接書き換えず**
   - 「パッチ候補ファイル」を介して、明示的な承認（手動適用）でのみ行う

3. すべての変更は Git 管理・レビュー対象とし、追跡可能にする

---

## 2. Scope (In-Scope / Out-of-Scope)

### 2.1 In-Scope（Step 7 でやること）

1. **提案結果の永続化フロー設計・実装**
   - 入力：`moncler_plp_pdp_outcome` + Self-Healing + Selector Discovery の結果
   - 出力：
     - `instance/self_healing/moncler/<RUN_ID>_analysis.json`
     - `instance/self_healing/moncler/<RUN_ID>_patch_candidate.json`
     - （オプション）`docs/reports/MONCLER_PATCH_<RUN_ID>.md`

2. **パッチ候補ファイルのフォーマット設計**
   - `overrides.local.json` の **差分だけを表現する JSON 形式** を定義する
   - 対象は **MONCLER_OFFICIAL ブロックのみ**
   - 「どのフィールドを、どの値に変更するか」を明示する構造にする

3. **パッチ候補生成ロジックの実装**
   - Self-Healing / Selector Discovery の結果から
     - `selectors.plp` / `selectors.pdp` / `navigation.trap_url_patterns` 等の候補値を抽出
   - 既存 `overrides.local.json` の MONCLER_OFFICIAL ブロックを読み込み
   - 差分（before / after）を計算し、パッチ JSON として出力

4. **人間がレビューしやすいレポート生成**
   - Markdown レポート（任意だが推奨）として
     - 変更前/後セレクタ
     - Self-Healing 分析結果の要約
     - 推奨適用可否（例：SAFE / REVIEW REQUIRED）
   - `docs/reports/` 配下に保存するフローを設計

5. **テスト**
   - パッチ生成ロジックのユニットテスト
   - 擬似的な Self-Healing/Selector 提案から、期待する JSON パッチが出ることの検証

### 2.2 Out-of-Scope（Step 7 ではやらないこと）

- **自動適用**：
  - 人間の承認なしに `overrides.local.json` を直接書き換える処理
- **Moncler 以外サイトへの一般化**：
  - Step 7 時点では **MONCLER_OFFICIAL 専用** とする
- **Web UI / GUI によるパッチ承認画面**
  - Streamlit / Flask UI でのパッチレビュー UI は将来の Step (Step 8+) の対象とする

---

## 3. Current Architecture (Step 6 時点の整理)

### 3.1 データフロー（現状）

1. `NavigationDriver.run_plp_flow()` 実行
2. `extract_moncler_pdp_links()` が
   - `layer_stats, rejection_stats, layers_used` を含む outcome を生成
3. `NavigationOutcome` に
   - `moncler_outcome`
   - `locale_corrections`
   を格納
4. `TelemetryService.record_moncler_plp_pdp_outcome()` が
   - `moncler_plp_pdp_outcome` として Telemetry に保存
5. Self-Healing / Selector Discovery が
   - `moncler_outcome` + `rejection_stats` + `selectors_current` + DOM snapshot path
   を入力として、**分析結果・セレクタ候補を返却**

### 3.2 Step 6 までの限界

- Self-Healing / Selector Discovery の結果は
  - プロセスメモリ上のオブジェクト、もしくはログにとどまる
- `site_config` への反映手順は
  - 「人間がログ・コードを見て手作業で書き換える」状態
- 変更内容が Git 上で一貫したフォーマットで追跡しづらい

---

## 4. Target Architecture (Step 7)

### 4.1 全体フロー

1. **Run 実行**
   - `python -m app.scripts.run_site moncler --query "down jacket" ...`

2. **Telemetry & Self-Healing トリガ**
   - `NavigationDriver` が `moncler_outcome` を Telemetry に保存
   - 失敗条件に合致した場合、Self-Healing / Selector Discovery にタスクを発行

3. **Self-Healing / Selector Discovery 実行**
   - `SelfHealingAgent.handle_moncler_failure(...)`
   - `SelectorDiscoveryAgent.propose_moncler_selectors(...)`

4. **Step 7 パッチ生成モジュール**
   - 新ユーティリティ（例: `app/agents/moncler_patch_builder.py` または `app/core/moncler_patch_builder.py`）
   - 入力：
     - `moncler_plp_pdp_outcome`
     - Self-Healing / Selector Discovery の結果
     - 現行 `overrides.local.json` の MONCLER_OFFICIAL ブロック
   - 出力：
     - `instance/self_healing/moncler/<RUN_ID>_analysis.json`
     - `instance/self_healing/moncler/<RUN_ID>_patch_candidate.json`
     - （任意）`docs/reports/MONCLER_PATCH_<RUN_ID>.md`

5. **人間レビュー & 手動適用**
   - 開発者が `*_patch_candidate.json` とレポートを確認
   - 必要に応じて `overrides.local.json` を編集（Git 管理）

### 4.2 ファイル構成（案）

```text
instance/
  self_healing/
    moncler/
      RUN_20251208_XXXX_analysis.json
      RUN_20251208_XXXX_patch_candidate.json

docs/
  reports/
    MONCLER_PATCH_20251208_XXXX.md
```

---

## 5. Data Model & File Format

### 5.1 Self-Healing 分析結果（analysis.json）

```json
{
  "run_id": "20251208_123456_789",
  "site": "MONCLER_OFFICIAL",
  "timestamp": "2025-12-08T12:34:56Z",
  "current_url": "https://www.moncler.com/en-int/...",
  "moncler_outcome": {
    "plp_materialized": false,
    "tiles_detected": true,
    "pdp_links_raw": 0,
    "pdp_links_accepted": 0,
    "layer_stats": {
      "primary": { "raw": 0, "accepted": 0 },
      "secondary": { "raw": 0, "accepted": 0 },
      "tertiary": { "raw": 0, "accepted": 0 }
    },
    "rejection_stats": {
      "external_domain": 0,
      "blocked_domain": 0,
      "no_en_int_path": 0,
      "no_products_path": 0,
      "double_locale_path": 3,
      "trap_pattern": 1,
      "other": 0
    },
    "locale_corrections": 3,
    "trap_detected": true
  },
  "self_healing": {
    "analysis": "PLP→PDP 抽出に失敗した主原因は double locale redirect...",
    "root_cause": "locale_redirect_misconfig",
    "suggested_actions": [
      {
        "id": "update_trap_url_patterns",
        "description": "/en-lt/en-int/search を trap に追加する...",
        "risk_level": "LOW"
      }
    ]
  },
  "selector_discovery": {
    "candidate_selectors": [
      "article[data-component*='ProductCard'] a[href*='/products/']",
      "div[data-testid='product-card'] a[href*='/products/']"
    ],
    "recommended_layer": "primary",
    "confidence_scores": {
      "article[data-component*='ProductCard'] a[href*='/products/']": 0.87
    }
  }
}
```

### 5.2 パッチ候補ファイル（patch_candidate.json）

```json
{
  "target_file": "app/config/sites/overrides.local.json",
  "site": "MONCLER_OFFICIAL",
  "run_id": "20251208_123456_789",
  "timestamp": "2025-12-08T12:34:56Z",
  "summary": "Moncler PLP→PDP 抽出失敗に対する selector/trap パッチ提案",
  "changes": {
    "selectors.plp": {
      "pdp_link_selectors": {
        "before": ["a[href*='/products/']"],
        "after": [
          "article[data-component*='ProductCard'] a[href*='/products/']",
          "div[data-testid='product-card'] a[href*='/products/']"
        ]
      },
      "tile_selectors": {
        "before": [
          "div:has(a[href$='.html'])",
          "section[role='region'] .product-list a[href$='.html']"
        ],
        "after": [
          "article[data-component*='ProductCard']",
          "div[data-testid='product-card']"
        ]
      }
    },
    "navigation.trap_url_patterns": {
      "append": [
        "/en-lt/en-int/search",
        "/en-de/en-int/search"
      ]
    }
  },
  "risk_assessment": {
    "overall": "MEDIUM",
    "notes": "PLP DOM が変化している可能性があり、本番反映前に 1 回ブラウザ検証を推奨。"
  }
}
```

※ `changes` は以下の原則に従う：

- `before` / `after` の両方を含める（差分が明示されるようにする）
- 追加のみの場合は `append` を使用
- 削除のみの場合は `remove` を使用（Step 7 では削除は原則慎重、必要最小限）

### 5.3 パッチレポート（Markdown, 任意）

`docs/reports/MONCLER_PATCH_<RUN_ID>.md` のテンプレート（概要のみ）：

- Run 情報（run_id, 日時, URL）
- 失敗の概要（Self-Healing 分析要約）
- 提案パッチ一覧（before / after）
- 推奨度（OK / 要レビュー / 危険）

---

## 6. Implementation Plan (Step 7)

### Step7-1: ストレージレイアウト定義

`instance/self_healing/moncler/` を Self-Healing 関連の保存ディレクトリとして利用

保存ファイル：

- `<RUN_ID>_analysis.json`
- `<RUN_ID>_patch_candidate.json`

### Step7-2: パッチ生成ユーティリティの実装

新規モジュール（例）：

- `app/agents/moncler_patch_builder.py`

主な関数：

- `build_moncler_analysis_payload(...)`
  - 入力：moncler_outcome, self-healing result, selector-discovery result
  - 出力：analysis.json 相当の dict

- `build_moncler_patch_candidate(...)`
  - 入力：
    - analysis.json の dict
    - 現行 `overrides.local.json` の MONCLER_OFFICIAL ブロック
  - 出力：patch_candidate.json 相当の dict

NavigationDriver または SelfHealingAgent から呼び出す切り分けは以下の方針：

- 失敗検出 → Self-Healing/Selector Discovery 呼び出し → 結果を moncler_patch_builder に渡し、ファイル保存まで行う

### Step7-3: ファイル保存・レポート生成ロジック

moncler_patch_builder 内で：

- `instance/self_healing/moncler/*.json` を保存
- 必要に応じて Markdown レポートも生成（v1 ではオプション）

### Step7-4: 手動適用フローのガイド文書化

`docs/official/` または `docs/spec/` に

「patch_candidate.json を確認し、overrides.local.json にどう反映するか」の手順書を追加

例：

- before/after を確認
- 変更範囲がトラップページやロケール部分に限定されていることをチェック
- git diff で変更内容を確認
- pytest / 実ブラウザ確認を行う

---

## 7. Testing Strategy

### 7.1 ユニットテスト

新規テストファイル（例）：

- `tests/test_moncler_patch_builder.py`

テスト観点：

- `build_moncler_patch_candidate` に対して
  - selector の before/after が期待どおりに構築されるか
  - trap_url_patterns の append が想定通りか
  - 不正な提案（外部ドメインや /products/ を含まないパス等）がフィルタされること

### 7.2 疑似インテグレーションテスト

テスト用の moncler_outcome / Self-Healing / Selector Discovery 出力をモック

patch_candidate.json を生成し、その JSON をもとに

手作業で overrides.local.json に反映

その後 `python -m pytest tests/test_moncler_pdp_url.py` が通ることを確認

### 7.3 回帰テスト

Step 3〜6 で作成した既存テスト（31件）がすべて引き続きパスすること

Step 7 実装により、moncler_plp_pdp_outcome など既存 Telemetry スキーマが破壊されていないこと

---

## 8. Risks & Mitigation

### 8.1 リスク

**パッチ候補の品質が不十分**
→ 誤った selector を before / after に含めてしまう可能性

**ストレージ膨張**
→ `instance/self_healing/moncler/` 以下に JSON が増加し続ける

**仕様変更時の互換性問題**
→ Step 6 以前に保存された moncler_plp_pdp_outcome と、新しいパッチビルダーの仕様が噛み合わない可能性

### 8.2 緩和策

**リスク1：**
- Step 7 ではあくまで「提案」レベルに留める（自動適用しない）
- `risk_assessment.overall` を必須フィールドとし、HIGH の場合は特に慎重にレビューする運用

**リスク2：**
- 直近 N 件のみ保持、それ以前は手動・スクリプトで削除
- 将来的に `tools/cleanup_self_healing_logs.py` のようなユーティリティ追加を検討

**リスク3：**
- analysis.json / patch_candidate.json のバージョンフィールドを追加し、将来のバージョン管理に備える

---

## 9. Acceptance Criteria (Step 7 完了条件)

Step 7 が完了したと見なす条件は以下：

1. Moncler run 実行時に失敗条件を満たすと、
   - `instance/self_healing/moncler/<RUN_ID>_analysis.json`
   - `instance/self_healing/moncler/<RUN_ID>_patch_candidate.json`
   が自動生成されること

2. `patch_candidate.json` を人間が見れば、
   - どのフィールドをどう変える案なのかが明確に理解できること
   - 変更前/後が一目で分かること

3. 少なくとも 1 つの実 run から得られた `patch_candidate.json` をもとに
   - 人手で `overrides.local.json` を編集
   - pytest / 実ブラウザ検証で成功することを確認できること

4. 既存の Step 3〜6 のテストがすべてパスすること

---

## 10. 今後の拡張（Step 8 以降候補）

- Moncler 以外サイトへの一般化（セレクタパッチの共通フォーマット化）
- Streamlit / Flask ダッシュボードでのパッチレビュー UI
- Approval Agent による半自動承認フロー
- Telemetry ベースの「サイトごとのヘルススコア」可視化

