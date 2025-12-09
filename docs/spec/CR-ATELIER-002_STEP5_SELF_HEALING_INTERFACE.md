# CR-ATELIER-002 Step 5-4: Self-Healing / Selector Discovery 連携設計

- **Status:** Design Document
- **Author:** [AI Assistant]
- **Date:** 2025-12-08
- **Related CR:** CR-ATELIER-002_STEP5_MONCLER_ROBUST_EXTRACTION_DESIGN.md

## 1. Telemetry キーの定義

### 1.1 Moncler PLP/PDP 関連メトリクス

**キー名**: `moncler_plp_pdp_outcome`

**データ構造**:
```json
{
  "plp_materialized": true,
  "tiles_detected": 12,
  "pdp_links_raw": 15,
  "pdp_links_accepted": 10,
  "selector_layers_used": {
    "primary": {"raw": 15, "accepted": 10},
    "secondary": {"raw": 0, "accepted": 0},
    "tertiary": {"raw": 0, "accepted": 0}
  },
  "locale_corrections": 2,
  "trap_detected": false,
  "current_url": "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/",
  "run_id": "20251208_150906_778"
}
```

**フィールド説明**:
- `plp_materialized`: PLP がマテリアライズされたか
- `tiles_detected`: 検出された商品タイル数
- `pdp_links_raw`: 抽出された生の PDP リンク数
- `pdp_links_accepted`: バリデーションを通過した PDP リンク数
- `selector_layers_used`: 各レイヤで使用されたセレクタの統計
- `locale_corrections`: Locale 補正の回数
- `trap_detected`: Trap ページが検出されたか
- `current_url`: 現在の URL
- `run_id`: 実行 ID

### 1.2 デバッグ情報

**キー名**: `moncler_pdp_links_debug`（既存）

**データ構造**:
```json
{
  "raw_hrefs": ["/en-int/products/xxx", ...],
  "rejection_stats": {
    "origin": 2,
    "locale": 1,
    "path": 1,
    "trap": 0,
    "other": 1
  },
  "rejection_details": {
    "no_href": 0,
    "external_domain": 2,
    "blocked_domain": 0,
    "double_locale_path": 1,
    "no_en_int_path": 0,
    "no_products_path": 1,
    "trap_pattern": 0,
    "other": 1
  },
  "layer_stats": {
    "primary_raw": 15,
    "primary_accepted": 10,
    "secondary_raw": 0,
    "secondary_accepted": 0,
    "tertiary_raw": 0,
    "tertiary_accepted": 0
  },
  "current_url": "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/",
  "run_id": "20251208_150906_778"
}
```

## 2. Self-Healing Agent へのインタフェース

### 2.1 タスクペイロード設計

**タスクタイプ**: `moncler_plp_pdp_extraction_failure`

**ペイロード構造**:
```json
{
  "task_type": "moncler_plp_pdp_extraction_failure",
  "site": "MONCLER_OFFICIAL",
  "run_id": "20251208_150906_778",
  "failure_type": "raw_zero" | "trap_detected" | "locale_corrections_exceeded",
  "telemetry_keys": {
    "outcome": "moncler_plp_pdp_outcome",
    "debug": "moncler_pdp_links_debug"
  },
  "context": {
    "current_url": "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/",
    "query": "down jacket",
    "site_config": {...},
    "dom_snapshot": "plp_dom_initial_materialized.html",
    "selector_counts": "selector_counts_plp_initial.json"
  },
  "failure_details": {
    "raw_hrefs_count": 0,
    "accepted_hrefs_count": 0,
    "rejection_stats": {...},
    "layer_stats": {...},
    "locale_corrections": 3,
    "trap_detected": false
  }
}
```

### 2.2 失敗タイプの定義

**`raw_zero`**:
- `pdp_links_raw == 0` が一定回数連続
- セレクタが要素を見つけられていない

**`trap_detected`**:
- Trap ページが検出された
- 404 ページ、ロケールゲート、検索トップページ

**`locale_corrections_exceeded`**:
- Locale 補正が一定回数（例: 3回）を超えた
- 再リダイレクトが繰り返される

## 3. Selector Discovery Agent へのインタフェース

### 3.1 タスクペイロード設計

**タスクタイプ**: `moncler_plp_selector_discovery`

**ペイロード構造**:
```json
{
  "task_type": "moncler_plp_selector_discovery",
  "site": "MONCLER_OFFICIAL",
  "run_id": "20251208_150906_778",
  "intent": "Moncler PLP から PDP リンクを抽出する",
  "failed_selectors": [
    "article[data-component*='ProductCard'] a[href*='/products/']",
    "[data-testid*='product-card'] a[href*='/products/']"
  ],
  "current_selectors": {
    "primary": [...],
    "secondary": [...],
    "tertiary": [...]
  },
  "dom_snapshot": "plp_dom_initial_materialized.html",
  "selector_counts": "selector_counts_plp_initial.json",
  "context": {
    "current_url": "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/",
    "query": "down jacket",
    "site_config": {...}
  }
}
```

### 3.2 期待される出力

**出力構造**:
```json
{
  "ok": true,
  "proposed_selectors": [
    {
      "layer": "primary" | "secondary" | "tertiary",
      "selector": "article[data-component='ProductCard'] a[href*='/products/']",
      "confidence": 0.95,
      "evidence": {
        "matched_count": 12,
        "sample_urls": ["/en-int/products/xxx", ...]
      }
    }
  ],
  "message": "12件の代替セレクタを提案します。"
}
```

## 4. エージェント間の連携フロー

### 4.1 失敗検出 → Self-Healing Agent

1. **Telemetry から失敗を検出**:
   - `moncler_plp_pdp_outcome` を確認
   - `pdp_links_raw == 0` または `trap_detected == true` の場合

2. **タスクペイロードを生成**:
   - 失敗タイプを判定
   - Telemetry から必要な情報を取得
   - タスクペイロードを構築

3. **Self-Healing Agent にタスクを渡す**:
   - `self_healing_agent.execute()` を呼び出す
   - タスクペイロードを渡す

### 4.2 Self-Healing Agent → Selector Discovery Agent

1. **物理的回復を試みる**:
   - `PageRecoveryAgent` でページを回復
   - 成功した場合は終了

2. **知的修復を試みる**:
   - `SelectorRepairAgent` でセレクタを修復
   - 成功した場合は終了

3. **Selector Discovery Agent にタスクを渡す**:
   - `selector_discovery_agent.run()` を呼び出す
   - タスクペイロードを渡す

### 4.3 Selector Discovery Agent → セレクタ提案

1. **DOM スナップショットを分析**:
   - `plp_dom_initial_materialized.html` を読み込む
   - DOM 構造を分析

2. **セレクタ候補を生成**:
   - LLM またはルールベースでセレクタを生成
   - 各セレクタの信頼度を計算

3. **セレクタ提案を返す**:
   - 提案されたセレクタを返す
   - Telemetry に記録

## 5. 実装方針

### 5.1 Telemetry への記録

1. **`moncler_plp_pdp_outcome` の記録**:
   - `navigation_driver.py` の `run_plp_flow()` で記録
   - PLP マテリアライズ後、PDP 抽出後、Trap 検出後に記録

2. **`moncler_pdp_links_debug` の記録**:
   - `extractor.py` の `extract_moncler_pdp_links()` で記録
   - `accepted == 0` の場合に記録

### 5.2 Self-Healing Agent の拡張

1. **Moncler 専用の失敗検出ロジック**:
   - `moncler_plp_pdp_extraction_failure` タスクタイプを追加
   - Telemetry から失敗を検出するロジックを追加

2. **タスクペイロードの生成**:
   - Telemetry から必要な情報を取得
   - タスクペイロードを構築

### 5.3 Selector Discovery Agent の拡張

1. **Moncler 専用のセレクタ発見ロジック**:
   - `moncler_plp_selector_discovery` タスクタイプを追加
   - DOM スナップショットを分析するロジックを追加

2. **セレクタ提案の生成**:
   - LLM またはルールベースでセレクタを生成
   - 各セレクタの信頼度を計算

## 6. 次のステップ

1. **実装**:
   - Telemetry への `moncler_plp_pdp_outcome` 記録を実装
   - Self-Healing Agent の Moncler 専用ロジックを実装
   - Selector Discovery Agent の Moncler 専用ロジックを実装

2. **テスト**:
   - 失敗検出のテスト
   - タスクペイロード生成のテスト
   - セレクタ提案のテスト

