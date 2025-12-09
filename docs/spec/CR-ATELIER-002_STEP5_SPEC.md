# CR-ATELIER-002 Step 5

Moncler Robust PLP→PDP Extraction — 渡航性・復元性・学習基盤の設計書

**Version:** 1.0

**Status:** 完了（次フェーズ：Step6 実装）

**Author:** AI Assistant (for Atelier Kyo Manager)

## 1. Overview & Context

### 問題点

Moncler公式サイトの PLP → PDP 抽出で発生している問題：

- リダイレクトにより `/en-lt/en-int/` のような二重ロケールが発生
- セレクタが DOM と一致しないケースがある
- `/search` など PLP 相当ページが増え、従来のPLP想定が破綻
- 抽出失敗時の Telemetry 情報が不十分で、AI の自己修復が困難

### Step5 の目的

- DOM・URL・動作特性の整理
- セレクタ戦略の体系化（レイヤリング）
- Locale / Trap / Search のポリシー定義
- Self-Healing & Selector Discovery との連携仕様を作る

**＝ Step6 以降の実装を可能にする「設計フェーズ」**

## 2. Scope

### In Scope（本設計に含む）

- PLP / Search / Trap の DOM 特徴の整理
- PDPリンク抽出のレイヤ戦略（Primary / Secondary / Tertiary）
- URL / Locale / Trap の政策整理
- Telemetry の設計
- Self-Healing / Selector Discovery を動かすためのインタフェース定義

### Out of Scope（実装は Step6）

- Telemetry 記録処理の実コード化
- Self-Healing Agent の本実装
- Selector Discovery Agent の学習パイプライン
- 実ページのスクレイピング・確認

## 3. Architecture & Dataflow

```
NavigationDriver
     │
     ▼
LocaleGuard — TrapDetector
     │
     │ URL OK
     ▼
Extractor (PLP → PDP)
     │
     ├─ Layer1: site_configベース
     ├─ Layer2: DOM構造ベース
     └─ Layer3: 汎用fallback
     │
     ▼
Telemetry
     ├─ moncler_pdp_links_debug
     └─ moncler_plp_pdp_outcome
     │
     ▼
Self-Healing Agent / Selector Discovery Agent
```

## 4. Selector Layering Design

### Layer 1（Primary）

site_config 準拠。最も信頼性が高い。

**例：**
- `article[data-component*='ProductCard'] a[href*='/products/']`
- `[data-testid*='product-card'] a[href*='/products/']`

### Layer 2（Secondary）

DOM上の構造的特徴を利用する。

**例：**
- `a[href*='/products/']:not([class*='breadcrumb'])`
- `section a[href*='/products/']`

### Layer 3（Tertiary: fallback）

最終手段。過検知も許容。

**例：**
- `a[href*='/products/']`

## 5. Locale / Trap Policy

| 種別 | 内容 |
|------|------|
| LocaleGuard | `/en-int/` に揃える。二重ロケールを修正する。 |
| TrapDetector | `/search`（検索トップ）, `/404`, ロケールゲート を除外 |
| PDP URL バリデーション | `origin=moncler.com`, `/en-int/.../products/...` |

**リダイレクト後も URL を再検証する。**

## 6. Telemetry Design

### moncler_plp_pdp_outcome（新規）

```json
{
  "plp_materialized": true/false,
  "tiles_detected": int,
  "pdp_links_raw": int,
  "pdp_links_accepted": int,
  "selector_layers_used": ["primary", ...],
  "locale_corrections": int,
  "trap_detected": false
}
```

### moncler_pdp_links_debug（既存強化）

- `raw_hrefs`
- `rejection_stats`
- `layer_stats`
- `current_url`
- `run_id`

## 7. Self-Healing Interface

### 失敗トリガー（例）

| 失敗種別 | 説明 |
|---------|------|
| raw_zero | raw=0 の場合 |
| secondary_used | Primary が全滅し Secondary に fallback |
| trap_detected | Trap ページ遷移 |
| locale_corrections_exceeded | 修正回数が上限超え |

### Self-Healing Agent へのペイロード

```json
{
  "site": "moncler",
  "failure_type": "...",
  "dom_snapshot": "...",
  "layer_stats": {...},
  "selectors_current": {...}
}
```

### Selector Discovery Agent

**入力：**
- `dom_snapshot`
- `current selectors`
- `failure evidence`

**出力：**
- 新しいセレクタ候補
- 信頼度
- site_configへ反映するパッチ

## 8. Implementation Plan（Step6）

- Telemetry の `moncler_plp_pdp_outcome` 記録実装
- Extractor から Self-Healing エージェントを発火
- Selector Discovery Agent の MVP 実装
- site_config の自動パッチ作成に向けた基盤整備
- pytest（SelectorLayering / FailureDetection / Trap / Locale）の実装

## 9. Testing Strategy

- **pytest**: URLバリデーション、セレクタレイヤ、trap検知
- **実ブラウザ**（WSL + Playwright）で成功 run を複数回確認
- **Telemetry JSON 構造**の静的チェック
- **Self-Healing の単体テスト**：入力→期待ペイロード生成で検証

## 10. Risks & Mitigation

| リスク | 対策 |
|--------|------|
| Moncler側のDOM変化 | Selector Discovery Agentで継続学習 |
| ロケールゲートの強化 | LocaleGuardの強化 |
| 過検知 | Primary→Secondary→Tertiaryの順で防御 |

## 11. Summary

Step5 の設計フェーズは以下を満たして完了：

- ✅ PLP→PDP 抽出ロジックのレイヤ構造が定義済み
- ✅ Locale / Trap Policy が明確
- ✅ Self-Healing / Selector Discovery 連携が仕様化
- ✅ Step6 の実装に必要な設計資料がすべて揃った

