# Decision Log: Self-Healing Agent設計の採用

## 決定事項
MonclerのPLP→PDP抽出において、DOM構造の変化やロケールトラップに対応するため、以下の自己修復基盤設計を採用した：

1. **Selector Layering（3層セレクタ戦略）**
   - Layer 1（Primary）: site_config準拠（最も信頼性が高い）
   - Layer 2（Secondary）: DOM構造ベース（フォールバック）
   - Layer 3（Tertiary）: 汎用fallback（最終手段）

2. **Telemetry基盤の設計**
   - `moncler_plp_pdp_outcome`: 抽出結果のサマリー
   - `moncler_pdp_links_debug`: デバッグ用詳細情報

3. **Self-Healing Agent / Selector Discovery Agentとの連携インターフェース定義**

## 判断理由
1. **Monclerサイトの変動性**
   - `/en-lt/en-int/`のような二重ロケールが発生
   - セレクタがDOMと一致しないケースが頻発
   - `/search`などPLP相当ページが増えるなど、従来のPLP想定が破綻

2. **継続的学習の必要性と設計フェーズの分離**
   - Step 5を「設計フェーズ」として位置づけ
   - Step 6以降の実装を可能にする基盤設計を先行実施
   - Telemetry設計により、AIの自己修復に必要な情報を提供

3. **過検知・検知もれのバランス**
   - Layer 1から3へ順にフォールバックすることで、精度とカバレッジを両立
   - Primary→Secondary→Tertiaryの順で防御することで、過検知を抑制

## 代替案
- **案A: 静的なセレクタリストで対処** → MonclerのDOM変化速度に対応不可
- **案B: 失敗時に手動でセレクタを更新** → 自動化ニーズに反する
- **案C: 機械学習で完全自動学習** → 設計コストとリスクが高すぎる

## 日時
2025-12-05（CR-ATELIER-002_STEP5_SPEC起草時点）

## 決定者
Atelier Kyo / NexusCore Line

## 関連リソース
- Spec: `docs/spec/CR-ATELIER-002_STEP5_SPEC.md`
- Spec: `docs/spec/CR-ATELIER-002_MONCLER_PLP_PDP_EXTRACTION_FIX.md`
- コード: `app/agents/browser/navigation_driver.py`
