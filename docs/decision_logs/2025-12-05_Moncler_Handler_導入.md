# Decision Log: Moncler専用Handlerの導入

## 決定事項
Moncler公式サイト専用のHandler（`MonclerDrissionHandler`、`MonclerPLPStrategy`など）を導入し、BrowserUseAgent内に散在するMoncler固有ロジックを集約することを決定した。

## 判断理由
1. **責務の混在問題の解決**
   - BrowserUseAgentが2,761行の巨大ファイルとなり、PLP/ナビゲーション処理とPDP抽出処理が混在していた
   - Moncler固有の`MONCLER_OFFICIAL`分岐が42箇所に散在し、保守性が低下していた

2. **サイト固有対応の分離必要性**
   - Monclerはロケールトラップ（`/en-lt/en-int/`二重ロケール）、404ページ、location gateなどの特有問題を抱えていた
   - 他ブランド（jtcmerson.jp、 Acne Studiosなど）との共存のため、コードの分離が必須であった

3. **段階的アプローチの採用**
   - Step 1〜8の段階でNavigationDriver、Self-Healing Agent、Selector Discovery Agentが実装済み
   - これらを活かしながら、Moncler固有ロジックを専用モジュールに集約する段階に来た

## 代替案
- **案A: すべてをsite_configのJSONで完結させる** → 複雑なロケール制御やDOM構造の変化にはコード制御が必要と判断
- **案B: BrowserUseAgent内にif分岐を残したまま保守** → 42箇所の分岐管理は非現実的と判断

## 日時
2025-12-05（CR-ATELIER-002_MONCLER_PLP_PDP_EXTRACTION_FIX起草時点）

## 決定者
Atelier Kyo / NexusCore Line

## 関連リソース
- Spec: `docs/spec/CR-ATELIER-002_MONCLER_PLP_PDP_EXTRACTION_FIX.md`
- Spec: `docs/spec/CR-ATELIER-002_STEP5_SPEC.md`
- コード: `app/agents/plugins/moncler_plp_v1.py`
- コード: `app/config/sites/overrides.local.json`
