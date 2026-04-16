# Decision Log: Orchestrator分割（BrowserUseAgentのリファクタリング）

## 決定事項
肥大化したBrowserUseAgent（2,761行）を以下の3層に分割することを決定した：

1. **BrowserRunOrchestrator**: 高レベルフロー制御（run、_run_plp_flow、_run_pdp_flowなど）
2. **UiHelpers / UI Helpers**: 低レベルUI操作（cookie.accept、modal.dismissなど）
3. **BrowserUseAgent（Facade）**: 薄いPublic APIラッパー

## 判断理由
1. **責務の混在問題の深刻化**
   - PLP/ナビゲーション処理とPDP抽出処理が同一ファイルに共存
   - UI操作ヘルパーとオーケストレーションロジックが混在
   - 変更の影響範囲が広く、テストが困難になっていた

2. **NavigationDriverとの重複解消**
   - `_ensure_plp_materialized`と`NavigationDriver.ensure_plp_materialized`が重複
   - `_collect_pdp_links`と`NavigationDriver.collect_pdp_links`が重複
   - 5つのメソッドで重複が発生していた

3. **保守性と拡張性の向上**
   - コードレビューが困難になっていた
   - 新規サイト追加時の影響範囲が大きい状態
   - 単一責任の原則を適用し、各コンポーネントの責務を明確化

## 代替案
- **案A: 一切都り合いでBrowserUseAgentを維持** → 技術的負債が蓄積しすぎると判断
- **案B: すべてを小さなファイルに分割しすぎる** → 管理コストが増大すると判断
- **案C: BrowserUseAgentは削除し、新モジュールで完全書き換え** → 互換性維持のため、薄いFacadeとして残す案を採用

## 日時
2025-12-05（CR-ATELIER-003_BROWSER_AGENT_REFACTOR起草時点）

## 決定者
Atelier Kyo / NexusCore Line

## 関連リソース
- Spec: `docs/spec/CR-ATELIER-003_BROWSER_AGENT_REFACTOR.md`
- 完了レポート: `docs/completion_reports/CR_ATELIER_002_STEP3_COMPLETION_REPORT.md`〜`STEP8`
- コード: `app/agents/browser_use_agent.py`
- コード: `app/agents/browser/navigation_driver.py`
