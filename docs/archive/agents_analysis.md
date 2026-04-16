# Atelier-Kyo Manager エージェント群 詳細分析

**生成日時**: 2025-01-XX  
**目的**: 各エージェントの役割と依存関係を整理し、アーキテクチャの理解を深める

---

## 📋 目次

1. [エージェント階層構造](#エージェント階層構造)
2. [コアエージェント（主要タスク実行）](#コアエージェント主要タスク実行)
3. [支援エージェント（メタ・修復・分析）](#支援エージェントメタ修復分析)
4. [データ処理エージェント（永続化・レポート）](#データ処理エージェント永続化レポート)
5. [依存関係マップ](#依存関係マップ)
6. [実行フロー図](#実行フロー図)

---

## エージェント階層構造

```
┌─────────────────────────────────────────────────────────┐
│  Orchestrator (app/utils/ai_research_orchestrator.py)   │
│  最高司令部：全体の作戦を統括                              │
└─────────────────┬───────────────────────────────────────┘
                  │
      ┌───────────┴───────────┐
      │                       │
┌─────▼──────┐        ┌───────▼────────┐
│ Supplier   │        │ Persistence    │
│ ScoutAgent │        │ Agent          │
│ (現場総指揮)│        │ (記録保管官)    │
└─────┬──────┘        └────────────────┘
      │
      ├─→ SelectorDiscoveryAgent (斥候)
      │   └─→ SelfHealingAgent (自己修復)
      │       ├─→ PageRecoveryAgent (物理的回復)
      │       └─→ SelectorRepairAgent (知的修復)
      │
      └─→ BrowserUseAgent (実戦部隊)
          └─→ SessionManager (セッション管理)
```

---

## コアエージェント（主要タスク実行）

### 1. SupplierScoutAgent
**ファイル**: `app/agents/supplier_scout_agent.py`  
**バージョン**: 21.0.0J (Evidential Reporting)  
**役割**: 現場総指揮官。作戦計画を策定・記録し、部隊に任務を割り当て、結果を報告する。

**主要機能**:
- サイトごとの候補URL生成
- エージェント選択（SelectorDiscoveryAgent / BrowserUseAgent）
- 失敗時のFailureAnalysisAgent呼び出し
- DiscoveryResultにrun_idを付与

**依存関係**:
- `FailureAnalysisAgent` - 失敗分析
- `SelectorDiscoveryAgent` - セレクタ発見モード
- `BrowserUseAgent` - 通常実行モード
- `RunContext` - 実行コンテキスト管理
- `DiscoveryResult` - 結果モデル

**呼び出し元**:
- `AiResearchOrchestrator.run()`

---

### 2. BrowserUseAgent
**ファイル**: `app/agents/browser_use_agent.py`  
**役割**: 実戦部隊。Playwrightを使用して実際のブラウザ操作を実行する。

**主要機能**:
- ブラウザセッション管理（SessionManager経由）
- ページナビゲーション
- Cookie同意・モーダル処理
- 商品情報抽出
- 視覚的回帰テスト（VRT）

**依存関係**:
- `SessionManager` - セッション管理
- `NavigationDriver` - ナビゲーション制御
- `extract_product_info` - 商品情報抽出
- `RunContext` - 実行コンテキスト

**呼び出し元**:
- `SupplierScoutAgent.run()` (通常モード)

---

### 3. SelectorDiscoveryAgent
**ファイル**: `app/agents/selector_discovery_agent.py`  
**バージョン**: 27.0.0J (Final Consolidated)  
**役割**: 自己進化ループを備えた、最先端の斥候エージェント。

**主要機能**:
- セレクタ自動発見
- 自己修復ループ（最大N回）
- 失敗時のSelfHealingAgent呼び出し
- 設定の動的更新

**依存関係**:
- `FailureAnalysisAgent` - 失敗分析
- `SelfHealingAgent` - 自己修復統括
- `extract_product_info` - 商品情報抽出
- `RunContext` - 実行コンテキスト

**呼び出し元**:
- `SupplierScoutAgent.run()` (discover_selectors=True時)

---

## 支援エージェント（メタ・修復・分析）

### 4. SelfHealingAgent
**ファイル**: `app/agents/self_healing_agent.py`  
**バージョン**: 9.0.0J (Final Consolidated)  
**役割**: 自己修復オペレーションの現場指揮官。物理的回復と知的修復の2段階戦略を統括。

**主要機能**:
- 戦略1: 物理的回復（PageRecoveryAgent）
- 戦略2: 知的修復（SelectorRepairAgent）
- 失敗コンテキストの分析

**依存関係**:
- `PageRecoveryAgent` - 物理的回復
- `SelectorRepairAgent` - 知的修復（セレクタ提案）
- `RunContext` - 実行コンテキスト

**呼び出し元**:
- `SelectorDiscoveryAgent.run()` (失敗時)

---

### 5. PageRecoveryAgent
**ファイル**: `app/agents/page_recovery_agent.py`  
**バージョン**: 8.1.0J (Survival Protocol + UI Fallback)  
**役割**: 物理的なページ回復を試みる工兵部隊。自己位置回復とサバイバル能力を持つ。

**主要機能**:
- ページ戻り（goBack）
- Cookie同意処理
- UI導線フォールバック（検索アイコン、ハンバーガーメニュー）
- PLP安定化（attached→visible待機、マイクロスクロール）

**依存関係**:
- `RunContext` - 実行コンテキスト
- Playwright `Page` - ブラウザ操作

**呼び出し元**:
- `SelfHealingAgent.execute()` (戦略1)

---

### 6. SelectorRepairAgent
**ファイル**: `app/agents/selector_repair_agent.py`  
**バージョン**: 8.0.0J (Async Fix)  
**役割**: 情報分析官。AIを用いて壊れたセレクタを修復・提案する。

**主要機能**:
- LLMによるセレクタ提案
- 失敗したセレクタの分析
- 代替セレクタの生成
- 提案のJSON保存

**依存関係**:
- `AILlmController` (スタブ実装) - LLM呼び出し
- `RunContext` - 実行コンテキスト

**呼び出し元**:
- `SelfHealingAgent.execute()` (戦略2)

---

### 7. FailureAnalysisAgent
**ファイル**: `app/agents/failure_analysis_agent.py`  
**バージョン**: 7.0.0J (Forensic Reporting)  
**役割**: エージェントの失敗を分析し、原因と対策を提案するメタエージェント。

**主要機能**:
- エラーの診断・原因分析
- 推奨アクションの提案
- 統合フォレンジックレポート生成
- 証跡リンクの保存（Trace, HAR, Logs, Screenshots）

**依存関係**:
- `AILlmController` (スタブ実装) - LLM呼び出し
- `RunContext` - 実行コンテキスト

**呼び出し元**:
- `SupplierScoutAgent.run()` (例外時)
- `SelectorDiscoveryAgent.run()` (失敗時)

---

### 8. CodeUpdateAgent
**ファイル**: `app/agents/code_update_agent.py`  
**バージョン**: 1.3.2-apply-orchestrator  
**役割**: 設定ファイル（overrides.local.json）へのパッチ適用と監査ログ保存。

**主要機能**:
- AI提案の適用（dry_run可）
- 監査ログの保存（exports/patches/）
- unified diffの生成
- Git staging（オプション）

**依存関係**:
- なし（独立）

**呼び出し元**:
- `ProxyScrapeAgent._self_evolve()` (自己進化時)

---

### 9. GPTIntegration
**ファイル**: `app/agents/gpt_integration.py`  
**バージョン**: 1.2.2-proposal-save-integrated  
**役割**: LLM呼び出しの強化版ファサード。設定主導、スキーマ検証、リトライ機能。

**主要機能**:
- LLM呼び出し（AILlmController経由）
- 出力JSONスキーマ検証
- 指数バックオフ・再試行
- PII/DOMの過剰流出防止
- 提案の自動保存

**依存関係**:
- `AILlmController` - LLM制御
- `app/utils/ai_llm_controller.py`

**呼び出し元**:
- `ProxyScrapeAgent._self_evolve()` (自己進化時)

---

### 10. ProxyScrapeAgent
**ファイル**: `app/agents/proxy_scrape_agent.py`  
**バージョン**: 1.5.0-apply-integration  
**役割**: Orchestratorの実行をラップし、失敗時の自己進化ループを実装。

**主要機能**:
- Orchestrator.run()の実行
- 失敗時のSelfHealingAgent呼び出し
- CodeUpdateAgent.apply()による設定更新
- 指数バックオフ付き再試行

**依存関係**:
- `CodeUpdateAgent` - 設定更新
- `GPTIntegration` - LLM呼び出し（間接的）

**呼び出し元**:
- 外部スクリプト（未確認）

---

## データ処理エージェント（永続化・レポート）

### 11. PersistenceAgent
**ファイル**: `app/agents/persistence_agent.py`  
**バージョン**: 1.1 (2025-09-10)  
**役割**: 収集データの永続化専任エージェント。CSVとDBをStrategyで切替。

**主要機能**:
- CSV保存（output/price_history_*.csv）
- DB保存（ProductPriceHistory、EMA計算）
- Strategy パターン（CsvSink / DbSink）

**依存関係**:
- `app.models.Product` (オプション)
- `app.models.ProductPriceHistory` (オプション)
- `app.extensions.db` (オプション)

**呼び出し元**:
- `AiResearchOrchestrator.run()` (成功時)

---

### 12. ReportingAgent
**ファイル**: `app/agents/reporting_agent.py`  
**バージョン**: 3.0.0J (Intelligence Briefing)  
**役割**: 作戦結果を集約・分析し、人間が意思決定を行うためのレポートを生成する情報将校。

**主要機能**:
- HTMLレポート生成（run_summary_*.html）
- 成功/失敗の集計
- AI分析結果の統合（ai_forensic_report.json読み込み）
- モダンなデザインのレポート

**依存関係**:
- `DiscoveryResult` - 結果モデル

**呼び出し元**:
- `AiResearchOrchestrator.run()` (作戦完了後)

---

## その他のエージェント

### 13. PriceIntelligenceAgent
**ファイル**: `app/agents/price_intelligence_agent.py`  
**バージョン**: 13.0.0J (Config-Driven Timeout)  
**役割**: Seleniumベースの価格情報収集（レガシー？）

**依存関係**:
- Selenium WebDriver
- `extract_product_info`

---

### 14. ProfitabilityAgent
**ファイル**: `app/agents/profitability_agent.py`  
**バージョン**: 2.0 (2025-09-08)  
**役割**: 商品の収益性を多角的に分析・評価する専門エージェント。

**依存関係**:
- `AILlmController` - LLM要約生成
- `ShippingAgent` - 送料計算
- `fx_utils` - 為替レート取得
- Pydantic - データ検証

---

### 15. AIVisionAgent
**ファイル**: `app/agents/ai_vision_agent.py`  
**役割**: パーセル画像検査（スタブ実装）

**依存関係**:
- なし（現状スタブ）

---

### 16. LearningProbe
**ファイル**: `app/agents/learning_probe.py`  
**役割**: 偵察モードで「価格とリンクの在処」を学習し、learned_selectors.jsonに保存

**依存関係**:
- Playwright `Page`

---

## 依存関係マップ

### コア依存（全エージェント共通）
```
app/core/run_context.py (RunContext)
  └─ 実行コンテキスト管理、スクリーンショット、ログ保存

app/models/result_models.py
  ├─ DiscoveryResult (偵察結果)
  └─ GenerateResult (LLM応答)

app/config/loader.py, config.py
  └─ サイト設定の読み込み
```

### エージェント間依存
```
SupplierScoutAgent
  ├─→ FailureAnalysisAgent
  ├─→ SelectorDiscoveryAgent
  │     ├─→ FailureAnalysisAgent
  │     ├─→ SelfHealingAgent
  │     │     ├─→ PageRecoveryAgent
  │     │     └─→ SelectorRepairAgent
  │     │           └─→ AILlmController (スタブ)
  │     └─→ extract_product_info
  └─→ BrowserUseAgent
        ├─→ SessionManager
        │     └─→ RunContext
        ├─→ NavigationDriver
        └─→ extract_product_info

PersistenceAgent
  ├─→ app.models.Product (オプション)
  └─→ app.extensions.db (オプション)

ReportingAgent
  └─→ DiscoveryResult

ProxyScrapeAgent
  ├─→ CodeUpdateAgent
  └─→ GPTIntegration
        └─→ AILlmController
```

---

## 実行フロー図

### 通常実行フロー
```
Orchestrator.run()
  │
  ├─→ SupplierScoutAgent.run()
  │     │
  │     ├─→ BrowserUseAgent.run()
  │     │     ├─→ SessionManager (セッション初期化)
  │     │     ├─→ NavigationDriver (ナビゲーション)
  │     │     └─→ extract_product_info (商品情報抽出)
  │     │
  │     └─→ DiscoveryResult生成
  │
  ├─→ PersistenceAgent.snapshot_full_results() (成功時)
  │
  └─→ ReportingAgent.build_run_summary_report()
```

### セレクタ発見モード
```
Orchestrator.run()
  │
  └─→ SupplierScoutAgent.run()
        │
        └─→ SelectorDiscoveryAgent.run()
              │
              ├─→ extract_product_info (試行)
              │
              └─→ 失敗時
                    │
                    └─→ SelfHealingAgent.execute()
                          │
                          ├─→ PageRecoveryAgent.execute() (戦略1)
                          │
                          └─→ SelectorRepairAgent.propose_fix() (戦略2)
                                │
                                └─→ AILlmController.generate()
```

### 失敗分析フロー
```
SupplierScoutAgent.run()
  │
  └─→ 例外発生
        │
        └─→ FailureAnalysisAgent.analyze()
              │
              ├─→ AILlmController.generate() (診断)
              │
              └─→ ai_forensic_report.json保存
                    │
                    └─→ ReportingAgent (レポートに統合)
```

---

## 重要な設計パターン

### 1. Strategy パターン
- `PersistenceAgent`: CsvSink / DbSink の切替
- `PluginRuntime`: サイト別プラグインの動的ロード

### 2. Chain of Responsibility
- `SelfHealingAgent`: 物理的回復 → 知的修復の順で試行

### 3. Factory パターン
- `SupplierScoutAgent`: 実行モードに応じてエージェントを選択

### 4. Observer パターン（部分的）
- `RunContext`: スクリーンショット、ログの自動保存

---

## 注意事項

### スタブ実装
- `SelectorRepairAgent` 内の `AILlmController` はスタブ
- `FailureAnalysisAgent` 内の `AILlmController` はスタブ
- 実際の実装は `app/utils/ai_llm_controller.py` を参照

### 非同期処理
- すべてのエージェントの主要メソッドは `async def`
- `await` の漏れに注意（特に `RunContext.take_screenshot()`）

### 設定の優先順位
1. `runtime_kwargs` (実行時引数)
2. `site_config.discovery_settings` (サイト設定)
3. デフォルト値

---

## まとめ

atelier-kyo-manager のエージェント群は、**階層的な指揮系統**と**自己修復能力**を持つ、高度に組織化されたマルチエージェントシステムです。

- **コアエージェント**: 実際のタスク実行（SupplierScout, BrowserUse, SelectorDiscovery）
- **支援エージェント**: 修復・分析・学習（SelfHealing, FailureAnalysis, SelectorRepair）
- **データ処理エージェント**: 永続化・レポート（Persistence, Reporting）

各エージェントは独立して動作可能ですが、`RunContext` と `DiscoveryResult` を通じて情報を共有し、協調的に動作します。

