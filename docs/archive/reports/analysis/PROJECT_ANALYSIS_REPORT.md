# Atelier-Kyo Manager プロジェクト分析レポート
## BUYMA Growth Hub / Multi-Agent Automation System モダナイゼーション準備

**生成日時**: 2025-01-XX  
**目的**: プロジェクト全体の構造分析、ボトルネック特定、モジュール化戦略の提案

---

## 📊 エグゼクティブサマリー

atelier-kyo-manager は、**マルチエージェント型のWebスクレイピング・自動化システム**です。FlaskベースのWebアプリケーションと、25個以上の専門エージェントが協調動作する複雑なアーキテクチャを持っています。

### 主要な特徴
- **マルチエージェントアーキテクチャ**: 階層的な指揮系統を持つ自己修復型エージェント群
- **ブラウザ自動化**: Playwrightベースの高度なナビゲーション制御
- **AI統合**: LLM（Gemini/OpenAI）による分析・修復・レポート生成
- **自己進化機能**: 失敗時の自動修復と設定更新

---

## 🏗️ 主要コンポーネント

### 1. Webアプリケーション層 (`app/`)

#### 1.1 ルーティング (`app/routes.py`)
- **役割**: Flask BlueprintによるHTTPエンドポイント管理
- **機能**:
  - 商品管理（CRUD）
  - CSVインポート/エクスポート
  - 自動リサーチ画面
  - 倉庫API（Forward2me連携）
- **依存**: `models.Product`, `forms.*`, `extensions.db`

#### 1.2 データモデル (`app/models.py`, `app/models/result_models.py`)
- **Product**: 商品情報（価格、在庫、URL等）
- **DiscoveryResult**: エージェントの偵察結果（標準化データ構造）
- **GenerateResult**: LLM応答の標準化データ構造

#### 1.3 拡張機能 (`app/extensions.py`)
- **SQLAlchemy**: データベースORM（単一インスタンス）
- **Flask-Migrate**: マイグレーション管理
- **CSRF保護**: Flask-WTF

### 2. エージェント層 (`app/agents/`)

#### 2.1 コアエージェント（実行部隊）

**SupplierScoutAgent** (`supplier_scout_agent.py`)
- **役割**: 現場総指揮官。作戦計画策定、エージェント選択、結果報告
- **依存**: `FailureAnalysisAgent`, `SelectorDiscoveryAgent`, `BrowserUseAgent`
- **リスク**: 中（エージェント選択ロジックが複雑）

**BrowserUseAgent** (`browser_use_agent.py`)
- **役割**: 実戦部隊。Playwrightによるブラウザ操作実行
- **行数**: **2000行以上**（最大のファイル）
- **依存**: `SessionManager`, `NavigationDriver`, `extract_product_info`, `RunContext`
- **リスク**: **極高**（巨大で複雑、多くの責務を持つ）

**SelectorDiscoveryAgent** (`selector_discovery_agent.py`)
- **役割**: セレクタ自動発見と自己修復ループ
- **依存**: `FailureAnalysisAgent`, `SelfHealingAgent`
- **リスク**: 高（自己修復ロジックが複雑）

#### 2.2 支援エージェント（修復・分析）

**SelfHealingAgent** (`self_healing_agent.py`)
- **役割**: 自己修復オペレーション統括
- **依存**: `PageRecoveryAgent`, `SelectorRepairAgent`
- **リスク**: 中

**FailureAnalysisAgent** (`failure_analysis_agent.py`)
- **役割**: 失敗の診断・原因分析・レポート生成
- **依存**: `AILlmController`（スタブ実装あり）
- **リスク**: 中

**PageRecoveryAgent** (`page_recovery_agent.py`)
- **役割**: 物理的なページ回復（goBack、Cookie処理等）
- **リスク**: 低

**SelectorRepairAgent** (`selector_repair_agent.py`)
- **役割**: AIによるセレクタ修復提案
- **依存**: `AILlmController`（スタブ実装あり）
- **リスク**: 中

#### 2.3 データ処理エージェント

**PersistenceAgent** (`persistence_agent.py`)
- **役割**: 収集データの永続化（CSV/DB）
- **依存**: `app.models.Product`（オプション）
- **リスク**: 低

**ReportingAgent** (`reporting_agent.py`)
- **役割**: 作戦結果の集約・HTMLレポート生成
- **依存**: `DiscoveryResult`
- **リスク**: 低

#### 2.4 その他のエージェント
- `PriceIntelligenceAgent`: 価格情報収集（Seleniumベース、レガシー？）
- `ProfitabilityAgent`: 収益性分析
- `AIVisionAgent`: パーセル画像検査（スタブ）
- `ProxyScrapeAgent`: 自己進化ループ実装
- `CodeUpdateAgent`: 設定ファイルへのパッチ適用

### 3. ユーティリティ層 (`app/utils/`)

#### 3.1 オーケストレーター (`ai_research_orchestrator.py`)
- **役割**: システム全体の調査プロセスを統括する最高司令部
- **依存**: `SupplierScoutAgent`, `PersistenceAgent`, `ReportingAgent`
- **リスク**: **高**（中核コンポーネント、変更影響が大きい）

#### 3.2 LLM制御 (`ai_llm_controller.py`)
- **役割**: LLM（Gemini/OpenAI）呼び出しの統一インターフェース
- **機能**: キャッシュ、コスト追跡、リトライ
- **依存**: `result_models.GenerateResult`
- **リスク**: 中（外部API依存）

#### 3.3 その他のユーティリティ
- `shipping_agent.py`: 送料計算（Buyandship API連携）
- `pricing_calculator.py`: 価格計算
- `visual_regression.py`: 視覚的回帰テスト
- `observability.py`: DOM保存、セレクタカウント等
- `fx_utils.py`: 為替レート取得

### 4. 設定管理層 (`app/config/`)

#### 4.1 設定ローダー (`loader.py`)
- **役割**: 3階層設定（base, default, overrides）のマージ
- **依存**: `config.py`, JSON設定ファイル
- **リスク**: **高**（複雑なフォールバック戦略、後方互換性維持）

#### 4.2 設定ファイル
- `base.json`: 基本設定
- `overrides.local.json`: ローカル上書き設定
- `secrets.py`: APIキー等の機密情報

### 5. ブラウザ制御層 (`app/agents/browser/`)

#### 5.1 SessionManager (`session_manager.py`)
- **役割**: Playwrightセッションの管理・再利用
- **依存**: `RunContext`
- **リスク**: 中（ブラウザリソース管理が複雑）

#### 5.2 NavigationDriver (`navigation_driver.py`)
- **役割**: ナビゲーション制御の抽象化
- **リスク**: 低

#### 5.3 Extractor (`extractor.py`)
- **役割**: 商品情報抽出の統一インターフェース
- **依存**: `extract_product_info`, `MonclerPDPExtractor`
- **リスク**: 中

### 6. コア機能 (`app/core/`)

#### 6.1 RunContext (`run_context.py`)
- **役割**: 実行コンテキスト管理、スクリーンショット、ログ保存
- **依存**: なし（独立）
- **リスク**: 低（安定）

---

## ⚠️ ボトルネック

### 1. **BrowserUseAgent の巨大化**
- **問題**: 2000行以上の巨大ファイル、複数の責務が混在
- **影響**: 
  - テストが困難
  - 変更時の影響範囲が広い
  - コードレビューが困難
- **対策**: 責務分離、複数ファイルへの分割

### 2. **設定ローダーの複雑性**
- **問題**: 複数のフォールバック戦略、後方互換性維持のためのエイリアス
- **影響**: 設定読み込みの挙動が予測困難
- **対策**: 設定ローダーの簡素化、バージョン管理の導入

### 3. **エージェント間の密結合**
- **問題**: エージェントが直接他のエージェントをimport
- **影響**: 変更の伝播が広範囲、テストが困難
- **対策**: インターフェース抽象化、依存性注入

### 4. **非同期処理の複雑性**
- **問題**: `async/await`の使用が一貫していない可能性
- **影響**: デッドロック、パフォーマンス問題
- **対策**: 非同期処理の標準化、型チェック強化

### 5. **スタブ実装の散在**
- **問題**: 複数のエージェントで`AILlmController`のスタブ実装が存在
- **影響**: 実装の不整合、デバッグ困難
- **対策**: 統一インターフェースの確立

---

## 🔴 最大リスクファイル

### 1. `app/agents/browser_use_agent.py` ⚠️⚠️⚠️
- **リスクレベル**: **極高**
- **理由**: 
  - 2000行以上の巨大ファイル
  - 複数の責務（ナビゲーション、抽出、修復、VRT等）
  - 多くの依存関係
  - 変更時の影響範囲が広い
- **推奨アクション**: 優先的にリファクタリング

### 2. `app/utils/ai_research_orchestrator.py` ⚠️⚠️
- **リスクレベル**: **高**
- **理由**: 
  - システムの中核コンポーネント
  - 多くのエージェントに依存
  - 変更時の影響が大きい
- **推奨アクション**: インターフェース抽象化、依存性注入

### 3. `app/config/loader.py` ⚠️⚠️
- **リスクレベル**: **高**
- **理由**: 
  - 複雑なフォールバック戦略
  - 後方互換性維持のための複雑なロジック
  - 設定読み込みの挙動が予測困難
- **推奨アクション**: 設定ローダーの簡素化、バージョン管理

### 4. `app/agents/supplier_scout_agent.py` ⚠️
- **リスクレベル**: **中-高**
- **理由**: 
  - エージェント選択ロジックが複雑
  - 多くのエージェントに依存
- **推奨アクション**: ファクトリーパターンの導入

### 5. `app/utils/ai_llm_controller.py` ⚠️
- **リスクレベル**: **中**
- **理由**: 
  - 外部API依存
  - キャッシュ管理が複雑
- **推奨アクション**: リトライ戦略の改善、モック化の容易化

---

## 🔗 高結合度ファイル

### 1. **BrowserUseAgent とその依存関係**
```
browser_use_agent.py
├─→ SessionManager
├─→ NavigationDriver
├─→ Extractor
├─→ RunContext
├─→ visual_regression
├─→ observability
├─→ SelectorDiscoveryAgent
└─→ extract_product_info
```
**結合度**: 極高（8個以上の直接依存）

### 2. **SupplierScoutAgent とエージェント群**
```
supplier_scout_agent.py
├─→ FailureAnalysisAgent
├─→ SelectorDiscoveryAgent
│     └─→ SelfHealingAgent
│           ├─→ PageRecoveryAgent
│           └─→ SelectorRepairAgent
└─→ BrowserUseAgent
```
**結合度**: 高（複数のエージェントに依存）

### 3. **設定システム**
```
loader.py
├─→ config.py
├─→ base.json
└─→ overrides.local.json
```
**結合度**: 中（設定ファイルへの依存）

### 4. **Orchestrator とエージェント群**
```
ai_research_orchestrator.py
├─→ SupplierScoutAgent
├─→ PersistenceAgent
└─→ ReportingAgent
```
**結合度**: 高（中核コンポーネント）

---

## 🎯 モジュール化戦略

### Phase 1: 緊急度の高いリファクタリング

#### 1.1 BrowserUseAgent の分割
**目標**: 2000行のファイルを責務ごとに分割

**提案構造**:
```
app/agents/browser_use/
├── __init__.py (BrowserUseAgent ファサード)
├── core/
│   ├── agent.py (コアロジック)
│   └── bootstrap.py (セッション初期化)
├── navigation/
│   ├── driver.py (NavigationDriver統合)
│   └── handlers.py (Cookie、モーダル処理)
├── extraction/
│   └── product_extractor.py (商品情報抽出)
├── recovery/
│   └── recovery_strategies.py (回復戦略)
└── plugins/
    └── plugin_manager.py (プラグイン管理)
```

**メリット**:
- テストが容易になる
- 変更の影響範囲が明確になる
- コードレビューが容易になる

#### 1.2 設定システムの簡素化
**目標**: フォールバック戦略の簡素化、バージョン管理の導入

**提案**:
- 設定ファイルのバージョン管理（`version`フィールド追加）
- フォールバック戦略の削減（最大2階層）
- 設定検証の強化（Pydantic等）

#### 1.3 エージェントインターフェースの抽象化
**目標**: エージェント間の直接依存を削減

**提案**:
```python
# app/agents/interfaces.py
from abc import ABC, abstractmethod
from app.models.result_models import DiscoveryResult

class Agent(ABC):
    @abstractmethod
    async def run(self, **kwargs) -> DiscoveryResult:
        pass

class HealingAgent(ABC):
    @abstractmethod
    async def heal(self, context: RunContext) -> bool:
        pass
```

**メリット**:
- モック化が容易
- テストが容易
- 依存関係が明確

### Phase 2: アーキテクチャの改善

#### 2.1 依存性注入（DI）の導入
**目標**: エージェント間の結合を緩和

**提案**:
```python
# app/agents/container.py
from dependency_injector import containers, providers

class AgentContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    llm_controller = providers.Singleton(
        AILlmController,
        api_key=config.llm.api_key
    )
    
    browser_agent = providers.Factory(
        BrowserUseAgent,
        llm_controller=llm_controller
    )
```

#### 2.2 イベント駆動アーキテクチャの導入
**目標**: エージェント間の疎結合化

**提案**:
```python
# app/core/events.py
from dataclasses import dataclass
from typing import Protocol

@dataclass
class AgentEvent:
    agent_name: str
    event_type: str
    payload: dict

class EventHandler(Protocol):
    async def handle(self, event: AgentEvent) -> None:
        pass

class EventBus:
    def __init__(self):
        self._handlers: dict[str, list[EventHandler]] = {}
    
    def subscribe(self, event_type: str, handler: EventHandler):
        ...
    
    async def publish(self, event: AgentEvent):
        ...
```

**メリット**:
- エージェント間の直接依存が不要
- 拡張が容易（新しいエージェントの追加が簡単）
- テストが容易

#### 2.3 プラグインアーキテクチャの拡張
**目標**: サイト固有のロジックをプラグイン化

**現状**: `app/agents/plugins/` に一部実装あり

**提案**:
- プラグインインターフェースの標準化
- プラグインの動的ロード機能の強化
- プラグインのバージョン管理

### Phase 3: モダナイゼーション

#### 3.1 型安全性の強化
**目標**: 型ヒントの完全化、mypyの導入

**現状**: 一部のファイルで型ヒントが不完全

**提案**:
- `mypy`の導入とCI/CDへの統合
- `typing.Protocol`の活用
- データクラスの活用（`result_models`は既に実装済み）

#### 3.2 非同期処理の標準化
**目標**: 非同期処理の一貫性確保

**提案**:
- 非同期コンテキストマネージャーの活用
- `asyncio.gather`の適切な使用
- タイムアウト処理の標準化

#### 3.3 ロギングとオブザーバビリティの強化
**目標**: システムの可観測性向上

**提案**:
- 構造化ロギング（JSON形式）
- OpenTelemetryの統合（一部実装済み）
- メトリクス収集（Prometheus等）

#### 3.4 テスト戦略の改善
**目標**: テストカバレッジの向上

**現状**: `tests/` ディレクトリに一部テストあり

**提案**:
- ユニットテストの拡充
- 統合テストの追加
- モック戦略の標準化

---

## 📈 依存関係グラフ（主要部分）

```
┌─────────────────────────────────────────┐
│  Flask Application (app/__init__.py)     │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼──────┐      ┌───────▼────────┐
│ Routes   │      │ Models         │
│ (routes) │      │ (Product,      │
└───┬──────┘      │  ResultModels) │
    │             └───────┬─────────┘
    │                     │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────────────────────┐
    │  Orchestrator                        │
    │  (ai_research_orchestrator)          │
    └──────────┬──────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼──────────┐  ┌───────▼────────┐
│ SupplierScout│  │ Persistence    │
│ Agent        │  │ Agent          │
└───┬──────────┘  └────────────────┘
    │
    ├─→ SelectorDiscoveryAgent
    │   └─→ SelfHealingAgent
    │       ├─→ PageRecoveryAgent
    │       └─→ SelectorRepairAgent
    │
    └─→ BrowserUseAgent
        ├─→ SessionManager
        ├─→ NavigationDriver
        └─→ Extractor
            └─→ extract_product_info
```

---

## 🚀 BUYMA Growth Hub への拡張提案

### 1. マルチテナント対応
- **現状**: 単一インスタンス
- **提案**: テナントごとの設定分離、データ分離

### 2. スケーラビリティの向上
- **現状**: シングルプロセス
- **提案**: 
  - タスクキュー（Celery/RQ）の導入
  - 分散実行のサポート
  - リソースプールの管理

### 3. API化
- **現状**: Flask Webアプリケーション
- **提案**: 
  - RESTful APIの追加
  - GraphQLの検討
  - WebSocketサポート（リアルタイム更新）

### 4. データ分析機能の強化
- **現状**: 基本的なレポート生成
- **提案**: 
  - 時系列分析
  - 予測モデルの統合
  - ダッシュボードの追加

### 5. セキュリティの強化
- **現状**: 基本的なCSRF保護
- **提案**: 
  - 認証・認可の強化（JWT等）
  - レート制限
  - 監査ログ

---

## 📝 次のステップ

### 即座に実行すべきこと
1. ✅ **BrowserUseAgent の分割**（最優先）
2. ✅ **設定システムの簡素化**
3. ✅ **エージェントインターフェースの抽象化**

### 短期（1-2ヶ月）
1. 依存性注入の導入
2. イベント駆動アーキテクチャの検討
3. テストカバレッジの向上

### 中期（3-6ヶ月）
1. プラグインアーキテクチャの拡張
2. 型安全性の強化
3. ロギングとオブザーバビリティの強化

### 長期（6ヶ月以上）
1. マルチテナント対応
2. スケーラビリティの向上
3. API化

---

## 📚 参考資料

- `docs/agents_analysis.md`: エージェント群の詳細分析
- `Development Policy/`: 開発方針ドキュメント
- `app/config/loader.py`: 設定ローダーの実装
- `app/models/result_models.py`: 標準データモデル

---

**レポート作成者**: AI Assistant  
**最終更新**: 2025-01-XX

