# 🏛️ エージェント設計図 (AI-Generated)
このドキュメントはプロジェクトのソースコードを静的解析し、自律的に動作する「専門家エージェント」の役割と連携関係を可視化したものです。
AIがコードの詳細を分析する前にこの設計図を読むことで、システム全体のアーキテクチャと目的を効率的に理解できます。
---

## 🧠 司令塔 (Orchestrators)
複数の専門家エージェントを連携させ、複雑なワークフローを実行する中心的役割を担います。

###  orchestrator: `ai_research_orchestrator.py`
- **役割:** クラス 'ResearchOrchestrator': シングルセッション戦略で動作する、BUYMAリサーチフロー統括司令塔。

## 🛠️ 専門家 (Specialists)
特定のタスク（例：価格計算、画像収集）に特化したエージェントです。司令塔から呼び出されて機能します。

- **`ai_image_crawler.py`**
  - **専門分野:** クラス 'CrawlerService': 対象サイトから商品画像URLを収集するクローラー

- **`buyma_catalog_manager.py`**
  - **専門分野:** Buyma Catalog Manager の機能を提供

- **`buyma_catalog_manager.py`**
  - **専門分野:** Buyma Catalog Manager の機能を提供

- **`buyma_catalog_manager.py`**
  - **専門分野:** Buyma Catalog Manager の機能を提供

- **`shipping_agent.py`**
  - **専門分野:** クラス 'ShippingAgent': 転送倉庫の情報を取得するエージェント（Playwright, Async, 永続セッション対応, UIテキスト優先ログイン）

- **`ai_image_collector.py`**
  - **専門分野:** Role):

- **`pricing_calculator.py`**
  - **専門分野:** Pricing Calculator の機能を提供

- **`ai_image_crawler.py`**
  - **専門分野:** Ai Image Crawler の機能を提供

- **`ai_image_crawler.py`**
  - **専門分野:** Ai Image Crawler の機能を提供

- **`ai_llm_controller.py`**
  - **専門分野:** クラス 'AILlmController': 複数のLLMエンジンを透過的に呼び出すための統合コントローラー。

- **`pricing_calculator.py`**
  - **専門分野:** Pricing Calculator の機能を提供

- **`pricing_calculator.py`**
  - **専門分野:** Pricing Calculator の機能を提供
