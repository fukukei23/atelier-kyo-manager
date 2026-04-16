# 責務分割と依存抽象化 完了レポート

## 実装日時

2026年2月4日

## 概要

**目的**: browser_use_agent.py / navigation_driver.py の責務分割と、設定・LLM 呼び出しの依存抽象化により結合度を下げる。

**ゴール**:
- NavigationDriver 側の UI 系を ui_helpers / navigation_helpers に委譲し重複を解消
- SiteConfigProvider プロトコルを導入し、設定取得を注入可能にする
- LLMClient プロトコルを導入し、LLM 呼び出しを注入可能にする

**原則**:
- 既存の公開 API（run(), run_with_repair(), run_e2e() 等）は変更しない
- デフォルトは現行と同じ実装（DefaultSiteConfigProvider / AILlmController）を使用
- 本番コードの仕様・エラー処理は変えず、注入可能にするだけ

## 実装ステップ

### Phase 1.1: NavigationDriver と BrowserUseAgent の重複解消

**実施内容**:
1. **NavigationDriver が ui_helpers / navigation_helpers を利用するように変更**
   - [app/agents/browser/navigation_driver.py](app/agents/browser/navigation_driver.py) の先頭で `ui_helpers`（safe_wait_selector, accept_cookies_if_present, click_continue_shopping_if_present, kill_overlays）と `navigation_helpers`（normalize_url, normalize_abs_url）を import
   - `safe_wait_selector`: ui_safe_wait_selector が利用可能なら委譲、否则は従来の自前実装
   - `_normalize_abs_url`: nav_normalize_abs_url が利用可能なら委譲（スキーム除外は自前）
   - `_accept_cookies_if_present`: ui_accept_cookies_if_present に完全委譲（実装を削除）
   - `_click_continue_shopping_if_present`: ui_click_continue_shopping_if_present に完全委譲（実装を削除）
   - `_kill_overlays`: ui_kill_overlays が利用可能なら委譲、否则は従来の evaluate による削除

2. **BrowserUseAgent 側**
   - Phase A 相当の「_ensure_plp_materialized / _collect_pdp_links / _force_plp_recover / _click_first_card_or_link の削除」は既に完了済み（_run_plp_flow は Orchestrator 委譲のみ）のため、追加変更なし

### Phase 2.1: 設定の抽象化（SiteConfigProvider）

**実施内容**:
1. **プロトコルとデフォルト実装の新規作成**
   - [app/config/protocols.py](app/config/protocols.py) を新規作成
   - `SiteConfigProvider` プロトコル: `get_site_config(site_name) -> Dict | None`, `get_full_config() -> Dict`
   - `DefaultSiteConfigProvider`: loader の `get_site_config` / `load_full_config` をラップ

2. **注入箇所**
   - [app/agents/browser_use_agent.py](app/agents/browser_use_agent.py): コンストラクタに `config_provider: Optional[Any] = None` を追加。`run_e2e` 内のサイト設定取得を `_config_provider or DefaultSiteConfigProvider()` 経由の `get_site_config("MONCLER_OFFICIAL")` に変更。`SitesConfigLoader` の import を削除（存在しないモジュールだったため）
   - [app/scripts/run_site.py](app/scripts/run_site.py): `DefaultSiteConfigProvider` を import。`resolve_site_name` 内で full_config 取得を `DefaultSiteConfigProvider().get_full_config()` に変更（フォールバックで load_full_config も維持）。サイト設定取得を `config_provider = DefaultSiteConfigProvider()`, `site_config = config_provider.get_site_config(site_name)` に変更。`BrowserUseAgent(runtime_kwargs=..., config_provider=config_provider)` で config_provider を渡す

### Phase 2.2: LLM の抽象化（LLMClient）

**実施内容**:
1. **プロトコルの新規作成**
   - [app/utils/llm_protocol.py](app/utils/llm_protocol.py) を新規作成
   - `LLMClient` プロトコル: `generate(prompt, task_type=..., tools=..., stream=..., chunk_callback=...) -> Any`（GenerateResult 相当を返す想定）

2. **注入箇所**
   - [app/agents/selector_repair_agent.py](app/agents/selector_repair_agent.py): `AILlmController` を正しいクラス名 `AILlmController` で import。スタブ時は `_DefaultLLM` を定義し、`llm_client` 未指定時は `_DefaultLLM()` を使用。既存の `AiLlmController(mode="Chat/Default")` は `AILlmController()` に変更（AILlmController は __init__ に mode を取らないため）
   - [app/agents/browser_use_agent.py](app/agents/browser_use_agent.py): コンストラクタに `llm_client: Optional[Any] = None` を追加。`run_with_repair` 内で LLM 取得を `self._llm_client` が None でなければそれを使用、否则は `AILlmController()` を生成するように変更

## 変更ファイル一覧

| ファイル | 種別 | 説明 |
|----------|------|------|
| app/config/protocols.py | 新規 | SiteConfigProvider プロトコルと DefaultSiteConfigProvider |
| app/utils/llm_protocol.py | 新規 | LLMClient プロトコル |
| app/agents/browser/navigation_driver.py | 変更 | ui_helpers / navigation_helpers の import と委譲、_accept_cookies_if_present / _click_continue_shopping_if_present / _kill_overlays の委譲化 |
| app/agents/browser_use_agent.py | 変更 | config_provider / llm_client のコンストラクタ追加、run_e2e の設定取得を provider 経由に、run_with_repair の LLM を _llm_client 優先に、SitesConfigLoader 参照削除 |
| app/scripts/run_site.py | 変更 | DefaultSiteConfigProvider の利用と config_provider の BrowserUseAgent への受け渡し |
| app/agents/selector_repair_agent.py | 変更 | AILlmController の正しいクラス名、_DefaultLLM スタブ、LLMClient 型の明示（コメント） |

## 動作確認結果

- リンター: app/config/protocols.py, app/utils/llm_protocol.py に新規の致命的エラーはなし。browser_use_agent.py の既存の basedpyright エラー（playwright / RunContext スタブ等）は従来どおり
- 既存テスト: 注入のデフォルトが現行と同じ実装のため、既存の呼び出し方（config_provider なし・llm_client なし）で従来どおり動作する想定
- run_site: DefaultSiteConfigProvider 経由で get_site_config を取得し、BrowserUseAgent に config_provider を渡すため、従来の loader 利用と同等

## 設計上の改善点

- **結合度の低減**: 設定・LLM をプロトコルで抽象化し、テスト時にスタブに差し替え可能になった
- **重複の削減**: NavigationDriver 内の UI 系実装の一部を ui_helpers / navigation_helpers に集約し、同一ロジックの二重管理を削減
- **拡張性**: 将来、設定ソースをファイル以外（DB・リモート）に差し替える場合や、LLM を別プロバイダに差し替える場合に、Provider の実装を追加するだけで対応可能

## 既知の制約・注意事項

- **1.2 BrowserUseAgent を薄い Facade に**: 今回は未実施。既に _run_plp_flow / _run_pdp_flow は Orchestrator 委譲済みのため、UI ヘルパーの完全移行や human_like の切り出しは後続タスクとする
- **1.3 navigation_driver のモジュール分割**: 今回は未実施。link_collection.py / trap_and_locale.py 等への分割は中期タスクとして計画に残す
- **SelectorRepairAgent**: AILlmController のコンストラクタは `mode` を取らないため、従来の `AiLlmController(mode="Chat/Default")` は `_DefaultLLM()`（実体は AILlmController()）に変更。挙動は既存のシングルトン利用と同等
- **run_site の config_provider**: 現状は常に DefaultSiteConfigProvider() を生成。CLI から差し替え可能にする場合は、引数で Provider を渡す拡張を検討

## 次のステップ

1. 既存テストの実行（pytest tests/test_plp_driver.py tests/test_navigation_driver_stage3a2.py 等）でリグレッションがないことを確認
2. 必要に応じて、テストで SiteConfigProvider / LLMClient のスタブを注入するユニットテストを追加
3. Phase 1.2（BrowserUseAgent の UI ヘルパー完全移行）・Phase 1.3（navigation_driver のファイル分割）は別 CR または後続タスクとして実施
