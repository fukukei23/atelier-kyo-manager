# CR-ATELIER-003 Phase D-1: PDP フロー分析レポート

## 実装日時

2024年12月10日

## 目的

Phase D-1 として、PDP フローの Orchestrator 化（`BrowserOrchestrator.run_pdp`）に向けて、既存の PDP 処理の依存関係・設計要素を抽出する。

## 分析対象関数

### 1. `BrowserUseAgent._run_pdp_flow`

**ファイル**: `app/agents/browser_use_agent.py` (行2095-2116)

**シグネチャ**:
```python
async def _run_pdp_flow(
    self,
    page: Page,
    site: str,
    query: str,
    settings: Dict,
    run_context: RunContext,
    site_config: Dict[str, Any],
) -> DiscoveryResult
```

**実装概要**:
- 非常にシンプルな実装（約22行）
- `_build_pdp_prepare_hook` を呼び出して準備フックを構築
- `self.extraction_service.extract_single_pdp` を呼び出すだけ
- 戻り値は `DiscoveryResult` をそのまま返す

**外部依存**:
- `page: Page` - Playwright Page オブジェクト（現在のPDPページ）
- `self._context: BrowserContext` - Playwright BrowserContext（`extract_single_pdp` に渡される）
- `self.extraction_service: BrowserExtractionService` - 抽出サービス
- `site_config: Dict[str, Any]` - サイト設定
- `settings: Dict` - 実行設定
- `run_context: RunContext` - 実行コンテキスト

**準備フック（prepare_hook）**:
- `_build_pdp_prepare_hook` で構築される非同期関数
- UI helpers を使用:
  - `kill_overlays(page)` - オーバーレイの削除
  - `click_continue_shopping_if_present(page, site_config)` - ショッピング継続ボタンのクリック
  - `dismiss_geo_modal(page, logger)` - ジオモーダルの閉じる
- Visual regression check（設定されている場合）: `_perform_vrt(page, "pdp", settings)`

**例外処理**:
- `extract_single_pdp` が `ValueError` を投げる可能性がある（価格が見つからない場合）
- 例外は `extract_single_pdp` 内で処理され、`DiscoveryResult(ok=False)` が返される

**戻り値**:
- `DiscoveryResult` - 抽出結果またはエラー情報

**PLP → PDP の引き継ぎデータ**:
- `page` - PLP から PDP に遷移した後の Page オブジェクト
- `page.url` - PDP の URL（`extract_single_pdp` の `target_url` として使用）

---

### 2. `BrowserExtractionService.extract_single_pdp`

**ファイル**: `app/agents/browser/extractor.py` (行564-602)

**シグネチャ**:
```python
async def extract_single_pdp(
    self,
    *,
    page: Page,
    context: Optional[BrowserContext],
    site: str,
    query: str,
    settings: Dict[str, Any],
    run_context: RunContext,
    site_config: Dict[str, Any],
    target_url: str,
    prepare_page: PreparePageCallable = None,
) -> DiscoveryResult
```

**実装概要**:
- 単一の PDP ページから商品情報を抽出
- `_extract_from_pdp` を呼び出して商品情報を取得
- DOM スナップショットを保存（`save_dom`）
- 抽出結果を `run_context.save_json("pdp_extracted_data.json", ...)` に保存
- `DiscoveryResult` を返す

**外部依存**:
- `page: Page` - Playwright Page オブジェクト
- `context: Optional[BrowserContext]` - Playwright BrowserContext（`_extract_from_pdp` に渡される）
- `run_context: RunContext` - 実行コンテキスト（DOM保存、JSON保存に使用）
- `site_config: Dict[str, Any]` - サイト設定（`ProductExtractor` に渡される）
- `settings: Dict[str, Any]` - 実行設定
- `prepare_page: PreparePageCallable` - ページ準備フック（オプション）

**準備フック（prepare_hook）**:
- `prepare_page` パラメータとして受け取る
- `_extract_from_pdp` 内で `ProductExtractor.extract` に渡される
- ページ遷移後の UI 操作（オーバーレイ削除、モーダル閉じるなど）を実行

**例外処理**:
- `_extract_from_pdp` が `None` を返した場合、`ValueError("Price not found on PDP.")` を投げる
- `save_dom` が失敗した場合、警告ログを出力して続行
- 例外は呼び出し元（`_run_pdp_flow`）に伝播する

**戻り値**:
- `DiscoveryResult(ok=True, ...)` - 抽出成功時
  - `evidence={"extracted_data": item, "final_url": item.get("url")}`
- `DiscoveryResult(ok=False, ...)` - 抽出失敗時（例外が投げられるため、通常は返されない）

**Telemetry 保存ポイント**:
- `save_dom(run_context, page, "pdp_dom")` - PDP DOM スナップショット
- `run_context.save_json("pdp_extracted_data.json", {"extracted_data": item})` - 抽出結果

---

### 3. `BrowserExtractionService.extract_from_pdp_list`

**ファイル**: `app/agents/browser/extractor.py` (行604-671)

**シグネチャ**:
```python
async def extract_from_pdp_list(
    self,
    *,
    page: Page,
    context: BrowserContext,
    site: str,
    query: str,
    pdp_links: List[str],
    site_config: Dict[str, Any],
    settings: Dict[str, Any],
    run_context: RunContext,
    start_t: float,
    budget_ms: int,
    prepare_page: PreparePageCallable = None,
) -> DiscoveryResult
```

**実装概要**:
- 複数の PDP リンクから並列に商品情報を抽出
- `asyncio.Semaphore` を使用して並列実行数を制限
- 各 PDP リンクに対して `worker` 関数を実行
- `worker` 関数内で `context.new_page()` を使用して新しいページを作成
- `_extract_from_pdp` を呼び出して商品情報を取得
- 有効な抽出結果を集約して `DiscoveryResult` を返す

**外部依存**:
- `page: Page` - Playwright Page オブジェクト（PLP ページ、`original_plp_url` の取得に使用）
- `context: BrowserContext` - Playwright BrowserContext（`new_page()` で新しいページを作成）
- `pdp_links: List[str]` - 抽出対象の PDP URL リスト
- `run_context: RunContext` - 実行コンテキスト（JSON保存に使用）
- `start_t: float` - 開始時刻（タイムアウト計算に使用）
- `budget_ms: int` - 予算時間（ミリ秒）
- `prepare_page: PreparePageCallable` - ページ準備フック（オプション）

**準備フック（prepare_hook）**:
- `prepare_page` パラメータとして受け取る
- `worker` 関数内で `_extract_from_pdp` に渡される
- 各 PDP ページ遷移後に実行される

**例外処理**:
- `left_ms <= 0` の場合、`ValueError("Timed out before PDP extraction (watchdog).")` を投げる
- `worker` 関数内で例外が発生した場合、警告ログを出力して `None` を返す
- `worker_page` は `finally` ブロックで確実に閉じられる
- 有効な抽出結果が1つもない場合、`ValueError("Found PDP links but price extraction failed after size-selection and retry.")` を投げる

**戻り値**:
- `DiscoveryResult(ok=True, ...)` - 抽出成功時
  - `message=f"PLP extracted {len(valid_items)} items"`
  - `evidence={"extracted_data": valid_items, "final_url": original_plp_url}`
- `DiscoveryResult(ok=False, ...)` - 抽出失敗時（例外が投げられるため、通常は返されない）

**Telemetry 保存ポイント**:
- `run_context.save_json("plp_extracted_items.json", {"extracted_data": valid_items})` - 抽出結果リスト

**並列処理の詳細**:
- `asyncio.Semaphore` で並列実行数を制限（デフォルト: `DEFAULT_PDP_PARALLEL_LIMIT`）
- `settings.get("pdp_parallel_limit", DEFAULT_PDP_PARALLEL_LIMIT)` で設定可能
- 各 worker は独立したページを作成して使用
- タイムアウトは `start_t` と `budget_ms` から計算

---

### 4. `BrowserExtractionService._extract_from_pdp` (内部メソッド)

**ファイル**: `app/agents/browser/extractor.py` (行673-753)

**シグネチャ**:
```python
async def _extract_from_pdp(
    self,
    *,
    page: Page,
    url: str,
    context: Optional[BrowserContext],
    site: str,
    settings: Dict[str, Any],
    site_config: Dict[str, Any],
    timeout_override: Optional[int] = None,
    prepare_page: PreparePageCallable = None,
    run_context: Optional[RunContext] = None,
) -> Optional[Dict[str, Any]]
```

**実装概要**:
- 単一の PDP ページから商品情報を抽出する内部メソッド
- `ProductExtractor` を優先的に使用
- フォールバック: Moncler 専用抽出 → 価格抽出 → JSON-LD / Meta タグ

**外部依存**:
- `page: Page` - Playwright Page オブジェクト
- `url: str` - 抽出対象の PDP URL（`page.url != url` の場合、`page.goto` を実行）
- `context: Optional[BrowserContext]` - Playwright BrowserContext（`ProductExtractor` に渡される）
- `site_config: Dict[str, Any]` - サイト設定（`ProductExtractor` に渡される）
- `run_context: Optional[RunContext]` - 実行コンテキスト（`ProductExtractor` に渡される）
- `prepare_page: PreparePageCallable` - ページ準備フック（`ProductExtractor.extract` に渡される）

**準備フック（prepare_hook）**:
- `prepare_page` パラメータとして受け取る
- `ProductExtractor.extract` に渡される
- ページ遷移後の UI 操作を実行

**例外処理**:
- `ProductExtractor` が失敗した場合、警告ログを出力してフォールバック処理に進む
- すべてのフォールバックが失敗した場合、`None` を返す

**戻り値**:
- `Dict[str, Any]` - 抽出成功時（商品情報）
  - `title`, `price`, `currency`, `url`, `images`, `sizes`, `colors`, `description`, `brand`, `list_price`, `discount_pct`, `raw_html_path`, `metadata`
- `None` - 抽出失敗時

**フォールバック処理**:
1. **ProductExtractor** (優先)
2. **Moncler 専用抽出** (`site.upper() == "MONCLER_OFFICIAL"` の場合)
3. **価格抽出 + サイズ選択** (`_extract_price_with_size_option`)
4. **JSON-LD フォールバック** (`_extract_ld_json_price`)
5. **Meta タグフォールバック** (`_extract_meta_price`)

---

## DiscoveryResult の構造

**定義場所**: `app/models/result_models.py` (推測)

**構造**:
```python
class DiscoveryResult:
    ok: bool  # 成功/失敗フラグ
    site: str  # サイトコード
    query: str  # 検索クエリ
    message: str  # メッセージ
    evidence: Dict[str, Any]  # 証拠データ
        # extract_single_pdp: {"extracted_data": item, "final_url": item.get("url")}
        # extract_from_pdp_list: {"extracted_data": valid_items, "final_url": original_plp_url}
    url: Optional[str] = None  # 最終URL（オプション）
```

**使用例**:
- `DiscoveryResult(ok=True, site="example", query="test", message="PDP extracted", evidence={...})`
- `DiscoveryResult(ok=False, site="example", query="test", message="Price not found on PDP.", evidence={...})`

---

## BrowserOrchestrator.run_pdp の I/F 初案

### 引数一覧

```python
async def run_pdp(
    self,
    *,
    page: Page,
    context: BrowserContext,
    site: str,
    query: str,
    site_config: Dict[str, Any],
    settings: Dict[str, Any],
    run_context: RunContext,
    target_url: Optional[str] = None,  # PDP URL（None の場合は page.url を使用）
    prepare_page: Optional[PreparePageCallable] = None,  # ページ準備フック（オプション）
    # 以下は extract_from_pdp_list 用（オプション）
    pdp_links: Optional[List[str]] = None,  # PDP リンクリスト（None の場合は単一PDP抽出）
    start_t: Optional[float] = None,  # 開始時刻（pdp_links が指定されている場合に必須）
    budget_ms: Optional[int] = None,  # 予算時間（pdp_links が指定されている場合に必須）
) -> DiscoveryResult
```

### 戻り値

- `DiscoveryResult` - 固定（`_run_pdp_flow` と同じ）

### エラー処理方針

1. **単一PDP抽出** (`pdp_links` が `None` の場合):
   - `extract_single_pdp` を呼び出す
   - `ValueError` が発生した場合、`DiscoveryResult(ok=False, ...)` を返す
   - その他の例外は呼び出し元に伝播

2. **複数PDP抽出** (`pdp_links` が指定されている場合):
   - `extract_from_pdp_list` を呼び出す
   - `ValueError` が発生した場合、`DiscoveryResult(ok=False, ...)` を返す
   - その他の例外は呼び出し元に伝播

3. **タイムアウト処理**:
   - `extract_from_pdp_list` 内でタイムアウトチェックが行われる
   - `left_ms <= 0` の場合、`ValueError` を投げる

### Telemetry 保存ポイント

1. **単一PDP抽出**:
   - `save_dom(run_context, page, "pdp_dom")` - PDP DOM スナップショット
   - `run_context.save_json("pdp_extracted_data.json", {"extracted_data": item})` - 抽出結果

2. **複数PDP抽出**:
   - `run_context.save_json("plp_extracted_items.json", {"extracted_data": valid_items})` - 抽出結果リスト

### 実装方針

1. **`BrowserExtractionService` のインスタンス化**:
   - `BrowserExtractionService(self.log, self.runtime_kwargs)` を作成

2. **`prepare_page` の構築**:
   - `prepare_page` が `None` の場合、`_build_pdp_prepare_hook` 相当の処理を実行
   - UI helpers を使用（`kill_overlays`, `click_continue_shopping_if_present`, `dismiss_geo_modal`）
   - Visual regression check（設定されている場合）

3. **単一/複数の判定**:
   - `pdp_links` が `None` または空リストの場合、`extract_single_pdp` を呼び出す
   - `pdp_links` が指定されている場合、`extract_from_pdp_list` を呼び出す

---

## 移行時の注意点（テスト破壊リスク）

### 1. `_run_pdp_flow` の delegator 化可能性

**結論**: ✅ **完全に delegator 化可能**

**理由**:
- `_run_pdp_flow` は既に非常にシンプル（約22行）
- `_build_pdp_prepare_hook` と `extract_single_pdp` の呼び出しのみ
- `_run_plp_flow` と同様のパターンで delegator 化可能

**移行後の構造**:
```python
async def _run_pdp_flow(
    self,
    page: Page,
    site: str,
    query: str,
    settings: Dict,
    run_context: RunContext,
    site_config: Dict[str, Any],
) -> DiscoveryResult:
    """
    CR-ATELIER-003 Phase D: BrowserOrchestrator への完全委譲
    
    PDP 抽出の全ロジックは BrowserOrchestrator.run_pdp に集約されています。
    このメソッドは Orchestrator への薄いラッパーとして機能します。
    """
    if self.orchestrator is None:
        raise ValueError("Orchestrator is not initialized...")
    
    return await self.orchestrator.run_pdp(
        page=page,
        context=self._context,
        site=site,
        query=query,
        site_config=site_config,
        settings=settings,
        run_context=run_context,
        target_url=page.url,
    )
```

### 2. 分離すべきロジック

**分離不要**:
- `_build_pdp_prepare_hook` - Orchestrator 内で直接実装可能（UI helpers を使用）
- `extract_single_pdp` / `extract_from_pdp_list` - `BrowserExtractionService` に既に分離済み

**分離検討**:
- `_build_pdp_prepare_hook` のロジックを Orchestrator 内に直接実装
- UI helpers を使用するため、`BrowserUseAgent` への依存を削減

### 3. テスト破壊リスク

**高リスク**:
- `_run_pdp_flow` を呼び出すテストが存在する場合、モック設定を更新する必要がある
- `BrowserExtractionService` のモック設定が Orchestrator 経由でも機能する必要がある

**中リスク**:
- `prepare_page` フックの動作が変更される可能性
- Visual regression check の動作が変更される可能性

**低リスク**:
- `DiscoveryResult` の構造は変更しないため、戻り値の検証は影響を受けない
- `extract_single_pdp` / `extract_from_pdp_list` の実装は変更しないため、抽出ロジックへの影響は最小限

### 4. モック設定の更新

**必要な変更**:
- `browser_orchestrator.BrowserExtractionService` もモックする必要がある
- `browser_orchestrator.extract_single_pdp` / `browser_orchestrator.extract_from_pdp_list` もモックする必要がある

**例**:
```python
with patch('app.agents.browser_use_agent.BrowserExtractionService') as mock_extraction_service_class, \
     patch("app.agents.browser_orchestrator.BrowserExtractionService") as mock_orchestrator_extraction_service_class, \
     ...
```

---

## 実装スケルトン（Phase D-2 への準備）

### BrowserOrchestrator.run_pdp のスケルトン

```python
async def run_pdp(
    self,
    *,
    page: Page,
    context: BrowserContext,
    site: str,
    query: str,
    site_config: Dict[str, Any],
    settings: Dict[str, Any],
    run_context: RunContext,
    target_url: Optional[str] = None,
    prepare_page: Optional[PreparePageCallable] = None,
    pdp_links: Optional[List[str]] = None,
    start_t: Optional[float] = None,
    budget_ms: Optional[int] = None,
) -> DiscoveryResult:
    """
    PDP 抽出フローを統括する。
    - 単一PDP抽出: extract_single_pdp を呼び出す
    - 複数PDP抽出: extract_from_pdp_list を呼び出す
    
    CR-ATELIER-003 Phase D: BrowserUseAgent._run_pdp_flow から移行
    """
    # BrowserExtractionService をインスタンス化
    extraction_service = BrowserExtractionService(self.log, self.runtime_kwargs)
    
    # prepare_page が None の場合、構築
    if prepare_page is None:
        async def prepare_hook(page: Page):
            """PDP ページの準備フック"""
            # UI helpers を使用
            from app.agents.browser import ui_helpers
            if ui_helpers.kill_overlays:
                await ui_helpers.kill_overlays(page)
            if ui_helpers.click_continue_shopping_if_present:
                await ui_helpers.click_continue_shopping_if_present(page, site_config)
            if ui_helpers.dismiss_geo_modal:
                await ui_helpers.dismiss_geo_modal(page, self.log)
            
            # Visual regression check（設定されている場合）
            if settings.get("enable_visual_regression_check") and "pdp" in (settings.get("vrt_scope") or ""):
                try:
                    from app.utils.visual_regression import compare_and_maybe_update
                    await compare_and_maybe_update(page, "pdp", settings)
                except Exception as e:
                    self.log.warning(f"[Orchestrator] Visual regression check failed: {e}", exc_info=True)
        
        prepare_page = prepare_hook
    
    # 単一/複数の判定
    if pdp_links is None or len(pdp_links) == 0:
        # 単一PDP抽出
        try:
            return await extraction_service.extract_single_pdp(
                page=page,
                context=context,
                site=site,
                query=query,
                settings=settings,
                run_context=run_context,
                site_config=site_config,
                target_url=target_url or page.url,
                prepare_page=prepare_page,
            )
        except ValueError as e:
            # 価格が見つからない場合
            return DiscoveryResult(
                ok=False,
                site=site,
                query=query,
                message=str(e),
                evidence={"final_url": page.url},
            )
    else:
        # 複数PDP抽出
        if start_t is None or budget_ms is None:
            raise ValueError("start_t and budget_ms are required for extract_from_pdp_list")
        
        try:
            return await extraction_service.extract_from_pdp_list(
                page=page,
                context=context,
                site=site,
                query=query,
                pdp_links=pdp_links,
                site_config=site_config,
                settings=settings,
                run_context=run_context,
                start_t=start_t,
                budget_ms=budget_ms,
                prepare_page=prepare_page,
            )
        except ValueError as e:
            # タイムアウトまたは抽出失敗
            return DiscoveryResult(
                ok=False,
                site=site,
                query=query,
                message=str(e),
                evidence={"final_url": page.url},
            )
```

---

## まとめ

### 分析結果

1. **`_run_pdp_flow` の delegator 化可能性**: ✅ **完全に可能**
   - 既に非常にシンプルな実装（約22行）
   - `_run_plp_flow` と同様のパターンで移行可能

2. **分離すべきロジック**: **なし**（既に適切に分離済み）
   - `BrowserExtractionService` に抽出ロジックが集約されている
   - UI helpers を使用するため、`BrowserUseAgent` への依存を削減可能

3. **テスト破壊リスク**: **中程度**
   - Orchestrator 経由のモック設定が必要
   - `BrowserExtractionService` のモック設定を更新する必要がある

### 次のステップ（Phase D-2）

1. **`BrowserOrchestrator.run_pdp` の実装スケルトン作成**
   - 上記のスケルトンを実装
   - 単一/複数PDP抽出の判定ロジックを実装

2. **`BrowserUseAgent._run_pdp_flow` の delegator 化**
   - Orchestrator 呼び出しのみにする
   - 既存のテストがパスすることを確認

3. **テストの更新**
   - Orchestrator 経由のモック設定を追加
   - `BrowserExtractionService` のモック設定を更新

---

## 関連ファイル

- `app/agents/browser_use_agent.py` - `_run_pdp_flow`, `_build_pdp_prepare_hook`
- `app/agents/browser/extractor.py` - `BrowserExtractionService`, `extract_single_pdp`, `extract_from_pdp_list`, `_extract_from_pdp`
- `app/agents/browser_orchestrator.py` - `BrowserOrchestrator` (将来の実装)
- `app/agents/browser/ui_helpers.py` - UI 操作ヘルパー
- `app/models/result_models.py` - `DiscoveryResult` の定義

