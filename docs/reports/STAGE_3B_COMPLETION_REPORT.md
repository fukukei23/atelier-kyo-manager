# Stage 3B 完了レポート: Telemetry 分離

## 実装日時
2025-11-27

## 概要

Stage 3B では、`BrowserUseAgent` と `NavigationDriver` から観測処理（DOM保存・スクリーンショット・失敗時スナップショット・JSONログ）を切り出し、`TelemetryClient` という統一インターフェースに集約しました。

### 目的
- 観測処理の責務を `TelemetryClient` に集約
- `run_context.save_*` の直接呼び出しを排除
- 将来的に Telemetry を差し替え可能な設計に

### 原則
- **挙動不変**: 保存されるファイル名、タイミング、内容は従来と同じ
- **差分は小さく**: 各ステップごとにパッチを分割
- **依存関係**: `TelemetryClient` は `run_context` のみを前提とし、循環 import を回避

---

## 実装ステップ

### Step B-1: TelemetryClient インターフェースと実装の作成 ✅

**ファイル**: `app/agents/browser/telemetry.py`

**追加内容**:
1. `TelemetryContext` データクラス
   ```python
   @dataclass
   class TelemetryContext:
       site: str
       query: str
       run_id: Optional[str] = None
       stage: Optional[str] = None  # "plp", "pdp", "fail_plp" など
   ```

2. `TelemetryClient` クラス
   - `TelemetryService` のラッパーとして実装
   - 以下のメソッドを提供:
     - `save_dom(page, name, tctx)`: DOM 保存
     - `save_json(name, payload, tctx)`: JSON 保存
     - `save_screenshot(page, name, tctx)`: スクリーンショット保存
     - `write_fail_snapshot(page, reason, tctx, extra)`: 失敗スナップショット生成

**実装方針**:
- 既存の `TelemetryService` を内部で使用
- 既存の `observability.py` の関数シグネチャと互換性を維持
- ファイル名や JSON キーは変更しない

---

### Step B-2: BrowserUseAgent から TelemetryClient を使うようにする ✅

**ファイル**: `app/agents/browser_use_agent.py`

**変更内容**:

1. **インポート追加**
   ```python
   from app.agents.browser.telemetry import TelemetryService, TelemetryClient, TelemetryContext
   ```

2. **`run()` メソッドで TelemetryClient を初期化**
   ```python
   # Stage 3B: TelemetryClient を初期化
   telemetry = TelemetryClient(run_context)
   tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id)
   ```

3. **`run_context.save_*` の呼び出しを TelemetryClient 経由に変更**
   - `save_dom(run_context, page, "plp_dom_initial_materialized")`
     → `await telemetry.save_dom(page, "plp_dom_initial_materialized", tctx)`
   - `save_dom(run_context, page, "plp_dom_search_fallback")`
     → `await telemetry.save_dom(page, "plp_dom_search_fallback", tctx)`
   - `save_dom(run_context, page, "learn_plp_dom_for_discovery")`
     → `await telemetry.save_dom(page, "learn_plp_dom_for_discovery", tctx)`

4. **失敗時の `write_fail_snapshot` を TelemetryClient 経由に変更**
   ```python
   # Before:
   await write_fail_snapshot(run_context, active_page, final_url_on_fail, e, site_config)
   
   # After:
   await telemetry.write_fail_snapshot(
       active_page,
       reason=str(e),
       tctx=tctx,
       extra={"site_config": site_config, "final_url": final_url_on_fail}
   )
   ```

**変更箇所**:
- `_run_plp_flow()`: PLP materialize 後の DOM 保存
- `_run_plp_flow()`: ヘッダ検索 fallback 後の DOM 保存
- `_handle_run_failure()`: 失敗スナップショット生成
- `_run_learning_flow()`: 学習フローでの DOM 保存

---

### Step B-3: NavigationDriver からの Telemetry 呼び出し ✅

**ファイル**: `app/agents/browser/navigation_driver.py`

**変更内容**:

1. **型アノテーションの更新**
   ```python
   # Before:
   telemetry: Optional["TelemetryService"] = None
   
   # After:
   telemetry: Optional["TelemetryClient"] = None
   ```

2. **`collect_pdp_links()` での JSON 保存を TelemetryClient 経由に変更**
   ```python
   # Before:
   if run_context and hasattr(run_context, "save_json"):
       run_context.save_json("raw_pdp_links_v85.5.json", {"links": cleaned, "sample": sample})
   
   # After:
   if self.telemetry and ctx.run_context:
       tctx = TelemetryContext(
           site=ctx.site,
           query=ctx.query,
           run_id=getattr(ctx.run_context, "run_id", None),
           stage="plp"
       )
       await self.telemetry.save_json("raw_pdp_links_v85.5", {"links": cleaned, "sample": sample}, tctx)
   ```

3. **BrowserUseAgent からの NavigationDriver 初期化時に TelemetryClient を渡す**
   ```python
   navigation_driver = NavigationDriver(
       page=page,
       telemetry=telemetry,  # Stage 3B: TelemetryClient を渡す
       trap_checker=lambda url: self._looks_like_trap_or_legal(url),
       strategy=plugin,
   )
   ```

---

### Step B-4: 失敗パス（例外時）の Telemetry を統一 ✅

**ファイル**: `app/agents/browser_use_agent.py`

**変更内容**:

1. **`_handle_run_failure()` で TelemetryClient を使用**
   ```python
   # Stage 3B: TelemetryClient を使用して失敗スナップショットを保存
   telemetry = TelemetryClient(run_context)
   tctx = TelemetryContext(site=site, query=query, run_id=run_context.run_id, stage="fail_plp")
   await telemetry.write_fail_snapshot(
       active_page,
       reason=str(e),
       tctx=tctx,
       extra={"site_config": site_config, "final_url": final_url_on_fail}
   )
   ```

2. **フォールバック処理も TelemetryClient を使用**
   - `TelemetryService.write_fail_snapshot` が失敗した場合も、`TelemetryClient` 経由で再試行

**統一された失敗パス**:
- PLP が materialize しない
- PDP リンクが 0 件
- trap からの復旧失敗
- 予期しない例外で Abort したとき

すべてのパスで `TelemetryClient.write_fail_snapshot()` を呼び出すように統一。

---

## 変更ファイル一覧

### 新規作成
- なし（`TelemetryClient` は既存の `telemetry.py` に追加）

### 変更ファイル
1. **`app/agents/browser/telemetry.py`**
   - `TelemetryContext` データクラスを追加
   - `TelemetryClient` クラスを追加

2. **`app/agents/browser_use_agent.py`**
   - `TelemetryClient` と `TelemetryContext` をインポート
   - `run()` メソッドで `TelemetryClient` を初期化
   - `run_context.save_*` の呼び出しを `TelemetryClient` 経由に変更
   - `NavigationDriver` に `TelemetryClient` を渡すように変更

3. **`app/agents/browser/navigation_driver.py`**
   - `TelemetryClient` と `TelemetryContext` をインポート
   - `__init__` の `telemetry` パラメータの型を `TelemetryClient` に変更
   - `run_context.save_json` の呼び出しを `TelemetryClient` 経由に変更

---

## 動作確認

### 静的解析
- ✅ リンターエラーなし（既存の警告のみ）
- ✅ 型アノテーションの整合性確認

### コードレビュー
- ✅ `TelemetryClient` が正しく実装されている
- ✅ `BrowserUseAgent` と `NavigationDriver` が `TelemetryClient` を使用している
- ✅ 既存のファイル名・タイミング・内容が維持されている
- ✅ フォールバック処理が正しく実装されている

### テスト
- ⚠️ 実ブラウザテストは未実施（環境の問題により）
- ✅ インポートテスト: 正常に動作
- ✅ 型チェック: エラーなし

---

## 設計上の改善点

### 1. 観測処理の集約
- 以前: `run_context.save_*` が各所に散在
- 現在: `TelemetryClient` 経由で統一

### 2. 将来の拡張性
- `TelemetryClient` を差し替えることで、S3 送信や外部オブザーバへの対応が可能
- インターフェースが明確に定義されているため、実装の差し替えが容易

### 3. テスト容易性
- `TelemetryClient` をモックすることで、観測処理をテストから分離可能

---

## 既知の制約・注意事項

1. **既存の `TelemetryService` との関係**
   - `TelemetryClient` は `TelemetryService` のラッパーとして実装
   - 既存の `TelemetryService` の機能はそのまま利用可能

2. **フォールバック処理**
   - `TelemetryClient` が失敗した場合、既存の `observability.py` 関数にフォールバック
   - これにより、既存の動作を維持

3. **`save_json` の非同期化**
   - `TelemetryClient.save_json` は非同期メソッドとして実装
   - 既存の同期的な `run_context.save_json` との互換性を維持

---

## 次のステップ（推奨）

### 1. 実ブラウザテスト
- Moncler などの代表的なサイトで動作確認
- 保存されるファイルが従来と同じか確認

### 2. テストカバレッジの向上
- `TelemetryClient` のユニットテストを追加
- `BrowserUseAgent` と `NavigationDriver` の統合テストを追加

### 3. ドキュメント整備
- `TelemetryClient` の使用方法をドキュメント化
- 既存の `TelemetryService` との違いを明確化

### 4. 段階的な移行
- 既存の `observability.py` 関数を段階的に非推奨化
- すべてのコードが `TelemetryClient` 経由になるまで継続

---

## まとめ

Stage 3B の実装により、観測処理が `TelemetryClient` に集約され、将来的な拡張性が向上しました。既存の動作は維持されており、段階的な移行が可能です。

**実装ステータス**: ✅ 完了
**動作確認**: ⚠️ 実ブラウザテストは未実施（環境の問題により）
**次のアクション**: 実ブラウザテストの実施を推奨

