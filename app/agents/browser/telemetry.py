# -*- coding: utf-8 -*-
"""
Stage 3B: TelemetryService - 観測機能（Telemetry）を一元管理するサービス

責務:
- DOM保存・スクリーンショット・失敗時アーティファクト生成
- 「いつ」「どの段階で」「何が起きたか」を記録
- ビジネスロジック（どの PDP を捨てるか等）は持たない

Version: 1.0.0 (Initial Implementation)
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page
    from app.core.run_context import RunContext
    from app.models.result_models import DiscoveryResult


class RunPhase(str, Enum):
    """実行フェーズを表すEnum"""
    PLP_DISCOVERY = "plp_discovery"      # PLP探索中
    PDP_EXTRACT = "pdp_extract"          # PDP抽出中
    RECOVERY = "recovery"                 # 回復試行中
    LEARNING = "learning"                 # 学習モード
    MATERIALIZE = "materialize"           # PLPマテリアライズ中


@dataclass
class FailureContext:
    """失敗時のコンテキスト情報"""
    site_code: str
    url: str
    phase: RunPhase
    exception: Optional[Exception] = None
    retry_count: int = 0
    query: Optional[str] = None
    site_config: Optional[Dict[str, Any]] = None
    intent_description: Optional[str] = None


@dataclass
class TelemetryContext:
    """
    Stage 3B: Telemetry 実行時のコンテキスト情報
    
    このデータクラスは、TelemetryClient の各メソッドに渡されるコンテキスト情報を保持します。
    """
    site: str
    query: str
    run_id: Optional[str] = None
    stage: Optional[str] = None  # "plp", "pdp", "fail_plp" など


class TelemetryService:
    """
    Stage 3B: 観測機能（Telemetry）を一元管理するサービス
    
    責務:
    - DOM保存・スクリーンショット・失敗時アーティファクト生成
    - 「いつ」「どの段階で」「何が起きたか」を記録
    - ビジネスロジックは持たない
    """
    
    def __init__(
        self,
        run_context: "RunContext",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        TelemetryService を初期化
        
        Args:
            run_context: 実行コンテキスト（DOM/JSON/スクショ保存先を管理）
            logger: ロガー（指定がない場合はモジュールロガーを使用）
        """
        self.run_context = run_context
        self.logger = logger or logging.getLogger(__name__)
    
    async def record_plp_state(
        self,
        page: "Page",
        *,
        name: str = "plp_dom_initial",
        selectors: Optional[List[str]] = None,
        site_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        PLP ロード直後の DOM/スクショ保存
        
        Args:
            page: Playwright Page オブジェクト
            name: 保存ファイル名のベース（拡張子は自動付与）
            selectors: セレクタカウント対象（オプション）
            site_config: サイト設定（selectors が None の場合、ここから自動取得）
        """
        if not page or page.is_closed():
            return
        
        # selectors が None で site_config がある場合、自動取得
        if selectors is None and site_config:
            pdp_cfg = (site_config.get("selectors") or {}).get("pdp", {}) or {}
            selectors = (
                (pdp_cfg.get("pdp_link_selectors") or []) +
                (pdp_cfg.get("plp_container_selectors") or [])
            )
        
        try:
            # DOM保存（ファイル名を固定: plp_dom_initial_materialized.html）
            if name == "plp_dom_initial" or name == "plp_dom_initial_materialized":
                await self._save_dom(page, "plp_dom_initial_materialized")
            else:
                await self._save_dom(page, name)
            
            # セレクタカウント（指定がある場合、ファイル名を固定: selector_counts_plp_initial.json）
            if selectors:
                if name == "plp_dom_initial" or name == "plp_dom_initial_materialized":
                    await self._count_selectors(page, selectors, name="selector_counts_plp_initial")
                else:
                    await self._count_selectors(page, selectors, name=f"selector_counts_{name}")
            
            # スクリーンショット（RunContextがサポートしている場合）
            if hasattr(self.run_context, "take_screenshot"):
                try:
                    await self.run_context.take_screenshot(page, f"30_{name}")
                except Exception as e:
                    self.logger.debug(f"[Telemetry] Screenshot failed for {name}: {e}")
        except Exception as e:
            self.logger.warning(f"[Telemetry] Failed to record PLP state '{name}': {e}")
    
    async def record_success(
        self,
        phase: RunPhase,
        *,
        result: Optional["DiscoveryResult"] = None,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        成功時のメタ情報や差分更新など
        
        Args:
            phase: 実行フェーズ
            result: DiscoveryResult（オプション）
            url: 成功時のURL（オプション）
            metadata: 追加メタデータ（オプション）
        """
        try:
            record = {
                "phase": phase.value,
                "timestamp": time.time(),
                "url": url,
            }
            
            if result:
                record["result"] = {
                    "ok": result.ok,
                    "site": result.site,
                    "query": result.query,
                    "message": result.message,
                }
            
            if metadata:
                record.update(metadata)
            
            await self._save_json(record, f"success_{phase.value}")
        except Exception as e:
            self.logger.warning(f"[Telemetry] Failed to record success for {phase}: {e}")
    
    async def record_failure(
        self,
        failure: FailureContext,
        *,
        page: Optional["Page"] = None,
    ) -> None:
        """
        失敗時の DOM / スクショ / ログ一括処理
        
        Args:
            failure: 失敗コンテキスト
            page: Playwright Page オブジェクト（オプション）
        """
        try:
            # DOM保存
            if page and not page.is_closed():
                await self._save_dom(page, "failure_dom")
            
            # スクリーンショット
            screenshot_path: Optional[str] = None
            if page and not page.is_closed() and hasattr(self.run_context, "take_screenshot"):
                try:
                    screenshot_path = await self.run_context.take_screenshot(page, "99_failure")
                except Exception as e:
                    self.logger.debug(f"[Telemetry] Failed to take failure screenshot: {e}")
            
            # セレクタカウント（site_configがある場合）
            visible_counts: Dict[str, int] = {}
            if page and not page.is_closed() and failure.site_config:
                try:
                    pdp_cfg = (failure.site_config.get("selectors") or {}).get("pdp", {}) or {}
                    markers = list(dict.fromkeys(
                        (pdp_cfg.get("title_selectors") or []) +
                        (pdp_cfg.get("price_selectors") or [])
                    ))
                    for s in markers:
                        try:
                            visible_counts[s] = await page.locator(s).count()
                        except Exception:
                            visible_counts[s] = -1
                except Exception as e:
                    self.logger.warning(f"[Telemetry] Could not count visible markers: {e}")
            
            # 失敗レポート生成
            await self._write_fail_snapshot(
                failure=failure,
                screenshot_path=screenshot_path,
                visible_counts=visible_counts,
            )
        except Exception as e:
            self.logger.error(f"[Telemetry] Failed to record failure: {e}", exc_info=True)
    
    # --- 内部メソッド（既存のobservability.pyから移行） ---
    
    async def _save_dom(self, page: "Page", name: str) -> None:
        """DOM保存の内部実装"""
        if not page or page.is_closed():
            return
        try:
            html = await page.content()
            await self._maybe_await(self.run_context.save_content(f"{name}.html", html))
        except Exception as e:
            self.logger.warning(f"[Telemetry] Failed to save DOM for '{name}': {e}")
    
    async def _save_json(self, data: Any, name: str) -> None:
        """JSON保存の内部実装"""
        try:
            await self._maybe_await(self.run_context.save_json(f"{name}.json", data))
        except Exception as e:
            self.logger.warning(f"[Telemetry] Failed to save JSON for '{name}': {e}")
    
    async def _count_selectors(
        self,
        page: "Page",
        selectors: List[str],
        *,
        name: str = "selector_counts",
    ) -> None:
        """セレクタカウントの内部実装"""
        if not page or page.is_closed():
            return
        counts: Dict[str, int] = {}
        for s in selectors or []:
            try:
                counts[s] = await page.locator(s).count()
            except Exception:
                counts[s] = -1
        await self._save_json({"selector_counts": counts}, name)
    
    async def _write_fail_snapshot(
        self,
        failure: FailureContext,
        *,
        screenshot_path: Optional[str] = None,
        visible_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """失敗スナップショットの生成"""
        note = f"Final URL: {failure.url}\nError: {failure.exception}"
        visible_counts = visible_counts or {}
        
        md_report = f"""# Failure Snapshot

- **Run ID:** `{getattr(self.run_context, 'run_id', 'unknown')}`
- **Timestamp:** `{time.strftime('%Y-%m-%d %H:%M:%S Z', time.gmtime())}`
- **Phase:** `{failure.phase.value}`
- **Site:** `{failure.site_code}`
- **Query:** `{failure.query or 'N/A'}`
- **Retry Count:** `{failure.retry_count}`
- **Note:** `{note}`

## Key Selector Counts at Failure
```json
{json.dumps(visible_counts, indent=2)}
```

## Available Artifacts
- `failure_dom.html` (if page was available)
- `{screenshot_path or '99_failure.png (if captured)'}`
- `plp_dom_initial_materialized.html`
- `plp_dom_search_fallback.html`
- `selector_counts_plp_initial.json`
- `selector_counts_after_search_fallback.json`
- `raw_hrefs_raw_abs.json`
- `network.har`
- `trace.zip`
"""
        await self._maybe_await(self.run_context.save_content("fail_snapshot.md", md_report))
    
    async def record_raw_hrefs(
        self,
        hrefs: List[str],
        *,
        name: str = "raw_hrefs",
        limit: int = 200,
    ) -> None:
        """
        URLリストをJSONファイルとして保存
        
        Args:
            hrefs: 保存するURLのリスト
            name: 保存ファイル名のベース（拡張子は自動付与）
            limit: 保存する最大URL数（デフォルト: 200）
        """
        try:
            data = {
                "count": len(hrefs or []),
                "raw_hrefs": list(hrefs or [])[:limit],
            }
            await self._save_json(data, name)
        except Exception as e:
            self.logger.warning(f"[Telemetry] Failed to save raw hrefs '{name}': {e}")
    
    # --- 互換性メソッド（observability.py の関数シグネチャに合わせる） ---
    # これらは既存コードとの互換性を保つために提供されます
    
    async def save_dom(self, page: "Page", name: str) -> None:
        """
        DOM保存（observability.py の save_dom と互換）
        
        Args:
            page: Playwright Page オブジェクト
            name: 保存ファイル名のベース（拡張子は自動付与）
        """
        await self._save_dom(page, name)
    
    async def count_selectors(
        self,
        page: "Page",
        selectors: List[str],
        *,
        name: str = "selector_counts",
    ) -> None:
        """
        セレクタカウント（observability.py の count_selectors と互換）
        
        Args:
            page: Playwright Page オブジェクト
            selectors: カウント対象のセレクタリスト
            name: 保存ファイル名のベース（拡張子は自動付与）
        """
        await self._count_selectors(page, selectors, name=name)
    
    async def save_raw_hrefs(
        self,
        hrefs: List[str],
        *,
        name: str = "raw_hrefs",
        limit: int = 200,
    ) -> None:
        """
        URLリスト保存（observability.py の save_raw_hrefs と互換）
        
        Args:
            hrefs: 保存するURLのリスト
            name: 保存ファイル名のベース（拡張子は自動付与）
            limit: 保存する最大URL数（デフォルト: 200）
        """
        await self.record_raw_hrefs(hrefs, name=name, limit=limit)
    
    async def write_fail_snapshot(
        self,
        page: Optional["Page"],
        final_url: Optional[str],
        error: Exception,
        site_config: Dict[str, Any],
    ) -> None:
        """
        失敗スナップショット生成（observability.py の write_fail_snapshot と互換）
        
        Args:
            page: Playwright Page オブジェクト（オプション）
            final_url: 失敗時の最終URL
            error: 発生した例外
            site_config: サイト設定
        """
        # FailureContext に変換
        # site_code は site_config から取得、なければ "unknown"
        site_code = site_config.get("site_code") or site_config.get("site") or "unknown"
        failure = FailureContext(
            site_code=site_code,
            url=final_url or "unknown",
            phase=RunPhase.PLP_DISCOVERY,  # デフォルト値（既存コードでは特定されていない）
            exception=error,
            site_config=site_config,
        )
        await self.record_failure(failure, page=page)
    
    async def _maybe_await(self, x: Any) -> Any:
        """引数が Awaitable であれば await し、そうでなければそのまま返す"""
        return (await x) if inspect.isawaitable(x) else x


class TelemetryClient:
    """
    Stage 3B: BrowserUseAgent / NavigationDriver から利用される Telemetry Facade
    
    実際の保存先は内部で run_context に委譲する。
    TelemetryService のラッパーとして機能し、よりシンプルなインターフェースを提供します。
    """
    
    def __init__(self, run_context: Any, base_dir: str = "") -> None:
        """
        TelemetryClient を初期化
        
        Args:
            run_context: RunContext オブジェクト（DOM/JSON/スクショ保存先を管理）
            base_dir: ベースディレクトリ（現在は未使用、将来の拡張用）
        """
        self.run_context = run_context
        self.base_dir = base_dir
        # TelemetryService を内部で使用
        self._service = TelemetryService(run_context=run_context)
        self.logger = logging.getLogger(__name__)
    
    async def save_dom(
        self,
        page: Any,
        name: str,
        tctx: TelemetryContext,
    ) -> None:
        """
        DOM を保存する
        
        Args:
            page: Playwright Page オブジェクト
            name: 保存ファイル名のベース（拡張子は自動付与）
            tctx: TelemetryContext（現在は未使用、将来の拡張用）
        """
        await self._service.save_dom(page, name)
    
    async def save_json(
        self,
        name: str,
        payload: Dict[str, Any],
        tctx: TelemetryContext,
    ) -> None:
        """
        JSON を保存する
        
        Args:
            name: 保存ファイル名のベース（拡張子は自動付与）
            payload: 保存するJSONデータ
            tctx: TelemetryContext（現在は未使用、将来の拡張用）
        """
        await self._service._save_json(payload, name)
    
    async def save_screenshot(
        self,
        page: Any,
        name: str,
        tctx: TelemetryContext,
    ) -> None:
        """
        スクリーンショットを保存する
        
        Args:
            page: Playwright Page オブジェクト
            name: 保存ファイル名のベース（拡張子は自動付与）
            tctx: TelemetryContext（現在は未使用、将来の拡張用）
        """
        if hasattr(self.run_context, "take_screenshot"):
            try:
                await self.run_context.take_screenshot(page, name)
            except Exception as e:
                self.logger.warning(f"[TelemetryClient] Failed to save screenshot '{name}': {e}")
        else:
            self.logger.debug(f"[TelemetryClient] RunContext does not support take_screenshot")
    
    async def record_plp_state(
        self,
        page: Any,
        *,
        name: str = "plp_dom_initial_materialized",
        selectors: Optional[List[str]] = None,
        site_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        PLP 初期状態の DOM とセレクタカウントを保存するための共通 API（Phase1 Moncler 診断用）
        
        Args:
            page: Playwright Page オブジェクト
            name: 保存ファイル名のベース（デフォルト: "plp_dom_initial_materialized"）
            selectors: セレクタカウント対象（オプション）
            site_config: サイト設定（selectors が None の場合、ここから自動取得）
        
        保存されるファイル:
        - plp_dom_initial_materialized.html（DOM スナップショット）
        - selector_counts_plp_initial.json（セレクタカウント、selectors が指定された場合のみ）
        """
        try:
            await self._service.record_plp_state(
                page=page,
                name=name,
                selectors=selectors,
                site_config=site_config,
            )
        except Exception as e:
            self.logger.warning(f"[TelemetryClient] Failed to record PLP state '{name}': {e}", exc_info=True)
    
    async def write_fail_snapshot(
        self,
        page: Any,
        reason: str,
        tctx: TelemetryContext,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        失敗スナップショット（DOM + SS + メタ情報まとめ）を生成する
        
        Args:
            page: Playwright Page オブジェクト（オプション）
            reason: 失敗理由（エラーメッセージなど）
            tctx: TelemetryContext
            extra: 追加情報（site_config など）
        """
        # extra から site_config を取得（既存コードとの互換性のため）
        site_config = extra.get("site_config", {}) if extra else {}
        
        # エラーオブジェクトを作成（既存の write_fail_snapshot シグネチャに合わせる）
        class ErrorWrapper(Exception):
            def __init__(self, message: str):
                self.message = message
                super().__init__(message)
        
        error = ErrorWrapper(reason)
        final_url = None
        if page and not page.is_closed():
            try:
                final_url = page.url
            except Exception:
                pass
        
        # TelemetryService の write_fail_snapshot を使用
        await self._service.write_fail_snapshot(
            page=page,
            final_url=final_url,
            error=error,
            site_config=site_config,
        )

