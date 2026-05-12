"""
Stage 3B: TelemetryService - 観測機能（Telemetry）を一元管理するサービス

責務:
- DOM保存・スクリーンショット・失敗時アーティファクト生成
- 「いつ」「どの段階で」「何が起きたか」を記録
- ビジネスロジック（どの PDP を捨てるか等）は持たない

Version: 1.1.0 (TelemetryClient/dataclasses extracted to telemetry_client.py)
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.async_api import Page

    from app.core.run_context import RunContext
    from app.models.result_models import DiscoveryResult

# Re-export for backward compatibility
from app.agents.browser.telemetry_client import (  # noqa: F401
    FailureContext,
    RunPhase,
    TelemetryClient,
    TelemetryContext,
)


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
        run_context: RunContext,
        logger: logging.Logger | None = None,
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
        page: Page,
        *,
        name: str = "plp_dom_initial",
        selectors: list[str] | None = None,
        site_config: dict[str, Any] | None = None,
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
            selectors = (pdp_cfg.get("pdp_link_selectors") or []) + (pdp_cfg.get("plp_container_selectors") or [])

        try:
            self.logger.info(
                f"[Telemetry] Starting record_plp_state: name={name}, selectors={selectors is not None}, run_context={self.run_context}"
            )

            # CR-ATELIER-001: name が "plp_dom_initial" または "plp_dom_initial_materialized" の場合のみ固定ファイル名を使用
            # それ以外の場合は name パラメータを使用（例: "plp_trap_page" → "plp_trap_page.html"）
            if name in ("plp_dom_initial", "plp_dom_initial_materialized"):
                dom_filename = "plp_dom_initial_materialized"
                selector_filename = "selector_counts_plp_initial"
            else:
                dom_filename = name
                selector_filename = f"selector_counts_{name}"

            # DOM保存
            await self._save_dom(page, dom_filename)

            # セレクタカウント（指定がある場合）
            if selectors:
                self.logger.info(f"[Telemetry] Counting selectors: {len(selectors)} selectors")
                await self._count_selectors(page, selectors, name=selector_filename)
            else:
                self.logger.info("[Telemetry] No selectors provided, skipping selector count")

            # スクリーンショット（RunContextがサポートしている場合）
            if hasattr(self.run_context, "take_screenshot"):
                try:
                    await self.run_context.take_screenshot(page, f"30_{name}")
                except Exception as e:
                    self.logger.debug(f"[Telemetry] Screenshot failed for {name}: {e}")
            self.logger.info(f"[Telemetry] Completed record_plp_state: name={name}")
        except Exception as e:
            self.logger.warning(f"[Telemetry] Failed to record PLP state '{name}': {e}", exc_info=True)

    async def record_success(
        self,
        phase: RunPhase,
        *,
        result: DiscoveryResult | None = None,
        url: str | None = None,
        metadata: dict[str, Any] | None = None,
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
        page: Page | None = None,
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
            screenshot_path: str | None = None
            if page and not page.is_closed() and hasattr(self.run_context, "take_screenshot"):
                try:
                    screenshot_path = await self.run_context.take_screenshot(page, "99_failure")
                except Exception as e:
                    self.logger.debug(f"[Telemetry] Failed to take failure screenshot: {e}")

            # セレクタカウント（site_configがある場合）
            visible_counts: dict[str, int] = {}
            if page and not page.is_closed() and failure.site_config:
                try:
                    pdp_cfg = (failure.site_config.get("selectors") or {}).get("pdp", {}) or {}
                    markers = list(
                        dict.fromkeys((pdp_cfg.get("title_selectors") or []) + (pdp_cfg.get("price_selectors") or []))
                    )
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

    async def _save_dom(self, page: Page, name: str) -> None:
        """DOM保存の内部実装"""
        if not page or page.is_closed():
            self.logger.warning(f"[Telemetry] Cannot save DOM for '{name}': page is None or closed")
            return
        try:
            html = await page.content()
            filepath = f"{name}.html"
            self.logger.info(f"[Telemetry] Saving DOM to '{filepath}' (size: {len(html)} bytes)")
            await self._maybe_await(self.run_context.save_content(filepath, html))
            self.logger.info(f"[Telemetry] Successfully saved DOM to '{filepath}'")
        except Exception as e:
            self.logger.warning(f"[Telemetry] Failed to save DOM for '{name}': {e}", exc_info=True)

    async def _save_json(self, data: Any, name: str) -> None:
        """JSON保存の内部実装"""
        try:
            filepath = f"{name}.json"
            self.logger.info(f"[Telemetry] Saving JSON to '{filepath}'")
            await self._maybe_await(self.run_context.save_json(filepath, data))
            self.logger.info(f"[Telemetry] Successfully saved JSON to '{filepath}'")
        except Exception as e:
            self.logger.warning(f"[Telemetry] Failed to save JSON for '{name}': {e}", exc_info=True)

    async def _count_selectors(
        self,
        page: Page,
        selectors: list[str],
        *,
        name: str = "selector_counts",
    ) -> None:
        """セレクタカウントの内部実装"""
        if not page or page.is_closed():
            return
        counts: dict[str, int] = {}
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
        screenshot_path: str | None = None,
        visible_counts: dict[str, int] | None = None,
    ) -> None:
        """失敗スナップショットの生成"""
        note = f"Final URL: {failure.url}\nError: {failure.exception}"
        visible_counts = visible_counts or {}

        md_report = f"""# Failure Snapshot

- **Run ID:** `{getattr(self.run_context, "run_id", "unknown")}`
- **Timestamp:** `{time.strftime("%Y-%m-%d %H:%M:%S Z", time.gmtime())}`
- **Phase:** `{failure.phase.value}`
- **Site:** `{failure.site_code}`
- **Query:** `{failure.query or "N/A"}`
- **Retry Count:** `{failure.retry_count}`
- **Note:** `{note}`

## Key Selector Counts at Failure
```json
{json.dumps(visible_counts, indent=2)}
```

## Available Artifacts
- `failure_dom.html` (if page was available)
- `{screenshot_path or "99_failure.png (if captured)"}`
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
        hrefs: list[str],
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

    async def save_dom(self, page: Page, name: str) -> None:
        """
        DOM保存（observability.py の save_dom と互換）

        Args:
            page: Playwright Page オブジェクト
            name: 保存ファイル名のベース（拡張子は自動付与）
        """
        await self._save_dom(page, name)

    async def count_selectors(
        self,
        page: Page,
        selectors: list[str],
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
        hrefs: list[str],
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
        page: Page | None,
        final_url: str | None,
        error: Exception,
        site_config: dict[str, Any],
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

    async def record_moncler_plp_pdp_outcome(
        self,
        outcome: dict[str, Any],
    ) -> None:
        """
        CR-ATELIER-002 Step 6: Moncler PLP→PDP 抽出結果を Telemetry に記録

        Args:
            outcome: 抽出結果の辞書。以下のフィールドを含む:
                - plp_materialized: bool
                - tiles_detected: int
                - pdp_links_raw: int
                - pdp_links_accepted: int
                - selector_layers_used: list[str]  # ["primary", "secondary", ...]
                - layer_stats: dict  # レイヤ別件数など
                - locale_corrections: int
                - trap_detected: bool
                - current_url: str
                - run_id: str
                - timestamp: str  # ISO8601 文字列（オプション、なければ自動生成）
        """
        try:
            # timestamp がなければ自動生成
            if "timestamp" not in outcome:
                from datetime import datetime

                outcome["timestamp"] = datetime.utcnow().isoformat() + "Z"

            # キー名は設計書に準拠
            await self._save_json(outcome, "moncler_plp_pdp_outcome")
            self.logger.info(
                f"[Telemetry][Moncler] Recorded PLP→PDP outcome: "
                f"raw={outcome.get('pdp_links_raw', 0)}, "
                f"accepted={outcome.get('pdp_links_accepted', 0)}, "
                f"layers={outcome.get('selector_layers_used', [])}"
            )
        except Exception as e:
            self.logger.warning(f"[Telemetry][Moncler] Failed to record PLP→PDP outcome: {e}", exc_info=True)
