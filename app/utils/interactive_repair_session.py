# スクレイピング失敗時の LLM駆動 自己修復セッション
from __future__ import annotations

import contextlib
import json
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

try:
    from app.utils.ai_llm_controller import AiLlmController
except Exception:
    AiLlmController = Any  # type: ignore

_SCROLL_JS = {
    "down": "window.scrollBy(0, window.innerHeight);",
    "bottom": "window.scrollTo(0, document.body.scrollHeight);",
    "up": "window.scrollBy(0, -window.innerHeight);",
}
_DEFAULT_SCROLL_JS = "window.scrollBy(0, window.innerHeight);"

_ACTION_SPEC = (
    '{"action_type": "click" | "goto" | "wait_for" | "scroll" | "done", '
    '"selector": "CSSセレクタ or URL or 方向", "rationale": "説明"}'
)


class InteractiveRepairSession:

    def __init__(
        self,
        ai_controller: AiLlmController,
        run_context: Any | None = None,
        max_steps: int = 5,
        snapshot_dir: str | None = None,
    ) -> None:
        self.ai = ai_controller
        self.rc = run_context
        self.max_steps = max_steps
        self.snapshot_root = Path(snapshot_dir) if snapshot_dir else Path("instance") / "repair_session"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

        self.session_id = f"repair_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.session_dir = self.snapshot_root / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.steps_log: list[dict[str, Any]] = []

    # ── Snapshot ──────────────────────────────────────────

    async def _save_dom_and_screenshot(self, page, label: str) -> tuple[str, str]:
        ts = int(time.time() * 1000)
        safe = label.replace(" ", "_").replace("/", "_")
        html_path = self.session_dir / f"{ts}_{safe}.html"
        png_path = self.session_dir / f"{ts}_{safe}.png"

        try:
            html = await page.content()
        except Exception as e:
            html = f"<!DOCTYPE html><html><body><pre>DOM_FETCH_ERROR: {e}</pre></body></html>"

        try:
            await page.screenshot(path=str(png_path), full_page=True)
        except Exception:
            with open(png_path, "wb") as f:
                f.write(b"")

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return str(html_path), str(png_path)

    # ── Action dispatch ───────────────────────────────────

    async def _dispatch_action(self, page, action_type: str, selector: str) -> None:
        if action_type == "click":
            await page.click(selector, timeout=5000)
        elif action_type == "goto":
            await page.goto(selector, timeout=15000)
        elif action_type == "wait_for":
            await page.wait_for_selector(selector, timeout=8000)
        elif action_type == "scroll":
            await page.evaluate(_SCROLL_JS.get(selector or "down", _DEFAULT_SCROLL_JS))

    async def _execute_action_step(self, page, action: dict[str, Any]) -> dict[str, Any]:
        action_type = action.get("action_type", "")
        exec_error = None
        try:
            await self._dispatch_action(page, action_type, action.get("selector", ""))
        except Exception as e:
            exec_error = f"Action execution failed: {e}\n{traceback.format_exc()}"

        current_url = page.url
        with contextlib.suppress(Exception):
            await page.content()

        html_path, screenshot_path = await self._save_dom_and_screenshot(page, f"after_{action_type}")
        step_record = {
            "action": action,
            "exec_error": exec_error,
            "after_url": current_url,
            "after_html_path": html_path,
            "after_screenshot_path": screenshot_path,
        }
        self.steps_log.append(step_record)
        return step_record

    # ── Prompt builders ───────────────────────────────────

    def _build_llm_prompt_for_next_action(
        self,
        site_key: str,
        intent: str,
        current_url: str,
        recent_dom_text: str,
        last_error: str | None,
        tried_selectors: dict[str, Any],
        steps_so_far: list[dict[str, Any]],
    ) -> str:
        dom_excerpt = recent_dom_text[:4000]
        steps_repr = json.dumps(steps_so_far[-5:], ensure_ascii=False, indent=2)

        return f"""
あなたはECサイト調査用の自己修復スクレイピング専門エージェントです。
目的: 「{site_key}」サイトで「{intent}」を達成することです。
現在のURL: {current_url}

これまでの行動ログ(直近5手):
{steps_repr}

直近のエラー/問題:
{last_error}

いま画面上に見えているDOMの冒頭抜粋(約4000文字):
\"\"\"HTML_START
{dom_excerpt}
HTML_END\"\"\"

いまのスクレイパーは以下のセレクタ/戦略を試して失敗しました:
{json.dumps(tried_selectors, ensure_ascii=False, indent=2)}

あなたの仕事:
1. 次にやるべきアクションを1つだけ提案してください。
2. 「なぜそれをやるのか」の理由も短く説明してください。
3. 必ず有効なJSONのみで返してください。

返すJSONの仕様:
{_ACTION_SPEC}

制約:
- "click": page.click に渡せるCSSセレクタ
- "goto": selectorに移動先URL
- "wait_for": 待機すべきCSSセレクタ
- "scroll": "down" / "up" / "bottom"
- "done": 目標到達と判断した場合。selectorは空文字

必ず上のJSONだけを出力してください。
""".strip()

    def _build_llm_prompt_for_finalize(
        self, site_key: str, intent: str, steps: list[dict[str, Any]],
    ) -> str:
        steps_json = json.dumps(steps, ensure_ascii=False, indent=2)

        return f"""
あなたはスクレイピング自己修復エージェントです。
以下は {site_key} サイトで「{intent}」を達成するまでに実行した手順ログです。

手順ログ:
{steps_json}

お願い:
1. いまの学びを元に、将来は自動で同じ操作に成功させたい。
2. 次の2つを返してください。必ず有効なJSONだけを返してください。

フォーマット:
{{
  "selectors_update": {{
    "{site_key}": {{
      "selectors": {{
        "pdp": {{
          "plp_container_selectors": ["...CSS..."],
          "pdp_link_selectors": ["...CSS for product links..."]
        }}
      }},
      "discovery_settings": {{
        "fallback_url": "安定して入れたURL。無ければ空文字"
      }}
    }}
  }},
  "code_patch": "```diff\\n...unified diff...\\n```"
}}

制約:
- selectors_update は overrides.local.json にマージ可能な形で
- code_patch は unified diff 形式、```diff で始まり ``` で終わる1ブロックのみ
- 余計な説明文は書かず、上のJSONだけを返してください。
""".strip()

    # ── Loop helpers ──────────────────────────────────────

    def _load_initial_state(self, initial_failure: dict[str, Any]) -> tuple[str, str, dict, str]:
        try:
            with open(initial_failure.get("page_html_path", ""), encoding="utf-8") as f:
                dom_text = f.read()
        except Exception:
            dom_text = ""

        current_url = initial_failure.get("final_url", "")
        tried_selectors = initial_failure.get("selectors_used", {})
        last_error = initial_failure.get("exception_message", "")

        self.steps_log.append({
            "action": {"action_type": "init_state", "selector": "", "rationale": "failure context captured"},
            "exec_error": None,
            "after_url": current_url,
            "after_html_path": initial_failure.get("page_html_path", ""),
            "after_screenshot_path": initial_failure.get("screenshot_path", ""),
        })

        return dom_text, current_url, tried_selectors, last_error

    def _query_llm_for_action(self, **kwargs) -> dict[str, Any]:
        prompt = self._build_llm_prompt_for_next_action(**kwargs)
        return self.ai.generate(task="analysis", prompt=prompt, expect_json=True)

    def _run_finalize(self, site_key: str, intent: str) -> dict[str, Any]:
        prompt = self._build_llm_prompt_for_finalize(site_key=site_key, intent=intent, steps=self.steps_log)
        return self.ai.generate(task="code", prompt=prompt, expect_json=True)

    # ── Main entry ────────────────────────────────────────

    async def run_repair_loop(
        self, page, site_key: str, intent: str, initial_failure: dict[str, Any],
    ) -> dict[str, Any]:
        dom_text, current_url, tried_selectors, last_error = self._load_initial_state(initial_failure)
        new_html_path = ""

        for _ in range(self.max_steps):
            llm_raw = self._query_llm_for_action(
                site_key=site_key, intent=intent, current_url=current_url,
                recent_dom_text=dom_text, last_error=last_error,
                tried_selectors=tried_selectors, steps_so_far=self.steps_log,
            )
            if llm_raw.get("action_type") == "done":
                break

            outcome = await self._execute_action_step(page, llm_raw)
            current_url = outcome.get("after_url", current_url)
            new_html_path = outcome.get("after_html_path", "")
            try:
                with open(new_html_path, encoding="utf-8") as f:
                    dom_text = f.read()
            except Exception:
                pass
            last_error = outcome.get("exec_error")

        finalize = self._run_finalize(site_key, intent)
        return {
            "status": "recovered",
            "selectors_update": finalize.get("selectors_update", {}),
            "code_patch": finalize.get("code_patch", ""),
            "steps_taken": self.steps_log,
            "final_url": current_url,
            "final_dom_snapshot": new_html_path,
            "session_dir": str(self.session_dir),
        }
