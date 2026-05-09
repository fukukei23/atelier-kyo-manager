# ==============================================================================
# ファイル名 : app/utils/interactive_repair_session.py
# 目的       : スクレイピング失敗時に "Atlas的エージェントループ" を回すコア
# バージョン : 2025-10-31 JST draft-1
#
# このモジュールがやること
# ------------------------
# 1. BrowserUseAgent 側で PLP/PDP 取得に失敗したら呼び出す
# 2. 失敗状況(現在URL, DOMスナップショット, エラーログ, 試したセレクタなど)をLLMに渡す
# 3. LLM(GPT-5優先)が「次の一手」を JSON で返す
#    例:
#    {
#      "action_type": "click",
#      "selector": "button[aria-label='Accept all cookies']",
#      "rationale": "Cookie壁を閉じるため"
#    }
# 4. この一手を Playwright page に実行する
# 5. ゴール(= 商品一覧が見えて商品リンクが取れる状態)になれば終了
# 6. 最後に LLM に「今回の突破手順をふまえて:
#       - selectors更新案(JSON)
#       - 差分パッチ(unified diff)
#    を返して」
# 7. 呼び出し元へまとめて返す
#
# 想定している呼び出し元
# ----------------------
# BrowserUseAgent 内の「PLPを開く→商品カードを読む」の処理で失敗したとき:
#
#   from app.utils.interactive_repair_session import InteractiveRepairSession
#
#   session = InteractiveRepairSession(ai_controller, run_context)
#   repair_result = await session.run_repair_loop(page, site_key, intent, failure_ctx)
#
# repair_result は dict で、ざっくりこんな形を想定:
#
# {
#   "status": "recovered" or "failed",
#   "selectors_update": {...},  # llm提案のselectors差分 (永続化OK領域)
#   "code_patch": "....diff....",  # browser_use_agent.pyの修正案 (パッチファイルとして保存する)
#   "steps_taken": [...],  # エージェントが実行した一連のアクション履歴
#   "final_url": "...",
#   "final_dom_snapshot": "/path/to/dom_after.html",
# }
#
# 注意
# ----
# - ここでは "次の一手" の種類をいくつか定義する:
#     click / goto / wait_for / scroll / done
# - "done" が返ってきたらゴールに到達したとみなしてループ終了する
# - ループが上限手数に達したら失敗として返す
#
# - LLM側の出力フォーマットは厳密にJSON (Python側でjson.loadsする)
#   → AiLlmControllerに "必ず有効なJSONだけで返せ" とプロンプトで縛る
#
# - run_context はあなたのシステムに既にある実行ログ・保存パス管理オブジェクトを想定
#   もし run_context が無い/簡略化したいなら、一旦 None でも動くようにしている。
#
# ==============================================================================
from __future__ import annotations

import contextlib
import json
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

# あなたの既存コード側にあるはずのもの:
# - AiLlmController: LLMへの統一インターフェース
#   from app.utils.ai_llm_controller import AiLlmController
# - run_context: 実行中のメタ(ログ保存ディレクトリ等)を管理するオブジェクト
#   これは ai_research_orchestrator / supplier_scout_agent が持ってるはず
try:
    from app.utils.ai_llm_controller import AiLlmController
except Exception:
    # テスト環境用のフォールバック (存在しないとImportErrorするので)
    AiLlmController = Any  # type: ignore


class InteractiveRepairSession:
    """
    GPT-5を"案内役"として、Playwright pageを段階的に操作しながら
    目的のPLP/PDP到達を試みるための小さなセッションオブジェクト。
    """

    def __init__(
        self,
        ai_controller: AiLlmController,
        run_context: Any | None = None,
        max_steps: int = 5,
        snapshot_dir: str | None = None,
    ) -> None:
        """
        ai_controller: AiLlmController インスタンス (GPT-5, Gemini, DeepSeekの優先順を持ってるやつ)
        run_context : 実行ログやスクリーンショット保存用のコンテキスト(無くてもOK)
        max_steps   : LLMに指示を何ターンまで実行するか (無限ループ防止)
        snapshot_dir: DOMやスクショを一時保存するディレクトリ。指定なければ ./instance/repair_session/
        """
        self.ai = ai_controller
        self.rc = run_context
        self.max_steps = max_steps

        if snapshot_dir:
            self.snapshot_root = Path(snapshot_dir)
        else:
            self.snapshot_root = Path("instance") / "repair_session"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

        # このセッションのユニークID（ログフォルダ分離に使う）
        self.session_id = f"repair_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.session_dir = self.snapshot_root / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.steps_log: list[dict[str, Any]] = []  # 各ターンの記録

    async def _save_dom_and_screenshot(
        self,
        page,
        label: str,
    ) -> tuple[str, str]:
        """
        現在のDOMとスクショをローカルに保存してパスを返す。
        page: PlaywrightのPage
        label: ファイル名に含める任意ラベル (ex: "before_step1", "after_click")
        """
        timestamp = int(time.time() * 1000)
        safe_label = label.replace(" ", "_").replace("/", "_")

        html_path = self.session_dir / f"{timestamp}_{safe_label}.html"
        png_path = self.session_dir / f"{timestamp}_{safe_label}.png"

        try:
            html = await page.content()
        except Exception as e:
            html = f"<!DOCTYPE html><html><body><pre>DOM_FETCH_ERROR: {e}</pre></body></html>"

        try:
            await page.screenshot(path=str(png_path), full_page=True)
        except Exception:
            # スクショ取れない場合もある (クロスオリジンpopupなど)
            with open(png_path, "wb") as f:
                f.write(b"")  # 空ファイルでも一応確保
            # we ignore screenshot failure, but keep going

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return str(html_path), str(png_path)

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
        """
        LLM (GPT-5優先) に「次の一手」を聞くためのプロンプト文字列を組み立てる。
        ポイント:
        - 必ず有効なJSONだけを返してほしいと明示する
        - "action_type" は click / goto / wait_for / scroll / done のいずれか
        - セレクタはCSSまたはXPathでも良いが、CSSを優先要求
        """
        # DOMはそのまま渡すと巨大なので先頭だけ(安全側)
        dom_excerpt = recent_dom_text[:4000]

        steps_repr = json.dumps(steps_so_far[-5:], ensure_ascii=False, indent=2)

        prompt = f"""
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
3. 必ず有効なJSONのみで返してください。余計な文章を書くと失敗扱いになります。

返すJSONの仕様:
{{
  "action_type": "click" | "goto" | "wait_for" | "scroll" | "done",
  "selector": "CSSセレクタ文字列 or URL or scroll-directionなど",
  "rationale": "短い説明"
}}

制約:
- "click" の場合: 現在のpage.clickに渡せるCSSセレクタを返してください。
- "goto"  の場合: selectorには移動先URLを入れてください。
- "wait_for" の場合: selectorには待機すべきCSSセレクタを入れてください。
- "scroll" の場合: selectorに "down" または "up" または "bottom" など方向を書く。
- "done" の場合: selectorは空文字 "" にしてください。これは「目標のPLPや商品カードが取得できる状態になった」と判断したことを意味します。

必ず上のJSONだけを出力してください。
"""
        return prompt.strip()

    async def _execute_action_step(
        self,
        page,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        """
        LLMから返された1手(action)をPlaywrightで実行する。
        サポートする action_type:
            - click:   page.click(selector)
            - goto:    page.goto(url)
            - wait_for: page.wait_for_selector(selector)
            - scroll:  page.evaluate("window.scrollTo(...)")
            - done:    何もしない
        実行後のURLとスクショ/DOMをキャプチャして返す。
        """
        action_type = action.get("action_type")
        selector = action.get("selector", "")
        action.get("rationale", "")

        exec_error = None

        try:
            if action_type == "click":
                await page.click(selector, timeout=5000)
            elif action_type == "goto":
                await page.goto(selector, timeout=15000)
            elif action_type == "wait_for":
                await page.wait_for_selector(selector, timeout=8000)
            elif action_type == "scroll":
                direction = selector or "down"
                if direction == "down":
                    await page.evaluate("window.scrollBy(0, window.innerHeight);")
                elif direction == "bottom":
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                elif direction == "up":
                    await page.evaluate("window.scrollBy(0, -window.innerHeight);")
                else:
                    # fallback: just scroll down
                    await page.evaluate("window.scrollBy(0, window.innerHeight);")
            elif action_type == "done":
                # 何もしない
                pass
            else:
                exec_error = f"Unknown action_type: {action_type}"

        except Exception as e:
            exec_error = f"Action execution failed: {e}\n{traceback.format_exc()}"

        # 実行後の状態をキャプチャ
        try:
            current_url = page.url
        except Exception:
            current_url = "UNKNOWN_URL_AFTER_ACTION"

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

        # ローカルのステップログにも積む
        self.steps_log.append(step_record)
        return step_record

    def _build_llm_prompt_for_finalize(
        self,
        site_key: str,
        intent: str,
        steps: list[dict[str, Any]],
    ) -> str:
        """
        ループが終了したあとに呼ぶ。
        - 今回の突破手順をもとに、差分パッチ(unified diff)と
          selectors更新案(JSON)をまとめて提案させる。
        """
        steps_json = json.dumps(steps, ensure_ascii=False, indent=2)

        prompt = f"""
あなたはスクレイピング自己修復エージェントです。
以下は {site_key} サイトで「{intent}」を達成するまでに実行した手順ログです。
この手順は実際のブラウザに対して実行され、最終的に目的に近い画面まで到達しました。

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
          "plp_container_selectors": ["...CSS...", "...CSS..."],
          "pdp_link_selectors": ["...CSS for product links..."]
        }}
      }},
      "discovery_settings": {{
        "fallback_url": "もし安定して入れたURLがあれば記載。無ければ空文字でも良い"
      }}
    }}
  }},
  "code_patch": "```diff\\n...unified diff for browser_use_agent.py...\\n```"
}}

制約:
- "selectors_update" は overrides.local.json や learned_selectors.json にマージ可能な形で書いてください。
- "code_patch" は Python関数(例: collect_plp_links)の修正案のみを書いてください。
- code_patch は unified diff 形式で、```diff で始まり ``` で終わる1ブロックだけにしてください。
- 余計な説明文は書かず、上のJSONだけをそのまま返してください。
"""
        return prompt.strip()

    async def run_repair_loop(
        self,
        page,
        site_key: str,
        intent: str,
        initial_failure: dict[str, Any],
    ) -> dict[str, Any]:
        """
        メインエントリ。
        - BrowserUseAgent から呼ばれることを想定
        - initial_failure: {
            "final_url": "...",
            "page_html_path": "...",
            "screenshot_path": "...",
            "exception_message": "...",
            "selectors_used": {...}
          }
        戻り値は repair_result(dict)
        """
        # 1. 初期状態のDOMを読み込む
        try:
            with open(initial_failure.get("page_html_path", ""), encoding="utf-8") as f:
                dom_text = f.read()
        except Exception:
            dom_text = ""

        current_url = initial_failure.get("final_url", "")
        tried_selectors = initial_failure.get("selectors_used", {})
        last_error = initial_failure.get("exception_message", "")

        # 初期スナップショットもsteps_logに積んでおく
        self.steps_log.append(
            {
                "action": {"action_type": "init_state", "selector": "", "rationale": "failure context captured"},
                "exec_error": None,
                "after_url": current_url,
                "after_html_path": initial_failure.get("page_html_path", ""),
                "after_screenshot_path": initial_failure.get("screenshot_path", ""),
            }
        )

        # 2. アクションループ
        for _step_i in range(self.max_steps):
            # LLMに「次の一手」を問い合わせ
            prompt = self._build_llm_prompt_for_next_action(
                site_key=site_key,
                intent=intent,
                current_url=current_url,
                recent_dom_text=dom_text,
                last_error=last_error,
                tried_selectors=tried_selectors,
                steps_so_far=self.steps_log,
            )

            # AiLlmController で GPT-5優先の"analysis"タスクとして問い合わせる
            llm_raw = self.ai.generate(
                task="analysis",
                prompt=prompt,
                expect_json=True,  # ← AiLlmController側で安全にjson.loadsするなら使える
            )
            # llm_raw は既に Pythonのdict になって返る想定
            # （もし ai.generate がまだ文字列を返す実装なら、ここで json.loads する）

            action_type = llm_raw.get("action_type")
            if action_type == "done":
                # 目標到達したとLLMが判断
                break

            # LLMの指示を実行
            step_outcome = await self._execute_action_step(page, llm_raw)

            # 次のループのためにURLとDOMを更新
            current_url = step_outcome.get("after_url", current_url)
            new_html_path = step_outcome.get("after_html_path", "")
            try:
                with open(new_html_path, encoding="utf-8") as f:
                    dom_text = f.read()
            except Exception:
                dom_text = dom_text  # fallback

            # いちおう last_error を更新(実行エラーがあれば)
            last_error = step_outcome.get("exec_error")

        # 3. ループ終了後、LLMに最終まとめ(パッチ/セレクタ更新案)を要求
        finalize_prompt = self._build_llm_prompt_for_finalize(
            site_key=site_key,
            intent=intent,
            steps=self.steps_log,
        )

        finalize_raw = self.ai.generate(
            task="code",  # コード差分も含むので"code"レーンを優先
            prompt=finalize_prompt,
            expect_json=True,
        )

        # finalize_raw の想定:
        # {
        #   "selectors_update": {...},
        #   "code_patch": "```diff\n ... \n```"
        # }

        selectors_update = finalize_raw.get("selectors_update", {})
        code_patch = finalize_raw.get("code_patch", "")

        result = {
            "status": "recovered",
            "selectors_update": selectors_update,
            "code_patch": code_patch,
            "steps_taken": self.steps_log,
            "final_url": current_url,
            "final_dom_snapshot": new_html_path if "new_html_path" in locals() else "",
            "session_dir": str(self.session_dir),
        }

        return result
