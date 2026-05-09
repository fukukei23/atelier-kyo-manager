# ==============================================================================
# ファイル名 (File Name): selector_repair_agent.py
# レジストリ (Registry): app/agents/selector_repair_agent.py
# 更新日時 (Date & Time JST): 2025年12月11日
# バージョン (Version): 9.0.0J (Phase D-10: Selector Auto-Healing)
#
# --- v9.0.0Jでの主な変更点 ---
# - [Phase D-10] propose_selector_patches メソッドを追加
# - [Phase D-10] failure_context / failure_analysis / DOM snapshot を入力として受け取る
# - [Phase D-10] 仕様書に準拠した JSON schema を返す
# - [Phase D-10] AiLlmController を実際の実装からインポート
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
from typing import Any

try:
    from app.core.run_context import RunContext
except Exception:
    from core.run_context import RunContext

try:
    from app.models.result_models import GenerateResult
    from app.utils.ai_llm_controller import AILlmController
    from app.utils.llm_protocol import LLMClient
except ImportError:
    AILlmController = None  # type: ignore
    GenerateResult = None  # type: ignore
    LLMClient = None  # type: ignore

try:
    if AILlmController is None:
        raise ImportError("AILlmController not available")
    _DefaultLLM = AILlmController
except Exception:
    # フォールバック: スタブクラス
    class GenerateResult:  # type: ignore
        def __init__(self, text: str, cost_usd: float = 0.0):
            self.text = text
            self.cost_usd = cost_usd

    class _DefaultLLM:  # type: ignore
        def generate(self, prompt: str, task_type: str = "default", **kwargs: Any) -> Any:
            logging.info("AiLlmController (Stub): AIへの非同期リクエストをシミュレートします。")
            import time

            time.sleep(0.1)  # I/Oバウンドな処理を模倣
            dummy_response = {
                "site": "MONCLER_OFFICIAL",
                "page_type": "pdp",
                "strategy": "llm_selector_healing_v1",
                "candidates": [
                    {
                        "target": "title",
                        "old_selector": "h1.product-title",
                        "new_selector": "h1[data-test='product-title']",
                        "confidence": 0.92,
                        "reason": "DOM に data-test='product-title' が追加されているため",
                    }
                ],
            }
            return GenerateResult(text=json.dumps(dummy_response, ensure_ascii=False, indent=2))


logger = logging.getLogger(__name__)


class SelectorRepairAgent:
    """
    CR-ATELIER-003 Phase D-10: Selector Auto-Healing エージェント

    DOM 変化に対する CSS セレクタ自動修復を LLM ベースで実行する。
    """

    def __init__(self, llm_client: Any | None = None):
        """
        SelectorRepairAgent を初期化

        Args:
            llm_client: LLMClient プロトコルを満たすインスタンス（省略時は AILlmController を新規作成）
        """
        if llm_client is None:
            try:
                self.llm_client = _DefaultLLM() if _DefaultLLM else None
            except Exception as e:
                logger.warning(f"[SelectorRepair] Failed to initialize LLM client: {e}, using stub")
                self.llm_client = _DefaultLLM() if _DefaultLLM else None
        else:
            self.llm_client = llm_client

    async def propose_selector_patches(
        self,
        *,
        site: str,
        page_type: str,  # "plp" or "pdp"
        failure_context: dict[str, Any],
        failure_analysis: dict[str, Any],
        dom_snapshot_html: str,
        current_selectors: dict[str, Any],
        site_config: dict[str, Any] | None = None,
        previous_successes: list[dict[str, Any]] | None = None,  # CR-ATELIER-003 Phase D-10.1 Integration
        previous_failures: list[dict[str, Any]] | None = None,  # CR-ATELIER-003 Phase D-10.1 Integration
    ) -> dict[str, Any]:
        """
        LLM を使って、新しい CSS セレクタ候補を提案する。

        Args:
            site: サイトコード（例: "MONCLER_OFFICIAL"）
            page_type: ページタイプ（"plp" または "pdp"）
            failure_context: 標準化された failure_context 辞書
            failure_analysis: FailureAnalysisAgent による分析結果
            dom_snapshot_html: DOM snapshot の HTML 文字列
            current_selectors: 現行の selectors 設定（例: site_config["selectors"]["pdp"]）

        Returns:
            仕様書に準拠した selector repair result:
            {
              "site": "MONCLER_OFFICIAL",
              "page_type": "pdp",
              "strategy": "llm_selector_healing_v1",
              "candidates": [
                {
                  "target": "title",
                  "old_selector": "h1.product-title",
                  "new_selector": "h1[data-test='product-title']",
                  "confidence": 0.92,
                  "reason": "DOM に data-test='product-title' が追加されているため"
                },
                ...
              ]
            }
        """
        try:
            # プロンプトを構築
            prompt = self._build_selector_repair_prompt(
                site=site,
                page_type=page_type,
                failure_context=failure_context,
                failure_analysis=failure_analysis,
                dom_snapshot_html=dom_snapshot_html,
                current_selectors=current_selectors,
                site_config=site_config,
                previous_successes=previous_successes or [],  # CR-ATELIER-003 Phase D-10.1 Integration
                previous_failures=previous_failures or [],  # CR-ATELIER-003 Phase D-10.1 Integration
            )

            logger.info(
                f"[SelectorRepair] Proposing selector patches for {site}/{page_type} "
                f"(error_type={failure_context.get('error_type')})"
            )

            # LLM を呼び出し
            result = await self.llm_client.generate(prompt, task_type="code")

            # JSON をパース
            try:
                proposal = json.loads(result.text)
            except json.JSONDecodeError:
                # JSON パースに失敗した場合、テキストから抽出を試みる
                logger.warning("[SelectorRepair] LLM response is not valid JSON, attempting extraction")
                proposal = self._extract_json_from_text(result.text)

            # 仕様書に準拠した形式に正規化
            normalized = self._normalize_proposal(proposal, site=site, page_type=page_type)

            # CR-ATELIER-003 Phase D-10.1: Selector Ranking
            # LLM が返した候補をランキングして優先順位を付ける
            candidates = normalized.get("candidates", [])
            if candidates:
                ranked_candidates = self._rank_selectors(
                    candidates=candidates,
                    site_config=site_config or {},
                )
                normalized["candidates"] = ranked_candidates
                logger.info(
                    f"[SelectorRepair] Ranked {len(ranked_candidates)} selector candidates "
                    f"(top: {ranked_candidates[0].get('new_selector') if ranked_candidates else 'N/A'})"
                )

            logger.info(f"[SelectorRepair] Generated {len(normalized.get('candidates', []))} selector candidates")

            return normalized

        except Exception as e:
            logger.error(f"[SelectorRepair] Failed to propose selector patches: {e}", exc_info=True)
            # エラー時は空の結果を返す
            return {
                "site": site,
                "page_type": page_type,
                "strategy": "llm_selector_healing_v1",
                "candidates": [],
                "error": str(e),
            }

    def _build_selector_repair_prompt(
        self,
        *,
        site: str,
        page_type: str,
        failure_context: dict[str, Any],
        failure_analysis: dict[str, Any],
        dom_snapshot_html: str,
        current_selectors: dict[str, Any],
        site_config: dict[str, Any] | None = None,
    ) -> str:
        """
        Selector repair 用の LLM プロンプトを構築する

        CR-ATELIER-003 Phase D-10.1: サイト固有制約をプロンプトに追加
        """
        error_type = failure_context.get("error_type", "unknown")
        error_message = failure_context.get("error_message", "")
        error_class = failure_context.get("error_class", "")

        # CR-ATELIER-003 Phase D-10.1: DOM Snippet 最適化
        # 失敗したセレクタの周辺に焦点を当て、script/style/comment を除去
        failed_selector = self._extract_failed_selector_from_error(error_message)
        html_snippet = self._optimize_dom_snippet(
            html=dom_snapshot_html,
            failed_selector=failed_selector,
            max_chars=8000,
        )

        # 現行 selectors を文字列化
        selectors_str = json.dumps(current_selectors, ensure_ascii=False, indent=2)

        # failure_analysis の要約
        analysis_summary = failure_analysis.get("summary", "")
        root_causes = failure_analysis.get("root_causes", [])
        suggested_fixes = failure_analysis.get("suggested_fixes", [])

        # CR-ATELIER-003 Phase D-10.1: サイト固有制約を抽出
        site_constraints = self._extract_site_constraints(
            site=site,
            site_config=site_config or {},
            page_type=page_type,
        )

        # CR-ATELIER-003 Phase D-10.1: 前回の成功/失敗ログをプロンプトに追加
        feedback_section = ""
        if previous_successes:
            feedback_section += "\n# Previous Successful Selectors (参考にしてください)\n"
            feedback_section += "以下のセレクタは過去に成功しました:\n\n"
            for success in previous_successes[:5]:  # 最大5件
                selector = success.get("selector", "")
                reason = success.get("reason", "")
                feedback_section += f"- `{selector}`: {reason}\n"
            feedback_section += "\n"

        if previous_failures:
            feedback_section += "\n# Previous Failed Selectors (避けてください)\n"
            feedback_section += "以下のセレクタは過去に失敗しました。**これらのセレクタは提案しないでください**:\n\n"
            for failure in previous_failures[:5]:  # 最大5件
                selector = failure.get("selector", "")
                reason = failure.get("reason", "")
                feedback_section += f"- `{selector}`: {reason}\n"
            feedback_section += "\n"

        # サイト固有制約セクションを構築
        constraints_section = ""
        if site_constraints:
            constraints_section = "\n# Site-Specific Constraints (必須遵守)\n"
            constraints_section += "以下のルールを**必ず守って**セレクタを提案してください:\n\n"

            if site_constraints.get("allowed_domain"):
                constraints_section += (
                    f"- **Domain**: URL のホストは `{site_constraints['allowed_domain']}` であること\n"
                )

            if site_constraints.get("url_patterns"):
                patterns = site_constraints["url_patterns"]
                constraints_section += "- **URL Path**: パスには以下のいずれかを含むこと:\n"
                for pattern in patterns:
                    constraints_section += f"  - `{pattern}`\n"

            if page_type == "plp":
                constraints_section += "- **Element Type**: `<a>` タグ（アンカータグ）であること\n"
                constraints_section += "- **Relative URL**: `href` が相対 URL なら絶対 URL に解決可能であること\n"
                constraints_section += "- **Image Priority**: 画像 (`<img>`) を子要素に持つリンクを優先すること\n"

            constraints_section += "\n"

        # CR-ATELIER-003 Phase D-10.1: 前回の成功/失敗ログをプロンプトに追加
        feedback_section = ""
        if previous_successes:
            feedback_section += "\n# Previous Successful Selectors (参考にしてください)\n"
            feedback_section += "以下のセレクタは過去に成功しました:\n\n"
            for success in previous_successes[:5]:  # 最大5件
                selector = success.get("selector", "")
                reason = success.get("reason", "")
                feedback_section += f"- `{selector}`: {reason}\n"
            feedback_section += "\n"

        if previous_failures:
            feedback_section += "\n# Previous Failed Selectors (避けてください)\n"
            feedback_section += "以下のセレクタは過去に失敗しました。**これらのセレクタは提案しないでください**:\n\n"
            for failure in previous_failures[:5]:  # 最大5件
                selector = failure.get("selector", "")
                reason = failure.get("reason", "")
                feedback_section += f"- `{selector}`: {reason}\n"
            feedback_section += "\n"

        prompt = f"""
# Role
あなたは、Web自動化とフロントエンド開発を専門とする世界トップクラスのAIエンジニアです。

# Goal
ウェブサイト「{site}」の {page_type.upper()} ページで、CSS セレクタが原因で自動化タスクが失敗しました。
あなたの任務は、提供された DOM snapshot と失敗情報を分析し、
新しい、より堅牢で、将来の変更に強い CSS セレクタを提案することです。

# Context
- Website: {site}
- Page Type: {page_type.upper()}
- Error Type: {error_type}
- Error Class: {error_class}
- Error Message: {error_message}

# Failure Analysis
- Summary: {analysis_summary}
- Root Causes: {json.dumps(root_causes, ensure_ascii=False)}
- Suggested Fixes: {json.dumps(suggested_fixes, ensure_ascii=False)}

{constraints_section}{feedback_section}# Current Selectors
```json
{selectors_str}
```

# DOM Snapshot (Relevant Snippet)
```html
{html_snippet}
```

# Instructions
1. DOM snapshot を分析し、現行セレクタが失敗した理由を特定してください。
2. 各セレクタフィールド（title, price, images など）に対して、新しい CSS セレクタを最大3つまで提案してください。
3. 各提案には以下を含めてください:
   - target: 対象フィールド名（例: "title", "price"）
   - old_selector: 現在失敗しているセレクタ
   - new_selector: 提案する新しいセレクタ
   - confidence: 信頼度（0.0〜1.0）
   - reason: なぜそのセレクタを提案したかの理由

4. **重要**: サイト固有制約（上記）を必ず遵守してください。制約に違反するセレクタは提案しないでください。

5. 応答は、以下の JSON 形式のみで、厳密に出力してください:
```json
{{
  "site": "{site}",
  "page_type": "{page_type}",
  "strategy": "llm_selector_healing_v1",
  "candidates": [
    {{
      "target": "title",
      "old_selector": "h1.product-title",
      "new_selector": "h1[data-test='product-title']",
      "confidence": 0.92,
      "reason": "DOM に data-test='product-title' が追加されているため"
    }}
  ]
}}
```

重要: JSON のみを出力し、説明文やマークダウンは含めないでください。
"""
        return prompt

    def _extract_site_constraints(
        self,
        *,
        site: str,
        site_config: dict[str, Any],
        page_type: str,
    ) -> dict[str, Any]:
        """
        CR-ATELIER-003 Phase D-10.1: サイト固有制約を抽出する

        Args:
            site: サイトコード
            site_config: サイト設定
            page_type: ページタイプ（"plp" または "pdp"）

        Returns:
            サイト固有制約の辞書
        """
        constraints: dict[str, Any] = {}

        # allowed_domain を抽出
        allowed_domain = site_config.get("allowed_domain")
        if allowed_domain:
            constraints["allowed_domain"] = allowed_domain

        # URL パスパターンを抽出
        url_rules = site_config.get("url_rules", {})
        allow_path_patterns = url_rules.get("allow_path_patterns", [])

        # パターンから実際のパス要件を抽出（例: "/products/" や "/p/"）
        url_patterns = []
        for pattern in allow_path_patterns:
            # 正規表現から実際のパス要件を抽出
            if "/products/" in pattern or "products" in pattern.lower():
                url_patterns.append("/products/")
            if "/p/" in pattern or r"/p/" in pattern:
                url_patterns.append("/p/")

        # 重複を除去
        url_patterns = list(set(url_patterns))
        if url_patterns:
            constraints["url_patterns"] = url_patterns

        return constraints

    def _extract_failed_selector_from_error(self, error_message: str) -> str | None:
        """
        CR-ATELIER-003 Phase D-10.1: エラーメッセージから失敗したセレクタを抽出

        Args:
            error_message: エラーメッセージ（例: "Timeout while waiting for selector: h1.product-title"）

        Returns:
            失敗したセレクタ（抽出できない場合は None）
        """
        import re

        # "selector: ..." パターンを探す
        match = re.search(r"selector[:\s]+([^\s,]+)", error_message, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _optimize_dom_snippet(
        self,
        html: str,
        failed_selector: str | None = None,
        max_chars: int = 8000,
    ) -> str:
        """
        CR-ATELIER-003 Phase D-10.1: DOM Snippet を最適化する

        - script/style/comment ノードを除去
        - 失敗したセレクタの周辺（最大4階層上の parentNode）に焦点を当て
        - 最大文字数上限を適用

        Args:
            html: DOM snapshot の HTML 文字列
            failed_selector: 失敗したセレクタ（オプション）
            max_chars: 最大文字数（デフォルト: 8000）

        Returns:
            最適化された HTML スニペット
        """
        try:
            from bs4 import BeautifulSoup, Comment
        except ImportError:
            # BeautifulSoup が利用できない場合、簡易的な処理
            # script/style/comment を正規表現で除去
            import re

            # script タグを除去
            html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            # style タグを除去
            html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
            # HTML コメントを除去
            html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

            # 失敗したセレクタの周辺を抽出（簡易実装）
            if failed_selector:
                # セレクタが含まれる行の前後を取得
                lines = html.split("\n")
                for i, line in enumerate(lines):
                    if failed_selector in line:
                        # 前後 50 行を取得
                        start = max(0, i - 50)
                        end = min(len(lines), i + 50)
                        html = "\n".join(lines[start:end])
                        break

            # 最大文字数制限
            if len(html) > max_chars:
                html = html[:max_chars]

            return html

        # BeautifulSoup を使用した最適化
        soup = BeautifulSoup(html, "html.parser")

        # script/style/comment を除去
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # 失敗したセレクタの周辺に焦点を当てる
        if failed_selector:
            try:
                # セレクタに一致する要素を探す
                elements = soup.select(failed_selector)
                if elements:
                    # 最初の要素の親要素（最大4階層上）を取得
                    element = elements[0]
                    parent = element
                    for _ in range(4):
                        if parent.parent:
                            parent = parent.parent
                        else:
                            break

                    # 親要素の HTML を取得
                    html = str(parent)
                else:
                    # セレクタが見つからない場合、元の HTML を使用
                    html = str(soup)
            except Exception:
                # セレクタ解析エラーの場合、元の HTML を使用
                html = str(soup)
        else:
            html = str(soup)

        # 最大文字数制限
        if len(html) > max_chars:
            html = html[:max_chars]

        return html

    def _rank_selectors(
        self,
        candidates: list[dict[str, Any]],
        site_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        CR-ATELIER-003 Phase D-10.1: Selector Ranking

        LLM が返した複数セレクタに対して以下の基準でスコアリングする：
        - 条件一致（サイト固有ルールに合致するほど高得点）
        - specificity（CSS specificity の高さ）
        - 安定性（data-* や testid 系を優先）
        - 汎用性（fallback は低優先）

        Args:
            candidates: LLM が返したセレクタ候補のリスト
            site_config: サイト設定

        Returns:
            ランキングされたセレクタ候補のリスト
        """
        if not candidates:
            return []

        # 各候補にスコアを付ける
        scored_candidates = []
        for candidate in candidates:
            selector = candidate.get("new_selector", "")
            score = 0.0

            # 1. サイト固有ルールに合致するか（+20点）
            if self._matches_site_constraints(selector, site_config):
                score += 20.0

            # 2. data-testid や data-* 属性がある（+15点）
            if "[data-testid" in selector or "[data-" in selector:
                score += 15.0

            # 3. BEM スタイル（.product-card__title）（+10点）
            if "__" in selector and selector.startswith("."):
                score += 10.0

            # 4. CSS specificity が高い（+5点）
            specificity_score = self._calculate_specificity(selector)
            score += min(specificity_score * 0.5, 5.0)

            # 5. 汎用すぎるセレクタは減点（div, span など単一要素）
            if selector.strip() in ["div", "span", "a", "p", "h1", "h2", "h3"]:
                score -= 10.0

            # 6. confidence を考慮（+0〜10点）
            confidence = candidate.get("confidence", 0.0)
            score += confidence * 10.0

            scored_candidates.append(
                {
                    **candidate,
                    "_ranking_score": score,
                }
            )

        # スコアの降順でソート
        ranked = sorted(scored_candidates, key=lambda x: x.get("_ranking_score", 0.0), reverse=True)

        # ランキングスコアを削除（内部使用のみ）
        for candidate in ranked:
            candidate.pop("_ranking_score", None)

        return ranked

    def _matches_site_constraints(self, selector: str, site_config: dict[str, Any]) -> bool:
        """
        セレクタがサイト固有制約に合致するかチェック

        Args:
            selector: CSS セレクタ
            site_config: サイト設定

        Returns:
            制約に合致する場合 True
        """
        # URL パスパターンチェック
        url_rules = site_config.get("url_rules", {})
        allow_path_patterns = url_rules.get("allow_path_patterns", [])

        if allow_path_patterns:
            # "/products/" または "/p/" を含むパターンがある場合
            has_products_pattern = any("/products/" in p or "/p/" in p for p in allow_path_patterns)
            if has_products_pattern:
                # セレクタに "/products/" または "/p/" が含まれるかチェック
                if "/products/" not in selector and "/p/" not in selector:
                    return False

        # allowed_domain チェック
        allowed_domain = site_config.get("allowed_domain")
        if allowed_domain:
            # セレクタにドメインが含まれる場合、allowed_domain と一致するかチェック
            if allowed_domain in selector:
                return True
            # セレクタに他のドメインが含まれる場合は False
            # （簡易実装: より厳密なチェックが必要な場合は拡張）

        return True  # デフォルトは True（制約がない場合は合格）

    def _calculate_specificity(self, selector: str) -> float:
        """
        CSS specificity を計算（簡易版）

        Args:
            selector: CSS セレクタ

        Returns:
            specificity スコア（高いほど良い）
        """
        score = 0.0

        # ID セレクタ（#id）
        score += selector.count("#") * 100.0

        # クラスセレクタ（.class）
        score += selector.count(".") * 10.0

        # 属性セレクタ（[attr]）
        score += selector.count("[") * 10.0

        # 疑似クラス（:pseudo）
        score += selector.count(":") * 10.0

        # 要素セレクタ（div, span など）
        # 単一要素は減点
        if selector.strip() in ["div", "span", "a", "p", "h1", "h2", "h3", "h4", "h5", "h6"]:
            score += 1.0
        else:
            score += selector.count(" ") * 1.0

        return score

    def _extract_json_from_text(self, text: str) -> dict[str, Any]:
        """
        LLM の応答テキストから JSON を抽出する（フォールバック）
        """
        import re

        # JSON ブロックを探す
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        # フォールバック: 空の結果を返す
        return {"candidates": []}

    def _normalize_proposal(
        self,
        proposal: dict[str, Any],
        site: str,
        page_type: str,
    ) -> dict[str, Any]:
        """
        LLM の応答を仕様書に準拠した形式に正規化する
        """
        # 既に正しい形式の場合はそのまま返す
        if "site" in proposal and "page_type" in proposal and "candidates" in proposal:
            return proposal

        # 旧形式（proposed_selectors）から変換
        normalized = {
            "site": site,
            "page_type": page_type,
            "strategy": "llm_selector_healing_v1",
            "candidates": [],
        }

        # proposed_selectors がある場合、candidates に変換
        if "proposed_selectors" in proposal:
            for idx, sel in enumerate(proposal.get("proposed_selectors", [])):
                normalized["candidates"].append(
                    {
                        "target": f"field_{idx}",
                        "old_selector": "",
                        "new_selector": sel,
                        "confidence": 0.8,
                        "reason": proposal.get("rationale", "LLM proposal"),
                    }
                )

        # candidates が既にある場合はそのまま使用
        if "candidates" in proposal:
            normalized["candidates"] = proposal["candidates"]

        return normalized

    # 後方互換性: 既存の propose_fix メソッドを保持
    async def propose_fix(
        self,
        *,
        intent: str,
        failed_selectors: list[str],
        html_content: str,
        site: str,
        site_config: dict,
        run_context: RunContext,
    ) -> dict[str, Any]:
        """
        既存の propose_fix メソッド（後方互換性のため保持）

        注意: このメソッドは Phase D-10 の新しい propose_selector_patches に置き換えられます。
        """
        logger.warning("[SelectorRepair] propose_fix is deprecated, use propose_selector_patches instead")

        # 簡易的な変換
        failure_context = {
            "error_type": "selector_not_found",
            "error_message": f"Failed selectors: {', '.join(failed_selectors)}",
        }
        failure_analysis = {
            "summary": intent,
            "root_causes": ["Selector may be outdated"],
            "suggested_fixes": ["Find new selector"],
        }
        current_selectors = site_config.get("selectors", {}).get("pdp", {})

        return await self.propose_selector_patches(
            site=site,
            page_type="pdp",
            failure_context=failure_context,
            failure_analysis=failure_analysis,
            dom_snapshot_html=html_content,
            current_selectors=current_selectors,
        )
