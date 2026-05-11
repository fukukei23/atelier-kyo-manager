"""Selector repair LLM prompt construction."""

from __future__ import annotations

import json
import re
from typing import Any

from bs4 import BeautifulSoup, Comment


def build_selector_repair_prompt(
    *,
    site: str,
    page_type: str,
    failure_context: dict[str, Any],
    failure_analysis: dict[str, Any],
    dom_snapshot_html: str,
    current_selectors: dict[str, Any],
    site_config: dict[str, Any] | None = None,
    previous_successes: list[dict[str, Any]] | None = None,
    previous_failures: list[dict[str, Any]] | None = None,
) -> str:
    """Selector repair 用の LLM プロンプトを構築する."""
    error_type = failure_context.get("error_type", "unknown")
    error_message = failure_context.get("error_message", "")
    error_class = failure_context.get("error_class", "")

    failed_selector = extract_failed_selector_from_error(error_message)
    html_snippet = optimize_dom_snippet(
        html=dom_snapshot_html,
        failed_selector=failed_selector,
        max_chars=8000,
    )

    selectors_str = json.dumps(current_selectors, ensure_ascii=False, indent=2)

    analysis_summary = failure_analysis.get("summary", "")
    root_causes = failure_analysis.get("root_causes", [])
    suggested_fixes = failure_analysis.get("suggested_fixes", [])

    site_constraints = extract_site_constraints(
        site=site,
        site_config=site_config or {},
        page_type=page_type,
    )

    # feedback section
    feedback_section = ""
    if previous_successes:
        feedback_section += "\n# Previous Successful Selectors (参考にしてください)\n"
        feedback_section += "以下のセレクタは過去に成功しました:\n\n"
        for success in previous_successes[:5]:
            selector = success.get("selector", "")
            reason = success.get("reason", "")
            feedback_section += f"- `{selector}`: {reason}\n"
        feedback_section += "\n"

    if previous_failures:
        feedback_section += "\n# Previous Failed Selectors (避けてください)\n"
        feedback_section += "以下のセレクタは過去に失敗しました。**これらのセレクタは提案しないでください**:\n\n"
        for failure in previous_failures[:5]:
            selector = failure.get("selector", "")
            reason = failure.get("reason", "")
            feedback_section += f"- `{selector}`: {reason}\n"
        feedback_section += "\n"

    # constraints section
    constraints_section = ""
    if site_constraints:
        constraints_section = "\n# Site-Specific Constraints (必須遵守)\n"
        constraints_section += "以下のルールを**必ず守って**セレクタを提案してください:\n\n"

        if site_constraints.get("allowed_domain"):
            constraints_section += f"- **Domain**: URL のホストは `{site_constraints['allowed_domain']}` であること\n"

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


def extract_site_constraints(
    *,
    site: str,
    site_config: dict[str, Any],
    page_type: str,
) -> dict[str, Any]:
    """サイト固有制約を抽出する."""
    constraints: dict[str, Any] = {}

    allowed_domain = site_config.get("allowed_domain")
    if allowed_domain:
        constraints["allowed_domain"] = allowed_domain

    url_rules = site_config.get("url_rules", {})
    allow_path_patterns = url_rules.get("allow_path_patterns", [])

    url_patterns = []
    for pattern in allow_path_patterns:
        if "/products/" in pattern or "products" in pattern.lower():
            url_patterns.append("/products/")
        if "/p/" in pattern or r"/p/" in pattern:
            url_patterns.append("/p/")

    url_patterns = list(set(url_patterns))
    if url_patterns:
        constraints["url_patterns"] = url_patterns

    return constraints


def extract_failed_selector_from_error(error_message: str) -> str | None:
    """エラーメッセージから失敗したセレクタを抽出."""
    match = re.search(r"selector[:\s]+([^\s,]+)", error_message, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def optimize_dom_snippet(
    html: str,
    failed_selector: str | None = None,
    max_chars: int = 8000,
) -> str:
    """DOM Snippet を最適化する（script/style/comment除去、セレクタ周辺に焦点）."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    if failed_selector:
        try:
            elements = soup.select(failed_selector)
            if elements:
                element = elements[0]
                parent = element
                for _ in range(4):
                    if parent.parent:
                        parent = parent.parent
                    else:
                        break
                html = str(parent)
            else:
                html = str(soup)
        except Exception:
            html = str(soup)
    else:
        html = str(soup)

    if len(html) > max_chars:
        html = html[:max_chars]

    return html
