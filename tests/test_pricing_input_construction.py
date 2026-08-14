"""AST リファレンステスト — PricingInput 構築の出所明示を機械検証。

ISSUE-101/102 Phase1 T3。app/ 内のすべての PricingInput(...) 構築を AST で列挘し:
- purchase_price_source / selling_price_source を渡していない構築は
  Phase1 許容リスト（skip_source_validation=True で動作維持中の6経路）以外では失敗
- 許容リストのファイルは skip_source_validation=True を実際に含むこと（stale防止）

Phase2 T10 で許容リストを空にする（source 未設定 = 0 が完了条件）。
"""

from __future__ import annotations

import ast
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"

# Phase1 許容リスト: ISSUE-101 の6経路（Phase2 T5-T9 で source 明示 → T10 で清算）
# ① models/order.py は T6 で source 明示済み（skip撤去・許容リストから除外）
PHASE1_ALLOWLIST = {
    "utils/sourcing_profitability.py",  # ② sourcing パイプライン
    "services/price_comparison_service.py",  # ③
    "services/brand_price_service.py",  # ④
    "services/price_monitor_service.py",  # ⑤
    "agents/profitability_agent.py",  # ⑥
}

SOURCE_KEYS = ("purchase_price_source", "selling_price_source")


def _unconfigured_constructions(tree: ast.Module) -> list[int]:
    """tree 内で source 引数を渡していない PricingInput(...) 構築の行番号一覧。"""
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "PricingInput":
            continue
        keys = {kw.arg for kw in node.keywords if kw.arg is not None}
        if not set(SOURCE_KEYS) <= keys:
            lines.append(node.lineno)
    return lines


def find_unconfigured_pricing_inputs(root: Path) -> dict[str, list[int]]:
    """root 配下の source 未設定構築を {相対パス: [行]} で返す（.bak 除外）。"""
    violations: dict[str, list[int]] = {}
    for py in sorted(root.rglob("*.py")):
        if ".bak" in py.name:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        lines = _unconfigured_constructions(tree)
        if lines:
            violations[str(py.relative_to(root))] = lines
    return violations


def test_no_unconfigured_pricing_input_outside_allowlist() -> None:
    """許容リスト外に source 未設定の PricingInput 構築があれば失敗。"""
    violations = find_unconfigured_pricing_inputs(APP_DIR)
    unexpected = {p: ln for p, ln in violations.items() if p not in PHASE1_ALLOWLIST}
    assert not unexpected, (
        "PricingInput を purchase_price_source/selling_price_source なしで構築: "
        f"{unexpected}。出所を明示するか、skip_source_validation=True を伴う経路として"
        "PHASE1_ALLOWLIST に追加してください（Phase2 では許容リスト自体を清算）。"
    )


def test_allowlist_entries_still_skip_validation() -> None:
    """許容リストの各ファイルが skip_source_validation=True を実含すること（stale防止）。"""
    stale = [
        rel
        for rel in sorted(PHASE1_ALLOWLIST)
        if "skip_source_validation=True" not in (APP_DIR / rel).read_text(encoding="utf-8")
    ]
    assert not stale, (
        f"PHASE1_ALLOWLIST が stale（skip_source_validation=True が無い）: {stale}。"
        "source 明示が済んだ経路は許容リストから外すこと。"
    )


def test_detector_catches_violation_and_passes_compliant() -> None:
    """検出器自体の自明性検証（偽陰性ゼロ担保）。"""
    snippet = (
        "from app.core.pricing.schemas import PricingInput, PriceSource\n"
        "bad = PricingInput(purchase_price=1.0, selling_price=2.0)\n"
        "good = PricingInput(\n"
        "    purchase_price=1.0,\n"
        "    selling_price=2.0,\n"
        "    purchase_price_source=PriceSource.MANUAL_INPUT,\n"
        "    selling_price_source=PriceSource.MANUAL_INPUT,\n"
        ")\n"
    )
    lines = _unconfigured_constructions(ast.parse(snippet))
    assert lines == [2], f"違反(2行目)のみ検出されるべき: {lines}"
