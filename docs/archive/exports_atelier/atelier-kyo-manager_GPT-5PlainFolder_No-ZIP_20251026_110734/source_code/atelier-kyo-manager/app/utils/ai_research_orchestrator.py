# -*- coding: utf-8 -*-
# ==============================================================================
# File       : app/utils/ai_research_orchestrator.py
# Purpose    : Research entrypoint with learn/run modes, loop-policy and sites_config injection
# Version    : 3.3.0 (SupplierScoutAgent.run signature aligned; sites_config deep-merge; loop-policy; log-level)
# Timestamp  : 2025-10-15 JST
#
# Usage (examples)
# ------------------------------------------------------------------------------
#  Headful debugging (Windows / Playwright):
#    python -u -m app.utils.ai_research_orchestrator `
#       --site MONCLER_OFFICIAL `
#       --query "down jacket" `
#       --headful --slow-mo 200 `
#       --log-level INFO `
#       --loop-policy auto `
#       --mode learn
#
#  Normal run (headless):
#    python -u -m app.utils.ai_research_orchestrator \
#       --site MONCLER_OFFICIAL \
#       --query "down jacket" \
#       --log-level INFO \
#       --mode run
#
# Notes
# ------------------------------------------------------------------------------
#  - SupplierScoutAgent.run(...) は run_context を**受け取りません**。
#    呼び出しは run(sites_config=..., sites=[...], query=...) に統一。
#  - sites_config は Config.SITES を基に overrides.local.json を**ディープマージ**して生成。
#  - Windows の --loop-policy は auto/proactor/selector を選択可（デフォルト auto）。
#  - --mode は learn/run を runtime_kwargs 経由で下層エージェントへ伝播。
# ==============================================================================

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# --- sys.path をプロジェクトルートに通す（保険） ---
try:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
except Exception:
    pass

# --- Config 読み込み（プロジェクトごとに配置差があるため二段トライ） ---
def _import_config_class():
    try:
        # 例: app/config.py に Config がある場合
        from app.config import Config  # type: ignore
        return Config
    except Exception:
        try:
            # 例: app/config/config.py に Config がある場合
            from app.config.config import Config  # type: ignore
            return Config
        except Exception as e:
            raise ImportError(f"Config class not found: {e}")

# --- 便利関数：ディープマージ ---
def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

# --- sites_config 構築：Config.SITES + overrides.local.json をマージ ---
def _load_sites_config() -> dict:
    Config = _import_config_class()
    base = dict(getattr(Config, "SITES", {}) or {})

    # overrides を複数候補から探索
    candidates = [
        Path("overrides.local.json"),
        Path("app/config/sites/overrides.local.json"),
        Path("app/config/overrides.local.json"),
    ]
    merged = dict(base)
    for p in candidates:
        if p.exists():
            try:
                overrides = json.loads(p.read_text(encoding="utf-8"))
                merged = _deep_merge(merged, overrides or {})
            except Exception as e:
                print(f"[WARN] Failed to load {p}: {e}")
    return merged

# --- Windows asyncio policy ---
def _apply_loop_policy(loop_policy: str) -> None:
    if platform.system().lower() != "windows":
        return
    lp = (loop_policy or "auto").lower().strip()
    try:
        if lp == "auto":
            # Python 3.8+ 既定（Proactor）に従う
            return
        elif lp == "proactor":
            # 明示指定（Modern Windows 推奨）
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())  # type: ignore[attr-defined]
        elif lp == "selector":
            # 互換用（古い依存に必要な場合のみ）
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to apply loop policy '{loop_policy}': {e}")

# --- ログ初期化 ---
def _setup_logging(level: str) -> None:
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.INFO)

# --- 依存（エージェント） ---
from app.agents.supplier_scout_agent import SupplierScoutAgent  # noqa: E402

@dataclass
class OrchestratorOptions:
    site: str
    query: str
    headful: bool = False
    slow_mo: int = 0
    loop_policy: str = "auto"     # auto | proactor | selector
    log_level: str = "INFO"
    mode: str = "run"             # run | learn
    extra_runtime_kwargs: Dict[str, Any] = field(default_factory=dict)

class AIResearchOrchestrator:
    """Researchタスク実行の薄いハブ。SupplierScoutAgent に sites_config を注入して走らせる。"""

    def __init__(self, options: OrchestratorOptions):
        self.options = options
        _apply_loop_policy(options.loop_policy)
        _setup_logging(options.log_level)
        self.log = logging.getLogger(self.__class__.__name__)

    async def run(self) -> List[Dict[str, Any]]:
        # runtime kwargs → 下層の BrowserUseAgent 等が参照
        runtime_kwargs: Dict[str, Any] = {
            "headless": not self.options.headful,
            "slow_mo": int(self.options.slow_mo or 0),
            "mode": (self.options.mode or "run").lower(),  # "learn" / "run"
            "enable_har": True,
            "enable_trace": True,
        }
        runtime_kwargs.update(self.options.extra_runtime_kwargs or {})

        self.log.info(
            "Start research | site=%s | query=%s | mode=%s | headful=%s | slow_mo=%sms | loop-policy=%s",
            self.options.site, self.options.query, runtime_kwargs.get("mode"),
            self.options.headful, self.options.slow_mo, self.options.loop_policy
        )

        # sites_config を構築（Config.SITES + overrides.local.json）
        sites_config = _load_sites_config()

        # エージェント実行
        scout_agent = SupplierScoutAgent(runtime_kwargs=runtime_kwargs)
        sites = [self.options.site]
        results_raw = await scout_agent.run(
            sites_config=sites_config,
            sites=sites,
            query=self.options.query,
        )

        # JSON化
        results: List[Dict[str, Any]] = []
        for r in results_raw:
            try:
                results.append(r.to_dict() if hasattr(r, "to_dict") else dict(r))
            except Exception:
                results.append({
                    "ok": getattr(r, "ok", False),
                    "site": getattr(r, "site", self.options.site),
                    "query": getattr(r, "query", self.options.query),
                    "message": getattr(r, "message", "unknown"),
                })

        # 参考：ここで結果を保存したい場合は任意の場所に書き出してOK
        # out_dir = Path("instance") / "orchestrator"
        # out_dir.mkdir(parents=True, exist_ok=True)
        # (out_dir / f"aggregate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json").write_text(
        #     json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8"
        # )

        return results


# -------------------------
# CLI Entrypoint
# -------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai_research_orchestrator.py",
        description="Research runner with learn/run modes, robust Windows loop-policy, and sites_config injection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--site", dest="site", required=True, help="Site key (e.g., MONCLER_OFFICIAL)")
    p.add_argument("--query", dest="query", required=True, help="Search query / seed keyword")
    p.add_argument("--headful", dest="headful", action="store_true", help="Run browser in headed mode")
    p.add_argument("--slow-mo", dest="slow_mo", type=int, default=0, help="Playwright slow motion (ms)")
    p.add_argument("--loop-policy", choices=["auto", "proactor", "selector"], default="auto",
                   help="Asyncio loop policy on Windows")
    p.add_argument("--log-level", dest="log_level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO",
                   help="Logging level")
    p.add_argument("--mode", choices=["run", "learn"], default="run",
                   help="learn: probe DOM/JSON-LD/network to learn selectors; run: use learned selectors.")
    return p


async def _amain(args: argparse.Namespace) -> None:
    opts = OrchestratorOptions(
        site=args.site,
        query=args.query,
        headful=bool(args.headful),
        slow_mo=int(args.slow_mo or 0),
        loop_policy=args.loop_policy,
        log_level=args.log_level,
        mode=args.mode,
        extra_runtime_kwargs={},  # 必要なら拡張
    )
    orchestrator = AIResearchOrchestrator(options=opts)
    await orchestrator.run()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        asyncio.run(_amain(args))
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        logging.getLogger(__name__).exception(f"Fatal error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
