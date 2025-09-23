# -*- coding: utf-8 -*-
"""
Unified runner for ResearchOrchestrator
- Windows: Playwright用に WindowsSelectorEventLoopPolicy を強制
- Orchestrator に存在しない引数は渡さず、属性があれば後からセット
- ログと結果JSONの保存、失敗時のエラーレポート出力
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import logging
import sys, asyncio
from datetime import datetime
from pathlib import Path

# --- Windows asyncio policy（Playwright対策） -------------------------------
# Playwright は Windows で Selector ポリシー必須。Proactorだと subprocess が NotImplementedError.
try:
    if sys.platform.startswith("win") and isinstance(
        asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy
    ):
        from asyncio import WindowsSelectorEventLoopPolicy  # type: ignore[attr-defined]
        asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
except Exception:
    pass  # 失敗しても致命的ではない

# --- アプリ本体 -------------------------------------------------------------
try:
    from app.utils.ai_research_orchestrator import ResearchOrchestrator
except Exception as e:
    print("[FATAL] ResearchOrchestrator の import に失敗: ", e, file=sys.stderr)
    sys.exit(99)

# --- デフォルト -------------------------------------------------------------
DEFAULT_ITEMS = 5
DEFAULT_TIMEOUT = 180  # seconds（サイト1件あたりの目安）
DEFAULT_LOG_DIR = Path("instance/logs")
DEFAULT_OUT_DIR = Path("output")


def setup_logging(level: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    logging.basicConfig(
        level=level_map.get(level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"runner_{datetime.now():%Y%m%d}.log", encoding="utf-8"),
        ],
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_orchestrator.py",
        description="ResearchOrchestrator 統合ランナー（本番/開発兼用）",
    )
    p.add_argument("brand", help="検索ブランド/キーワード（例: GUCCI、'Gucci GG0637SK' など）")

    head = p.add_mutually_exclusive_group()
    head.add_argument("--headless", dest="headless", action="store_true", help="ヘッドレス実行（デフォルト）")
    head.add_argument("--headful", dest="headless", action="store_false", help="ブラウザ表示ありで実行")
    p.set_defaults(headless=True)

    p.add_argument("--items", type=int, default=DEFAULT_ITEMS, help=f"収集するアイテム上限（既定: {DEFAULT_ITEMS}）")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"サイト単位の目安タイムアウト秒（既定: {DEFAULT_TIMEOUT}）")
    p.add_argument("--retries", type=int, default=1, help="致命的失敗時のリトライ回数（既定: 1）")
    p.add_argument("--log-level", default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], help="ログ出力レベル")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="結果JSONの出力ディレクトリ（既定: output/）")
    p.add_argument("--tag", default="", help="ファイル名に付ける任意タグ（例: nightly, test01）")
    return p.parse_args(argv)


def dump_result_json(out_dir: Path, brand: str, data: list | dict, tag: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_brand = "".join(c for c in brand if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
    tag_part = f"_{tag}" if tag else ""
    path = out_dir / f"research_{safe_brand}_{datetime.now():%Y%m%d_%H%M%S}{tag_part}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level, DEFAULT_LOG_DIR)

    logging.info("=== Unified Runner start ===")
    logging.info("brand=%s headless=%s items=%s timeout=%s retries=%s",
                 args.brand, args.headless, args.items, args.timeout, args.retries)

    # Orchestrator を初期化（存在しない引数は渡さない）
    orch = ResearchOrchestrator(headless=args.headless)

    # 任意パラメータは「属性があればセット」
    if hasattr(orch, "item_limit"):
        try:
            setattr(orch, "item_limit", int(args.items))
            logging.debug("Set orch.item_limit=%s", args.items)
        except Exception:
            logging.exception("Failed to set orch.item_limit")

    if hasattr(orch, "timeout"):
        try:
            setattr(orch, "timeout", int(args.timeout))
            logging.debug("Set orch.timeout=%s", args.timeout)
        except Exception:
            logging.exception("Failed to set orch.timeout")

    # リトライ付き実行
    last_err: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            results = orch.run(args.brand)  # list を想定
            logging.info("Runner: orchestrator finished with %d result(s).", len(results) if hasattr(results, "__len__") else -1)
            out_file = dump_result_json(args.out, args.brand, results, args.tag)
            logging.info("Runner: wrote result -> %s", out_file)
            print(str(out_file))  # 標準出力にも出す
            return 0  # 成功
        except KeyboardInterrupt:
            logging.warning("Runner: KeyboardInterrupt - abort.")
            return 130
        except Exception as e:
            last_err = e
            logging.exception("Runner: attempt %s/%s failed.", attempt, args.retries)

    # すべて失敗
    err_report = {
        "brand": args.brand,
        "attempts": args.retries,
        "error": repr(last_err),
        "time": datetime.now().isoformat(),
        "platform": sys.platform,
        "python": sys.version,
    }
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    err_path = DEFAULT_LOG_DIR / f"orchestrator_last_error_{datetime.now():%Y%m%d_%H%M%S}.json"
    err_path.write_text(json.dumps(err_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.error("Runner: all attempts failed. error report -> %s", err_path)
    print(str(err_path), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
