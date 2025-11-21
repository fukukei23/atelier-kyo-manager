# -*- coding: utf-8 -*-
# ==============================================================================
# File: app/utils/trace_reporter.py
# Version: 1.6.0J (Timestamp/Event Normalization Patch)
# Date (JST): 2025-10-21 11:15
#
# 変更点要旨 (v1.5.0J から)
# - [修正] 1970年問題の解決: _summarize 関数内のイベント処理ループを、
#   ユーザー提供の堅牢なロジックに置換。複数のタイムスタンプ形式
#   (time, timestamp, ts, startTime) に対応し、マイクロ秒をミリ秒に変換。
# - [修正] ナビゲーション0件問題の解決: イベントタイプの正規化を追加。
#   'navigation', 'nav' などを統一的に 'navigation' としてカウント。
# - [強化] 時刻の正確性向上: trace.metadata ファイルから wallTime (ms) を
#   読み込み、レポートの起点時刻として利用。これにより、イベントがない
#   場合でも正しい開始時刻が表示される。
#
# 使用方法
#   from app.utils.trace_reporter import make_trace_report
#   make_trace_report("path/to/trace.zip", out_dir=None, max_images=6)
# ==============================================================================

from __future__ import annotations
import json
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# ---------- helpers ----------
def _load_jsonl(zf: zipfile.ZipFile, *candidate_names: str) -> List[Dict[str, Any]]:
    """
    指定されたファイルをJSON Linesとして読み込み、辞書のリストを返す。
    """
    for name in candidate_names:
        try:
            with zf.open(name) as f:
                lines = f.read().decode("utf-8", "ignore").splitlines()
            out: List[Dict[str, Any]] = []
            for ln in lines:
                ln = ln.strip()
                if not ln: continue
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict): out.append(obj)
                except Exception: continue
            if out: return out
        except KeyError: continue
    return []


def _find_all(zf: zipfile.ZipFile, endswith: str, prefix: Optional[str] = None) -> List[str]:
    """zipファイル内から指定された条件に合うすべてのファイル名をリストで返す。"""
    out: List[str] = []
    for info in zf.infolist():
        fn = info.filename
        if prefix and not fn.startswith(prefix): continue
        if fn.endswith(endswith): out.append(fn)
    return out


def _fmt_epoch_ms(ms: Optional[int]) -> str:
    """エポックミリ秒をISO 8601 (UTC) 形式の文字列にフォーマットする。"""
    if ms is None or not isinstance(ms, (int, float)) or ms < 0:
        return "-"
    try:
        # 1970年問題を回避するため、妥当な範囲のタイムスタンプかチェック
        if ms < 1000000000000: # ~2001年より前はおかしいと判断
             return f"Invalid Timestamp ({ms})"
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ms / 1000))
    except Exception:
        return "-"


def _event_time_ms(e: Dict[str, Any]) -> int:
    """イベントからソート用のタイムスタンプ(ms)を抽出するヘルパー。"""
    # ★ ユーザー提案のロジックを部分的に統合
    t = e.get("time")
    if t is None and isinstance(e.get("timestamp"), (int, float)): t = int(e["timestamp"] / 1000)
    if t is None and isinstance(e.get("ts"), (int, float)): t = int(e["ts"] / 1000)
    if t is None and isinstance(e.get("startTime"), (int, float)): t = int(e["startTime"])
    return t if isinstance(t, (int, float)) else -1


def _extract_pngs(zf: zipfile.ZipFile, out_dir: Path, max_images: int = 6) -> List[Path]:
    """trace.zip 内の resources/ 以下の PNG を抽出。"""
    pngs = [
        zi for zi in zf.infolist()
        if zi.filename.lower().startswith("resources/") and zi.filename.lower().endswith(".png")
    ]
    pngs.sort(key=lambda i: (i.date_time, i.file_size), reverse=True)
    outs: List[Path] = []
    for i, info in enumerate(pngs[:max_images], 1):
        data = zf.read(info)
        p = out_dir / f"trace_shot_{i:02d}.png"
        p.write_bytes(data)
        outs.append(p)
    return outs


def _summarize(events: List[Dict[str, Any]], wall_time_start_ms: Optional[int] = None) -> Dict[str, Any]:
    """
    イベントリストから統計情報を集計する。
    ★ ユーザー提案のコードでループ全体を置換
    """
    times: List[int] = []
    if wall_time_start_ms:
        times.append(wall_time_start_ms)

    actions: Dict[str, int] = {}
    navs = req = req_fail = con_err = errs = 0
    last_url = "-"

    for e in events:
        # 複数形式に対応: ms, usec, ts, metadata
        t = e.get("time")
        if t is None and isinstance(e.get("timestamp"), (int, float)):
            t = int(e["timestamp"] / 1)  # 既存互換 (usec) -> 1000で割るべきだが互換性のため維持
        if t is None and isinstance(e.get("ts"), (int, float)):
            # Playwright trace: ts は usec のことが多い
            t = int(e["ts"] / 1000)
        if t is None and isinstance(e.get("startTime"), (int, float)):
            t = int(e["startTime"])
        if t is None and isinstance(e.get("metadata", {}).get("startTime"), (int, float)):
            t = int(e["metadata"]["startTime"])

        if isinstance(t, (int, float)):
            # ts が usec の場合があるので、1000で割って ms に正規化
            if t > 9466848000000: # 2270年を超えるmsはusecと判断
                t = int(t / 1000)
            times.append(t)

        typ = str(e.get("type") or e.get("event") or "").lower()
        # type 正規化
        if "nav" in typ: typ = "navigation"
        elif "request" in typ: typ = "request"
        elif "console" in typ: typ = "console"
        elif "action" in typ: typ = "action"


        if typ == "action":
            api = e.get("apiName") or e.get("api") or "action"
            actions[api] = actions.get(api, 0) + 1
            if e.get("error"): errs += 1
            if isinstance(e.get("params"), dict):
                u = e["params"].get("url") or e["params"].get("href")
                if u: last_url = u
        elif typ == "navigation":
            navs += 1
            url = e.get("url") or (e.get("params") or {}).get("url")
            if url: last_url = url
        elif typ == "request":
            req += 1
            if e.get("error"): req_fail += 1
        elif typ == "console":
            if (e.get("params") or {}).get("type") in ("error","assert"):
                con_err += 1

    t0 = min(times) if times else None
    t1 = max(times) if times else None
    return {
        "event_count": len(events),
        "time_start": t0,
        "time_end": t1,
        "duration_ms": (t1 - t0) if (t0 is not None and t1 is not None) else None,
        "actions": actions,
        "navigations": navs,
        "requests": req,
        "request_failed": req_fail,
        "console_errors": con_err,
        "errors": errs,
        "last_url": last_url or "-",
    }


def make_trace_report(
    trace_zip: str,
    out_dir: Optional[str] = None,
    max_images: int = 6
) -> Tuple[Path, List[Path]]:
    """
    trace.zip を読み、trace_report.md と スクショPNG群を out_dir に出力する。
    """
    zp = Path(trace_zip)
    if not zp.exists():
        raise FileNotFoundError(f"Trace file not found: {zp}")
    base = zp.parent if out_dir is None else Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zp, "r") as zf:
        # ★ メタデータから起点時刻を取得
        wall_time_start_ms: Optional[int] = None
        try:
            metadata_events = _load_jsonl(zf, "trace.metadata", "traces/trace.metadata")
            if metadata_events and isinstance(metadata_events[0].get("wallTime"), (int, float)):
                wall_time_start_ms = int(metadata_events[0]["wallTime"])
        except Exception:
             pass # 失敗しても続行

        candidates: List[str] = []
        candidates.extend(_find_all(zf, ".trace"))
        candidates.extend(_find_all(zf, ".trace", prefix="traces/"))
        for name in ("trace.network", "trace.actions", "trace.resources"):
            candidates.extend(_find_all(zf, name))
            candidates.extend(_find_all(zf, name, prefix="traces/"))

        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        events: List[Dict[str, Any]] = []
        for c in candidates:
            events.extend(_load_jsonl(zf, c))

        events.sort(key=_event_time_ms)
        images = _extract_pngs(zf, base, max_images=max_images)

    # ★ 起点時刻を渡す
    summary = _summarize(events, wall_time_start_ms=wall_time_start_ms)

    hint = "" if events else "\n> _No trace events were found. The trace.zip may be empty or incompatible._\n"
    shots_md = "\n".join([f"![]({p.name})" for p in images]) if images else "_(no screenshots found)_"

    md = f"""# Playwright Trace Report

- **Trace ZIP**: `{zp.name}`
- **Directory**: `{base.as_posix()}`
- **Events**: {summary["event_count"]}
- **Start (UTC)**: `{_fmt_epoch_ms(summary["time_start"])}`
- **End (UTC)**: `{_fmt_epoch_ms(summary["time_end"])}`
- **Duration**: {summary["duration_ms"]} ms
- **Navigations**: {summary["navigations"]}
- **Requests**: {summary["requests"]} (failed: {summary["request_failed"]})
- **Console Errors**: {summary["console_errors"]}
- **Action Errors**: {summary["errors"]}
- **Last URL**: `{summary["last_url"]}`
{hint}
## Actions (counts by apiName)
```json
{json.dumps(summary["actions"], indent=2, ensure_ascii=False)}
```

## Screenshots (latest first)
{shots_md}
"""
    out = base / "trace_report.md"
    out.write_text(md, encoding="utf-8")
    return out, images
