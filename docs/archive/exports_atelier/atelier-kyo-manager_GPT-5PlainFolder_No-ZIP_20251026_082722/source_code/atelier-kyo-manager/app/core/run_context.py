# -*- coding: utf-8 -*-
# ==============================================================================
# File Name : app/core/run_context.py
# Version   : 1.6.0J (Logger + Text/Bytes Save + Auto Diagnostic Bundle)
# Date (JST): 2025-10-20 00:20
#
# 変更点要旨
# - (v1.6.0J) logger を標準搭載し、他モジュールから rc.logger を期待しても落ちない。
# - (v1.6.0J) save_content / save_json / save_bytes を提供（observability の期待API）。
# - (v1.6.0J) 実行結果を「アップロードしやすい診断バンドル」に自動集約する
#             build_upload_bundle() を追加。manifest.json 付き。
# - (継続) 非同期スクショ take_screenshot() は維持。
#
# 使用方法（メモ）
# - BrowserUseAgent 側は何も変更不要。失敗/終了時に rc.build_upload_bundle() が走り、
#   instance/runs/<run_id>/upload_bundle.zip が生成されます。
#   その zip を ChatGPT に1ファイルでアップロードすれば解析に十分です。
# ==============================================================================

from __future__ import annotations
import datetime
import json
import logging
import zipfile
from pathlib import Path
from typing import Optional, Union, Iterable, Dict, Any

try:
    from playwright.async_api import Page as AsyncPage
except ImportError:
    AsyncPage = type(None)
try:
    from playwright.sync_api import Page as SyncPage
except ImportError:
    SyncPage = type(None)

PageObject = Union[AsyncPage, SyncPage]


class RunContext:
    """エージェント実行ごとに成果物の入れ物を用意し、保存や収集を担当。"""
    def __init__(self, base_path: str = "instance/runs"):
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.run_path = Path(base_path) / self.run_id
        self.screenshots_path = self.run_path / "screenshots"
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.screenshots_path.mkdir(exist_ok=True)
        self.screenshot_counter = 0

        # ---- logger を標準装備（他モジュールが rc.logger を要求してもOK）----
        self.logger = logging.getLogger(f"RunContext[{self.run_id}]")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.run_path / "system.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        self.logger.addHandler(fh)
        # 既に親にハンドラがあれば二重出力を避ける
        self.logger.propagate = False

    # ---------- 基本パス ----------
    def get_path(self, filename: str) -> Path:
        return self.run_path / filename

    # ---------- 保存系（observability 互換） ----------
    def save_content(self, filename: str, content: str) -> Optional[str]:
        """任意テキストをUTF-8で保存（observability 期待API名）。"""
        try:
            path = self.get_path(filename)
            path.write_text(content, encoding="utf-8")
            self.logger.info(f"[RunContext] Text saved: {path}")
            return str(path)
        except Exception as e:
            self.logger.error(f"[RunContext] Failed to save text to {filename}: {e}")
            return None

    def save_json(self, filename: str, data: Dict[str, Any]) -> Optional[str]:
        """辞書をJSONで保存（ファイル名は呼び出し元が拡張子を決める流儀）。"""
        try:
            path = self.get_path(filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[RunContext] JSON saved: {path}")
            return str(path)
        except Exception as e:
            self.logger.error(f"[RunContext] Failed to save JSON to {filename}: {e}")
            return None

    def save_bytes(self, filename: str, data: bytes) -> Optional[str]:
        """バイナリを保存（将来のHAR/追加添付などにも使用）。"""
        try:
            path = self.get_path(filename)
            path.write_bytes(data)
            self.logger.info(f"[RunContext] Bytes saved: {path}")
            return str(path)
        except Exception as e:
            self.logger.error(f"[RunContext] Failed to save bytes to {filename}: {e}")
            return None

    # ---------- ログ追記 ----------
    def append_log(self, filename: str, content: str) -> Optional[str]:
        path = self.get_path(filename)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(content + "\n")
            return str(path)
        except Exception as e:
            self.logger.error(f"[RunContext] Failed to append to {filename}: {e}")
            return None

    # ---------- スクショ ----------
    async def take_screenshot(self, page: Optional[PageObject], name: str) -> Optional[str]:
        if not page or getattr(page, "is_closed", lambda: True)():
            self.logger.warning(f"Screenshot '{name}' skipped: page is not available.")
            return None
        path = self.screenshots_path / f"{self.screenshot_counter:02d}_{name}.png"
        try:
            await page.screenshot(path=str(path))
            self.screenshot_counter += 1
            self.logger.info(f"Screenshot taken: {path}")
            return str(path)
        except Exception as e:
            self.logger.error(f"Failed to take screenshot to {path}: {e}")
            return None

    # ---------- 診断バンドル ----------
    def _glob_many(self, patterns: Iterable[str]) -> Iterable[Path]:
        for pat in patterns:
            yield from self.run_path.glob(pat)

    def build_upload_bundle(
        self,
        bundle_name: str = "upload_bundle.zip",
        extra_patterns: Optional[Iterable[str]] = None
    ) -> Optional[str]:
        """
        実行成果物を1つのZIPに集約。
        生成物: upload_bundle.zip と manifest.json
        """
        try:
            patterns = [
                "ai_forensic_report.json",
                "navigation_plan.json",
                "raw_hrefs_raw_abs.json",
                "selector_counts_*.json",
                "trace_report.md",
                "trace.zip",
                "network.har",
                "fail_snapshot.md",
                "pdp_extracted_data.json",
                "plp_extracted_items.json",
                "system.log",
                "trace_shot_*.png",
                "screenshots/*.png",
            ]
            if extra_patterns:
                patterns.extend(list(extra_patterns))

            files = sorted(set(p for p in self._glob_many(patterns) if p.exists()))
            manifest = {
                "run_id": self.run_id,
                "count": len(files),
                "files": [str(p.relative_to(self.run_path)) for p in files],
            }
            self.save_json("manifest.json", manifest)

            out = self.run_path / bundle_name
            with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
                # まず manifest を
                z.write(self.run_path / "manifest.json", arcname="manifest.json")
                for p in files:
                    z.write(p, arcname=str(p.relative_to(self.run_path)))
            self.logger.info(f"[RunContext] Upload bundle created: {out}")
            return str(out)
        except Exception as e:
            self.logger.error(f"[RunContext] Failed to build upload bundle: {e}")
            return None
