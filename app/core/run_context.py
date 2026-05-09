# 2025年09月21日 16:28 JST
# ファイル名: run_context.py
# フォルダパス: app/core/
# バージョン: 1.2.0J (Async Hardened)
#
# --- v1.2.0Jでの主な変更点 ---
# - [非同期対応] あなたの指摘に基づき、`take_screenshot`を`async def`に
#   変更し、内部で`await page.screenshot()`を呼び出すように修正。
#   これにより`RuntimeWarning`を解消し、非同期アーキテクチャに完全対応します。

import datetime
import json
import logging
from pathlib import Path
from typing import Union

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
    """AIエージェントの単一実行コンテキストを管理するクラス。"""

    def __init__(self, base_path: str = "instance/runs"):
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.run_path = Path(base_path) / self.run_id
        self.screenshots_path = self.run_path / "screenshots"
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.screenshots_path.mkdir(exist_ok=True)
        self.screenshot_counter = 0

    def save_json(self, filename: str, data: dict):
        path = self.run_path / filename
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.debug(f"JSON saved: {path}")
        except Exception as e:
            logging.error(f"Failed to save JSON to {path}: {e}")

    def get_path(self, filename: str) -> Path:
        return self.run_path / filename

    def append_log(self, filename: str, content: str):
        path = self.run_path / filename
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(content + "\n")
        except Exception as e:
            logging.error(f"Failed to append to log {path}: {e}")

    async def take_screenshot(self, page: PageObject | None, name: str) -> str | None:
        """
        [ASYNC] 命名規約に沿った名前でスクリーンショットを撮影し、そのパスを返す。
        """
        if not page or page.is_closed():
            logging.warning(f"Screenshot '{name}' was skipped because the page is not available.")
            return None

        path_str = str(self.screenshots_path / f"{self.screenshot_counter:02d}_{name}.png")
        try:
            # --- ★★★ `await`を追加 ★★★ ---
            await page.screenshot(path=path_str)
            self.screenshot_counter += 1
            logging.info(f"Screenshot taken: {path_str}")
            return path_str
        except Exception as e:
            logging.error(f"Failed to take screenshot to {path_str}: {e}")
            return None

    def setup_system_logger(self) -> logging.FileHandler:
        log_path = self.get_path("system.log")
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        return handler

    # --- Utility: save arbitrary text content ---
    def save_content(self, filename: str, content: str):
        """
        Save text content to a file under the current run directory.
        """
        path = self.get_path(filename)
        try:
            path.write_text(content, encoding="utf-8")
            logging.debug(f"Content saved: {path}")
        except Exception as e:
            logging.error(f"Failed to save content to {path}: {e}")
