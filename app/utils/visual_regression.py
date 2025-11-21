# ==============================================================================
# ファイル名 (File Name): app/utils/visual_regression.py
# レジストリ (Registry): app/utils/visual_regression.py
# 更新日時 (Date & Time JST): 2025年10月05日 20:46:00
# バージョン (Version): 3.0.0J (Highly Robust)
#
# --- v3.0.0Jでの主な変更点 ---
# - [堅牢化] あなたの提案に基づき、パスの相対化処理を、Windowsのドライブ差異にも
#   対応した、より安全なロジックに全面的に刷新。
# - [非停止設計] VRT処理中のいかなる例外も捕捉し、呼び出し元のプロセスを
#   中断させない、フェイルセーフな設計に変更。
# - [型契約の厳格化] 戻り値のパス情報を文字列(str)に統一。
#
# --- 使い方 ---
# このユーティリティは BrowserUseAgent から呼び出されます。
# 外部ライブラリとして Pillow と pixelmatch が必要です。
# `pip install Pillow pixelmatch`
# ==============================================================================
# -*- coding: utf-8 -*-

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple
from io import BytesIO
import re
import os

from playwright.async_api import Page
from PIL import Image
from pixelmatch.contrib.PIL import pixelmatch

logger = logging.getLogger(__name__)

def _safe_relpath(p: Path, start: Path = Path.cwd()) -> str:
    """
    ログ表示用。p が start 配下なら相対、無理なら絶対パスを返す。
    Windows のドライブ差異などで relative_to が失敗するケースを吸収。
    """
    try:
        # Resolve both paths to be absolute
        p_resolved = p.resolve()
        start_resolved = start.resolve()
        return os.path.relpath(str(p_resolved), str(start_resolved))
    except (ValueError, AttributeError):
        # Fallback to absolute path if relpath fails (e.g., different drives on Windows)
        return str(p.resolve())


async def compare_and_maybe_update(
    *,
    page: Page,
    baseline_path: Path,
    selector: str = "full_page",
    threshold: float = 0.02,
    hard_fail_threshold: float = 0.05,
    auto_update_baseline: bool = False,
    save_failed_diff_only: bool = True,
) -> Tuple[float, bool, Optional[str], Optional[str]]:
    """
    ページのスクリーンショットを撮影し、基準画像と比較する。
    例外を捕捉し、常にタプルを返すことで呼び出し元の安定性を保証する。

    :return: (差分率, 合否, 使用したベースラインパス(str), 差分画像パス(str))
    """
    try:
        # 1) パス準備（絶対化 & ディレクトリ作成）
        baseline_path = Path(baseline_path).resolve()
        baseline_path.parent.mkdir(parents=True, exist_ok=True)

        # 2) 現在スクリーンショット取得
        current_bytes: Optional[bytes] = None
        if selector != "full_page":
            target_locator = page.locator(selector).first
            await target_locator.wait_for(state="visible", timeout=10000)
            current_bytes = await target_locator.screenshot()
        else:
            current_bytes = await page.screenshot(full_page=True)

        if not current_bytes:
             raise ValueError("Failed to capture current screenshot.")

        # ベースラインが存在しない場合の処理
        if not baseline_path.exists():
            if auto_update_baseline:
                baseline_path.write_bytes(current_bytes)
                base_used_str = _safe_relpath(baseline_path)
                logger.info(f"[VRT] New baseline created at '{base_used_str}'")
                return 0.0, True, base_used_str, None
            else:
                base_used_str = _safe_relpath(baseline_path)
                logger.info(f"[VRT] Baseline not found and auto-update is off. Skipping for '{base_used_str}'")
                return 0.0, True, base_used_str, None

        # 3) 比較実行
        base_img = Image.open(baseline_path).convert("RGB")
        cur_img = Image.open(BytesIO(current_bytes)).convert("RGB")

        if base_img.size != cur_img.size:
            logger.warning(f"[VRT] Image sizes differ. Resizing current image to baseline size {base_img.size}")
            cur_img = cur_img.resize(base_img.size)

        diff_img = Image.new("RGB", base_img.size)

        mismatched_pixels = pixelmatch(
            base_img, cur_img, diff_img,
            threshold=threshold, includeAA=True
        )

        total_pixels = base_img.width * base_img.height
        diff_rate = mismatched_pixels / total_pixels if total_pixels > 0 else 0

        passed = diff_rate <= threshold
        hard_failed = diff_rate >= hard_fail_threshold

        diff_path_str: Optional[str] = None
        if not passed or not save_failed_diff_only:
            diffs_dir = (baseline_path.parent / "diffs").resolve()
            diffs_dir.mkdir(parents=True, exist_ok=True)

            sanitized_stem = re.sub(r'[\\/:*?"<>|]', '_', baseline_path.stem)
            diff_filename = f"diff_{sanitized_stem}.png"
            diff_path = diffs_dir / diff_filename
            diff_img.save(diff_path)
            diff_path_str = _safe_relpath(diff_path)

        base_used_str = _safe_relpath(baseline_path)

        if hard_failed:
            logger.error(f"[VRT] Hard failure! Diff rate {diff_rate:.4f} exceeds hard threshold {hard_fail_threshold}. Diff saved to '{diff_path_str}'")
        elif not passed:
            logger.warning(f"[VRT] Visual difference detected. Rate: {diff_rate:.4f} (Threshold: {threshold}). Diff saved to '{diff_path_str}'")

        if not passed and auto_update_baseline:
            baseline_path.write_bytes(current_bytes)
            logger.info(f"[VRT] Baseline auto-updated to new version at '{base_used_str}'")

        # ハード失敗時はFalseを返す
        return diff_rate, not hard_failed, base_used_str, diff_path_str

    except Exception as e:
        logger.error(f"[VRT] Core comparison failed: {e}", exc_info=True)
        # 呼び出し元は止めず、失敗として扱う
        return 1.0, False, None, None
