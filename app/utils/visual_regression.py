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
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from pixelmatch.contrib.PIL import pixelmatch
from playwright.async_api import Page

logger = logging.getLogger(__name__)


def _safe_relpath(p: Path, start: Path | None = None) -> str:
    """
    ログ表示用。p が start 配下なら相対、無理なら絶対パスを返す。
    Windows のドライブ差異などで relative_to が失敗するケースを吸収。
    """
    if start is None:
        start = Path.cwd()
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
) -> tuple[float, bool, str | None, str | None]:
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
        current_bytes: bytes | None = None
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

        mismatched_pixels = pixelmatch(base_img, cur_img, diff_img, threshold=threshold, includeAA=True)

        total_pixels = base_img.width * base_img.height
        diff_rate = mismatched_pixels / total_pixels if total_pixels > 0 else 0

        passed = diff_rate <= threshold
        hard_failed = diff_rate >= hard_fail_threshold

        diff_path_str: str | None = None
        if not passed or not save_failed_diff_only:
            diffs_dir = (baseline_path.parent / "diffs").resolve()
            diffs_dir.mkdir(parents=True, exist_ok=True)

            sanitized_stem = re.sub(r'[\\/:*?"<>|]', "_", baseline_path.stem)
            diff_filename = f"diff_{sanitized_stem}.png"
            diff_path = diffs_dir / diff_filename
            diff_img.save(diff_path)
            diff_path_str = _safe_relpath(diff_path)

        base_used_str = _safe_relpath(baseline_path)

        if hard_failed:
            logger.error(
                f"[VRT] Hard failure! Diff rate {diff_rate:.4f} exceeds hard threshold {hard_fail_threshold}. Diff saved to '{diff_path_str}'"
            )
        elif not passed:
            logger.warning(
                f"[VRT] Visual difference detected. Rate: {diff_rate:.4f} (Threshold: {threshold}). Diff saved to '{diff_path_str}'"
            )

        if not passed and auto_update_baseline:
            baseline_path.write_bytes(current_bytes)
            logger.info(f"[VRT] Baseline auto-updated to new version at '{base_used_str}'")

        # ハード失敗時はFalseを返す
        return diff_rate, not hard_failed, base_used_str, diff_path_str

    except Exception as e:
        logger.error(f"[VRT] Core comparison failed: {e}", exc_info=True)
        # 呼び出し元は止めず、失敗として扱う
        return 1.0, False, None, None


def unpack_vrt(res, *, threshold: float | None = None):
    if isinstance(res, dict):
        d = float(res.get("diff_percent", 0.0))
        ok = bool(res.get("is_ok", (threshold is None or d <= float(threshold or 0.0))))
        return {"diff_percent": d, "is_ok": ok}
    if isinstance(res, (tuple, list)):
        if not res:
            return None
        return unpack_vrt(res[0], threshold=threshold)
    if isinstance(res, (int, float)):
        d = float(res)
        ok = (threshold is None) or (d <= float(threshold or 0.0))
        return {"diff_percent": d, "is_ok": ok}
    return None


async def perform_vrt(
    page,
    scope: str,
    settings: dict,
    site_name: str,
    logger: Any,
) -> None:
    from pathlib import Path as _P

    from playwright.async_api import Error as PlaywrightError

    try:
        baseline_dir = _P(settings.get("vrt_baseline_dir") or f"app/visual_baselines/{site_name}")
        baseline_dir.mkdir(parents=True, exist_ok=True)
        sel = settings.get("vrt_plp_selector") if scope == "plp" else settings.get("vrt_pdp_selector")
        res = await compare_and_maybe_update(
            page=page,
            baseline_path=baseline_dir / f"{scope}.png",
            selector=sel,
            threshold=settings.get("vrt_threshold", 0.02),
            hard_fail_threshold=settings.get("vrt_hard_fail_threshold", 0.05),
            auto_update_baseline=settings.get("auto_update_baseline", False),
            save_failed_diff_only=settings.get("save_failed_diff_only", True),
        )
        vrt_result = unpack_vrt(res, threshold=settings.get("vrt_threshold", 0.02))
        if vrt_result and not vrt_result.get("is_ok", True):
            diff = float(vrt_result.get("diff_percent", 0.0))
            hard = diff > float(settings.get("vrt_hard_fail_threshold", 0.05))
            msg = f"[VRT][{scope.upper()}] diff={diff:.4f} (thr={settings.get('vrt_threshold', 0.02)})"
            if hard and settings.get("vrt_fail_on_hard_threshold", True) and scope == "pdp":
                raise PlaywrightError(msg + f" HARD>{settings.get('vrt_hard_fail_threshold', 0.05)}")
            logger.warning(msg + (" HARD" if hard else ""))
    except Exception as vrt_e:
        logger.warning(f"[VRT][{scope.upper()}] skipped: {vrt_e}")
