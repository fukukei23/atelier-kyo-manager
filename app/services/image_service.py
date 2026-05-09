# ======================================================================
# File: image_service.py
# Purpose: 画像ダウンロード + 背景除去のサービス層ラッパー
# Date: 2026-04-20
# ======================================================================
from __future__ import annotations

import io
import logging
from pathlib import Path

import requests
from PIL import Image
from rembg import remove

from app.config.config import AppConfig

logger = logging.getLogger(__name__)


class ImageService:
    """画像のダウンロード・背景除去を提供するサービス"""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir: Path = base_dir or AppConfig.DATA_DIR / "catalog_images"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 画像ダウンロード
    # ------------------------------------------------------------------
    def download_images(self, urls: list[str], product_id: int) -> list[Path]:
        """URLリストから画像をダウンロードしてローカルに保存"""
        dest_dir = self.base_dir / str(product_id)
        dest_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        for i, url in enumerate(urls, 1):
            try:
                img = self._fetch_image(url)
                if img is None:
                    logger.warning("DL failed (returned None): %s", url)
                    continue
                out_path = dest_dir / f"{i:03d}.jpg"
                img.save(out_path, "JPEG", quality=90)
                saved.append(out_path)
                logger.info("DL saved: %s", out_path.name)
            except Exception as e:
                logger.warning("DL error for %s: %s", url, e)
        return saved

    def _fetch_image(self, url: str, timeout: int = 20, max_bytes: int = 5_000_000) -> Image.Image | None:
        """URLから画像を取得（UA偽装・サイズ制限付き）"""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        }
        r = requests.get(url, headers=headers, timeout=timeout, stream=True)
        r.raise_for_status()
        content = r.content if len(r.content) <= max_bytes else r.raw.read(max_bytes)
        return Image.open(io.BytesIO(content)).convert("RGB")

    # ------------------------------------------------------------------
    # 背景除去
    # ------------------------------------------------------------------
    def remove_backgrounds(self, input_paths: list[Path], product_id: int) -> list[Path]:
        """rembg で背景除去。出力先: data/catalog_images/{product_id}/processed/"""
        proc_dir = self.base_dir / str(product_id) / "processed"
        proc_dir.mkdir(parents=True, exist_ok=True)

        results: list[Path] = []
        for p in input_paths:
            try:
                out_path = proc_dir / f"{p.stem}_nobg.png"
                self._remove_bg_single(p, out_path)
                results.append(out_path)
                logger.info("BG removed: %s", out_path.name)
            except Exception as e:
                logger.warning("BG remove failed for %s: %s", p.name, e)
                # 背景除去失敗時は元画像をそのまま使用
                results.append(p)
        return results

    @staticmethod
    def _remove_bg_single(input_path: Path, output_path: Path) -> None:
        """1枚の画像の背景を除去して保存"""
        img = Image.open(input_path).convert("RGB")
        rgba = remove(img)
        if rgba.mode != "RGBA":
            rgba = rgba.convert("RGBA")

        # 横幅1200px上限
        if rgba.width > 1200:
            h = int(rgba.height * 1200 / rgba.width)
            rgba = rgba.resize((1200, h), Image.LANCZOS)

        rgba.save(output_path, "PNG", optimize=True)
