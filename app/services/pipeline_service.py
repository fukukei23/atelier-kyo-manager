# ======================================================================
# File: pipeline_service.py
# Purpose: 画像収集→背景除去→説明文生成→出品テキスト生成の統合パイプライン
# Date: 2026-04-20
# ======================================================================
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config.config import AppConfig
from app.extensions import db
from app.models.product import Product
from app.services.image_service import ImageService

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    product_id: int
    status: str = "pending"           # success / partial / failed
    collected_urls: list[str] = field(default_factory=list)
    downloaded_paths: list[str] = field(default_factory=list)
    processed_paths: list[str] = field(default_factory=list)
    description: str = ""
    listing_text: str = ""
    errors: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0


class PipelineService:
    """商品出品パイプラインの統合オーケストレータ"""

    def __init__(self, image_service: ImageService | None = None):
        self.image_svc = image_service or ImageService()

    def run(self, product_id: int, site_key: str = "") -> PipelineResult:
        """パイプライン全体を実行"""
        t0 = time.time()
        result = PipelineResult(product_id=product_id)
        product = db.session.get(Product, product_id)
        if product is None:
            result.status = "failed"
            result.errors.append(f"Product id={product_id} not found")
            return result

        # ステータス更新: running
        product.pipeline_status = "running"
        product.pipeline_error = None
        db.session.commit()

        try:
            # Step 1: 画像URL収集
            urls = self._step_collect_images(product, site_key)
            result.collected_urls = urls

            # Step 2: 画像ダウンロード
            downloaded = self._step_download_images(urls, product_id)
            result.downloaded_paths = [str(p) for p in downloaded]

            # Step 2b: 手動アップロード画像も検索
            uploaded = self._find_uploaded_images(product_id)
            all_images = downloaded + uploaded

            # Step 3: 背景除去
            if all_images:
                processed = self._step_remove_backgrounds(all_images, product_id)
                result.processed_paths = [str(p) for p in processed
                                          if "processed" in str(p) or "_nobg" in str(p)]

            # Step 4: 説明文生成
            description = self._step_generate_description(product)
            if description:
                result.description = description
                product.description = description

            # Step 5: 出品テキスト生成
            listing_text = self._step_generate_listing(product)
            result.listing_text = listing_text

            # 結果保存
            product.processed_images = json.dumps(result.processed_paths)
            product.pipeline_run_at = datetime.utcnow()

            # ステータス判定
            if result.errors:
                product.pipeline_status = "partial"
                result.status = "partial"
            else:
                product.pipeline_status = "success"
                result.status = "success"

            db.session.commit()

        except Exception as e:
            db.session.rollback()
            result.status = "failed"
            result.errors.append(str(e))
            product.pipeline_status = "failed"
            product.pipeline_error = str(e)[:500]
            db.session.commit()
            logger.exception("Pipeline failed for product %d", product_id)

        result.elapsed_sec = round(time.time() - t0, 2)
        return result

    # ------------------------------------------------------------------
    # 各ステップ
    # ------------------------------------------------------------------
    def _step_collect_images(
        self, product: Product, site_key: str
    ) -> list[str]:
        """Step 1: 画像URL収集（CrawlerService）"""
        if not site_key or not AppConfig.SITES:
            logger.info("No site config, skipping image collection")
            return []

        try:
            from app.utils.ai_image_crawler import CrawlerService
            crawler = CrawlerService(
                sites_config=AppConfig.SITES,
                headless=AppConfig.HEADLESS,
            )
            query = f"{product.brand or ''} {product.name or ''}".strip()
            urls = crawler.search_and_collect_images(
                site_key=site_key,
                query=query,
                max_results=6,
            )
            logger.info("Collected %d image URLs", len(urls))
            return urls
        except Exception as e:
            logger.warning("Image collection failed: %s", e)
            return []

    def _step_download_images(
        self, urls: list[str], product_id: int
    ) -> list[Path]:
        """Step 2: 画像ダウンロード"""
        if not urls:
            return []
        try:
            return self.image_svc.download_images(urls, product_id)
        except Exception as e:
            logger.warning("Image download failed: %s", e)
            return []

    def _find_uploaded_images(self, product_id: int) -> list[Path]:
        """手動アップロード済み画像を検索"""
        upload_dir = self.image_svc.base_dir / str(product_id) / "uploaded"
        if not upload_dir.exists():
            return []
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(upload_dir.glob(ext))
        return sorted(images)

    def _step_remove_backgrounds(
        self, paths: list[Path], product_id: int
    ) -> list[Path]:
        """Step 3: 背景除去"""
        try:
            return self.image_svc.remove_backgrounds(paths, product_id)
        except Exception as e:
            logger.warning("Background removal failed: %s", e)
            return paths

    def _step_generate_description(self, product: Product) -> str:
        """Step 4: AI説明文生成"""
        try:
            from app.utils.ai_generate_descriptions import DescriptionGenerator
            generator = DescriptionGenerator()
            product_info = {
                "brand": product.brand or "",
                "name": product.name or "",
                "category": product.item_category or "",
                "features": f"{product.color or ''} {product.material or ''}".strip(),
            }
            return generator.generate(product_info, model_name="gemini")
        except Exception as e:
            logger.warning("Description generation failed: %s", e)
            return ""

    def _step_generate_listing(self, product: Product) -> str:
        """Step 5: 出品テキスト生成"""
        try:
            from app.services.template_service import generate_listing_text
            return generate_listing_text(product)
        except Exception as e:
            logger.warning("Listing text generation failed: %s", e)
            return ""

    # ------------------------------------------------------------------
    # 手動画像アップロード処理
    # ------------------------------------------------------------------
    def save_uploaded_images(
        self, product_id: int, files: list
    ) -> list[Path]:
        """手動アップロード画像を保存"""
        upload_dir = self.image_svc.base_dir / str(product_id) / "uploaded"
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        for f in files:
            filename = f.filename or ""
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            dest = upload_dir / filename
            f.save(dest)
            saved.append(dest)
            logger.info("Uploaded: %s", dest.name)
        return saved
