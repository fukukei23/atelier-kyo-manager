# ======================================================================
# File: test_pipeline_service.py
# Purpose: PipelineService のユニットテスト（全コンポーネントモック）
# Date: 2026-04-20
# ======================================================================
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app import create_app
from app.extensions import db as _db
from app.models.product import Product


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------
@pytest.fixture(scope="function")
def app():
    """テスト用Flaskアプリ"""
    app = create_app()
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["WTF_CSRF_ENABLED"] = False
    with app.app_context():
        _db.create_all()
        yield app
        _db.session.remove()
        _db.drop_all()


@pytest.fixture
def sample_product(app):
    """テスト用Product"""
    p = Product(
        name="テストバッグ",
        brand="GUCCI",
        purchase_price=50000,
        selling_price=80000,
        stock_status=True,
        listing_status="draft",
        brand_tier="medium",
    )
    _db.session.add(p)
    _db.session.commit()
    return p


# -----------------------------------------------------------------------
# PipelineService.run() テスト
# -----------------------------------------------------------------------
class TestPipelineService:
    """PipelineService のテスト"""

    @patch("app.services.pipeline_service.PipelineService._step_generate_listing")
    @patch("app.services.pipeline_service.PipelineService._step_generate_description")
    @patch("app.services.pipeline_service.PipelineService._step_remove_backgrounds")
    @patch("app.services.pipeline_service.PipelineService._step_download_images")
    @patch("app.services.pipeline_service.PipelineService._step_collect_images")
    def test_run_full_success(
        self,
        mock_collect,
        mock_download,
        mock_remove_bg,
        mock_desc,
        mock_listing,
        app,
        sample_product,
    ):
        """全ステップ成功"""
        from app.services.pipeline_service import PipelineService

        mock_collect.return_value = ["https://example.com/img1.jpg"]
        mock_download.return_value = [Path("/tmp/1/001.jpg")]
        mock_remove_bg.return_value = [Path("/tmp/1/processed/001_nobg.png")]
        mock_desc.return_value = "素晴らしいバッグです"
        mock_listing.return_value = "【GUCCI】テストバッグ\nカラー: "

        svc = PipelineService()
        result = svc.run(sample_product.id)

        assert result.status == "success"
        assert result.collected_urls == ["https://example.com/img1.jpg"]
        assert result.description == "素晴らしいバッグです"
        assert len(result.processed_paths) == 1
        assert result.errors == []

        # DBの確認
        updated = _db.session.get(Product, sample_product.id)
        assert updated.pipeline_status == "success"
        assert updated.description == "素晴らしいバッグです"

    def test_run_product_not_found(self, app, sample_product):
        """存在しないProduct ID"""
        from app.services.pipeline_service import PipelineService

        svc = PipelineService()
        result = svc.run(product_id=9999)

        assert result.status == "failed"
        assert any("not found" in e for e in result.errors)

    @patch("app.services.pipeline_service.PipelineService._step_generate_listing")
    @patch("app.services.pipeline_service.PipelineService._step_generate_description")
    @patch("app.services.pipeline_service.PipelineService._step_remove_backgrounds")
    @patch("app.services.pipeline_service.PipelineService._step_download_images")
    @patch("app.services.pipeline_service.PipelineService._step_collect_images")
    def test_run_partial_success_no_images(
        self,
        mock_collect,
        mock_download,
        mock_remove_bg,
        mock_desc,
        mock_listing,
        app,
        sample_product,
    ):
        """画像収集失敗 → 説明文のみ生成（partial）"""
        from app.services.pipeline_service import PipelineService

        mock_collect.return_value = []
        mock_download.return_value = []
        mock_remove_bg.return_value = []
        mock_desc.return_value = "説明文のみ"
        mock_listing.return_value = "【GUCCI】テストバッグ"

        svc = PipelineService()
        result = svc.run(sample_product.id)

        assert result.status == "success"  # エラーがなくても画像0
        assert result.collected_urls == []
        assert result.description == "説明文のみ"

    @patch("app.services.pipeline_service.PipelineService._step_generate_listing")
    @patch("app.services.pipeline_service.PipelineService._step_generate_description")
    @patch("app.services.pipeline_service.PipelineService._step_remove_backgrounds")
    @patch("app.services.pipeline_service.PipelineService._step_download_images")
    @patch("app.services.pipeline_service.PipelineService._step_collect_images")
    def test_run_desc_failure_uses_fallback(
        self,
        mock_collect,
        mock_download,
        mock_remove_bg,
        mock_desc,
        mock_listing,
        app,
        sample_product,
    ):
        """説明文生成失敗 → 空文字でも継続"""
        from app.services.pipeline_service import PipelineService

        mock_collect.return_value = []
        mock_download.return_value = []
        mock_remove_bg.return_value = []
        mock_desc.return_value = ""  # 生成失敗
        mock_listing.return_value = "テンプレートのみ"

        svc = PipelineService()
        result = svc.run(sample_product.id)

        assert result.listing_text == "テンプレートのみ"


# -----------------------------------------------------------------------
# ImageService テスト
# -----------------------------------------------------------------------
class TestImageService:
    """ImageService のテスト"""

    def test_download_images_empty_urls(self, app, tmp_path):
        """URL空リスト → 空結果"""
        from app.services.image_service import ImageService

        svc = ImageService(base_dir=tmp_path)
        result = svc.download_images([], product_id=1)
        assert result == []

    @patch("app.services.image_service.ImageService._fetch_image")
    def test_download_images_success(self, mock_fetch, app, tmp_path):
        """正常DL"""
        from PIL import Image

        mock_fetch.return_value = Image.new("RGB", (100, 100), "red")

        from app.services.image_service import ImageService

        svc = ImageService(base_dir=tmp_path)
        result = svc.download_images(
            ["https://example.com/a.jpg", "https://example.com/b.jpg"],
            product_id=42,
        )
        assert len(result) == 2
        assert result[0].parent.name == "42"

    @patch("app.services.image_service.ImageService._fetch_image")
    def test_download_images_partial_failure(self, mock_fetch, app, tmp_path):
        """一部DL失敗 → 成功分のみ"""
        from PIL import Image

        mock_fetch.side_effect = [Image.new("RGB", (50, 50)), None]

        from app.services.image_service import ImageService

        svc = ImageService(base_dir=tmp_path)
        result = svc.download_images(
            ["https://example.com/a.jpg", "https://example.com/b.jpg"],
            product_id=1,
        )
        assert len(result) == 1

    @patch("app.services.image_service.remove")
    def test_remove_backgrounds(self, mock_rembg, app, tmp_path):
        """背景除去正常"""
        from PIL import Image

        mock_rembg.return_value = Image.new("RGBA", (100, 100), (255, 0, 0, 128))

        # 入力画像を用意
        input_dir = tmp_path / "1"
        input_dir.mkdir()
        input_img = input_dir / "001.jpg"
        Image.new("RGB", (100, 100), "blue").save(input_img)

        from app.services.image_service import ImageService

        svc = ImageService(base_dir=tmp_path)
        result = svc.remove_backgrounds([input_img], product_id=1)
        assert len(result) == 1
        assert "_nobg" in result[0].name


# -----------------------------------------------------------------------
# Route テスト
# -----------------------------------------------------------------------
class TestPipelineRoutes:
    """パイプライン関連ルートのテスト"""

    @patch("app.services.pipeline_service.PipelineService.run")
    def test_run_pipeline_route(self, mock_run, app, sample_product):
        """POST /products/<id>/run-pipeline"""
        from app.services.pipeline_service import PipelineResult

        mock_run.return_value = PipelineResult(
            product_id=sample_product.id,
            status="success",
            processed_paths=["/tmp/1/processed/001_nobg.png"],
            elapsed_sec=5.0,
        )

        with app.test_client() as client:
            # login_required をバイパスするため login_context を使用
            with app.test_request_context():
                from app.models.user import User

                # テストユーザー作成
                test_user = User(username="test")
                test_user.set_password("test")
                _db.session.add(test_user)
                _db.session.commit()

            with client.session_transaction() as sess:
                sess["_user_id"] = str(test_user.id)

            resp = client.post(
                f"/products/{sample_product.id}/run-pipeline",
                data={"csrf_token": "dummy"},
            )
            assert resp.status_code == 302  # redirect

    def test_pipeline_result_route(self, app, sample_product):
        """GET /products/<id>/pipeline-result"""
        from app.models.user import User

        test_user = User(username="test2")
        test_user.set_password("test")
        _db.session.add(test_user)
        _db.session.commit()

        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess["_user_id"] = str(test_user.id)
            resp = client.get(f"/products/{sample_product.id}/pipeline-result")
            assert resp.status_code == 200
            assert "パイプライン結果" in resp.data.decode("utf-8")

    def test_pipeline_result_not_found(self, app):
        """GET /products/9999/pipeline-result → 404"""
        from app.models.user import User

        test_user = User(username="test3")
        test_user.set_password("test")
        _db.session.add(test_user)
        _db.session.commit()

        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess["_user_id"] = str(test_user.id)
            resp = client.get("/products/9999/pipeline-result")
            assert resp.status_code == 404
