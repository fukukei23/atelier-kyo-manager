"""
pipeline_service のテスト — バッチ実行・アップロード・フォールバックの検証
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.pipeline_service import BatchResult, PipelineResult, PipelineService


class TestRunBatch:
    """run_batch の全パターン検証"""

    @patch("app.services.pipeline_service.db")
    def test_mixed_results(self, mock_db):
        """success + skipped + failed の混在バッチ"""
        svc = PipelineService(image_service=MagicMock())

        # run_batch is sequential: pid=1 found, pid=2 not found (skip), pid=3 found
        p1 = MagicMock()
        p1.id = 1
        p3 = MagicMock()
        p3.id = 3

        # db.session.get called per pid that exists: 1, 2(skip), 3
        mock_db.session.get.side_effect = [p1, None, p3]

        run_result_ok = PipelineResult(product_id=1, status="success")
        with patch.object(
            svc,
            "run",
            side_effect=[
                run_result_ok,
                Exception("crash"),
            ],
        ):
            result = svc.run_batch([1, 2, 3])

        assert result.total == 3
        assert result.success == 1
        assert result.skipped == 1
        assert result.failed == 1
        assert result.elapsed_sec >= 0

    @patch("app.services.pipeline_service.db")
    def test_all_success(self, mock_db):
        """全件成功"""
        svc = PipelineService(image_service=MagicMock())

        products = [MagicMock(id=i) for i in range(3)]
        mock_db.session.get.side_effect = products

        with patch.object(svc, "run", return_value=PipelineResult(product_id=0, status="success")):
            result = svc.run_batch([1, 2, 3])

        assert result.total == 3
        assert result.success == 3
        assert result.failed == 0

    @patch("app.services.pipeline_service.db")
    def test_empty_batch(self, mock_db):
        """空リスト → 0件"""
        svc = PipelineService(image_service=MagicMock())
        result = svc.run_batch([])
        assert result.total == 0
        assert result.results == []

    @patch("app.services.pipeline_service.db")
    def test_partial_in_batch(self, mock_db):
        """partial 成功のカウント"""
        svc = PipelineService(image_service=MagicMock())
        p = MagicMock()
        mock_db.session.get.return_value = p

        with patch.object(svc, "run", return_value=PipelineResult(product_id=1, status="partial")):
            result = svc.run_batch([1])

        assert result.partial == 1
        assert result.success == 0


class TestSaveUploadedImages:
    """save_uploaded_images の検証"""

    def test_saves_valid_images(self, tmp_path):
        svc = PipelineService(image_service=MagicMock())
        svc.image_svc.base_dir = tmp_path

        f1 = MagicMock()
        f1.filename = "photo1.jpg"
        f2 = MagicMock()
        f2.filename = "photo2.png"

        saved = svc.save_uploaded_images(42, [f1, f2])

        assert len(saved) == 2
        f1.save.assert_called_once()
        f2.save.assert_called_once()

    def test_skips_non_image_files(self, tmp_path):
        svc = PipelineService(image_service=MagicMock())
        svc.image_svc.base_dir = tmp_path

        doc = MagicMock()
        doc.filename = "document.pdf"
        img = MagicMock()
        img.filename = "photo.jpg"

        saved = svc.save_uploaded_images(42, [doc, img])

        assert len(saved) == 1
        doc.save.assert_not_called()

    def test_empty_files_list(self, tmp_path):
        svc = PipelineService(image_service=MagicMock())
        svc.image_svc.base_dir = tmp_path

        saved = svc.save_uploaded_images(42, [])
        assert saved == []


class TestFindUploadedImages:
    """_find_uploaded_images のディレクトリ探索検証"""

    def test_finds_images(self, tmp_path):
        svc = PipelineService(image_service=MagicMock())
        svc.image_svc.base_dir = tmp_path

        upload_dir = tmp_path / "42" / "uploaded"
        upload_dir.mkdir(parents=True)
        (upload_dir / "a.jpg").write_text("img")
        (upload_dir / "b.png").write_text("img")
        (upload_dir / "c.txt").write_text("not img")

        result = svc._find_uploaded_images(42)
        names = [p.name for p in result]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.txt" not in names

    def test_returns_empty_when_no_dir(self, tmp_path):
        svc = PipelineService(image_service=MagicMock())
        svc.image_svc.base_dir = tmp_path

        result = svc._find_uploaded_images(999)
        assert result == []


class TestStepRemoveBackgrounds:
    """_step_remove_backgrounds のフォールバック検証"""

    def test_fallback_returns_original_paths_on_failure(self):
        mock_svc = MagicMock()
        mock_svc.remove_backgrounds.side_effect = RuntimeError("rembg failed")

        svc = PipelineService(image_service=mock_svc)
        paths = [Path("/tmp/img1.jpg"), Path("/tmp/img2.png")]

        result = svc._step_remove_backgrounds(paths, 1)

        # フォールバックで元のパスを返す
        assert result == paths

    def test_returns_processed_paths_on_success(self):
        mock_svc = MagicMock()
        processed = [Path("/tmp/1/img1_nobg.png")]
        mock_svc.remove_backgrounds.return_value = processed

        svc = PipelineService(image_service=mock_svc)
        paths = [Path("/tmp/img1.jpg")]

        result = svc._step_remove_backgrounds(paths, 1)
        assert result == processed


class TestPipelineResultDataclass:
    """PipelineResult / BatchResult の基本検証"""

    def test_pipeline_result_defaults(self):
        r = PipelineResult(product_id=1)
        assert r.status == "pending"
        assert r.collected_urls == []
        assert r.errors == []
        assert r.elapsed_sec == 0.0

    def test_batch_result_defaults(self):
        r = BatchResult()
        assert r.total == 0
        assert r.success == 0
        assert r.results == []
