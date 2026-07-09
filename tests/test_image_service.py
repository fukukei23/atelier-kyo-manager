"""Tests for app/services/image_service.py"""

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


def _make_fake_image() -> bytes:
    """Create a minimal valid JPEG in memory."""
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def service(tmp_path):
    from app.services.image_service import ImageService

    return ImageService(base_dir=tmp_path / "images")


class TestImageServiceInit:
    def test_base_dir_created(self, tmp_path):
        from app.services.image_service import ImageService

        ImageService(base_dir=tmp_path / "new_dir")
        assert (tmp_path / "new_dir").exists()

    def test_custom_base_dir(self, tmp_path):
        from app.services.image_service import ImageService

        custom = tmp_path / "custom"
        svc = ImageService(base_dir=custom)
        assert svc.base_dir == custom


class TestFetchImage:
    @patch("app.services.image_service.requests.get")
    def test_fetch_valid_image(self, mock_get, service):
        fake_bytes = _make_fake_image()
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [fake_bytes]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = service._fetch_image("https://example.com/image.jpg")
        assert result is not None
        assert result.mode == "RGB"

    def test_fetch_non_https_blocked(self, service):
        result = service._fetch_image("http://example.com/image.jpg")
        assert result is None

    def test_fetch_invalid_url_blocked(self, service):
        result = service._fetch_image("not-a-url")
        assert result is None

    @patch("app.services.image_service.requests.get")
    def test_fetch_too_large_returns_none(self, mock_get, service):
        # 6MB chunk exceeds the 5MB limit
        large_chunk = b"x" * 6_000_001
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [large_chunk]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = service._fetch_image("https://example.com/huge.jpg")
        assert result is None


class TestDownloadImages:
    @patch("app.services.image_service.ImageService._fetch_image")
    def test_download_saves_images(self, mock_fetch, service, tmp_path):
        fake_img = Image.new("RGB", (10, 10))
        mock_fetch.return_value = fake_img

        urls = ["https://example.com/img1.jpg", "https://example.com/img2.jpg"]
        saved = service.download_images(urls, product_id=42)
        assert len(saved) == 2

    @patch("app.services.image_service.ImageService._fetch_image")
    def test_download_skips_failed_fetch(self, mock_fetch, service):
        mock_fetch.return_value = None

        urls = ["https://example.com/fail.jpg"]
        saved = service.download_images(urls, product_id=1)
        assert saved == []

    @patch("app.services.image_service.ImageService._fetch_image")
    def test_download_empty_urls(self, mock_fetch, service):
        saved = service.download_images([], product_id=1)
        assert saved == []
        mock_fetch.assert_not_called()
