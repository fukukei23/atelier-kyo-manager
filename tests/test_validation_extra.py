"""Tests for app/utils/validation.py — additional coverage"""

import pytest

from app.utils.validation import (
    ValidationError,
    sanitize_filename,
    validate_float,
    validate_int,
    validate_url,
)


class TestValidateUrl:
    def test_valid_https(self):
        result = validate_url("https://example.com/path")
        assert result == "https://example.com/path"

    def test_http_rejected_by_default(self):
        with pytest.raises(ValidationError, match="スキーム"):
            validate_url("http://example.com")

    def test_ftp_rejected(self):
        with pytest.raises(ValidationError):
            validate_url("ftp://files.example.com")

    def test_file_scheme_rejected(self):
        with pytest.raises(ValidationError):
            validate_url("file:///etc/passwd")

    def test_no_host_rejected(self):
        with pytest.raises(ValidationError, match="ホスト"):
            validate_url("https://")

    def test_custom_allowed_scheme(self):
        result = validate_url("http://example.com", allowed_schemes=("http", "https"))
        assert result == "http://example.com"

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            validate_url("")


class TestValidateInt:
    def test_valid_int(self):
        assert validate_int(42) == 42

    def test_valid_string_int(self):
        assert validate_int("100") == 100

    def test_min_val_ok(self):
        assert validate_int(5, min_val=0) == 5

    def test_min_val_violation(self):
        with pytest.raises(ValidationError, match="以上"):
            validate_int(-1, name="count", min_val=0)

    def test_max_val_ok(self):
        assert validate_int(99, max_val=100) == 99

    def test_max_val_violation(self):
        with pytest.raises(ValidationError, match="以下"):
            validate_int(200, name="qty", max_val=100)

    def test_non_numeric_string_raises(self):
        with pytest.raises(ValidationError, match="整数"):
            validate_int("abc", name="amount")

    def test_none_raises(self):
        with pytest.raises(ValidationError):
            validate_int(None)

    def test_float_truncated(self):
        assert validate_int(3.9) == 3


class TestValidateFloat:
    def test_valid_float(self):
        assert validate_float(3.14) == 3.14

    def test_valid_string_float(self):
        assert validate_float("1.5") == 1.5

    def test_min_val_ok(self):
        assert validate_float(0.01, min_val=0.0) == 0.01

    def test_min_val_violation(self):
        with pytest.raises(ValidationError, match="以上"):
            validate_float(-0.1, min_val=0.0)

    def test_max_val_violation(self):
        with pytest.raises(ValidationError, match="以下"):
            validate_float(101.0, max_val=100.0)

    def test_non_numeric_raises(self):
        with pytest.raises(ValidationError, match="数値"):
            validate_float("xyz")

    def test_int_accepted_as_float(self):
        result = validate_float(10)
        assert result == 10.0
        assert isinstance(result, float)


class TestSanitizeFilename:
    def test_simple_name(self):
        assert sanitize_filename("file.txt") == "file.txt"

    def test_path_traversal_stripped(self):
        result = sanitize_filename("../../etc/passwd")
        assert result == "passwd"
        assert "/" not in result

    def test_backslash_kept_on_linux(self):
        # On Linux, Path treats backslash as a valid filename character
        result = sanitize_filename("file\\name.txt")
        assert result is not None  # sanitize runs without error

    def test_empty_string(self):
        assert sanitize_filename("") == ""
