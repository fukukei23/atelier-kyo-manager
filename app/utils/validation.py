from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


class ValidationError(ValueError):
    """バリデーション失敗時の例外。"""


def validate_url(url: str, *, allowed_schemes: tuple[str, ...] = ("https",)) -> str:
    """URLのスキームを検証する。SSRF対策で許可スキーム以外を拒否。"""
    parsed = urlparse(url)
    if parsed.scheme not in allowed_schemes:
        raise ValidationError(f"許可されていないスキーム: {parsed.scheme}")
    if not parsed.netloc:
        raise ValidationError("ホストが指定されていません")
    return url


def validate_int(
    value: str | int, *, name: str = "value", min_val: int | None = None, max_val: int | None = None
) -> int:
    """文字列/intをintに変換し、範囲チェック。"""
    try:
        result = int(value)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"{name} は整数で指定してください") from e
    if min_val is not None and result < min_val:
        raise ValidationError(f"{name} は {min_val} 以上にしてください")
    if max_val is not None and result > max_val:
        raise ValidationError(f"{name} は {max_val} 以下にしてください")
    return result


def validate_float(
    value: str | float, *, name: str = "value", min_val: float | None = None, max_val: float | None = None
) -> float:
    """文字列/floatをfloatに変換し、範囲チェック。"""
    try:
        result = float(value)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"{name} は数値で指定してください") from e
    if min_val is not None and result < min_val:
        raise ValidationError(f"{name} は {min_val} 以上にしてください")
    if max_val is not None and result > max_val:
        raise ValidationError(f"{name} は {max_val} 以下にしてください")
    return result


def sanitize_filename(name: str) -> str:
    """パストラバーサル対策。Path.nameでディレクトリ部分を除去。"""
    return Path(name).name
