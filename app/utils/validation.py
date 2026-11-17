from __future__ import annotations

import math
import re
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


# 表計算（Excel/LibreOffice/Google Sheets）がセル先頭で式として解釈する接頭辞。
# `=` `+` `@` タブ CR は常に危険。`-` は「負の数値」で正当なため数値風のみ除外する。
_CSV_FORMULA_PREFIXES = ("=", "+", "@", "\t", "\r")
_CSV_NUMERIC_PATTERN = re.compile(r"-[0-9][0-9.,%\s]*")


def sanitize_csv_cell(value: object) -> object:
    """CSVセルの文字列から表計算の式インジェクションを無効化する。

    スクレイピングで取得した商品名などは攻撃者制御下にあるため、
    セル先頭に `=` 等があると、エクスポート後に表計算で式が実行される。
    先頭にアポストフィを付してテキストとして扱う（Strix vuln-0019）。

    数値・None 等はそのまま返す（`str` 以外の値は書き換えない）。
    """
    if not isinstance(value, str) or not value:
        return value
    stripped = value.lstrip()
    if not stripped:
        return value
    head = stripped[0]
    if head in _CSV_FORMULA_PREFIXES:
        return "'" + value
    if head == "-" and not _CSV_NUMERIC_PATTERN.fullmatch(stripped):
        return "'" + value
    return value


def validate_fetch_url(url: str, *, allowed_schemes: tuple[str, ...] = ("http", "https")) -> str:
    """取得先URLを検証する。

    `file://` などのローカルファイルスキームや内部ホストへの読み取りを防ぐ。
    Strix vuln-0016 / vuln-0020（取得先URLの未検証）。
    """
    if not url or not isinstance(url, str):
        raise ValidationError("URLが空です")
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in allowed_schemes:
        raise ValidationError(f"許可されていないスキーム: {parsed.scheme}")
    if not parsed.netloc:
        raise ValidationError("ホストが指定されていません")
    return url


def validate_fx_rate(rate: object) -> float | None:
    """為替レートを検証する。

    None / 0 / 負値 / NaN / inf は「有効なレートなし」と判定し、
    1.0 への黙示フォールバックを防ぐ（Strix vuln-0008 / vuln-0010 / vuln-0011）。
    """
    if rate is None:
        return None
    try:
        value = float(rate)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


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
