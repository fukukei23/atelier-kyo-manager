from __future__ import annotations

from datetime import datetime, timezone


def _utcnow() -> datetime:
    """naive UTC現在時刻を返す。datetime.utcnow()の置き換え用。

    SQLiteの既存データがnaive datetimeのため、tzinfoは付けない。
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
