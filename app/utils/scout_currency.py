"""通貨検出・価格変換ユーティリティ"""

from __future__ import annotations

import re

_CURRENCY_SIGNS = {
    "¥": "JPY",
    "￥": "JPY",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "₩": "KRW",
    "₫": "VND",
    "A$": "AUD",
    "C$": "CAD",
}
_CURR_WORDS = {"JPY": ["JPY", "円"], "USD": ["USD"], "EUR": ["EUR"], "GBP": ["GBP"], "AUD": ["AUD"], "CAD": ["CAD"]}

_PRICE_RE = re.compile(r"([0-9][0-9\.,\s]*)")


def detect_currency(text: str, currency_hint: str | None = None) -> str:
    if currency_hint:
        return currency_hint.upper()
    for sign, code in _CURRENCY_SIGNS.items():
        if sign in text:
            return code
    up = text.upper()
    for code, words in _CURR_WORDS.items():
        if any(w in up for w in words):
            return code
    return "UNKNOWN"


def to_number(price_text: str) -> int | None:
    t = price_text.replace(" ", " ")
    m = _PRICE_RE.search(t)
    if not m:
        return None
    digits = re.sub(r"[^\d]", "", m.group(1))
    return int(digits) if digits else None


def convert_price(price: int, currency: str, fx_to: str | None, fx_table: dict[str, float]) -> float | None:
    if not fx_to:
        return None
    fx_to, currency = fx_to.upper(), (currency or "").upper()
    if fx_to == "JPY":
        if currency == "JPY":
            return float(price)
        rate = fx_table.get(currency)
        return price * rate if rate else None
    return None
