"""
Shared product name matching utilities.
Extracted from buyma_price_scraper for reuse across services.

Usage:
    from app.utils.product_matching import cross_site_match_score, extract_model_numbers

The generalized cross_site_match_score() works across any two product name strings,
not limited to BUYMA matching like the original _match_score() in buyma_price_scraper.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOISE_WORDS: set[str] = {
    "中古",
    "USED",
    "used",
    "新品",
    "未使用",
    "即発",
    "国内発送",
    "送料込",
    "関税込",
    "送料・関税込",
    "送料・関税込み",
    "直営店買付",
    "セール",
    "人気",
    "入手困難",
    "限定",
    "レッド",
    "ディズ",
    " unisex",
    "100%",
    "☆",
    "★",
    "♪",
    "【",
    "】",
    "！",
    "全色",
    "全カラー",
    "カラバリあり",
    "再入荷",
    "完売",
    "残りわずか",
    "在庫あり",
    "残少",
    "あすつく",
    "翌日配送",
    "正規品",
    "本物保証",
}

MODEL_NUMBER_RE = re.compile(r"\b([A-Z0-9]{5,12})\b")

SIZE_RE = re.compile(
    r"\b(FREE|ONE\s*SIZE|ONE SIZE|F|XS|S|M|L|XL|XXL|2XL|3XL|(\d{2,3})\s*cm)\b",
    re.IGNORECASE,
)

COLOR_NORMALIZE: dict[str, str] = {
    "BLACK": "BLACK",
    "NERO": "BLACK",
    "黒": "BLACK",
    "ブラック": "BLACK",
    "NOIR": "BLACK",
    "WHITE": "WHITE",
    "BIANCO": "WHITE",
    "白": "WHITE",
    "ホワイト": "WHITE",
    "RED": "RED",
    "ROSSO": "RED",
    "赤": "RED",
    "レッド": "RED",
    "BLUE": "BLUE",
    "BLU": "BLUE",
    "青": "BLUE",
    "ブルー": "BLUE",
    "BEIGE": "BEIGE",
    "ベージュ": "BEIGE",
    "ECRU": "BEIGE",
    "BROWN": "BROWN",
    "MARRONE": "BROWN",
    "茶": "BROWN",
    "ブラウン": "BROWN",
    "CARAMEL": "BROWN",
    "GREEN": "GREEN",
    "VERDE": "GREEN",
    "緑": "GREEN",
    "グリーン": "GREEN",
    "PINK": "PINK",
    "ROSA": "PINK",
    "ピンク": "PINK",
    "GREY": "GREY",
    "GRAY": "GREY",
    "グレー": "GREY",
    "GRIGIO": "GREY",
    "NAVY": "NAVY",
    "ネイビー": "NAVY",
    "MARINE": "NAVY",
    "GOLD": "GOLD",
    "ゴールド": "GOLD",
    "ORO": "GOLD",
    "SILVER": "SILVER",
    "シルバー": "SILVER",
    "ARGENTO": "SILVER",
}


# ---------------------------------------------------------------------------
# Token / Model extraction
# ---------------------------------------------------------------------------


def extract_model_numbers(name: str) -> set[str]:
    """Extract alphanumeric model numbers (5-12 chars) from product name."""
    upper = name.upper()
    numbers: set[str] = set()
    for m in MODEL_NUMBER_RE.finditer(upper):
        token = m.group(1)
        has_alpha = any(c.isalpha() for c in token)
        has_digit = any(c.isdigit() for c in token)
        if has_alpha and has_digit:
            numbers.add(token)
    return numbers


def extract_tokens(name: str) -> list[str]:
    """Tokenize product name by removing noise words and splitting."""
    clean = name
    for noise in NOISE_WORDS:
        clean = clean.replace(noise, " ")
    clean = re.sub(r"[★☆♪【】!！・/／\s]+", " ", clean)
    tokens = [t for t in clean.split() if len(t) >= 2]
    return tokens


def normalize_color(name: str) -> str | None:
    """Extract and normalize color from product name."""
    upper = name.upper()
    for key, canonical in COLOR_NORMALIZE.items():
        if key in upper:
            return canonical
    return None


def extract_size(name: str) -> str | None:
    """Extract size keyword from product name."""
    m = SIZE_RE.search(name)
    return m.group(0).strip() if m else None


# ---------------------------------------------------------------------------
# Matching score functions
# ---------------------------------------------------------------------------


def cross_site_match_score(
    query_name: str,
    candidate_name: str,
    brand: str | None = None,
    require_brand_in_candidate: bool = False,
) -> float:
    """
    Generalized cross-site matching score between two product names.

    Args:
        query_name: The canonical/product name being searched (e.g. from official site)
        candidate_name: The candidate name to score (e.g. from another site)
        brand: Optional brand name; if provided and require_brand_in_candidate=True,
               the brand must appear in candidate_name
        require_brand_in_candidate: If True, return 0.0 when brand doesn't appear in candidate

    Returns:
        Score 0.0-1.0 indicating match confidence.
        Higher = more likely to be the same product.

    Scoring logic:
    - Model number overlap: +0.4 (strongest signal)
    - Color match: +0.1
    - Token overlap (60% weight) + SequenceMatcher (40% weight)
    - Clamped to max 1.0
    """
    candidate_upper = candidate_name.upper()

    # Optional brand requirement (useful for BUYMA matching)
    if require_brand_in_candidate and brand and brand.upper() not in candidate_upper:
        return 0.0

    # Model number bonus
    query_models = extract_model_numbers(query_name)
    cand_models = extract_model_numbers(candidate_name)
    model_bonus = 0.4 if (query_models and cand_models and query_models & cand_models) else 0.0

    # Color bonus
    query_color = normalize_color(query_name)
    cand_color = normalize_color(candidate_name)
    color_bonus = 0.1 if (query_color and cand_color and query_color == cand_color) else 0.0

    # Token overlap
    query_tokens = extract_tokens(query_name.upper())
    cand_tokens = set(extract_tokens(candidate_upper))

    if not query_tokens:
        return min(1.0, model_bonus + color_bonus)

    hits = sum(1 for t in query_tokens if t in cand_tokens or t in candidate_upper)
    token_score = hits / len(query_tokens)

    # Sequence similarity
    seq_score = SequenceMatcher(None, query_name.upper(), candidate_upper).ratio()

    return min(1.0, 0.6 * token_score + 0.4 * seq_score + model_bonus + color_bonus)


def match_score_buyma(official_name: str, buyma_name: str, brand: str) -> float:
    """
    BUYMA-specific matching score.
    Requires brand to appear in buyma_name (as per original _match_score logic).

    This is the same algorithm as the original _match_score() in buyma_price_scraper.
    Kept as a thin wrapper for backward compatibility with buyma_price_scraper.py.
    """
    return cross_site_match_score(
        query_name=official_name,
        candidate_name=buyma_name,
        brand=brand,
        require_brand_in_candidate=True,
    )
