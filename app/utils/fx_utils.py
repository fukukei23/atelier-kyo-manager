# ==============================================================================
# File: fx_utils.py
# Registry: C:\Users\USER\tools\atelier-kyo-manager\app\utils\fx_utils.py
# Date & Time (JST): 2025-09-16 13:05:00
# Version: 2.1J (Revised)
#
# --- What's New (v2.1J) ---
#  - [Fix] Made the 'manual_table' argument in get_fx_table_jpy optional with a
#    default value of None. This resolves the TypeError when called without this
#    argument, making the function more flexible and robust.
#
# --- 機能 ---
# - ECBから為替を取得（EURベース）→ JPY換算テーブルへ変換
# - ローカルキャッシュ（TTL既定12h）
# - 手動テーブル（--fx-rates）のパース
# - 優先順位: 手動 > 自動(ECB) > なし
# ==============================================================================
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from app.config.constants import FX_CACHE_TTL_HOURS, FX_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


# --- 共通 ---
def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _app_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _cache_paths():
    cache_dir = os.path.join(_app_root(), "instance", "cache")  # 修正: ".." を削除
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir, os.path.join(cache_dir, "ecb_fx.json")


def _load_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[FX] Failed to load JSON from {path}: {e}")
        return None


def _save_json(path: str, payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[FX] Failed to save JSON to {path}: {e}")


# --- 手動レート "USD=155,EUR=170,..." を dict へ ---
def parse_fx_rates_str(rates_str: str) -> dict[str, float]:
    table: dict[str, float] = {}
    if not rates_str:
        return table
    for part in rates_str.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().upper()
        try:
            table[k] = float(v.strip())
        except ValueError:
            continue
    return table


# --- ECB ---
_ECB_XML_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"


def _ecb_fetch_xml(timeout_sec: int = FX_REQUEST_TIMEOUT) -> tuple[str, str]:
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(_ECB_XML_URL, timeout=timeout_sec, context=ctx) as resp:
        xml_bytes = resp.read()
    xml_text = xml_bytes.decode("utf-8", errors="ignore")
    asof = "unknown"
    try:
        root = ET.fromstring(xml_text)
        for node in root.iter():
            if "Cube" in node.tag and node.attrib.get("time"):
                asof = node.attrib["time"]
                break
    except Exception as e:
        logger.warning(f"[FX] Failed to parse ECB XML timestamp: {e}")
    return asof, xml_text


def _ecb_xml_to_eur_table(xml_text: str) -> dict[str, float]:
    table = {"EUR": 1.0}
    try:
        root = ET.fromstring(xml_text)
        for node in root.iter():
            if "Cube" in node.tag and "currency" in node.attrib and "rate" in node.attrib:
                ccy = node.attrib["currency"].upper()
                rate = float(node.attrib["rate"])
                table[ccy] = rate
    except Exception as e:
        logger.warning(f"[FX] Failed to parse ECB XML rates: {e}")
    return table


def build_jpy_table_from_eur(eur_table: dict[str, float]) -> dict[str, float]:
    """
    EURベース → 1通貨単位→JPY の表に変換。
    例: 1 USD = (JPY_per_EUR) / (USD_per_EUR)
    """
    if "JPY" not in eur_table:
        return {}
    jpy_per_eur = eur_table["JPY"]
    out: dict[str, float] = {"JPY": 1.0}
    for ccy, per_eur in eur_table.items():
        if ccy == "JPY":
            continue
        try:
            out[ccy] = jpy_per_eur / per_eur
        except Exception:
            continue
    return out


def get_fx_table_jpy(
    manual_table: dict[str, float] | None = None, auto: bool = True, ttl_hours: int = FX_CACHE_TTL_HOURS
) -> tuple[dict[str, float], dict]:
    """
    最終レート表（1通貨→JPY）とメタ情報を返す。
    優先: manual > auto(ECB+cache) > {}
    meta = {"source": "manual|ecb(live)|ecb(cache)|ecb(cache-fallback)|None", "asof": iso, "cache": bool}
    """
    meta = {"source": None, "asof": None, "cache": None}

    # 1) manual
    if manual_table:
        meta.update({"source": "manual", "asof": _now_iso(), "cache": False})
        return {k.upper(): float(v) for k, v in manual_table.items()}, meta

    # 2) auto
    if auto:
        cache_dir, cache_file = _cache_paths()
        cached = _load_json(cache_file)
        # cache
        if cached:
            try:
                asof_str = cached.get("asof")
                if asof_str:
                    asof = datetime.fromisoformat(asof_str.replace("Z", "+00:00"))
                    if datetime.now(timezone.utc).replace(tzinfo=None) - asof.replace(tzinfo=None) <= timedelta(hours=ttl_hours):
                        tbl = cached.get("table") or {}
                        if tbl:
                            meta.update({"source": "ecb(cache)", "asof": asof_str, "cache": True})
                            return {k.upper(): float(v) for k, v in tbl.items()}, meta
            except Exception as e:
                logger.warning(f"[FX] Failed to parse cache: {e}")
        # live
        try:
            _asof, xml_text = _ecb_fetch_xml()
            eur_table = _ecb_xml_to_eur_table(xml_text)
            jpy_table = build_jpy_table_from_eur(eur_table)
            if jpy_table:
                payload = {"asof": _now_iso(), "table": jpy_table}
                _save_json(cache_file, payload)
                meta.update({"source": "ecb(live)", "asof": payload["asof"], "cache": False})
                return jpy_table, meta
        except Exception as e:
            logger.warning(f"[FX] Live ECB fetch failed: {e}")
            # fallback to old cache
            if cached and cached.get("table"):
                meta.update({"source": "ecb(cache-fallback)", "asof": cached.get("asof"), "cache": True})
                return {k.upper(): float(v) for k, v in cached["table"].items()}, meta

    # 3) nothing
    return {}, meta
