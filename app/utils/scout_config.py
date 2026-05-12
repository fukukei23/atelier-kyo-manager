"""サイト設定レイヤー読み込み（base → overrides → legacy のマージ）"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from dataclasses import fields as dataclass_fields
from pathlib import Path

from app.utils.scout_models import SiteConfig, SiteSelectors

APP_ROOT = Path(__file__).resolve().parents[2]
SITES_DIR = APP_ROOT / "config" / "sites"
CFG_BASE = SITES_DIR / "base.json"
CFG_OVR = SITES_DIR / "overrides.local.json"
CFG_LEGACY = APP_ROOT / "config" / "crawler_sites.json"


def _update_dict_recursive(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _update_dict_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Could not load/parse JSON {path.name}: {e}")
        return {}


def _dict_to_siteconfig(d: dict) -> SiteConfig:
    from app.utils.scout_models import default_sites

    base_obj = SiteConfig(name="", home_url="")
    full_dict = asdict(base_obj)
    full_dict = _update_dict_recursive(full_dict, d)
    sel_dict = asdict(SiteSelectors())
    sel_dict = _update_dict_recursive(sel_dict, full_dict.get("selectors", {}))
    full_dict["selectors"] = SiteSelectors(**sel_dict)
    valid_keys = {f.name for f in dataclass_fields(SiteConfig)}
    final_dict = {k: v for k, v in full_dict.items() if k in valid_keys}
    return SiteConfig(**final_dict)


def load_config_sites() -> list[SiteConfig]:
    from app.utils.scout_models import default_sites

    log = logging.getLogger("ConfigLoader")
    loaded_files = []
    sites_dict_map = {s.name.upper(): asdict(s) for s in default_sites()}
    base_data = _load_json_if_exists(CFG_BASE)
    if base_data:
        loaded_files.append(str(CFG_BASE.relative_to(APP_ROOT)))
        for site_conf in base_data.get("sites", []):
            name = (site_conf.get("name") or "").upper()
            if name in sites_dict_map:
                sites_dict_map[name] = _update_dict_recursive(sites_dict_map[name], site_conf)
    overrides_data = _load_json_if_exists(CFG_OVR)
    if overrides_data:
        loaded_files.append(str(CFG_OVR.relative_to(APP_ROOT)))
        for site_conf in overrides_data.get("sites", []):
            name = (site_conf.get("name") or "").upper()
            if name in sites_dict_map:
                sites_dict_map[name] = _update_dict_recursive(sites_dict_map[name], site_conf)
    legacy_data = _load_json_if_exists(CFG_LEGACY)
    if legacy_data:
        loaded_files.append(str(CFG_LEGACY.relative_to(APP_ROOT)))
        for site_conf in legacy_data.get("sites", []):
            name = (site_conf.get("name") or "").upper()
            if name not in sites_dict_map:
                sites_dict_map[name] = site_conf
                log.info(f"Loaded '{name}' from legacy config as fallback.")
    log.info(f"Config loaded from: {', '.join(loaded_files) if loaded_files else 'defaults only'}")
    return [_dict_to_siteconfig(d) for d in sites_dict_map.values()]
