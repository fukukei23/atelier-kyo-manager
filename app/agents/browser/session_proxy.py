"""
SessionManager から分離されたプロキシ管理ロジック。

プロキシプールの読み込み・サイト別選択・ラウンドロビン・引数生成を担当。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROXY_POOL_PATH = Path("app/config/proxy_pool.json")
MAX_PROXY_RETRIES_PER_RUN = 3

logger = logging.getLogger(__name__)


@dataclass
class ProxyEntry:
    label: str
    server: str
    username: str | None = None
    password: str | None = None
    country: str | None = None
    active: bool = True


class ProxyManager:
    """プロキシプールの読み込み・選択・ラウンドロビンを管理"""

    def __init__(self, proxy_pool_path: Path = PROXY_POOL_PATH) -> None:
        self._pool: dict[str, list[ProxyEntry]] = self._load(proxy_pool_path)
        self._index: dict[str, int] = {}

    @property
    def pool(self) -> dict[str, list[ProxyEntry]]:
        return self._pool

    def get_list_for_site(self, site: str) -> list[ProxyEntry]:
        if not site:
            return []
        candidates: list[str] = [site, site.upper(), site.lower()]
        base = site
        for suffix in ["_OFFICIAL", "-OFFICIAL", "_official", "-official"]:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if base and base != site:
            candidates.extend([base, base.upper(), base.lower()])
        seen: set[str] = set()
        ordered: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        for key in ordered:
            if key.upper() in self._pool:
                return self._pool[key.upper()]
        return self._pool.get("DEFAULT", [])

    def choose_for_site(self, site: str) -> ProxyEntry | None:
        proxies = self.get_list_for_site(site)
        if not proxies:
            return None
        key = (site or "").upper()
        idx = self._index.get(key, 0)
        for _ in range(len(proxies)):
            entry = proxies[idx % len(proxies)]
            idx += 1
            if entry.active:
                self._index[key] = idx % len(proxies)
                return entry
        return None

    def get_for_site(self, site: str, site_config: dict[str, Any], runtime_kwargs: dict[str, Any] | None = None) -> ProxyEntry | None:
        if not self._pool:
            return None
        site_key = site or (site_config.get("id")) or (runtime_kwargs or {}).get("site_name")
        self.get_list_for_site(site_key or "")
        chosen = self.choose_for_site(site_key or "")
        if chosen:
            logger.info("[ProxyManager] chosen proxy for site=%s → %s", site_key, chosen.server)
        return chosen

    @staticmethod
    def build_playwright_arg(entry: ProxyEntry) -> dict[str, str]:
        proxy: dict[str, str] = {"server": entry.server}
        if entry.username:
            proxy["username"] = entry.username
        if entry.password:
            proxy["password"] = entry.password
        return proxy

    def _load(self, path: Path) -> dict[str, list[ProxyEntry]]:
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("[ProxyManager] failed to load proxy pool: %s", e)
            return {}
        pool: dict[str, list[ProxyEntry]] = {}
        for key, cfg in (raw or {}).items():
            entries: list[ProxyEntry] = []
            for p in (cfg or {}).get("proxies") or []:
                try:
                    entries.append(
                        ProxyEntry(
                            label=p.get("label") or f"{key}_proxy",
                            server=p["server"],
                            username=p.get("username"),
                            password=p.get("password"),
                            country=p.get("country"),
                            active=bool(p.get("active", True)),
                        )
                    )
                except Exception as e:
                    logger.warning("[ProxyManager] invalid proxy entry under '%s': %s (%s)", key, p, e)
            if entries:
                pool[key.upper()] = entries
        return pool
