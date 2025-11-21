from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx


class Forward2meClient:
    """Thin async wrapper around the Forward2me REST API."""

    def __init__(self, api_base: str, api_key: str, *, timeout: float = 10.0) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def list_parcels(
        self,
        *,
        status: str = "in_warehouse",
        updated_since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"status": status}
        if updated_since:
            params["updated_since"] = updated_since.isoformat()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.api_base}/api/parcels",
                headers=self._headers(),
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []

    async def get_parcel(self, parcel_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.api_base}/api/parcels/{parcel_id}",
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else {}
